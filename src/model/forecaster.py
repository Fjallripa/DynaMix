import torch
import torch.nn.functional as F
from tqdm import tqdm
from .preprocessing import DataPreprocessor


class DynaMixForecaster:
    """
    Forecasting pipeline for DynaMix models with batch processing support.
    """
    
    def __init__(self, model):
        """
        Initialize the forecaster with a DynaMix model.
        
        Args:
            model: DynaMix model instance
        """
        
        self.model = model
        
    
    
    def _init_latent_state(self, initial_condition):
        """
        Initialize the latent state from the initial condition.
        
        Args:
            initial_condition: Initial state of shape (batch_size, N)
            
        Returns:
            Initial latent state z
        """
        
        N = self.model.N

        
        # Initialize latent state
        z = torch.matmul(initial_condition, self.model.B).t()  # (M, batch_size)
        z[:N] = initial_condition.t()
        
        return z
    
    
    
    def _reshape_for_model(self, context, initial_x=None, device=None):
        """
        Prepare and reshape input data for the model.
        Handles tensor conversion, dimension adjustments, and reshaping when feature_dim > model_dim.
        
        Args:
            context: Context data tensor of shape (seq_length, batch_size, feature_dim) or (seq_length, feature_dim)
            initial_x: Optional initial condition of shape (batch_size, feature_dim) or (feature_dim,)
            device: Device to place tensors on
            
        Returns:
            Processed context, initial_x, dimensions, and reshaping metadata
        """            
        
        # Get the dtype from model parameters
        model_dtype = next(self.model.parameters()).dtype           
        
        # Convert to torch tensor if needed
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=model_dtype, device=device)
        elif context.device != device or context.dtype != model_dtype:
            context = context.to(device=device, dtype=model_dtype)
        
        if initial_x is not None and not isinstance(initial_x, torch.Tensor):
            initial_x = torch.tensor(initial_x, dtype=model_dtype, device=device)
        elif initial_x is not None and (initial_x.device != device or initial_x.dtype != model_dtype):
            initial_x = initial_x.to(device=device, dtype=model_dtype)
            
        # Check data dimensions and reshape if needed
        original_dim = context.dim()
        if original_dim == 2:
            context = context.unsqueeze(1)  # (seq_length, feature_dim) -> (seq_length, 1, feature_dim)
        elif original_dim != 3:
            raise ValueError(f"Expected 2D or 3D tensor for context, got shape {context.shape} with {context.dim()} dimensions")
        if initial_x is not None and initial_x.dim() == 1:
            initial_x = initial_x.unsqueeze(0)  # (feature_dim,) -> (1, feature_dim)
            if initial_x.shape[1] != context.shape[2]:
                raise ValueError(f"Initial condition has {initial_x.shape[1]} features, but context has {context.shape[2]} features")
        
        # Data shape
        seq_length, batch_size, feature_dim = context.shape
        
        # Check if reshaping is needed for model dimension
        if feature_dim <= self.model.N:
            return context, initial_x, (batch_size, feature_dim, False, None, None, original_dim)


        # Case: feature_dim > self.model.N            
        print(f"Warning: Input feature dimension {feature_dim} exceeds model dimension {self.model.N}. "
              f"This may lead to performance degradation."
              f"Reshaping data to treat each feature as separate time series.")
        
        # Store original dimensions for reshaping back later
        original_batch_size = batch_size
        original_feature_dim = feature_dim
        
        # Reshape context to (seq_length, batch_size * feature_dim, 1)
        transposed = context.permute(0, 2, 1)
        new_batch_size = batch_size * feature_dim
        reshaped_context = transposed.reshape(seq_length, new_batch_size, 1)
        
        # Similarly reshape initial_x if provided
        reshaped_initial_x = initial_x
        if initial_x is not None:
            # Reshape from (batch_size, feature_dim) to (batch_size * feature_dim, 1)
            reshaped_initial_x = initial_x.transpose(0, 1).reshape(new_batch_size, 1)        
        
        return (reshaped_context, reshaped_initial_x, 
            (new_batch_size, 1, True, original_batch_size, original_feature_dim, original_dim)
        )
    
    
        
    def _reshape_to_original(self, output, reshape_metadata):
        """
        Reshape output back to original dimensions.
        Handles both high-dimensional reshaping and 2D input restoration.
        
        Args:
            output: Model output of shape (T, batch_size, N)
            reshape_metadata: Tuple containing (was_reshaped, original_batch_size, original_feature_dim, original_dim)
            
        Returns:
            Output with original shape restored
        """
        
        _, _, was_reshaped, original_batch_size, original_feature_dim, original_dim = reshape_metadata
        
        
        # Step 1: Reshape back to original dimensions if needed
        if was_reshaped:
            # Current shape: (T, batch_size=original_batch_size*original_feature_dim, 1)
            T = output.shape[0]
            
            # First reshape to (T, original_feature_dim, original_batch_size)
            # by treating the batch dimension as (original_feature_dim, original_batch_size)
            reshaped = output.reshape(T, original_feature_dim, original_batch_size, -1)
            
            # Then permute to (T, original_batch_size, original_feature_dim)
            output = reshaped.permute(0, 2, 1, 3).squeeze(-1)
        
        
        # Step 2: If input was 2D, remove batch dimension from output
        if original_dim == 2 and output.shape[1] == 1:
            output = output.squeeze(1)
            
        return output
    
    
        
    @torch.no_grad()
    def forecast(self, context, horizon, preprocessing_method="pos_embedding", 
                standardize=True, fit_nonstationary=False, initial_x=None):
        """
        Efficient batched forecasting with the DynaMix model.
        
        This method implements a complete forecasting pipeline including:
        - Data preprocessing (Box-Cox, detrending, standardization)
        - Embedding techniques for dimensionality matching
        - DynaMix model prediction
        - Data postprocessing (inverse transformations)
        
        Args:
            context: Context data tensor of shape (seq_length, batch_size, feature_dim) or (seq_length, feature_dim)
            horizon: Forecast horizon (number of steps to predict)
            preprocessing_method: Data preprocessing method ('pos_embedding', 'zero_embedding',
                                  'delay_embedding', or 'delay_embedding_random') (default: 'pos_embedding')
            standardize: Whether to standardize the data (default: True)
            fit_nonstationary: Whether to fit a non-stationary time series (default: False)
            initial_x: Optional initial condition of shape (batch_size, feature_dim) or (feature_dim,)
            
        Returns:
            Predicted sequence of shape (horizon, batch_size, feature_dim)
        """
        
        # Get model dimensions
        M = self.model.M
        N = self.model.N
        device = context.device if isinstance(context, torch.Tensor) else self.model.B.device
        model_dtype = next(self.model.parameters()).dtype
        
        
        # Apply context reshaping if needed
        context, initial_x, shape_metadata = self._reshape_for_model(context, initial_x, device)
        
        
        # Create data preprocessor
        preprocessor = DataPreprocessor(
            standardize=standardize,
            box_cox=fit_nonstationary,
            detrending=fit_nonstationary,
            preprocessing_method=preprocessing_method
        )

        
        # Step 1: Apply preprocessing pipeline
        context_embedded, initial_condition = preprocessor.preprocess(context, self.model.N, initial_x)
        
        
        # Step 2: Initialize latent state
        z = self._init_latent_state(initial_condition)
        
        
        # Step 3: Perform forecasting loop
        Z_gen = torch.empty(horizon, M, shape_metadata[0], device=device, dtype=model_dtype)
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda'):
            precomputed_cnn = self.model.precompute_cnn(context_embedded)
            for t in range(horizon):
                z = self.model(z, context_embedded, precomputed_cnn=precomputed_cnn)
                Z_gen[t] = z

        
        # Step 4: Apply observation generation
        output = Z_gen[:, :shape_metadata[1], :].permute(0, 2, 1)  # (horizon, batch_size, feature_dim)
        
        
        # Step 5: Apply inverse data transformations (e.g. standardization, ...)
        output = preprocessor.postprocess(output)
        
        
        # Step 6: Reshape back to original dimensions if needed
        output = self._reshape_to_original(output, shape_metadata)
        
        
        return output



    @torch.no_grad()
    def forecast_att_noise(self, context, horizon, num_samples, ensemble_forecast_length=1, attention_noise=True,
        preprocessing_method="pos_embedding", standardize=True, fit_nonstationary=False, initial_x=None
    ):
        """
        **Variant of DynamixForecaster.forecast()**
        This variant introduces attention noise into each batch element and returns a resulting ensemble forecast 
        `ensemble_forecast_length` timesteps into the future for each timestep it actually moves forward. Each timestep,
        the median forecast of the batch is used as input for the next step.

        Extra Args:
            num_samples: int >= 1. size of the batch for the ensemble forecast. each element has their own attention noise
            ensemble_forecast_length: int >=1. how many steps into the future to do the ensemble forecast *at each time step*. 
                returns last timestep of each forecast as output but uses the first forecast step as input for the next 
                forecast.
            attention_noise: boolean.

        
        **unchanged docstring of normal forecast()**
        Efficient batched forecasting with the DynaMix model.
        
        This method implements a complete forecasting pipeline including:
        - Data preprocessing (Box-Cox, detrending, standardization)
        - Embedding techniques for dimensionality matching
        - DynaMix model prediction
        - Data postprocessing (inverse transformations)
        
        Args:
            context: Context data tensor of shape (seq_length, batch_size, feature_dim) or (seq_length, feature_dim)
            horizon: Forecast horizon (number of steps to predict)
            preprocessing_method: Data preprocessing method ('pos_embedding', 'zero_embedding',
                                  'delay_embedding', or 'delay_embedding_random') (default: 'pos_embedding')
            standardize: Whether to standardize the data (default: True)
            fit_nonstationary: Whether to fit a non-stationary time series (default: False)
            initial_x: Optional initial condition of shape (batch_size, feature_dim) or (feature_dim,)
            
        Returns:
            Predicted sequence of shape (horizon, batch_size, feature_dim)
        """
        
        # Get model dimensions
        M = self.model.M
        N = self.model.N
        device = context.device if isinstance(context, torch.Tensor) else self.model.B.device
        model_dtype = next(self.model.parameters()).dtype     
        
        # Apply context reshaping if needed
        context, initial_x, shape_metadata = self._reshape_for_model(context, initial_x, device)

        # Create a batch of num_samples out of the context data
        if shape_metadata[0] > 1:   # check if batch_size > 1
            raise ValueError(f"forecast_att_noise() only works for len(context.size()) == 2 (i.e. no batches).")
        shape_metadata = tuple([num_samples, *shape_metadata[1:]])   # changing batch_size from 1 to num_samples
        context = context.repeat(1, shape_metadata[0], 1)
         
        # Create data preprocessor
        preprocessor = DataPreprocessor(
            standardize=standardize,
            box_cox=fit_nonstationary,
            detrending=fit_nonstationary,
            preprocessing_method=preprocessing_method
        )

        
        # Step 1: Apply preprocessing pipeline
        context_embedded, initial_condition = preprocessor.preprocess(context, self.model.N, initial_x)
                
        # Step 2: Initialize latent state
        z = self._init_latent_state(initial_condition)   # (feature_dim, batch_size)
           
        # Step 3: Perform forecasting loop with uncertainty sampling
        Z_gen = torch.empty(horizon, M, shape_metadata[0], device=device, dtype=model_dtype)
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda'):
            precomputed_cnn = self.model.precompute_cnn(context_embedded)
            for t in tqdm(range(horizon)):
                #z = z[:,0:1].repeat(1, shape_metadata[0])   # input of each step is first of prev. output batch
                z = torch.quantile(z, 0.5, dim=1, keepdim=True).repeat(1, shape_metadata[0])  
                    # input of each step is median of prev. output batch

                # forecast next step
                z = self.model(z, context_embedded, 
                               precomputed_cnn=precomputed_cnn, 
                               attention_noise=attention_noise)

                # log observations
                predicted_batch = z
                if ensemble_forecast_length > 1:
                    for i in range(ensemble_forecast_length-1):
                        predicted_batch = self.model(predicted_batch, context_embedded, precomputed_cnn=precomputed_cnn)
                Z_gen[t] = predicted_batch
   
        # Step 4: Apply observation generation
        output = Z_gen[:, :shape_metadata[1], :].permute(0, 2, 1)  # (horizon, batch_size, feature_dim)
          
        # Step 5: Apply inverse data transformations (e.g. standardization, ...)
        output = preprocessor.postprocess(output)
                
        # Step 6: Reshape back to original dimensions if needed
        output = self._reshape_to_original(output, shape_metadata)
        
        
        return output



    #@torch.no_grad()
    def forecast_posterior(self, context, horizon, num_samples, prior_size, prior_positions=True,
        preprocessing_method="pos_embedding", standardize=True, fit_nonstationary=False, initial_x=None
    ):
        """
        **Variant of DynamixForecaster.forecast()**
        [Insert description]
        This variant only works on 3D data!

        Extra Args:
            num_samples: int > 1. size of the batch for the ensemble forecast. each element has their own attention noise
            prior_size: float, usually smaller than 0.05. size of the epsilon ball prior around a (normalized) point x_t.
            prior_position: bool or 1D boolean array of length `horizon`. 
                if bool then all points in the forecast will get a prior. if array, then only points where the value is True
                will get a prior.
        
        
        **unchanged docstring of normal forecast()**
        Efficient batched forecasting with the DynaMix model.
        
        This method implements a complete forecasting pipeline including:
        - Data preprocessing (Box-Cox, detrending, standardization)
        - Embedding techniques for dimensionality matching
        - DynaMix model prediction
        - Data postprocessing (inverse transformations)
        
        Args:
            context: Context data tensor of shape (seq_length, batch_size, feature_dim) or (seq_length, feature_dim)
            horizon: Forecast horizon (number of steps to predict)
            preprocessing_method: Data preprocessing method ('pos_embedding', 'zero_embedding',
                                  'delay_embedding', or 'delay_embedding_random') (default: 'pos_embedding')
            standardize: Whether to standardize the data (default: True)
            fit_nonstationary: Whether to fit a non-stationary time series (default: False)
            initial_x: Optional initial condition of shape (batch_size, feature_dim) or (feature_dim,)
            
        Returns:
            Predicted sequence of shape (horizon, batch_size, feature_dim)
        """


        # Prep work
        # -------------
        ## Get model dimensions
        M = self.model.M
        N = self.model.N
        device = context.device if isinstance(context, torch.Tensor) else self.model.B.device
        model_dtype = next(self.model.parameters()).dtype
        
        ## Apply context reshaping if needed (this function variant is not compatible with such reshapings)
        context, initial_x, shape_metadata = self._reshape_for_model(context, initial_x, device)
        
        ## Check that data dimensions are sound.
        if shape_metadata[0] > 1:   # check if batch_size > 1
            raise ValueError(f"forecast_posterior() only works without batches (i.e. a 2D context tensor).")
        if context.size(2) != 3:
            raise ValueError(f"context data has {context.size(2)} dimensions but forecast_posterior() can only handle {self.model.N}D data at the moment (the model's data dimension).")
        if num_samples < 2:
            raise ValueError(f"forecast_posterior() only works with a sample size `num_samples` > 1")
        
        ## Create a batch of num_samples out of the context data
        shape_metadata = tuple([num_samples, *shape_metadata[1:]])   # changing batch_size from 1 to num_samples
        context = context.repeat(1, shape_metadata[0], 1)
        
        ## Create data preprocessor
        preprocessor = DataPreprocessor(
            standardize=standardize,
            box_cox=fit_nonstationary,
            detrending=fit_nonstationary,
            preprocessing_method=preprocessing_method
        )


        # Forecasting
        # -----------
        ## Step 1: Apply preprocessing pipeline
        context_embedded, initial_condition = preprocessor.preprocess(context, self.model.N, initial_x)
        
        ## Step 2: Initialize latent state
        z = self._init_latent_state(initial_condition)
          
        ## Step 3: Perform forecasting loop
        Z_posterior = torch.empty(horizon, M, num_samples, device=device, dtype=model_dtype)
        Z_prior = Z_posterior.clone()
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda'):
            def model(z_data):
                return self.model(z_data, context_embedded, 
                    precomputed_cnn=self.model.precompute_cnn(context_embedded), attention_noise=False
                )
            for t in tqdm(range(horizon)):
                if prior_positions == True or prior_positions[t] == True:
                    z = z[:, 0].unsqueeze(1).repeat(1, num_samples)
                    z[:N, 1:] += prior_size * torch.randn(N, num_samples-1, dtype=z.dtype, device=z.device)
                Z_prior[t] = z
                z = model(z)
                Z_posterior[t] = z
        
        ## Step 4: Apply observation generation
        Z_prior = Z_prior.permute(0, 2, 1)   # (horizon, M, num_samples) -> (horizon, num_samples, M)
        Z_posterior = Z_posterior.permute(0, 2, 1)
        X_prior = Z_prior[:,:,:N]
        X_posterior = Z_posterior[:,:,:N]
             
        ## Step 5: Apply inverse data transformations (standardization)
        X_prior = preprocessor.postprocess(X_prior)
        X_posterior = preprocessor.postprocess(X_posterior)
          
   
        return Z_prior, Z_posterior, X_prior, X_posterior



    
    def forecast_jacobian(self, context, horizon, num_samples, prior_size, prior_positions=True,
        preprocessing_method="pos_embedding", standardize=True, fit_nonstationary=False, initial_x=None
    ):
        """
        **Variant of DynamixForecaster.forecast_posterior()**
        [Insert description]
        This variant only works on 3D data!

        Extra Args:
            num_samples: int > 1. size of the batch for the ensemble forecast. each element has their own attention noise
            prior_size: float, usually smaller than 0.05. size of the epsilon ball prior around a (normalized) point x_t.
            prior_position: bool or 1D boolean array of length `horizon`. 
                if bool then all points in the forecast will get a prior. if array, then only points where the value is True
                will get a prior.
        
        
        **unchanged docstring of normal forecast()**
        Efficient batched forecasting with the DynaMix model.
        
        This method implements a complete forecasting pipeline including:
        - Data preprocessing (Box-Cox, detrending, standardization)
        - Embedding techniques for dimensionality matching
        - DynaMix model prediction
        - Data postprocessing (inverse transformations)
        
        Args:
            context: Context data tensor of shape (seq_length, batch_size, feature_dim) or (seq_length, feature_dim)
            horizon: Forecast horizon (number of steps to predict)
            preprocessing_method: Data preprocessing method ('pos_embedding', 'zero_embedding',
                                  'delay_embedding', or 'delay_embedding_random') (default: 'pos_embedding')
            standardize: Whether to standardize the data (default: True)
            fit_nonstationary: Whether to fit a non-stationary time series (default: False)
            initial_x: Optional initial condition of shape (batch_size, feature_dim) or (feature_dim,)
            
        Returns:
            Predicted sequence of shape (horizon, batch_size, feature_dim)
        """

        
        # Prep work
        # -------------
        ## Get model dimensions
        M = self.model.M
        N = self.model.N
        device = context.device if isinstance(context, torch.Tensor) else self.model.B.device
        model_dtype = next(self.model.parameters()).dtype
        
        ## Apply context reshaping if needed (this function variant is not compatible with such reshapings)
        context, initial_x, shape_metadata = self._reshape_for_model(context, initial_x, device)
        
        ## Check that data dimensions are sound.
        if shape_metadata[0] > 1:   # check if batch_size > 1
            raise ValueError(f"forecast_posterior() only works without batches (i.e. a 2D context tensor).")
        if context.size(2) != 3:
            raise ValueError(f"context data has {context.size(2)} dimensions but forecast_posterior() can only handle {self.model.N}D data at the moment (the model's data dimension).")
        if num_samples < 2:
            raise ValueError(f"forecast_posterior() only works with a sample size `num_samples` > 1")
        
        ## Create a batch of num_samples out of the context data
        shape_metadata = tuple([num_samples, *shape_metadata[1:]])   # changing batch_size from 1 to num_samples
        context = context.repeat(1, shape_metadata[0], 1)
        
        ## Create data preprocessor
        preprocessor = DataPreprocessor(
            standardize=standardize,
            box_cox=fit_nonstationary,
            detrending=fit_nonstationary,
            preprocessing_method=preprocessing_method
        )


        # Forecasting
        # -----------
        ## Step 1: Apply preprocessing pipeline
        context_embedded, initial_condition = preprocessor.preprocess(context, self.model.N, initial_x)
        
        ## Step 2: Initialize latent state
        z = self._init_latent_state(initial_condition)
          
        ## Step 3: Perform forecasting loop
        Z_posterior = torch.empty(horizon, M, num_samples, device=device, dtype=model_dtype)
        Z_prior = Z_posterior.clone()
        Jacobians = []
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda'):
            def model(z_data):
                return self.model(z_data, context_embedded, 
                    precomputed_cnn=self.model.precompute_cnn(context_embedded), attention_noise=False
                )
            for t in tqdm(range(horizon)):
                if prior_positions == True or prior_positions[t] == True:
                    z = z[:, 0].unsqueeze(1).repeat(1, num_samples)
                    z[:N, 1:] += prior_size * torch.randn(N, num_samples-1, dtype=z.dtype, device=z.device)
                Z_prior[t] = z
                Jacobians.append(torch.autograd.functional.jacobian(model, z))
                z = model(z)
                Z_posterior[t] = z
        
        ## Step 4: Apply observation generation
        Z_prior = Z_prior.permute(0, 2, 1)   # (horizon, M, num_samples) -> (horizon, num_samples, M)
        Z_posterior = Z_posterior.permute(0, 2, 1)
        X_prior = Z_prior[:,:,:N]
        X_posterior = Z_posterior[:,:,:N]
             
        ## Step 5: Apply inverse data transformations (standardization)
        X_prior = preprocessor.postprocess(X_prior)
        X_posterior = preprocessor.postprocess(X_posterior)
          
   
        return Z_prior, Z_posterior, X_prior, X_posterior, torch.stack(Jacobians)
