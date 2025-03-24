import numpy as np
from numba import njit

@njit
def _convolve1d(x, filt):
    """
    Optimized 1D convolution for series data with numba.
    
    Parameters
    ----------
    x : ndarray, shape (N,)
        Input array (flat)
    filt : ndarray, shape (M,)
        Filter coefficients (flat)
        
    Returns
    -------
    ndarray, shape (N-M+1,)
        Filtered array
    """
    n_x = len(x)
    n_filt = len(filt)
    n_out = n_x - n_filt + 1
    
    result = np.zeros(n_out)
    
    # Pre-compute reversed filter (make it contiguous)
    filt_rev = np.ascontiguousarray(filt[::-1])
    
    # Loop optimization for series data
    for i in range(n_out):
        # Create a contiguous copy of the slice for better performance
        x_slice = np.ascontiguousarray(x[i:i+n_filt])
        result[i] = np.dot(x_slice, filt_rev)
            
    return result

@njit
def _pad_nans_1d(arr, trim_head=0, trim_tail=0):
    """
    Pad 1D array with NaNs at the beginning and/or end.
    
    Parameters
    ----------
    arr : ndarray, shape (N,)
        Input array
    trim_head : int
        Number of NaNs to pad at the beginning
    trim_tail : int
        Number of NaNs to pad at the end
        
    Returns
    -------
    ndarray, shape (N+trim_head+trim_tail,)
        Padded array
    """
    n = len(arr)
    result = np.empty(n + trim_head + trim_tail)
    result.fill(np.nan)
    
    result[trim_head:trim_head+n] = arr
        
    return result

def convolution_filter(x, filt, nsides=2):
    """
    Linear filtering via convolution optimized for series data with numba.
    
    Parameters
    ----------
    x : array_like, shape (N,) or (N,1)
        Time series data
    filt : array_like, shape (M,)
        Filter coefficients
    nsides : int, optional
        If 2, a centered moving average is computed.
        If 1, the filter is for past values only.
        
    Returns
    -------
    ndarray, shape (N,) or (N,1)
        Filtered array with the same shape as input
    """
    # Convert inputs to flat numpy arrays
    x_array = np.asarray(x).squeeze()
    filt_array = np.asarray(filt).squeeze()
    
    # Store original shape for later
    original_shape = x.shape
    
    # Handle trim values based on nsides
    if nsides == 1:
        trim_head = len(filt_array) - 1
        trim_tail = 0
    elif nsides == 2:
        filt_len = len(filt_array)
        trim_head = int(np.ceil(filt_len/2.) - 1)
        trim_tail = int(np.ceil(filt_len/2.) - filt_len % 2)
    else:
        raise ValueError("nsides must be 1 or 2")
    
    # Apply convolution
    if nsides == 2:
        # For 2-sided filter, use the filter as is
        result = _convolve1d(x_array, filt_array)
    elif nsides == 1:
        # For 1-sided filter, prepend a zero
        temp_filt = np.zeros(len(filt_array) + 1)
        temp_filt[1:] = filt_array
        result = _convolve1d(x_array, temp_filt)
    
    # Pad with NaNs as needed
    if trim_head > 0 or trim_tail > 0:
        result = _pad_nans_1d(result, trim_head, trim_tail)
    
    # Reshape to match input if necessary
    if len(original_shape) > 1:
        result = result.reshape(-1, 1)
    
    return result

@njit
def _extrapolate_trend_1d(trend, npoints):
    """
    Extrapolate trend for 1D arrays.
    """
    result = trend.copy()
    
    # Find first non-NaN index
    front = 0
    while front < len(result) and np.isnan(result[front]):
        front += 1
        
    # Find last non-NaN index
    back = len(result) - 1
    while back >= 0 and np.isnan(result[back]):
        back -= 1
        
    if front >= back:
        return result  # All NaNs or only one valid point
    
    # Define regression points at the beginning
    front_last = min(front + npoints, back)
    x_front = np.arange(front, front_last)
    y_front = result[front:front_last]
    
    # Linear regression at the beginning
    sum_x = np.sum(x_front)
    sum_y = np.sum(y_front)
    sum_xy = np.sum(x_front * y_front)
    sum_xx = np.sum(x_front * x_front)
    n_pts = len(x_front)
    
    # Calculate slope (k) and intercept (n)
    det = n_pts * sum_xx - sum_x * sum_x
    k = (n_pts * sum_xy - sum_x * sum_y) / det if det != 0 else 0
    n = (sum_y - k * sum_x) / n_pts
    
    # Extrapolate beginning points
    for i in range(front):
        result[i] = k * i + n
        
    # Define regression points at the end
    back_first = max(front, back - npoints)
    x_back = np.arange(back_first, back)
    y_back = result[back_first:back]
    
    # Linear regression at the end
    sum_x = np.sum(x_back)
    sum_y = np.sum(y_back)
    sum_xy = np.sum(x_back * y_back)
    sum_xx = np.sum(x_back * x_back)
    n_pts = len(x_back)
    
    # Calculate slope (k) and intercept (n)
    det = n_pts * sum_xx - sum_x * sum_x
    k = (n_pts * sum_xy - sum_x * sum_y) / det if det != 0 else 0
    n = (sum_y - k * sum_x) / n_pts
    
    # Extrapolate end points
    for i in range(back + 1, len(result)):
        result[i] = k * i + n
    
    return result

@njit
def _extrapolate_trend_2d(trend, npoints):
    """
    Extrapolate trend for 2D arrays.
    """
    result = trend.copy()
    n_rows, n_cols = result.shape
    
    for col in range(n_cols):
        # Find first non-NaN index
        front = 0
        while front < n_rows and np.isnan(result[front, col]):
            front += 1
            
        # Find last non-NaN index
        back = n_rows - 1
        while back >= 0 and np.isnan(result[back, col]):
            back -= 1
            
        if front >= back:
            continue  # All NaNs or only one valid point
        
        # Define regression points at the beginning
        front_last = min(front + npoints, back)
        x_front = np.arange(front, front_last)
        y_front = result[front:front_last, col]
        
        # Linear regression at the beginning
        sum_x = np.sum(x_front)
        sum_y = np.sum(y_front)
        sum_xy = np.sum(x_front * y_front)
        sum_xx = np.sum(x_front * x_front)
        n_pts = len(x_front)
        
        # Calculate slope (k) and intercept (n)
        det = n_pts * sum_xx - sum_x * sum_x
        k = (n_pts * sum_xy - sum_x * sum_y) / det if det != 0 else 0
        n = (sum_y - k * sum_x) / n_pts
        
        # Extrapolate beginning points
        for i in range(front):
            result[i, col] = k * i + n
            
        # Define regression points at the end
        back_first = max(front, back - npoints)
        x_back = np.arange(back_first, back)
        y_back = result[back_first:back, col]
        
        # Linear regression at the end
        sum_x = np.sum(x_back)
        sum_y = np.sum(y_back)
        sum_xy = np.sum(x_back * y_back)
        sum_xx = np.sum(x_back * x_back)
        n_pts = len(x_back)
        
        # Calculate slope (k) and intercept (n)
        det = n_pts * sum_xx - sum_x * sum_x
        k = (n_pts * sum_xy - sum_x * sum_y) / det if det != 0 else 0
        n = (sum_y - k * sum_x) / n_pts
        
        # Extrapolate end points
        for i in range(back + 1, n_rows):
            result[i, col] = k * i + n
    
    return result

def extrapolate_trend(trend, npoints):
    """
    Replace NaN values on trend's end-points with least-squares extrapolated values.
    
    Parameters
    ----------
    trend : ndarray
        Trend component with NaN values at the edges
    npoints : int
        Number of points to use for extrapolation
        
    Returns
    -------
    ndarray
        Trend with extrapolated values
    """
    # Determine dimensionality and call appropriate function
    if trend.ndim == 1:
        return _extrapolate_trend_1d(trend, npoints)
    else:
        return _extrapolate_trend_2d(trend, npoints)
    
def add_trend(x, trend="c", prepend=True, has_constant="skip"):
    """
    Add a trend and/or constant to a NumPy array.

    Parameters
    ----------
    x : array_like
        Original array of data (assumed to be a NumPy array).
    trend : str {'n', 'c', 't', 'ct', 'ctt'}
        The trend to add: 'n' (none), 'c' (constant), 't' (linear trend),
        'ct' (constant and linear), 'ctt' (constant, linear, quadratic).
    prepend : bool
        If True, prepend trend columns; otherwise, append them.
    has_constant : str {'raise', 'add', 'skip'}
        Controls behavior when trend includes 'c' and x has a constant column:
        'raise' raises an error, 'skip' skips adding the constant, 'add' adds it.

    Returns
    -------
    ndarray
        The original data with added trend columns.
    """
    x = np.asarray(x)  # Faster than asanyarray since we don't need subclass preservation
    nobs = x.shape[0]
    
    # Early return for 'n' trend
    if trend == "n":
        return x.copy()
    
    # Determine which trend columns to add
    add_const = 'c' in trend
    add_trend = 't' in trend
    add_squared = trend == 'ctt'
    
    # Check for existing constant if needed
    if add_const and has_constant in ("raise", "skip"):
        # More efficient constant detection
        if x.ndim == 1:
            # For 1D arrays
            is_const = (x == x[0]).all()
            has_const = is_const and x[0] != 0
        else:
            # For 2D arrays
            const_cols = np.all(x == x[0:1], axis=0) & (x[0] != 0)
            has_const = np.any(const_cols)
        
        if has_const:
            if has_constant == "raise":
                if x.ndim == 1:
                    raise ValueError(f"x is constant. Adding a constant with trend='{trend}' is not allowed.")
                else:
                    const_cols_indices = np.where(const_cols)[0]
                    const_cols_str = ", ".join(map(str, const_cols_indices))
                    raise ValueError(f"x contains constant columns: {const_cols_str}. Adding a constant with trend='{trend}' is not allowed.")
            elif has_constant == "skip":
                add_const = False
    
    # Prepare trend columns
    trend_cols = []
    
    if add_const:
        trend_cols.append(np.ones((nobs, 1)))
    
    if add_trend:
        trend_cols.append(np.arange(1, nobs + 1).reshape(-1, 1))
    
    if add_squared:
        # Use broadcasting for efficient squaring
        trend_cols.append((np.arange(1, nobs + 1)**2).reshape(-1, 1))
    
    # If no columns to add, return original
    if not trend_cols:
        return x.copy()
    
    # Combine trend columns
    trendarr = np.hstack(trend_cols) if len(trend_cols) > 1 else trend_cols[0]
    
    # Handle 1D arrays correctly
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # Concatenate based on prepend
    if prepend:
        return np.hstack([trendarr, x])  # trendarr first, then x
        
    else:
        return np.hstack([x, trendarr])  # x first, then trendarr

