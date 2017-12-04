import numpy as np
# Define conversions in x and y from pixels space to meters
ym_per_pix = 3/80 # meters per pixel in y dimension
xm_per_pix = 3.7/585 # meters per pixel in x dimension

def get_real_lanes_curvature(ploty, leftx, lefty, rightx, righty):
    ploty_scaled = ploty*ym_per_pix
    # Fit new polynomials to x,y in world space
    left_fit_cr,right_fit_cr,dummy = fit_polylines(ploty[-1]+1,leftx, lefty, rightx, righty,x_scale_factor=xm_per_pix,y_scale_factor=ym_per_pix)
    # Calculate the new radii of curvature
    left_curverad, right_curverad = get_polylines_curve(ploty_scaled,left_fit_cr,right_fit_cr)
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    return left_curverad,right_curverad


def fit_polylines(y_n_samples,leftx, lefty, rightx, righty,x_scale_factor=1,y_scale_factor=1):
    # Fit a second order polynomial to each
    ploty = np.linspace(0, y_n_samples - 1, y_n_samples)
    left_fit = np.polyfit(lefty*y_scale_factor, leftx*x_scale_factor, 2)
    right_fit = np.polyfit(righty*y_scale_factor, rightx*x_scale_factor, 2)
    return left_fit, right_fit,ploty


def get_polylines_points(ploty, left_fit,right_fit):
    # Generate x and y values for plotting
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return left_fitx,right_fitx

def get_polylines_curve(ploty,left_fit,right_fit):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad,right_curverad
