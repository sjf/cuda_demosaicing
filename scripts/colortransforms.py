#!/usr/bin/python

#-------------------------------------
# Code is given as is.
# Use at your own risk and discretion.
#
# The formulas for the functions defined below was obtained from the EasyRGB website: http://www.easyrgb.com/math.html
# I've just taken the "neutral programming function" and written it in Python.
#-------------------------------------


ref_X =  95.047
ref_Y = 100.000
ref_Z = 108.883

#-------------------------------------------------#
def rgb_to_xyz(R, G, B):
    '''
    Convert from RGB to XYZ.
    '''
    var_R = ( R / 255.)
    var_G = ( G / 255.)
    var_B = ( B / 255.)

    if var_R > 0.04045:
        var_R = ( ( var_R + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_R /= 12.92

    if var_G > 0.04045:
        var_G = ( ( var_G + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_G /= 12.92
    if var_B > 0.04045:
        var_B = ( ( var_B + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_B /= 12.92

    var_R *= 100
    var_G *= 100
    var_B *= 100

    #Observer. = 2 deg, Illuminant = D65
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    return X,Y,Z


#-------------------------------------------------#
def xyz_to_rgb(X, Y, Z):
    '''
    Convert from XYZ to RGB.
    '''
    var_X = X / 100.
    var_Y = Y / 100.
    var_Z = Z / 100.

    var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415
    var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570

    if var_R > 0.0031308: 
        var_R = 1.055 * ( var_R ** ( 1. / 2.4 ) ) - 0.055
    else:
        var_R *= 12.92

    if var_G > 0.0031308:
        var_G = 1.055 * ( var_G ** ( 1. / 2.4 ) ) - 0.055
    else:
        var_G *= 12.92

    if var_B > 0.0031308:
        var_B = 1.055 * ( var_B ** ( 1. / 2.4 ) ) - 0.055
    else:
        var_B *= 12.92

    R = var_R * 255
    G = var_G * 255
    B = var_B * 255

    return R, G, B


#-------------------------------------------------#
def xyz_to_cielab(X, Y, Z):
    '''
    Convert from XYZ to CIE-L*a*b*
    '''
    var_X = X / ref_X
    var_Y = Y / ref_Y
    var_Z = Z / ref_Z

    if var_X > 0.008856:
        var_X **= ( 1./3. )
    else:
        var_X = ( 7.787 * var_X ) + ( 16. / 116. )
    if var_Y > 0.008856:
        var_Y **= ( 1./3. )
    else:
        var_Y = ( 7.787 * var_Y ) + ( 16. / 116. )
    if var_Z > 0.008856:
        var_Z **= ( 1./3. )
    else:
        var_Z = ( 7.787 * var_Z ) + ( 16. / 116. )

    CIE_L = ( 116 * var_Y ) - 16.
    CIE_a = 500. * ( var_X - var_Y )
    CIE_b = 200. * ( var_Y - var_Z )

    return CIE_L, CIE_a, CIE_b


#-------------------------------------------------#
def cielab_to_xyz(CIE_L, CIE_a, CIE_b):
    '''
    Convert from CIE-L*a*b* to XYZ 
    '''

    var_Y = ( CIE_L + 16. ) / 116.
    var_X = CIE_a / 500. + var_Y
    var_Z = var_Y - CIE_b / 200.

    if var_Y**3 > 0.008856:
        var_Y **= 3.
    else:
        var_Y = ( var_Y - 16. / 116. ) / 7.787

    if var_X**3 > 0.008856:
        var_X **= 3.
    else:
        var_X = ( var_X - 16. / 116. ) / 7.787

    if var_Z**3 > 0.008856:
        var_Z **= 3
    else:
        var_Z = ( var_Z - 16. / 116. ) / 7.787

    X = ref_X * var_X     
    Y = ref_Y * var_Y     
    Z = ref_Z * var_Z     

    return X,Y,Z

#-------------------------------------------------#
def rgb_to_cielab(R, G, B):
    '''
    Convert from RGB to CIE-L*a*b*.
    '''
    X,Y,Z = rgb_to_xyz(R,G,B)
    return xyz_to_cielab(X,Y,Z)

#-------------------------------------------------#
def cielab_to_rgb(CIE_L, CIE_a, CIE_b):
    '''
    Convert from CIE-L*a*b* to RGB.
    '''
    X,Y,Z = cielab_to_xyz(CIE_L, CIE_a, CIE_b)
    return xyz_to_rgb(X,Y,Z)

#-------------------------------------------------#
def rgb_to_hsv(R, G, B):
    '''
    Convert from RGB to HSV.
    '''
    var_R = ( R / 255.)
    var_G = ( G / 255.)
    var_B = ( B / 255.)

    var_Min = min( var_R, var_G, var_B )    # Min. value of RGB
    var_Max = max( var_R, var_G, var_B )    # Max. value of RGB
    del_Max = var_Max - var_Min             # Delta RGB value

    V = var_Max

    if del_Max == 0:    # This is a gray, no chroma...
        H = 0
        S = 0
    else:               # Chromatic data...
        S = del_Max / var_Max
        
        del_R = ( ( ( var_Max - var_R ) / 6.) + ( del_Max / 2.) ) / del_Max
        del_G = ( ( ( var_Max - var_G ) / 6.) + ( del_Max / 2.) ) / del_Max
        del_B = ( ( ( var_Max - var_B ) / 6.) + ( del_Max / 2.) ) / del_Max
        
        if var_R == var_Max:
            H = del_B - del_G
        elif var_G == var_Max:
            H = ( 1. / 3. ) + del_R - del_B
        elif var_B == var_Max:
            H = ( 2. / 3. ) + del_G - del_R
        
        if H < 0:
            H += 1
        if H > 1:
            H -= 1

    return H,S,V


#-------------------------------------------------#
def hsv_to_rgb(H, S, V):
    '''
    Convert from HSV to RGB.
    '''
    if S == 0:
        R = V * 255.
        G = V * 255.
        B = V * 255.

    else:
        var_h = H * 6.
        if var_h == 6:
            var_h = 0      # H must be < 1
        var_i = int( var_h )
        var_1 = V * ( 1. - S )
        var_2 = V * ( 1. - S * ( var_h - var_i ) )
        var_3 = V * ( 1. - S * ( 1. - ( var_h - var_i ) ) )


        if var_i == 0:
            var_r = V
            var_g = var_3
            var_b = var_1

        elif var_i == 1:
            var_r = var_2
            var_g = V
            var_b = var_1

        elif var_i == 2:
            var_r = var_1
            var_g = V
            var_b = var_3

        elif var_i == 3:
            var_r = var_1
            var_g = var_2
            var_b = V

        elif var_i == 4:
            var_r = var_3
            var_g = var_1
            var_b = V

        else:
            var_r = V
            var_g = var_1
            var_b = var_2


        R = var_r * 255.
        G = var_g * 255.
        B = var_b * 255.

    return R,G,B

#-------------------------------------------------#
def hsv_to_cielab(H, S, V):
    '''
    Convert from HSV to CIE-L*a*b*.
    '''
    R,G,B = hsv_to_rgb(H,S,V)
    X,Y,Z = rgb_to_xyz(R,G,B)
    return xyz_to_cielab(X,Y,Z)

#-------------------------------------------------#
def cielab_to_hsv(CIE_L, CIE_a, CIE_b):
    '''
    Convert from CIE-L*a*b* to HSV.
    '''
    X,Y,Z = cielab_to_xyz(CIE_L, CIE_a, CIE_b)
    R,G,B = xyz_to_rgb(X,Y,Z)
    return rgb_to_hsv(R,G,B)

#-------------------------------------------------#
