# griddata vs SmoothBivariateSpline
# http://stackoverflow.com/questions/3526514/
#   problem-with-2d-interpolation-in-scipy-non-rectangular-grid

# http://www.scipy.org/Cookbook/Matplotlib/Gridding_irregularly_spaced_data
# http://en.wikipedia.org/wiki/Natural_neighbor
# http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html

from __future__ import division
import sys
import numpy as np
from scipy.interpolate import SmoothBivariateSpline  # $scipy/interpolate/fitpack2.py
from matplotlib.mlab import griddata

__date__ = "2010-10-08 Oct"  # plot diffs, ypow
    # "2010-09-13 Sep"  # smooth relative

def avminmax( X ):
    absx = np.abs( X[ - np.isnan(X) ])
    av = np.mean(absx)
    m, M = np.nanmin(X), np.nanmax(X)
    histo = np.histogram( X, bins=5, range=(m,M) ) [0]
    return "av %.2g  min %.2g  max %.2g  histo %s" % (av, m, M, histo)

def cosr( x, y ):
    return 10 * np.cos( np.hypot(x,y) / np.sqrt(2) * 2*np.pi * cycle )

def cosx( x, y ):
    return 10 * np.cos( x * 2*np.pi * cycle )

def dipole( x, y ):
    r = .1 + np.hypot( x, y )
    t = np.arctan2( y, x )
    return np.cos(t) / r**3

#...............................................................................
testfunc = cosx
Nx = Ny = 20  # interpolate random Nx x Ny points -> Newx x Newy grid
Newx = Newy = 100
cycle = 3
noise = 0
ypow = 2  # denser => smaller error
imclip = (-5., 5.)  # plot trierr, splineerr to same scale
kx = ky = 3
smooth = .01  # Spline s = smooth * z2sum, see note
    # s is a target for sum (Z() - spline())**2  ~ Ndata and Z**2;
    # smooth is relative, s absolute
    # s too small => interpolate/fitpack2.py:580: UserWarning: ier=988, junk out
    # grr error message once only per ipython session
seed = 1
plot = 0

exec "\n".join( sys.argv[1:] )  # run this.py N= ...
np.random.seed(seed)
np.set_printoptions( 1, threshold=100, suppress=True )  # .1f

print 80 * "-"
print "%s  Nx %d Ny %d -> Newx %d Newy %d  cycle %.2g noise %.2g  kx %d ky %d smooth %s" % (
    testfunc.__name__, Nx, Ny, Newx, Newy, cycle, noise, kx, ky, smooth)

#...............................................................................

    # interpolate X Y Z to xnew x ynew --
X, Y = np.random.uniform( size=(Nx*Ny, 2) ) .T
Y **= ypow
    # 1d xlin ylin -> 2d X Y Z, Ny x Nx --
    # xlin = np.linspace( 0, 1, Nx )
    # ylin = np.linspace( 0, 1, Ny )
    # X, Y = np.meshgrid( xlin, ylin )
Z = testfunc( X, Y )  # Ny x Nx
if noise:
    Z += np.random.normal( 0, noise, Z.shape )
# print "Z:\n", Z
z2sum = np.sum( Z**2 )

xnew = np.linspace( 0, 1, Newx )
ynew = np.linspace( 0, 1, Newy )
Zexact = testfunc( *np.meshgrid( xnew, ynew ))
if imclip is None:
    imclip = np.min(Zexact), np.max(Zexact)
xflat, yflat, zflat = X.flatten(), Y.flatten(), Z.flatten()

#...............................................................................
print "SmoothBivariateSpline:"
fit = SmoothBivariateSpline( xflat, yflat, zflat, kx=kx, ky=ky, s = smooth * z2sum )
Zspline = fit( xnew, ynew ) .T  # .T ??

splineerr = Zspline - Zexact
print "Zspline - Z:", avminmax(splineerr)
print "Zspline:    ", avminmax(Zspline)
print "Z:          ", avminmax(Zexact)
res = fit.get_residual()
print "residual %.0f  res/z2sum %.2g" % (res, res / z2sum)
# print "knots:", fit.get_knots()
# print "Zspline:", Zspline.shape, "\n", Zspline
print ""

#...............................................................................
print "griddata:"
Ztri = griddata( xflat, yflat, zflat, xnew, ynew,interp='linear' )
        # 1d x y z -> 2d Ztri on meshgrid(xnew,ynew)

nmask = np.ma.count_masked(Ztri)
if nmask > 0:
    print "info: griddata: %d of %d points are masked, not interpolated" % (
        nmask, Ztri.size)
    Ztri = Ztri.data  # Nans outside convex hull
trierr = Ztri - Zexact
print "Ztri - Z:", avminmax(trierr)
print "Ztri:    ", avminmax(Ztri)
print "Z:       ", avminmax(Zexact)
print ""

#...............................................................................

import pylab as pl
nplot = 2
fig = pl.figure( figsize=(10, 10/nplot + .5) )
pl.suptitle( "Interpolation error: griddata - %s, BivariateSpline - %s" % (
    testfunc.__name__, testfunc.__name__ ), fontsize=11 )

def subplot( z, jplot, label ):
    ax = pl.subplot( 1, nplot, jplot )
    im = pl.imshow(
        np.clip( z, *imclip ),  # plot to same scale
        cmap=pl.cm.RdYlBu,
        interpolation="nearest" )
            # nearest: squares, else imshow interpolates too
            # todo: centre the pixels
    ny, nx = z.shape
    pl.scatter( X*nx, Y*ny, edgecolor="y", s=1 )  # for random XY
    pl.xlabel(label)
    return [ax, im]

subplot( trierr, 1,
    "griddata, Delaunay triangulation + Natural neighbor: max %.2g" %
    np.nanmax(np.abs(trierr)) )

ax, im = subplot( splineerr, 2,
    "SmoothBivariateSpline kx %d ky %d smooth %.3g: max %.2g" % (
    kx, ky, smooth, np.nanmax(np.abs(splineerr)) ))

pl.subplots_adjust( .02, .01, .92, .98, .05, .05 )  # l b r t
cax = pl.axes([.95, .05, .02, .9])  # l b w h
pl.colorbar( im, cax=cax )  # -1.5 .. 9 ??
if plot >= 2:
    pl.savefig( "tmp.png" )
pl.show()