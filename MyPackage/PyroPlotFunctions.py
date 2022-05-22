import pyro
import torch
import numpy as np
#import random 

#seed = 52
#pyro.set_rng_seed(seed)
#torch.manual_seed(seed);
#np.random.seed(seed)
#random.seed(seed)

import holoviews as hv
from holoviews.core.layout import Layout
from holoviews.core.overlay import Overlay
from holoviews import opts
hv.extension('plotly', logo=False)
opts.defaults(opts.Surface(width=1000, height=800),
             opts.Curve(width=700, height=500),
             opts.Area(width=700, height=500))

from typing import Union, Tuple, Callable

import Pyro_BO


def extractPlotData(X_test: torch.Tensor,
                    model: Pyro_BO.PyroBO) -> Tuple[torch.Tensor, # X_test
                                                            torch.Tensor, # X_train
                                                            torch.Tensor, # y_train
                                                            torch.Tensor, # pred_mean
                                                            torch.Tensor, # pred_unc
                                                            torch.Tensor, # acq_values
                                                            torch.Tensor]: # next_sample
    return (X_test,
            model.X.data,
            model.y.data,
            model(X_test)[0].data,
            model(X_test)[1].data,
            model.acq_fun.get_acq_values(mu_vector=model(X_test)[0], sigma_vector=model(X_test)[1].sqrt()).data,
            model.next_sample(X_test))



def simplePlot(X_test: torch.Tensor,
               X_train: torch.Tensor,
               y_train: torch.Tensor,
               pred_mean: torch.Tensor,
               pred_unc: torch.Tensor,
               acq_values: torch.Tensor,
               next_sample: torch.Tensor = None,
               show_legend: bool=True,
               xlabel=None,
               ylabel=None,
               obj_plot=None,
               init_samples=None) -> Union[Layout, Overlay]:
    hv.extension('plotly', logo=False)
    
    if X_test.shape[-1] == 1:
        return simple1D_Plot(X_test=X_test,
                             X_train=X_train,
                             y_train=y_train,
                             pred_mean=pred_mean,
                             pred_unc=pred_unc,
                             acq_values=acq_values,
                             next_sample=next_sample,
                             show_legend=show_legend,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             obj_plot=obj_plot,
                             init_samples=init_samples)
    
    elif X_test.shape[-1] == 2:        
        return _simple2D_Plot(X_test=X_test,
                              X_train=X_train,
                              y_train=y_train,
                              pred_mean=pred_mean,
                              pred_unc=pred_unc,
                              acq_values=acq_values,
                              init_samples=init_samples)
    
    else:
        raise NotImplementedError('not implemented yet')

        
def scale(from_, to_):
    """
    Converts scale of input from_ to the scales of input to_.
    """
    a1 = from_ - from_.min()
    a2 = a1 / a1.max()
    a3 = a2 * abs(to_.max() - to_.min())
    a4 = a3 + to_.min()
    return a4


# inputs: Union[torch.tensor, numpy.array]
def simple1D_Plot(X_test: torch.Tensor,
                  X_train: torch.Tensor,
                  y_train: torch.Tensor,
                  pred_mean: torch.Tensor,
                  pred_unc: torch.Tensor,
                  acq_values: torch.Tensor,
                  next_sample: torch.Tensor,
                  show_legend: bool,
                  xlabel=None,
                  ylabel=None,
                  obj_plot=None,
                  init_samples=None) -> Layout:
    
    def customize_legend_area(plot, element):
        for content in plot.state['data']:
            if(
                content['legendgroup'] == 'Area'
                and content['fill'] is None
            ):
                content['showlegend'] = False
    #acq_values = scale(acq_values,
    #                   np.array([pred_mean.min() - pred_unc[pred_mean.argmin().item()], 
    #                                 pred_mean.max() + pred_unc[pred_mean.argmax().item()]]))
    
    # plot prediction uncertainty
    unc_plot = hv.Area((X_test,
                        pred_mean + pred_unc,
                        pred_mean - pred_unc),
                    label='Prediction Uncertainty',
                    vdims=['Mean/ Uncertainty', 'point'],
    ).opts(color='cornflowerblue', height=300, show_legend=show_legend, hooks=[customize_legend_area],
           xlabel='x' if xlabel is None else xlabel,
           ylabel='y' if ylabel is None else ylabel)

    # plot prediction mean
    pred_plot = hv.Curve((X_test,
                          pred_mean),
                    label='Mean Prediction',
    ).opts(color='blue', title='Gaussian Process', show_legend=show_legend)

    # plot training points
    train_plot = hv.Scatter((X_train,
                             y_train),
                    label='Iteration Points',
    ).opts(color='black', marker='circle', show_legend=show_legend)
    
    if init_samples is not None:
        train_plot = train_plot * hv.Scatter((init_samples[0], init_samples[1]),
                                             label='Initial Points',
                                            ).opts(color='orange', marker='circle', show_legend=show_legend)
    
    acq_plot = hv.Curve((X_test,
                        acq_values),
    ).opts(color='red', title='Acquisition Function', height=200,
           xlabel='x' if xlabel is None else xlabel,
           ylabel='y' if ylabel is None else ylabel)
    
    acq_next_point = acq_values[(abs(X_test - next_sample)).argmin().item()].reshape(-1)
    
    next_point =  hv.Scatter((next_sample,
                              acq_next_point + abs(0.05*acq_next_point)),
                    label='Next Sample',
    ).opts(color='red', marker='triangle-down', size=8, show_legend=show_legend)
                              
    if obj_plot is None:
        return hv.Layout(unc_plot.opts(height=300) * pred_plot * train_plot + acq_plot.opts(height=200)*next_point.opts(show_legend=show_legend)).opts(shared_axes=False).cols(1)
    else:
        return hv.Layout(obj_plot.opts(height=300) * unc_plot * pred_plot * train_plot + acq_plot.opts(height=200)*next_point.opts(show_legend=show_legend)).opts(shared_axes=False).cols(1)

    
def _simple2D_Plot(X_test: torch.Tensor,
                   X_train: torch.Tensor,
                   y_train: torch.Tensor,
                   pred_mean: torch.Tensor,
                   pred_unc: torch.Tensor,
                   acq_values: torch.Tensor) -> Overlay:
        
    n = len(X_test.T[0].unique())
    m = len(X_test.T[1].unique())
    
    acq_values = acq_values.reshape(n,m).T
    
    def customize_surface_color(plot, element):
        del plot.handles["components"]["traces"][0]["colorscale"]
        plot.handles["components"]["traces"][0]["surfacecolor"] = acq_values.tolist()
        plot.handles["components"]["traces"][0]["cmin"] = acq_values.min().item()
        plot.handles["components"]["traces"][0]["cmax"] = acq_values.max().item()
    
    pred_plot = hv.Surface((X_test.T[0].unique(),
                            X_test.T[1].unique(),
                            pred_mean.reshape(n, m).T),
                    label='Prediction',
    ).opts(show_legend=True, hooks=[customize_surface_color], colorbar=True)
    
    train_plot = hv.Scatter3D((X_train.T[0],
                               X_train.T[1],
                               y_train),
                    label='Iteration Points',
    ).opts(color='black', marker='circle', show_legend=True, size=5)
    
    if init_samples is not None:
        train_plot = train_plot * hv.Scatter3D((init_samples[0][0],
                                                init_samples[0][1],
                                                init_samples[1]),
                                               label='Initial Points',
                                              ).opts(color='orange')
    
    return pred_plot * train_plot
    
    
def HeatMapPlot(X_space: torch.Tensor,
                  y_obj: torch.Tensor,
                  model: Pyro_BO.PyroBO,
                  X_context: torch.Tensor = None,
                  var: bool = False) -> Layout:
    hv.extension('bokeh', logo=False)
    
    data = [(x.item(),y.item(), z.item()) for x,y,z in zip(*X_space.T, y_obj.T.flatten())]
    obj_heat = hv.HeatMap(data,
    ).opts(title='Objective Function', cmap='RdBu_r', colorbar=True, show_legend=True, width=400, height=400)
    
    obj_max = y_obj.max().item()
    obj_min = y_obj.min().item()
    
    data = [(x.item(), y.item(), z.item()) for x,y,z in zip(*X_space.T, model(X_space)[0])]
    pred_heat = hv.HeatMap(data,
    ).opts(title='Mean Prediction', cmap='RdBu_r', show_legend=False, colorbar=True, width=400, height=400)
    pred_heat = pred_heat.redim.range(z=(obj_min, obj_max))
    
    data = [(x.item(), y.item()) for x,y in zip(*model.X.T)]

    train_samples = hv.Points(data,
                              label='Iteration Samples',
    ).opts(color='black', marker='circle', legend_position='bottom', show_legend=True)
    
    data = [(x.item(), y.item(), z.item()) for x,y,z in zip(*X_space.T, model.get_acq_values(X_space))]
    acq_heat = hv.HeatMap(data,
    ).opts(title='Acquisition Function', cmap='plasma',show_legend=False, colorbar=True, width=400, height=400)
    
    next_sample = model.next_sample(X_space) if X_context is None else model.next_sample(X_context)
    next_point = hv.Points((next_sample.T[0].detach().numpy(),
                            next_sample.T[1].detach().numpy()),
                            label='next sample',
    ).opts(marker='diamond', legend_position='bottom', show_legend=True, size=10, color='red')
    
    best_sample = model.X[int(torch.argmax(model.y))]
    best_point = hv.Points((best_sample.T[0],
                            best_sample.T[1]),
                            label='best sample',
    ).opts(marker='circle', legend_position='bottom', show_legend=True, size=10, color='greenyellow')
    
    plot11 = pred_heat*best_point*next_point*train_samples.opts(show_legend=True)
    plot11 = plot11.opts(show_legend=True)
    plot12 = obj_heat*best_point*next_point*train_samples
    plot12 = plot12.opts(show_legend=False)
    
    plot2 = acq_heat*best_point*next_point*train_samples
    plot2 = plot2.opts(show_legend=False)
    
    if var:
        data = [(x.item(), y.item(), z.item()) for x,y,z in zip(*X_space.T, model(X_space)[1])]
        variance_plot = hv.HeatMap(
            data
        ).opts(title='Standard Deviation of Prediction', cmap='RdBu_r', colorbar=True, width=400, height=400)
        variance_plot = (variance_plot*best_point*next_point*train_samples).opts(show_legend=False)
        
        return Layout(plot11+plot12+variance_plot+plot2).cols(2)
    
    return Layout(plot11+plot12 + plot2).cols(2)

