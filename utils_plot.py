import plotly.graph_objs as go
import plotly.express as px
import numpy as np

def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig


def mean_std_df(df, group_columns, columns_to_agg):
    xdf = df.groupby(group_columns).agg({column : [np.mean, np.std] for column in columns_to_agg})
    xdf.columns = xdf.columns.map("_".join)
    return xdf.reset_index()


# def line_plot_for_multi_columns(
#         df,
#         x_axis='epoch',
#         y_axiss = ['train_loss', 'val_loss'],
#         x_axis_title = 'Epoch',
#         y_axis_title = 'Loss',
#         legend_title = 'Loss',
#         legend_names={
#             'train_loss': 'Train',
#             'val_loss': 'Validation'
#         }
#     ):
    
def nice_plot_multi_columns(
        df,
        x_axis,
        y_axiss,
        plot_type='line',
        x_axis_title=None,
        y_axis_title=None,
        legend_title=None,
        legend_names=None
    ):

    _orig_legend_title = legend_title
    assert x_axis in df.columns
    if(x_axis_title is None):
        x_axis_title = x_axis
       
    if(legend_names is None):
        legend_names = dict(zip(y_axiss, map(str, y_axiss)))
    if(legend_title is None):
        legend_title = 'Legend'

    for y_axis in y_axiss:
        assert y_axis in df.columns
        assert y_axis in legend_names.keys()
    assert len(y_axiss) == len(legend_names)


    df_melt = df.melt(id_vars=[x_axis], 
                    value_vars=y_axiss, 
                    value_name='value',
                    var_name=legend_title)
    
    df_melt[legend_title] = df_melt[legend_title].map(legend_names)
    df_mean = mean_std_df(df_melt, [x_axis, legend_title], ['value'])

    if(plot_type == 'line'):
        fig = line(data_frame = df_mean,
                x=x_axis,
                y='value_mean',
                error_y='value_std',
                color=legend_title,
                error_y_mode='bands')
    elif(plot_type == 'bar'):
        fig = px.bar(df_mean, 
                        x=x_axis, 
                        y='value_mean', 
                        color=legend_title, 
                        error_y='value_std')
        fig.update_layout(barmode='group')
    
    fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            legend_title=_orig_legend_title,
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
            ),
            width=800,
            height=400,
            hovermode="x"
        )
    return fig


def nice_plot(
        df,
        x_axis,
        y_axis,
        plot_type='line',
        x_axis_title=None,
        y_axis_title=None,
        group_by=None,
        legend_title=None,
        legend_names=None
    ):

    assert y_axis in df.columns
    if(x_axis_title is None):
        x_axis_title = x_axis
    if(y_axis_title is None):
        y_axis_title = y_axis
    if(group_by is not None):
        assert group_by in df.columns
        if(legend_names is not None):
            assert set(df[group_by].unique()) == set(legend_names.keys())
        else:
            legend_names = dict(zip(df[group_by].unique(), map(str, df[group_by].unique())))
        if(legend_title is None):
            legend_title = group_by

    if(group_by is not None):
        df_mean = mean_std_df(df, [x_axis, group_by], [y_axis])
    else:
        df_mean = mean_std_df(df, [x_axis], [y_axis])
    if(group_by is not None):
        df_mean[group_by] = df_mean[group_by].map(legend_names)
    
    if(plot_type == 'line'):
        fig = line(data_frame = df_mean,
                x=x_axis,
                y=f'{y_axis}_mean',
                error_y=f'{y_axis}_std',
                color=group_by,
                error_y_mode='bands')
    elif(plot_type == 'bar'):
        fig = px.bar(df_mean, 
                        x=x_axis, 
                        y=f'{y_axis}_mean', 
                        color=group_by, 
                        error_y=f'{y_axis}_std')
        fig.update_layout(barmode='group')
    else:
        raise NotImplementedError
    
    
    fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
            ),
            legend_title=legend_title,
            width=800,
            height=400,
            hovermode="x"
        )
    
    
    return fig



def bar_plot_for_multi_columns(
        df,
        x_axis='epoch',
        y_axiss = ['train_loss', 'val_loss'],
        x_axis_title = 'Epoch',
        y_axis_title = 'Loss',
        legend_title = 'Loss',
        legend_names={
            'train_loss': 'Train',
            'val_loss': 'Validation'
        }
    ):
    
    for y_axis in y_axiss:
        assert y_axis in df.columns
        assert y_axis in legend_names.keys()
    assert len(y_axiss) == len(legend_names)


    df_melt = df.melt(id_vars=[x_axis], 
                    value_vars=y_axiss, 
                    value_name='value',
                    var_name=legend_title)
    
    df_melt[legend_title] = df_melt[legend_title].map(legend_names)
    df_mean = mean_std_df(df_melt, [x_axis, legend_title], ['value'])

    fig = px.bar(df_mean, x=x_axis, y='value_mean', color=legend_title, error_y='value_std')
    
    fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
            ),
            width=800,
            height=400,
            hovermode="x"
        )
    return fig

