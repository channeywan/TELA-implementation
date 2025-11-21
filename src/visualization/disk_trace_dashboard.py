import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from data.loader import DiskDataLoader
from config.settings import DirConfig, DataConfig, WarehouseConfig
import joblib


def iterate_first_day(timestamps) -> int:
    """
    计算第一周第一天的行数
    """
    target_time = "2023-05-09 00:00:00"
    target_time = pd.to_datetime(target_time)
    line_len = -1
    for timestamp in timestamps:
        line_len += 1
        if timestamp.weekday() == target_time.weekday() and timestamp.time() == target_time.time():
            break
    return line_len


def get_circular_trace(disk_trace_bandwidth: pd.DataFrame, disk_capacity: int, begin_line: int, trace_len: int) -> np.ndarray:
    """
    return :DataFrame, shape(trace_len, [disk_capacity, bandwidth])
    """
    timestamps = disk_trace_bandwidth["timestamp"]
    last_timestamp = timestamps.iloc[-1]
    target_timestamp = last_timestamp + pd.Timedelta("5 min")
    target_weekday = target_timestamp.weekday()
    target_time = target_timestamp.time()
    all_weekdays = timestamps.dt.weekday
    all_times = timestamps.dt.time
    mask_match = (all_weekdays == target_weekday) & (
        all_times == target_time)
    matches = np.where(mask_match)[0]
    if len(matches) == 0:
        return None
    start_index = matches[0]
    base_trace = disk_trace_bandwidth.iloc[begin_line:]["bandwidth"].to_numpy(
    )
    append_trace = disk_trace_bandwidth.iloc[start_index:]["bandwidth"].to_numpy(
    )
    if len(append_trace) == 0:
        return None
    len_base = len(base_trace)

    if len_base >= trace_len:
        circular_bandwidth_trace = base_trace[:trace_len]
    else:
        len_needed = trace_len - len_base
        num_repeats = int(np.ceil(len_needed / len(append_trace)))
        padding = np.tile(append_trace, num_repeats)[:len_needed]
        circular_bandwidth_trace = np.concatenate([base_trace, padding])
    return circular_bandwidth_trace


def get_time_label(x_index, time_range):
    if time_range == '30days':
        # 30天模式：每天288个数据点，显示日期和小时
        day = x_index // 288
        hour = (x_index % 288) // 12
        minute = ((x_index % 288) % 12) * 5
        return f"{hour:02d}:{minute:02d}"
    elif time_range == '7days':
        # 7天模式：每天288个数据点
        day = x_index // 288
        hour = (x_index % 288) // 12
        minute = ((x_index % 288) % 12) * 5
        return f"{hour:02d}:{minute:02d}"
    else:  # 1day
        # 1天模式：288个数据点，每5分钟一个点
        hour = x_index // 12
        minute = (x_index % 12) * 5
        return f"{hour:02d}:{minute:02d}"


def show_dashboard(host='127.0.0.1', port=8050, debug=True):
    disk_data_loader = DiskDataLoader()
    global_data, global_disks_trace = disk_data_loader.load_items_and_trace(
        cluster_index_list=DataConfig.CLUSTER_DIR_LIST)
    global_data = pd.read_csv(os.path.join(
        DirConfig.PLACEMENT_DIR, "motivation", 'selected_items.csv'))
    app = dash.Dash(__name__)
    app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[

        html.H1("CVD disk trace visualization", style={
                'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div(
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'alignItems': 'flex-end',
                'justifyContent': 'space-around',
                'padding': '20px',
                'backgroundColor': '#f4f4f4',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            },
            children=[
                # --- 左栏：主要控件 ---
                html.Div(
                    style={
                        'flex': 3,
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '15px'
                    },
                    children=[

                        # 控件行 1: Cluster, Disk Count, Disk ID, Buttons
                        html.Div(
                            style={'display': 'flex', 'gap': '15px',
                                   'flexWrap': 'wrap'},  # [MODIFIED] 允许换行
                            children=[
                                html.Div([
                                    html.Label("Select Cluster Index:", style={
                                        'display': 'block', 'marginBottom': '5px'}),
                                    dcc.Dropdown(
                                        id='cluster-input',
                                        options=[{'label': f'Cluster {i}', 'value': i}
                                                 for i in DataConfig.SELECT_CLUSTER_INDEX_LIST],
                                        value=400,
                                        clearable=False
                                    )
                                ], style={'flex': 1, 'minWidth': '150px'}),  # [MODIFIED] 使用 flex 布局

                                html.Div([
                                    html.Label("Select Disk Count:", style={  # [MODIFIED] 澄清
                                        'display': 'block', 'marginBottom': '5px'}),
                                    dcc.Input(
                                        id='disk-count-input',
                                        type='number',
                                        value=1,
                                        min=1,
                                        max=5,
                                        style={'width': '100%'}
                                    )
                                ], style={'flex': 1, 'minWidth': '150px'}),  # [MODIFIED]

                                html.Div([
                                    html.Label("Line Width:", style={
                                        'display': 'block', 'marginBottom': '5px'}),
                                    dcc.Input(
                                        id='line-width-input',
                                        type='number',
                                        value=1.8,
                                        step=0.1,
                                        min=0.1,
                                        max=2,
                                        style={'width': '100%'}
                                    )
                                ], style={'flex': 1, 'minWidth': '100px'}),  # [MODIFIED]

                                # [NEW] Disk ID 输入框
                                html.Div([
                                    html.Label("Specific Disk ID:", style={
                                        'display': 'block', 'marginBottom': '5px'}),
                                    dcc.Input(
                                        id='disk-id-input',
                                        type='text',
                                        placeholder='Enter exact disk_ID...',
                                        style={'width': '100%'}
                                    )
                                ], style={'flex': 2, 'minWidth': '200px'}),  # [NEW]

                                html.Div([
                                    html.Label("", style={
                                        'display': 'block', 'marginBottom': '5px'}),
                                    html.Button(
                                        '刷新 (Random)',  # [MODIFIED] 澄清
                                        id='refresh-button',
                                        n_clicks=0,
                                        style={
                                            'width': '100%',
                                            'padding': '10px',
                                            'backgroundColor': '#009db1',
                                            'color': 'white',
                                            'border': 'none',
                                            'borderRadius': '4px',
                                            'cursor': 'pointer',
                                            'fontSize': '14px',
                                            'fontWeight': 'bold'
                                        }
                                    )
                                ], style={'flex': 1, 'minWidth': '120px'}),  # [MODIFIED]

                                # [NEW] "生成" 按钮
                                html.Div([
                                    html.Label("", style={
                                        'display': 'block', 'marginBottom': '5px'}),
                                    html.Button(
                                        '生成 (Specific)',  # [NEW]
                                        id='generate-button',  # [NEW]
                                        n_clicks=0,
                                        style={
                                            'width': '100%',
                                            'padding': '10px',
                                            # [NEW] 绿色
                                            'backgroundColor': '#4CAF50',
                                            'color': 'white',
                                            'border': 'none',
                                            'borderRadius': '4px',
                                            'cursor': 'pointer',
                                            'fontSize': '14px',
                                            'fontWeight': 'bold'
                                        }
                                    )
                                ], style={'flex': 1, 'minWidth': '120px'})  # [NEW]
                            ]
                        ),

                        # 控件行 2: Time Range
                        html.Div([
                            html.Label("Select Time Range:", style={
                                'display': 'block', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='time-range-dropdown',
                                options=[
                                    {'label': '30 天 (完整)', 'value': '30days'},
                                    {'label': '7 天 (缩放)', 'value': '7days'},
                                    {'label': '1 天 (缩放)', 'value': '1day'},
                                ],
                                value='7days',
                                clearable=False
                            )
                        ]),

                        # 控件行 3: Data Type
                        html.Div([
                            html.Label("Data Type:", style={
                                'display': 'block', 'marginBottom': '5px'}),
                            dcc.RadioItems(
                                id='normalization-radio',
                                options=[
                                    {'label': '绝对带宽', 'value': 'absolute'},
                                    {'label': '归一化带宽', 'value': 'normalized'},
                                ],
                                value='normalized',
                                labelStyle={'display': 'inline-block',
                                            'marginRight': '20px'}
                            )
                        ])
                    ]
                ),
            ]
        ),

        # --- 图表区域 ---
        dcc.Loading(
            id="loading-icon",
            type="circle",
            children=[
                dcc.Graph(id='trace-graph',
                          style={'height': '70vh', 'marginTop': '30px'})
            ]
        )
    ])

    @app.callback(
        Output('trace-graph', 'figure'),
        [Input('cluster-input', 'value'),
         Input('disk-count-input', 'value'),
         Input('time-range-dropdown', 'value'),
         Input('normalization-radio', 'value'),
         Input('line-width-input', 'value'),
         Input('refresh-button', 'n_clicks'),
         Input('generate-button', 'n_clicks')],
        [State('disk-id-input', 'value')]
    )
    def update_graph(cluster_index, disk_count, time_range, normalization_type, line_width,
                     n_clicks_refresh, n_clicks_generate, disk_id_input):  # [NEW] 添加新参数

        # [NEW] 确定哪个按钮触发了回调
        trigger_id = ctx.triggered_id

        fig = go.Figure()

        # --- 1. 通用设置 (时间轴) ---
        if time_range == '30days':
            trace_length = 8640
            x_ticks_pos = [i * 288 * 5 for i in range(7)]
            x_ticks_labels = [f'{i*5}d' for i in range(7)]
        elif time_range == '7days':
            trace_length = 288 * 7
            x_ticks_pos = [i * 288 for i in range(8)]
            x_ticks_labels = [f'Day {i}' for i in range(8)]
        else:  # 1day
            trace_length = 288
            x_ticks_pos = [i * 48 for i in range(7)]
            x_ticks_labels = ['0:00', '4:00', '8:00',
                              '12:00', '16:00', '20:00', '0:00']
        x_axis_data = list(range(trace_length))

        # --- 2. [NEW] 核心逻辑：根据触发按钮选择要绘制的 disk ---

        # 获取当前集群的所有 disk
        disks_in_cluster = global_data[global_data['cluster_index']
                                       == cluster_index]

        disks_to_plot = pd.DataFrame()  # 初始化一个空的 DataFrame

        if trigger_id == 'generate-button':
            # --- 逻辑 A: "生成" 按钮被点击 ---
            if disk_id_input:  # 检查输入框是否非空
                # 筛选特定的 disk_ID
                specific_disk = disks_in_cluster[disks_in_cluster['disk_ID']
                                                 == disk_id_input]
                if not specific_disk.empty:
                    disks_to_plot = specific_disk
                else:
                    # [NEW] 错误处理：未找到 Disk ID
                    fig.update_layout(
                        title=f'Error: Disk ID "{disk_id_input}" not found in Cluster {cluster_index}',
                        template='plotly_white'
                    )
                    return fig
            else:
                # [NEW] 错误处理：输入框为空
                fig.update_layout(
                    title=f'Error: Please enter a Disk ID before clicking "Generate"',
                    template='plotly_white'
                )
                return fig

        else:
            # --- 逻辑 B: "刷新" 按钮被点击 (或任何其他输入改变) ---
            if not disks_in_cluster.empty:
                # 确保 disk_count 不超过集群中的 disk 总数
                actual_count = min(disk_count, len(disks_in_cluster))
                disks_to_plot = disks_in_cluster.sample(n=actual_count)

        # --- 3. 绘图循环 (现在使用 disks_to_plot) ---

        plotted_labels = set()

        # [MODIFIED] 循环的主体现在是 disks_to_plot
        for _, disk in disks_to_plot.iterrows():
            try:  # [NEW] 添加 try...except 块以处理坏数据
                bandwidth_trace = get_circular_trace(
                    disk_trace_bandwidth=global_disks_trace[cluster_index][disk['disk_ID']],
                    disk_capacity=disk['disk_capacity'],
                    begin_line=iterate_first_day(
                        global_disks_trace[cluster_index][disk['disk_ID']]['timestamp']),
                    trace_len=trace_length
                )

                # [NEW] 检查 get_circular_trace 是否失败
                if bandwidth_trace is None:
                    continue  # 跳过这个 disk

                if normalization_type == 'absolute':
                    y_axis_data = bandwidth_trace
                else:
                    max_bw = np.max(bandwidth_trace)
                    y_axis_data = bandwidth_trace if max_bw == 0 else bandwidth_trace / max_bw

                label = f"Disk {disk['disk_ID']}"

                hover_texts = []
                for i in range(len(x_axis_data)):
                    time_label = get_time_label(x_axis_data[i], time_range)
                    hover_texts.append(
                        f"Time: {time_label}<br>Bandwidth: {bandwidth_trace[i]:.2f}")  # [MODIFIED]

                show_legend = False
                if label not in plotted_labels:
                    plotted_labels.add(label)
                    show_legend = True

                fig.add_trace(go.Scatter(
                    x=x_axis_data,
                    y=y_axis_data,
                    mode='lines',
                    line=dict(width=line_width),
                    opacity=0.7,
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend,
                    hovertext=hover_texts,
                    hoverinfo="text+name"
                ))
            except Exception as e:
                # [NEW] 捕获循环中的意外错误
                print(f"Error processing disk {disk['disk_ID']}: {e}")
                continue  # 跳过这个出错的 disk

        # --- 4. 最终布局 ---
        fig.update_layout(
            # [MODIFIED]
            title=f'Trace {len(disks_to_plot)} disk(s) in Cluster {cluster_index}',
            xaxis_title="TimeStamp",
            # [MODIFIED]
            yaxis_title="Bandwidth (Normalized)" if normalization_type == 'normalized' else "Bandwidth (Absolute)",
            xaxis=dict(
                tickmode='array',
                tickvals=x_ticks_pos,
                ticktext=x_ticks_labels
            ),
            hovermode="closest",
            legend_title_text='Disk',
            template='plotly_white'
        )
        return fig

        # 生成时间标签函数

    app.run(debug=debug, host=host, port=port)
