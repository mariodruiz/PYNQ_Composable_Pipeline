# Copyright (C) 2023 AMD, Inc
#
# SPDX-License-Identifier: BSD-3-Clause

from pynq_composable import VideoStream, VSource, VSink
import pytest
import time

if pytest.board == 'KV260':
    vin = 'cpipe.ps_video_in'
    vout = 'cpipe.ps_video_out'
else:
    vin = 'cpipe.hdmi_sink_in'
    vout = 'cpipe.hdmi_sink_out'

app_construct = [
    ("cpipe.pr_0.dilate_accel", f"[{vin}, cpipe.pr_0.dilate_accel, {vout}]"),
    ("cpipe.pr_0.dilate_accel", f"[{vin}, cpipe.pr_0.erode_accel, {vout}]"),
    ("cpipe.pr_0.filter2d_accel", f"[{vin}, cpipe.pr_0.filter2d_accel, {vout}]"),
    ("cpipe.pr_0.filter2d_accel", f"[{vin}, cpipe.pr_0.axis_data_fifo_1, {vout}]"),
    ("cpipe.pr_0.fast_accel", f"[{vin}, cpipe.pr_0.fast_accel, {vout}]"),
    ("cpipe.pr_0.fast_accel", f"[{vin}, cpipe.pr_0.axis_data_fifo_0, {vout}]"),
    ("cpipe.pr_1.dilate_accel", f"[{vin}, cpipe.pr_1.dilate_accel, {vout}]"),
    ("cpipe.pr_1.dilate_accel", f"[{vin}, cpipe.pr_1.erode_accel, {vout}]"),
    ("cpipe.pr_1.cornerHarris_accel", f"[{vin}, cpipe.pr_1.cornerHarris_accel, {vout}]"),
    ("cpipe.pr_1.cornerHarris_accel", f"[{vin}, cpipe.pr_1.axis_data_fifo_0, {vout}]"),
    ("cpipe.pr_1.rgb2xyz_accel", f"[{vin}, cpipe.pr_1.rgb2xyz_accel, {vout}]"),
    ("cpipe.pr_1.rgb2xyz_accel", f"[{vin}, cpipe.pr_1.axis_data_fifo_1, {vout}]"),
    ("cpipe.pr_2.add_accel", f"[{vin}, cpipe.duplicate_accel, [[cpipe.lut_accel], [1]], cpipe.pr_2.add_accel, {vout}]"),
    ("cpipe.pr_2.absdiff_accel", f"[{vin}, cpipe.duplicate_accel, [[cpipe.lut_accel], [1]], cpipe.pr_2.absdiff_accel, {vout}]"),
    ("cpipe.pr_2.bitwise_and_accel", f"[{vin}, cpipe.duplicate_accel, [[cpipe.lut_accel], [1]], cpipe.pr_2.bitwise_and_accel, {vout}]"),
    ("cpipe.pr_2.subtract_accel", f"[{vin}, cpipe.duplicate_accel, [[cpipe.lut_accel], [1]], cpipe.pr_2.subtract_accel, {vout}]"),
]


@pytest.mark.skipif(not pytest.webcam and not pytest.videofile,
                    reason='Web Camera or Video file not found')
@pytest.mark.parametrize('app', app_construct)
def test_app(app, create_composable):
    """This test will compose apps using only static IP & start a video stream

    """
    ol, cpipe = create_composable
    if pytest.board == 'KV260':
        sink = VSink.DP
    else:
        sink = VSink.HDMI

    cpipe.load([eval(app[0])])
    pipeline = eval(app[1])
    cpipe._graph_debug = True
    cpipe.compose(pipeline)
    file = '../mountains.mp4' if pytest.videofile else 0
    video = VideoStream(ol, VSource.OpenCV, sink, file=file)

    try:
        video.start()
        time.sleep(5)
        status = video._video._started and video._video._running
        label = 'FAILED\n'
        color = 'red'
        if status:
            label = 'PASSED\n'
            color = 'green'
        label = label + pytest.overlay + '\n' + app[0]
        cpipe.graph.attr(label=label, _attributes={'fontcolor': color})

        name = ('passed' if status else 'failed') + \
            f'/dynamic_{app_construct.index(app)}'

        cpipe.graph.render(format='png', outfile=f'result_tests/{name}.png')
        video.stop()
        assert status
    except pytest.PytestUnhandledThreadExceptionWarning:
        assert False, f'App {app} failed'
