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
    f"[{vin}, cpipe.rgb2gray_accel, {vout}]",
    f"[{vin}, cpipe.gray2rgb_accel, {vout}]",
    f"[{vin}, cpipe.rgb2hsv_accel, {vout}]",
    f"[{vin}, cpipe.filter2d_accel, {vout}]",
    f"[{vin}, cpipe.colorthresholding_accel, {vout}]",
    f"[{vin}, cpipe.lut_accel, {vout}]",
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

    cpipe._graph_debug = True
    cpipe.compose(eval(app))
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
        label = label + pytest.overlay
        cpipe.graph.attr(label=label, _attributes={'fontcolor': color})

        name = ('passed' if status else 'failed') + \
            f'/static_{app_construct.index(app)}'

        cpipe.graph.render(format='png', outfile=f'result_tests/{name}.png')
        video.stop()
        assert status
    except pytest.PytestUnhandledThreadExceptionWarning:
        assert False, f'App {app} failed'
