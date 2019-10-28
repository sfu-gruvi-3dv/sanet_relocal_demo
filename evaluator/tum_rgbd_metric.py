from frame_seq_data import FrameSeqData
import os
from seq_data.tum_rgbd.tum_seq2ares import export_to_tum_format


def rgbd_rpe(pred_seq: FrameSeqData, gt_seq: FrameSeqData, cmdline_options=None):
    """Runs the rgbd command line tool for the RPE error

    gt_poses: list of Pose
    pr_poses: list of Pose
    timestamps: list of float

    cmdline_options: str
        Options passed to the evaluation tool
        Default is '--fixed_delta'

    """
    import tempfile
    import shlex
    from evaluator.tum_rgbd_module.evaluate_rpe import evaluate_rpe

    f, gt_txt = tempfile.mkstemp()
    os.close(f)
    export_to_tum_format(frames=gt_seq, output_path=gt_txt)

    f, pr_txt = tempfile.mkstemp()
    os.close(f)
    export_to_tum_format(frames=pred_seq, output_path=pr_txt)

    if cmdline_options is None:
        cmdline_options = '--fixed_delta'

    cmdline = '{0} {1} {2}'.format(cmdline_options, gt_txt, pr_txt)
    result = evaluate_rpe(shlex.split(cmdline))
    os.remove(gt_txt)
    os.remove(pr_txt)
    return result


def rgbd_ate(pred_seq: FrameSeqData, gt_seq: FrameSeqData, cmdline_options=None):
    """Runs the rgbd command line tool for the ATE error

    gt_poses: list of Pose
    pr_poses: list of Pose
    timestamps: list of float

    cmdline_options: str
        Options passed to the evaluation tool
        Default is ''

    """
    import tempfile
    import shlex
    from evaluator.tum_rgbd_module.evaluate_ate import evaluate_ate

    f, gt_txt = tempfile.mkstemp()
    os.close(f)
    export_to_tum_format(frames=gt_seq, output_path=gt_txt)

    f, pr_txt = tempfile.mkstemp()
    os.close(f)
    export_to_tum_format(frames=pred_seq, output_path=pr_txt)

    if cmdline_options is None:
        cmdline_options = ''

    cmdline = '{0} {1} {2}'.format(cmdline_options, gt_txt, pr_txt)
    result = evaluate_ate(shlex.split(cmdline))
    os.remove(gt_txt)
    os.remove(pr_txt)
    return result