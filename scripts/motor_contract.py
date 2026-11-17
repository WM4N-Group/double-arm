"""Read-only motor metadata and offline binary-command interpretation.

No simulator imports, setters, policy inputs, reward or action modification.
"""
import math
import hashlib
from pathlib import Path


def json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    return str(value)


def snapshot_motor_contract(robot):
    report = {'source': 'read_only_runtime_controller_attributes_NOT_policy_inputs', 'errors': [],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report['robot_action_normalize'] = json_value(getattr(robot, '_action_normalize', None))
    for name in ('base', 'trunk', 'arm_left', 'arm_right', 'gripper_left', 'gripper_right'):
        try:
            controller = robot.controllers[name]
            # The pinned MultiFingerGripperController exposes _motor_type but
            # not motor_type. Missing one optional attribute must not discard
            # its independently available mode/limits or invent their values.
            motor_type_attribute = 'motor_type'
            try:
                motor_type = controller.motor_type
            except AttributeError:
                motor_type_attribute = '_motor_type'
                motor_type = getattr(controller, '_motor_type', None)
                if motor_type is None:
                    report['errors'].append(f'{name}.motor_type: neither public nor private attribute available')
            report[name] = {'class': type(controller).__name__, 'motor_type': json_value(motor_type),
                'motor_type_attribute': motor_type_attribute if motor_type is not None else None,
                'mode': json_value(getattr(controller, '_mode', None)),
                'inverted': json_value(getattr(controller, '_inverted', None)),
                'use_delta_commands_raw_flag': json_value(getattr(controller, '_use_delta_commands', None)),
                'command_input_limits': json_value(getattr(controller, '_command_input_limits', None)),
                'command_output_limits': json_value(getattr(controller, '_command_output_limits', None))}
        except Exception as error:
            report[name] = None
            report['errors'].append(f'{name}: {type(error).__name__}: {error}')
    return report


def binary_close_flags(commands, threshold):
    """For a verified non-inverted binary controller: open >= threshold."""
    if not math.isfinite(threshold):
        raise ValueError('Finite verified command-space threshold required')
    values=[float(value) for value in commands]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Unknown/nonfinite commands must not be treated as open/close')
    return [value < threshold for value in values]


def runtime_binary_threshold(contract, name='gripper_left'):
    """None when runtime metadata cannot establish this exact convention."""
    c = contract.get(name) if contract else None
    if not c or c.get('mode') != 'binary' or c.get('inverted') is not False:
        return None
    def scalar(x):
        while isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError('Scalar gripper limits required')
            x = x[0]
        return float(x)
    try:
        lo, hi = [scalar(x) for x in c['command_input_limits']]
        out_lo, out_hi = [scalar(x) for x in c['command_output_limits']]
        if not all(math.isfinite(x) for x in (lo,hi,out_lo,out_hi)) or hi <= lo or not out_lo < 0 < out_hi:
            return None
        return lo + (-out_lo)/(out_hi-out_lo)*(hi-lo)
    except (KeyError, TypeError, ValueError):
        return None


def choose_close_threshold(runtime_contract, assume_pinned=False):
    threshold=runtime_binary_threshold(runtime_contract)
    if threshold is not None:
        return threshold, 'actual saved runtime motor metadata'
    if runtime_contract:
        return None, 'runtime contract unknown/unsupported; source assumption MUST NOT override it'
    if assume_pinned:
        return .5, 'INFERRED pinned R1 binary input[0,1]/output[-1,1], action_normalizeFalse, lower grasp direction; not captured runtime instance'
    return None, 'unknown; only raw-negative stats'
