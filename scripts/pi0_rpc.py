"""Small localhost-only cross-Python RPC schema, no simulator / model imports."""
import base64
import json
from urllib.request import Request,urlopen
import numpy as np
from momagen_pi0_schema import rgb_image


def encode_observation(data):
    result={'state':np.asarray(data['state'],dtype=np.float32).tolist(),'prompt':str(data['prompt'])}
    for name in ('head_image','left_wrist_image','right_wrist_image'):
        image=rgb_image(data[name])
        result[name]={'shape':list(image.shape),'uint8_b64':base64.b64encode(image.tobytes()).decode('ascii')}
    return result


def decode_observation(data):
    state=np.asarray(data['state'],dtype=np.float32)
    if state.shape!=(21,) or not np.isfinite(state).all():raise ValueError('Expected finite R1 proprioception')
    result={'state':state,'prompt':str(data['prompt'])}
    for name in ('head_image','left_wrist_image','right_wrist_image'):
        record=data[name];shape=tuple(record['shape'])
        if shape not in ((256,256,3),(224,224,3)):raise ValueError('Unexpected RGB image shape')
        raw=base64.b64decode(record['uint8_b64'],validate=True)
        if len(raw)!=int(np.prod(shape)):raise ValueError('Invalid RGB byte count')
        result[name]=np.frombuffer(raw,dtype=np.uint8).reshape(shape).copy()
    return result


def infer_local(data,token,port=18731,timeout=180):
    body=json.dumps(encode_observation(data)).encode()
    request=Request(f'http://127.0.0.1:{int(port)}/infer',data=body,
                    headers={'Content-Type':'application/json','Authorization':'Bearer '+token},method='POST')
    with urlopen(request,timeout=timeout) as response:result=json.load(response)
    if result.get('model')!='pi0' or result.get('teacher_enabled') is not False:
        raise ValueError('Unexpected inference service / teacher fallback')
    actions=np.asarray(result['actions'],dtype=np.float32)
    if actions.shape!=(50,21) or not np.isfinite(actions).all():raise ValueError('Expected finite 50x21 native R1 pi0 action chunk')
    return actions,result
