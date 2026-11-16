"""Episode split provenance. Training diagnostics must never become heldout runs."""
from pathlib import Path


def verify_evaluation_episode(provenance,dataset,demo,split):
    if split not in ('train','validation') or provenance.get('split')!=split:
        raise ValueError('Evaluation split does not match provenance')
    matches=[entry for entry in provenance.get('episodes',[])
        if Path(entry['source']).resolve()==Path(dataset).resolve() and entry['source_demo']==demo and entry.get('split')==split]
    if len(matches)!=1:raise ValueError('Evaluation source/demo not uniquely in declared split')
    return matches[0]
