"""Observe explicit task-object pose writes; never modify object poses ourselves."""


class ObjectPoseAudit:
    def __init__(self, objects):
        self.originals=[];self.events=[];self.step=-1
        for name,wrapped in objects.items():
            obj=getattr(wrapped,'unwrapped',wrapped)
            if not hasattr(obj,'set_position_orientation'):continue
            original=obj.set_position_orientation
            self.originals.append((obj,original))
            def observed(*args,_method=original,_name=name,**kwargs):
                self.events.append({'object':_name,'step':self.step,'keyword_arguments':list(kwargs)})
                return _method(*args,**kwargs)
            obj.set_position_orientation=observed

    def close(self):
        for obj,method in self.originals:obj.set_position_orientation=method
        self.originals.clear()
