class InstanceTracker:
    """
    Tracks all the instances of a class.
    Each instance has a index that can be tracked back the instance list.
    """
    total: int = 0
    instances: list = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.instances = []

    def __init__(self):
        self.index = None
        self.register_instance(self)

    @classmethod
    def register_instance(cls, instance):
        instance.index = cls.total
        cls.total += 1
        cls.instances.append(instance)

    @classmethod
    def reset_class(cls):
        cls.total = 0
        cls.instances = []
