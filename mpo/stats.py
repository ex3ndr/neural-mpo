class Stats:
    def __init__(self):
        self.scalars = {}

    def push(self, name, value):
        if name not in self.scalars:
            self.scalars[name] = []
        self.scalars[name].append(value)
