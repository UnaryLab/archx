from loguru import logger

class Event:
    def __init__(self):
        self.events = {'event': {}}

    def add_event(self, name: str, subevent: list, performance: str):
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert name not in self.events, f"Event '{name}' already exists in event."
        assert isinstance(subevent, list), "'subevent' parameter must be of type 'list'."
        assert isinstance(performance, str), "'performance' parameter must be of type 'str' or not provided."

        self.events['event'][name] = {'subevent': subevent, 'performance': performance}

        logger.info(f"Added event: {name}")
        for sub in subevent:
            logger.debug(f"\tSubevent: {sub}")
        logger.debug(f"\tPerformance: {performance}")

    def to_yaml(self):
        return self.events