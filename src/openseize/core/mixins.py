import inspect
import pprint
import reprlib

class ViewInstance:
    """Mixin endowing inheritors with echo and print str representations."""

    __slots__ = ()

    def _fetch_attributes(self):
        """Returns a dict of all non-protected attrs."""

        if '__dict__' in dir(self):
            return {attr: val for attr, val in self.__dict__.items()
                    if not attr.startswith('_')}
        else:
            #slotted instance
            return {attr: getattr(self, attr) for attr in self.__slots__}
            
    def _fetch_methods(self):
        """Returns non-protected instance and class methods."""
        
        methods = dict(inspect.getmembers(self, inspect.ismethod))
        return {name: m for name, m in methods.items() 
                if not name.startswith('_')}

    def _fetch_properties(self):
        """Returns non_protected properties."""

        def isproperty(item):
            """Helper returning True if item is a property."""

            return isinstance(item, property)

        props = inspect.getmembers(type(self), isproperty)
        return {name: getattr(self, name) for name,_ in props 
                if not name.startswith('_')}

    def __repr__(self):
        """Returns the __init__'s signature as the echo representation.
        
        Returns: str
        """

        #build a signature and get its args and class name
        signature = inspect.signature(self.__init__)
        args = str(signature)
        cls_name = type(self).__name__
        return '{}{}'.format(cls_name, args)

    def __str__(self):
        """Returns this instances print representation."""

        cls_name = type(self).__name__
        #get the attributes & properties
        attrs = self._fetch_attributes()
        props = self._fetch_properties()
        #add properties
        attrs.update(props)
        #build a pretty printed msg
        msg_start = cls_name + ' Object\n' + '---Attributes & Properties---'
        pp = pprint.PrettyPrinter(sort_dicts=False, compact=True)
        msg_body = pp.pformat(attrs) 
        msg_end = '\nType help({}) for full documentation'.format(cls_name)
        return '\n'.join([msg_start, msg_body, msg_end])


class ViewContainer:
    """Mixin endowing data containers with str and echo representations."""

    __slots__ = ()

    def _fetch_attributes(self):
        """Returns a dict of all non-protected attrs."""

        if '__dict__' in dir(self):
            return {attr: val for attr, val in self.__dict__.items()
                    if not attr.startswith('_')}
        else:
            #slotted instance
            return {attr: getattr(self, attr) for attr in self.__slots__}

    def __repr__(self):
        """Returns this containers echo representation."""

        cls_name = type(self).__name__
        #get containers attributes
        attrs = self._fetch_attributes()
        #create a constrained repr instance
        r = reprlib.aRepr
        r.maxdict = 2
        return '{}: {}'.format(cls_name, r.repr(attrs))

    def __str__(self):
        """Returns this instances print representation."""

        cls_name = type(self).__name__
        #get containers attributes
        attrs = self._fetch_attributes()
        #build a pretty printed msg
        msg_start = cls_name + ' Object:'
        pp = pprint.PrettyPrinter(sort_dicts=False, compact=True)
        msg_body = pp.pformat(attrs) 
        msg_end = '\nType help({}) for full documentation'.format(cls_name)
        return '\n'.join([msg_start, msg_body, msg_end])


