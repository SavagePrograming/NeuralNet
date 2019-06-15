class InvalidInnovationNumber(LookupError):
    def __init__(self, innovation_number, is_node):
        super(InvalidInnovationNumber, self).__init__(
            "Inovation number '%d' which is a %s" % (innovation_number, "node" if is_node else "connection"))
