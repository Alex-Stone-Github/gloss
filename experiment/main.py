

# Oh yeah the type that these graph members deal with is a float32, so yea
# TYPE = flaot32

"""
Kind of like and interface for the graph structure in this module
"""
# yea borrowed that one from java
class Node:
    def evaluate(self) -> float:
        """
        This is a method that when implemented should return a value of type TYPE
        """
        raise NotImplementedError("You need to implement evaluate on your nodes")
    def __str__(self) -> str:
        raise NotImplementedError("You need to implement a str method for your class")
    def requires_grad(self) -> bool:
        raise NotImplementedError("You need to implement a requires grad method for your class")


"""
A leaf class has a value and when evaluated just returns that value
"""
class Leaf(Node):
    def __init__(self, value, requires_grad=False):
        self.value = value
        self._requires_grad = requires_grad
    def evaluate(self):
        return self.value
    def __str__(self):
        return str(self.evaluate())
    def requires_grad(self):
        return self._requires_grad



# These are just binary operation nodes -- they do what you think
class Addition(Node):
    def __init__(self, refa, refb):
        self.refa = refa
        self.refb = refb
    def evaluate(self):
        return self.refa.evaluate() + self.refb.evaluate()
    def __str__(self):
        return str(self.evaluate())
    def requires_grad(self):
        return self.refa.requires_grad() or self.refb.requires_grad()
class Subtraction(Node):
    def __init__(self, refa, refb):
        self.refa = refa
        self.refb = refb
    def evaluate(self):
        return self.refa.evaluate() - self.refb.evaluate()
    def __str__(self):
        return str(self.evaluate())
    def requires_grad(self):
        return self.refa.requires_grad() or self.refb.requires_grad()
class Multiplication(Node):
    def __init__(self, refa, refb):
        self.refa = refa
        self.refb = refb
    def evaluate(self):
        return self.refa.evaluate() * self.refb.evaluate()
    def __str__(self):
        return str(self.evaluate())
    def requires_grad(self):
        return self.refa.requires_grad() or self.refb.requires_grad()



if __name__ == "__main__":
    # make sure we are in main
    leaf = Leaf(3, requires_grad=True)
    leaf2 = Leaf(4)

    inter = Multiplication(leaf, leaf2)

    print(leaf, leaf.requires_grad())
    print(leaf2, leaf2.requires_grad())
    print(inter, inter.requires_grad())




