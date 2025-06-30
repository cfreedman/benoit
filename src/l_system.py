class LSystem:
    def __init__(self, axiom: str, rules: dict[str, str]):
        self.axiom = axiom
        self.rules = rules

    def next(self, input: str) -> str:
        result = ""

        for char in input:
            result += self.rules[char] if char in self.rules else char

        return result

    def get_generation(self, n: int) -> str:
        input = self.axiom

        for iteration in range(n):
            input = self.next(input)

        return input
