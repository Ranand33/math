class Proposition:
    # Proposition types: Universal Affirmative (A), Universal Negative (E),
    # Particular Affirmative (I), Particular Negative (O)
    TYPES = {"A": "All {} are {}.", 
             "E": "No {} are {}.", 
             "I": "Some {} are {}.", 
             "O": "Some {} are not {}."}
    
    def __init__(self, prop_type, subject, predicate):
        if prop_type not in self.TYPES:
            raise ValueError("Proposition type must be A, E, I, or O")
        self.type = prop_type
        self.subject = subject
        self.predicate = predicate
    
    def __str__(self):
        return self.TYPES[self.type].format(self.subject, self.predicate)


class Syllogism:
    # Valid syllogistic figures
    VALID_FIGURES = {
        # Figure 1
        ("A", "A", "A", 1): True,  # Barbara
        ("E", "A", "E", 1): True,  # Celarent
        ("A", "I", "I", 1): True,  # Darii
        ("E", "I", "O", 1): True,  # Ferio
        # Figure 2
        ("E", "A", "E", 2): True,  # Cesare
        ("A", "E", "E", 2): True,  # Camestres
        ("E", "I", "O", 2): True,  # Festino
        ("A", "O", "O", 2): True,  # Baroco
        # Figure 3
        ("A", "A", "I", 3): True,  # Darapti
        ("E", "A", "O", 3): True,  # Felapton
        ("I", "A", "I", 3): True,  # Disamis
        ("A", "I", "I", 3): True,  # Datisi
        ("O", "A", "O", 3): True,  # Bocardo
        ("E", "I", "O", 3): True,  # Ferison
        # Figure 4
        ("A", "A", "I", 4): True,  # Bramantip
        ("A", "E", "E", 4): True,  # Camenes
        ("I", "A", "I", 4): True,  # Dimaris
        ("E", "A", "O", 4): True,  # Fesapo
        ("E", "I", "O", 4): True,  # Fresison
    }
    
    def __init__(self, major_premise, minor_premise, conclusion):
        self.major_premise = major_premise
        self.minor_premise = minor_premise
        self.conclusion = conclusion
        
        # Determine the figure based on term positions
        # In syllogisms, we have three terms: major, minor, and middle
        # The figure is determined by the position of the middle term
        if (major_premise.subject == minor_premise.predicate and 
            major_premise.predicate == conclusion.predicate and 
            minor_premise.subject == conclusion.subject):
            self.figure = 1
        elif (major_premise.predicate == minor_premise.predicate and 
              major_premise.subject == conclusion.predicate and 
              minor_premise.subject == conclusion.subject):
            self.figure = 2
        elif (major_premise.subject == minor_premise.subject and 
              major_premise.predicate == conclusion.predicate and 
              minor_premise.predicate == conclusion.subject):
            self.figure = 3
        elif (major_premise.predicate == minor_premise.subject and 
              major_premise.subject == conclusion.predicate and 
              minor_premise.predicate == conclusion.subject):
            self.figure = 4
        else:
            self.figure = None
    
    def is_valid(self):
        if self.figure is None:
            return False
        
        key = (self.major_premise.type, self.minor_premise.type, 
               self.conclusion.type, self.figure)
        return self.VALID_FIGURES.get(key, False)
    
    def __str__(self):
        validity = "VALID" if self.is_valid() else "INVALID"
        return (f"Major Premise: {self.major_premise}\n"
                f"Minor Premise: {self.minor_premise}\n"
                f"Conclusion: {self.conclusion}\n"
                f"Figure: {self.figure}\n"
                f"Validity: {validity}")


# Example usage
if __name__ == "__main__":
    # All men are mortal.
    # All Greeks are men.
    # Therefore, all Greeks are mortal.
    major = Proposition("A", "men", "mortal")
    minor = Proposition("A", "Greeks", "men")
    conclusion = Proposition("A", "Greeks", "mortal")
    
    result = Syllogism(major, minor, conclusion)
    print("Example 1: Barbara")
    print(result)
    print()
    
    # Example 2: Invalid syllogism
    # All cats are animals.
    # Some dogs are animals.
    # Therefore, some dogs are cats.
    major = Proposition("A", "cats", "animals")
    minor = Proposition("I", "dogs", "animals")
    conclusion = Proposition("I", "dogs", "cats")
    
    invalid = Syllogism(major, minor, conclusion)
    print("Example 2: Invalid syllogism")
    print(invalid)