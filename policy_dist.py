from agents.student import Student
from agents.teacher import Teacher


class SingleDistillation():
    def __init__(self,teacher,student):
        self.teacher=teacher

        self.student=student

    def distill(self):
        pass


