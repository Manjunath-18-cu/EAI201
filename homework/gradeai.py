def ai_grading_assistant():
    print("Hello..! I am your Grading AI Assistant and I will help you get your grades based on your marks./n Please Enter the options below..")
    
    n=input("Enter student's name:")

    subs=int(input("How many subjects?:"))
    dict={}
    
    for i in range(0,subs):
        subject=input(f"Enter name of subject {i+1}: ")
        marks=float(input(f"Enter marks for {subject} (out of 100): "))
        dict[subject]=marks

    print("Grade report for :",n)
    total=0
    for subject, marks in dict.items():
        total=total+marks

    avg=total/subs

    print(f"Average Marks: {avg:.2f}")

    if avg>=90:
        print("Final Grade: A")
    elif avg>=80:
        print("Final Grade: B")
    elif avg>=70:
        print("Final Grade: C")
    elif avg>=60:
        print("Final Grade: D")
    else:
        print("Final Grade: W")
ai_grading_assistant()
