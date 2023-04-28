# To ensure that code, med, age_on_admittance, and disposition all has the same length
# NOTE:
# initial length:
# code = age
# med = disposition
def handle_arrays(array1, array2, age_on_admittance,disposition):
    sublists1 = []
    sublists2 = []
    age_sublists = []
    disposition_sublists = []
    temp = []
    temp_age = []
    temp_disposition = []



    final_result1 = []
    final_result2 = []
    age_final_result = []
    disposition_final_result = []


    for item, age in zip(array1,age_on_admittance):
        if item == "SEP":
            sublists1.append(temp)
            age_sublists.append(temp_age)
            temp = []
            temp_age = []
        else:
            temp.append(item)
            temp_age.append(age)
    if temp:
        sublists1.append(temp)
        age_sublists.append(temp_age)

    for item,disp in zip(array2,disposition):
        if item == "SEP":
            sublists2.append(temp)
            disposition_sublists.append(temp_disposition)
            temp = []
            temp_disposition = []
        else:
            temp.append(item)
            temp_disposition.append(disp)
    if temp:
        sublists2.append(temp)
        disposition_sublists.append(temp_disposition)

    # print(len(age_sublists))
    # for a in sublists1:
    #     print(len(a))
    # for b in age_sublists:
    #     print(len(b))

    if len(sublists1) == len(sublists2):
        print("CASE 1")
        for a,b,c,d in zip(sublists1,sublists2, age_sublists, disposition_sublists):
            if len(a) > len(b):
                diff = len(a) - len(b)
                for _ in range(diff):
                    b.append('UNK')
                    # c.append(c[0])
                    d.append(d[0])
            if len(b) > len(a):
                diff = len(b) - len(a)
                for _ in range(diff):
                    a.append('UNK')
                    c.append(c[0])
                    # d.append(d[0])

            

    elif len(sublists1) > len(sublists2):
        print("CASE 2")            

        for _ in range(len(array1)-len(array2)):
            sublists2[-1].append('UNK')

        for a,b in zip(sublists1,age_sublists):
            while (len(a)>len(b)):
                b.append(b[0])

        # sublists2[-1].append('SEP')


    elif len(sublists2) > len(sublists1):
        print("CASE 3")

        for _ in range(len(array2)-len(array1)):
            sublists1[-1].append('UNK')

        for a,b in zip(sublists2,age_sublists):
            while (len(a)>len(b)):
                b.append(b[0])


        # sublists1[-1].append('SEP')


                
    for sublist in sublists1:
        final_result1.extend(sublist)
        final_result1.append('SEP')

    for sublist in sublists2:
        final_result2.extend(sublist)
        final_result2.append('SEP')

    for sublist in age_sublists:
        age_final_result.extend(sublist)
        age_final_result.append(sublist[0])

    for sublist in disposition_sublists:
        disposition_final_result.extend(sublist)
        disposition_final_result.append(sublist[0])


    print(len(final_result1))
    print(len(final_result2))
    print(len(age_final_result))
    print(len(disposition_final_result))
    print("=======")


    if(len(final_result1)!=len(final_result1)!=len(age_final_result)):
        print('NOT SAME LENGTH')
        print(array1)
        print(array2)
        print(final_result1)
        print(final_result2)

    return final_result1, final_result2, age_final_result, disposition