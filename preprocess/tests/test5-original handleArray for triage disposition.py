# GOAL: To ensure that code, med, age_on_admittance, and disposition all has the same length
# NOTE:
# initial length:
# code = age
# med = disposition
# NOTE 2:
# CASE 1 : both code and medicine has same amount of visit (same amount of SEP)
# CASE 2 : code has more visit (more SEP)
# CASE 3 : med has more visit (more SEP)
def handle_arrays(icd_code, medicine, age_on_admittance, disposition, triage):
    code_sublists = []
    med_sublists = []
    age_sublists = []
    disposition_sublists = []
    triage_sublists = []

    temp = []
    temp_age = []
    temp_disposition = []
    temp_triage = []


    code_final_result = []
    med_final_result = []
    age_final_result = []
    disposition_final_result = []
    triage_final_result = []


    for item, age in zip(icd_code,age_on_admittance):
        if item == "SEP":
            code_sublists.append(temp)
            age_sublists.append(temp_age)
            temp = []
            temp_age = []
        else:
            temp.append(item)
            temp_age.append(age)
    if temp:
        code_sublists.append(temp)
        age_sublists.append(temp_age)

    for item,disp in zip(medicine,disposition):
        if item == "SEP":
            med_sublists.append(temp)
            disposition_sublists.append(temp_disposition)
            temp = []
            temp_disposition = []
        else:
            temp.append(item)
            temp_disposition.append(disp)
    if temp:
        med_sublists.append(temp)
        disposition_sublists.append(temp_disposition)

    for item in triage:
        if item == "SEP":
            triage_sublists.append(temp_triage)
            temp_triage = []
        else:
            temp_triage.append(item)
    if temp_triage:
        triage_sublists.append(item)



    for a,b,c,d,e in zip(code_sublists,med_sublists, age_sublists, disposition_sublists, triage_sublists):
        if len(a) > len(b):
            diff = len(a) - len(b)
            for _ in range(diff):
                # a.append('UNK')
                b.append('UNK')
                # c.append(c[0])
                d.append(d[0])
        
        if len(b) > len(a):
            diff = len(b) - len(a)
            for _ in range(diff):
                a.append('UNK')
                # b.append('UNK')
                c.append(c[0])
                # d.append(d[0])
                # e.append('UNK')
        
        # Since the step before ensures that len(a) = b = c = d,
        # we just need to compare to triage that has fixed length
        
        if len(a) > len(e):
            diff2 = len(a) - len(e)
            for _ in range(diff2):
                e.append('UNK')
        if len(e) > len(a):
            diff2 = len(e) - len(a)
            for _ in range(diff2):
                a.append('UNK')
                b.append('UNK')
                c.append(c[0])
                d.append(d[0])

    # Since if it is CASE 3, length of med, disposition and triage need to be corrected
    if len(med_sublists) > len(code_sublists):
        for b,d,e in zip(med_sublists, disposition_sublists, triage_sublists):
            if len(b) > len(e):
                diff2 = len(b) - len(e)
                for _ in range(diff2):
                    e.append('UNK')
            if len(e) > len(b):
                diff2 = len(e) - len(b)
                for _ in range(diff2):
                    b.append('UNK')
                    d.append(d[0])


                
    for sublist in code_sublists:
        code_final_result.extend(sublist)
        code_final_result.append('SEP')

    for sublist in med_sublists:
        med_final_result.extend(sublist)
        med_final_result.append('SEP')

    for sublist in age_sublists:
        age_final_result.extend(sublist)
        age_final_result.append(sublist[0])

    for sublist in disposition_sublists:
        disposition_final_result.extend(sublist)
        disposition_final_result.append(sublist[0])

    for sublist in triage_sublists:
        triage_final_result.extend(sublist)
        triage_final_result.append('SEP')


    if len(code_final_result) > len(med_final_result):
        # print("CASE 2")
        for _ in range(len(code_final_result)-len(med_final_result)-1):
            med_final_result.append('UNK')
            disposition_final_result.append(disposition_final_result[-1])


        med_final_result.append('SEP')
        disposition_final_result.append(disposition_final_result[-1])
        
        for _ in range(len(code_final_result)-len(triage_final_result)-1):
            triage_final_result.append('UNK')
        triage_final_result.append('UNK')


    elif len(med_final_result) > len(code_final_result):
        # print("CASE 3")
        # print(len(code_final_result))
        # print(len(age_final_result))
        # print(len(med_final_result))
        # print(len(disposition_final_result))
        # print(len(triage_final_result))
        # print("====")
        # print(len(code_sublists))
        # print(len(age_sublists))
        # print(len(med_sublists))
        # print(len(disposition_sublists))
        # print(len(triage_sublists))
        for _ in range(len(med_final_result)-len(code_final_result)-1):
            code_final_result.append('UNK')
            age_final_result.append(age_final_result[-1])

        code_final_result.append('SEP')
        age_final_result.append(age_final_result[-1])

        
    else:  
        # print("CASE 1")
        pass

    # print(len(final_result1))
    # print(len(final_result2))
    # print(len(age_final_result))
    # print(len(disposition_final_result))
    # print("=======")


    if(len(code_final_result)==len(med_final_result)==len(age_final_result)==len(disposition_final_result)==len(triage_final_result)):
        # print("ALL HAS SAME LENGTH")
        pass
    else:
        print('NOT SAME LENGTH')
        print(code_final_result)
        print(age_final_result)
        print(med_final_result)
        print(disposition_final_result)
        print(triage_final_result)
        print(len(code_final_result))
        print(len(age_final_result))
        print(len(med_final_result))
        print(len(disposition_final_result))
        print(len(triage_final_result))

    return code_final_result, med_final_result, age_final_result, disposition_final_result, triage_final_result

