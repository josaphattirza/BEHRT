# GOAL: To ensure that code, med, age_on_admittance, and disposition all has the same length
# NOTE:
# initial length:
# code = age
# med = disposition
# NOTE 2:
# CASE 1 : both code and medicine has same amount of visit (same amount of SEP)
# CASE 2 : code has more visit (more SEP)
# CASE 3 : med has more visit (more SEP)
def handle_arrays(icd_code, medicine, age_on_admittance, disposition, revisit72, triage):
    code_sublists = []
    med_sublists = []
    age_sublists = []
    disposition_sublists = []
    revisit72_sublists = []
    triage_sublists = []

    temp = []
    temp_age = []
    temp_disposition = []
    temp_revisit72 = []
    temp_triage = []


    code_final_result = []
    med_final_result = []
    age_final_result = []
    disposition_final_result = []
    revisit72_final_result = []
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

    for item,disp,rev in zip(medicine,disposition,revisit72):
        if item == "SEP":
            med_sublists.append(temp)
            disposition_sublists.append(temp_disposition)
            revisit72_sublists.append(temp_revisit72)
            temp = []
            temp_disposition = []
            temp_revisit72 = []
        else:
            temp.append(item)
            temp_disposition.append(disp)
            temp_revisit72.append(rev)
    if temp:
        med_sublists.append(temp)
        disposition_sublists.append(temp_disposition)
        revisit72_sublists.append(temp_revisit72)

    for item in triage:
        if item == "SEP":
            triage_sublists.append(temp_triage)
            temp_triage = []
        else:
            temp_triage.append(item)
    if temp_triage:
        triage_sublists.append(item)

    data = [code_sublists, med_sublists, age_sublists, 
            disposition_sublists, revisit72_sublists, 
            triage_sublists]

    max_length_2nd_tier = 0

    for sublist in data:
        for inner_list in sublist:
            if len(inner_list) > max_length_2nd_tier:
                max_length_2nd_tier = len(inner_list)

    max_len = max(len(code_sublists), 
                  len(med_sublists), 
                  len(age_sublists), 
                  len(disposition_sublists),
                  len(revisit72_sublists), 
                  len(triage_sublists))

    # Create the new 2D list with 'UNK' elements
    code_sublists_new = [['UNK'] * max_length_2nd_tier for _ in range(max_len)]
    med_sublists_new = [['UNK'] * max_length_2nd_tier for _ in range(max_len)]
    age_sublists_new = [['UNK'] * max_length_2nd_tier for _ in range(max_len)]
    disposition_sublists_new = [['UNK'] * max_length_2nd_tier for _ in range(max_len)]
    revisit72_sublists_new = [['UNK'] * max_length_2nd_tier for _ in range(max_len)]
    triage_sublists_new = [['UNK'] * max_length_2nd_tier for _ in range(max_len)]

    # Transfer data from old_code to new_code
    for i in range(len(code_sublists)):
        for j in range(len(code_sublists[i])):
            code_sublists_new[i][j] = code_sublists[i][j]

    # Transfer data from old_code to new_code
    for i in range(len(med_sublists)):
        for j in range(len(med_sublists[i])):
            med_sublists_new[i][j] = med_sublists[i][j]

    # Transfer data from old_code to new_code
    for i in range(len(age_sublists)):
        for j in range(len(age_sublists[i])):
            age_sublists_new[i][j] = age_sublists[i][j]

    # Transfer data from old_code to new_code
    for i in range(len(disposition_sublists)):
        for j in range(len(disposition_sublists[i])):
            disposition_sublists_new[i][j] = disposition_sublists[i][j]
    
    # Transfer data from old_code to new_code
    for i in range(len(revisit72_sublists)):
        for j in range(len(revisit72_sublists[i])):
            revisit72_sublists_new[i][j] = revisit72_sublists[i][j]

    # Transfer data from old_code to new_code
    for i in range(len(triage_sublists)):
        for j in range(len(triage_sublists[i])):
            triage_sublists_new[i][j] = triage_sublists[i][j]

    for a,b,c,d,e,f in zip(code_sublists,
                         med_sublists, 
                         age_sublists, 
                         disposition_sublists, 
                         revisit72_sublists,
                         triage_sublists):
        # if len(a) > 8:
        #     print("LEN A MORE THAN 8", len(a))
        # if len(b) > 8:
        #     print("LEN B MORE THAN 8", len(b))
        # if len(c) > 8:
        #     print("LEN C MORE THAN 8", len(c))
        # if len(d) > 8:
        #     print("LEN D MORE THAN 8", len(d))
        # if len(e) > 8:
        #     print("LEN D MORE THAN 8", len(e))
        # if len(f) > 8:
        #     print("LEN D MORE THAN 8", len(f))
        pass
                
    for sublist in code_sublists_new:
        code_final_result.extend(sublist)
        code_final_result.append('SEP')

    for sublist in med_sublists_new:
        med_final_result.extend(sublist)
        med_final_result.append('SEP')

    for sublist in age_sublists_new:
        age_final_result.extend(sublist)
        age_final_result.append(sublist[0])

    for sublist in disposition_sublists_new:
        disposition_final_result.extend(sublist)
        disposition_final_result.append(sublist[0])

    for sublist in revisit72_sublists_new:
        revisit72_final_result.extend(sublist)
        revisit72_final_result.append(sublist[0])

    for sublist in triage_sublists_new:
        triage_final_result.extend(sublist)
        triage_final_result.append('SEP')

    # print(len(final_result1))
    # print(len(final_result2))
    # print(len(age_final_result))
    # print(len(disposition_final_result))
    # print("=======")


    if(len(code_final_result)==len(med_final_result)== \
       len(age_final_result)==len(disposition_final_result)== \
       len(revisit72_final_result)== \
        len(triage_final_result)):
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

    return code_final_result, med_final_result, age_final_result, disposition_final_result, revisit72_final_result, triage_final_result

