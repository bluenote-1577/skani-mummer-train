import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

skani_file_elgg2 = './c125-elgg_part_2_skani_triangle.txt'
anim_file_elgg2 = './anim-elgg_part_2.txt'
#c200_skani_file_elgg2 = 'results/c200-elgg_part_2_skani_triangle.txt'

skani_file_nayfach = './c125_nayfach_skani.txt'
anim_file_nayfach = './anim-nayfach.txt'
#c200_skani_file_nayfach = 'results/c200_nayfach_skani.txt'


c125skani_files = [skani_file_nayfach, skani_file_elgg2]
#c200skani_files = [c200_skani_file_nayfach, c200_skani_file_elgg2]
anim_files = [anim_file_nayfach, anim_file_elgg2]
all_skani = [c125skani_files]

train_file = 'c125_latest_train.csv'
test_file = 'c125_latest_test.csv'

train_f = open(train_file,'w')
test_f = open(test_file,'w')

diff_cut = 2.50

for (skani_c,skani_files) in enumerate(all_skani):
    for i in range(len(anim_files)):
        skani_file = skani_files[i]
        l = 0
        anim_file = anim_files[i]
        refs_to_vec_dict = dict()
        for line in open(skani_file,'r'):
            if 'ANI' in line:
                continue
            spl = line.split('\t')
            main_ref = spl[0]
            main_ref = main_ref.split("/")[-1]
            other_ref = spl[1]
            other_ref = other_ref.split("/")[-1]
            contig_quants_r = spl[-8:-5]
            contig_quants_q = spl[-5:-2]
            num_contigs_r = float(spl[7])
            num_contigs_q = float(spl[8])
            #ci_std = [float(x) for x in spl[-9:-6]]
            std = float(spl[11])
            ci_lower = float(spl[9])
            ci_upper = float(spl[10])
            avg_chain_length = float(spl[-2])
            total_bases = float(spl[-1])
            if float(contig_quants_r[1]) > float(contig_quants_q[1]):
                #refs_to_vec_dict[(main_ref,other_ref)] = [float(spl[2]), float(spl[3]), float(spl[4]), ci_lower, ci_upper, num_contigs_r, num_contigs_q, std] + [float(x) for x in contig_quants_r] + [float(x) for x in contig_quants_q]
                #refs_to_vec_dict[(main_ref,other_ref)] = [float(spl[2]), float(spl[3]), float(spl[4]), ci_lower, ci_upper, std] + [float(x) for x in contig_quants_r] + [float(x) for x in contig_quants_q] + [avg_chain_length]
                refs_to_vec_dict[(main_ref,other_ref)] = [float(spl[2]), float(spl[3]), float(spl[4]), std] + [float(x) for x in contig_quants_r] + [float(x) for x in contig_quants_q]
            else:
                #refs_to_vec_dict[(main_ref,other_ref)] = [float(spl[2]), float(spl[4]), float(spl[3]), ci_lower, ci_upper, num_contigs_q, num_contigs_r, std] + [float(x) for x in contig_quants_q] + [float(x) for x in contig_quants_r]
                #refs_to_vec_dict[(main_ref,other_ref)] = [float(spl[2]), float(spl[4]), float(spl[3]), ci_lower, ci_upper, std] + [float(x) for x in contig_quants_q] + [float(x) for x in contig_quants_r] + [avg_chain_length]
                refs_to_vec_dict[(main_ref,other_ref)] = [float(spl[2]), float(spl[4]), float(spl[3]), std] + [float(x) for x in contig_quants_q] + [float(x) for x in contig_quants_r] 
            l = len(refs_to_vec_dict[(main_ref,other_ref)]) + 1


        count = 0
        for line in open(anim_file,'r'):
            if 'reference' in line:
                continue

            spl = line.split('\t')
            main_ref = spl[0]
            main_ref = main_ref.split("/")[-1]
            other_ref = spl[1]
            other_ref = other_ref.split("/")[-1]
            if (main_ref,other_ref) not in refs_to_vec_dict:
                print(skani_files[i], anim_files[i], main_ref, other_ref)
            else:
                refs_to_vec_dict[(main_ref,other_ref)].append(float(spl[2]))

        eprint(l)

        for (key,vec) in refs_to_vec_dict.items():
            if len(vec) == l:
                count += 1
                if abs(vec[0] - vec[-1]) < diff_cut:
                    string = ','.join([str(x) for x in vec])
                    if len(anim_files) > 1:
                        if i != len(anim_files) - 1:
                            train_f.write(string + '\n')
                        else:
                            test_f.write(string + '\n')
                    else:
                        if hash(key[0]) % 4 == 0 and hash(key[1]) % 4 == 0:
                            test_f.write(string + '\n')
                        elif hash(key[0]) % 4 != 0 and hash(key[1]) %4 != 0:
                            train_f.write(string + '\n')

