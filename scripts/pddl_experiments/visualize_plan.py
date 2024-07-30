import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Problem info (Input)
    parser.add_argument('--mt_file_name', default='one_tet_MT_layer_0_contact.json',
                        help='The name of the multi tangent file to solve (json file\'s name, e.g. "box_MT_layer_1.json")')

    # Planning Problem Scope
    parser.add_argument('--planning_cases', metavar='N', type=int, nargs='+',
                        help='Which planning case to parse')

    args = parser.parse_args()

    # Load process file
    mt_file_name = args.mt_file_name
    mt = parse_mt(mt_file_name)

    mt_name = os.path.splitext(os.path.basename(mt_file_name))[0]
    problem_name = mt_name

 