\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.16518 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.913793 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.04651 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.02093 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.954092 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.09252 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.04915 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.840567 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.40752 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.00871 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.745477 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.30887 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.00785 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.891196 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.31986 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.804312 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.747772 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.834071 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.921844 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.15145 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.868553 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.698194 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.905014 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.25375 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.897509 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.01539 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.620311 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.12667 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.859387 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.03883 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.10979 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.07177 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.00733 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.52699 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.07791 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.78575 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.12674 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.29905 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19582 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.855174 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.949044 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.12176 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.883581 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.34563 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.08529 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.759699 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.82161 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.16397 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.661895 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.44418 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.24647 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.948152 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.20055 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.698094 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.613401 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.760549 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.05543 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.57946 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.978428 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.555921 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.892563 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.3799 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.918726 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.862652 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.75801 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.27949 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.688524 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.13491 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.09818 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.836873 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.826062 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.4397 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.16039 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.733165 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.02174 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.37465 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25036 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.844224 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.03125 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.09812 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.832611 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.30536 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.26818 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.743513 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 2.0274 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.857306 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.623352 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.72114 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.12249 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.835068 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.26643 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.729546 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.570762 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.799636 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.02031 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.57674 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.873881 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.477924 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.893566 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.6329 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.76875 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.920736 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.751174 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.27778 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.733182 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.15394 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.19877 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.894598 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.908246 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.65956 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.27795 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.634615 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.14941 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.46569 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15273 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.902907 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.03918 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.980613 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.891626 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.19041 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.21459 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.747277 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.54182 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.06562 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.75502 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.4656 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.11859 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.924014 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.2827 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.749467 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.684667 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.910258 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.024 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.31781 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.960752 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.652656 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.0178 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.27708 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.880013 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.06347 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.698897 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.35381 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.730298 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.05507 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.20266 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.937834 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.838065 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.32691 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.956668 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.83317 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.962725 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.32484 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17798 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.913887 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.77841 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.24875 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.906565 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.21785 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.946988 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.14242 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.66931 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.773221 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.663096 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.14351 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.32425 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.00606 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.893895 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.09198 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.804129 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.734008 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.25964 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.62269 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.02714 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.570568 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.772269 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.41226 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.923117 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.530743 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.31632 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.673533 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.837287 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.971106 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.765814 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.79914 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.15635 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.47584 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 2.02299 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.59502 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.54633 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.09333 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12774 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.944359 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.03744 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.06138 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.95605 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01713 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.20932 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.785105 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.26822 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.936225 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.826341 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.24939 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.00098 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.03941 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.919114 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.778019 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.824113 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.930498 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.05161 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.0732 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.957106 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.599581 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.888074 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.09312 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.901526 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.961279 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.836616 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.989333 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.776632 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.916955 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.20051 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.981855 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.07804 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.38349 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.17429 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.944438 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.16593 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.46112 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1437 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.866557 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.892896 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.17678 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.979982 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.27726 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.16426 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.941934 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.70381 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.90942 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.722512 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.53499 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.18031 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.986696 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.970767 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.889825 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.682356 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.776277 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.02286 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.47112 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.929121 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.593426 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.904157 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.33532 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.853475 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.759813 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.919148 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.866458 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.776608 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.06776 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.01804 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.776111 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.07621 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.29456 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.45227 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.61621 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.33091 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.16151 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.2309 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.826444 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.775412 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.29795 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.949413 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.54416 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.970103 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.11047 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.07386 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.927147 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.459459 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.49079 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.46956 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.00256 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.12488 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.916792 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.56356 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.707911 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.29768 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.50231 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.14771 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.585035 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.84939 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.61773 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.967984 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.532317 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.23171 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.873495 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.761955 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.941084 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.875578 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.645647 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.04688 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.54647 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.92362 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.43975 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.28482 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.910699 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15658 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.919641 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.03057 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.970395 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.913422 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.08786 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.24814 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.726756 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.39371 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.917112 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.762384 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.37378 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.03274 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.905292 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.17575 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.862841 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.681666 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.938135 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.04691 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.08421 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.936807 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.731649 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.972324 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.27925 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.887779 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.997148 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.820212 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.22906 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.868012 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.13445 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.1609 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.955287 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.02291 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.26763 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.07058 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.829122 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.13004 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.27049 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13651 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.949999 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.906256 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.08502 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.92527 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10808 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.15588 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.880599 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.44187 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.991007 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.83264 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.34897 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.12558 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.03152 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.06373 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.819332 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.779648 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.777937 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.935842 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.10542 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.956168 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.662879 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.948139 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.23682 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.869455 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.916274 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.860283 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.07888 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.851161 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.03775 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.12594 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.918258 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.997416 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.15874 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.163 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.78698 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.05345 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.17972 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.291999639531637 z_k1_s0000
                                 - 0.178999650207964 z_k1_s0001
                                 - 0.503998637932548 z_k1_s0002
                                 - 0.380999372216199 z_k1_s0003
                                 - 0.494000004538804 z_k1_s0004
                                 - 0.676997500606936 z_k1_s0005
                                 - 0.135999739624478 z_k1_s0006
                                 - 0.551998508211838 z_k1_s0007
                                 - 0.562000547730749 z_k1_s0010
                                 - 0.383001344958927 z_k1_s0011
                                 - 0.179000621177121 z_k1_s0012
                                 - 0.304001134200546 z_k1_s0013
                                 - 0.355002024348648 z_k1_s0014
                                 - 0.673001400449034 z_k1_s0015
                                 - 0.276001104387617 z_k1_s0016
                                 - 0.814001525249762 z_k1_s0017
                                 - 0.210000360015901 z_k1_s0020
                                 - 0.122000145090113 z_k1_s0021
                                 - 0.284000510173914 z_k1_s0022
                                 - 0.139000232472423 z_k1_s0023
                                 - 0.0660001306193044 z_k1_s0024
                                 - 0.0750000761627588 z_k1_s0025
                                 - 0.155000760750428 z_k1_s0026
                                 - 0.172000481232296 z_k1_s0027
                                 - 0.178000034994139 z_k1_s0030
                                 - 0.174000150955042 z_k1_s0031
                                 - 0.413999427606047 z_k1_s0032
                                 - 0.245001267526658 z_k1_s0033
                                 - 0.383001076728674 z_k1_s0034
                                 - 0.144000212511143 z_k1_s0035
                                 - 0.162000037037367 z_k1_s0036
                                 - 0.609001567393515 z_k1_s0037
                                 - 0.59400066756026 z_k1_s0040
                                 - 0.386999239476022 z_k1_s0041
                                 - 0.363999577876978 z_k1_s0042
                                 - 0.552000825762741 z_k1_s0043
                                 - 0.582999450162924 z_k1_s0044
                                 - 0.750998817352999 z_k1_s0045
                                 - 0.33199965967893 z_k1_s0046
                                 - 0.503998101570225 z_k1_s0047
                                 - 0.241999882426992 z_k1_s0050
                                 - 0.143000041843217 z_k1_s0051
                                 - 0.573999041075626 z_k1_s0052
                                 - 0.515000184945851 z_k1_s0053
                                 - 0.294999178648639 z_k1_s0054
                                 - 0.161999872718898 z_k1_s0055
                                 - 0.220000026697259 z_k1_s0056
                                 - 0.297999203502692 z_k1_s0057
                                 - 0.602000393031966 z_k1_s0060
                                 - 0.156999983034928 z_k1_s0061
                                 - 0.351999545842688 z_k1_s0062
                                 - 0.229999955914228 z_k1_s0063
                                 - 0.241999996579313 z_k1_s0064
                                 - 0.317999979624885 z_k1_s0065
                                 - 0.388000216837883 z_k1_s0066
                                 - 0.455999349273489 z_k1_s0067
                                 - 0.657002961800673 z_k1_s0070
                                 - 0.258000634144882 z_k1_s0071
                                 - 0.203000553785035 z_k1_s0072
                                 - 0.264000462187418 z_k1_s0073
                                 - 0.503000748829177 z_k1_s0074
                                 - 0.236000372620491 z_k1_s0075
                                 - 0.994001076971772 z_k1_s0076
                                 - 0.708001882462981 z_k1_s0077
                                 - 0.496999959915096 z_k1_s0100
                                 - 0.445000098064079 z_k1_s0101
                                 - 0.790000379199243 z_k1_s0102
                                 - 0.912998854723233 z_k1_s0103
                                 - 0.625998832499003 z_k1_s0104
                                 - 0.924998575042589 z_k1_s0105
                                 - 0.762001031787413 z_k1_s0106
                                 - 0.411000353592334 z_k1_s0107
                                 - 0.289999072646113 z_k1_s0110
                                 - 0.300999417994528 z_k1_s0111
                                 - 0.153999994880745 z_k1_s0112
                                 - 0.147000003913433 z_k1_s0113
                                 - 0.169999863606096 z_k1_s0114
                                 - 0.378998874830557 z_k1_s0115
                                 - 0.113999935867601 z_k1_s0116
                                 - 0.381000697620114 z_k1_s0117
                                 - 0.147999799214253 z_k1_s0120
                                 - 0.141999807354216 z_k1_s0121
                                 - 0.55499924705345 z_k1_s0122
                                 - 0.194999735451212 z_k1_s0123
                                 - 0.137999812780858 z_k1_s0124
                                 - 0.381999481755708 z_k1_s0125
                                 - 0.143999804640895 z_k1_s0126
                                 - 0.295999598428506 z_k1_s0127
                                 - 0.514000078154715 z_k1_s0130
                                 - 0.569001650968674 z_k1_s0131
                                 - 0.47000113950316 z_k1_s0132
                                 - 0.48599864483135 z_k1_s0133
                                 - 0.21400007465896 z_k1_s0134
                                 - 0.551001557974944 z_k1_s0135
                                 - 0.575999109799999 z_k1_s0136
                                 - 0.48599864483135 z_k1_s0137
                                 - 0.525000216450549 z_k1_s0140
                                 - 0.539997676127411 z_k1_s0141
                                 - 0.551999071859685 z_k1_s0142
                                 - 0.776998674911237 z_k1_s0143
                                 - 0.203999388714083 z_k1_s0144
                                 - 0.281999205398744 z_k1_s0145
                                 - 0.168999602816765 z_k1_s0146
                                 - 0.37899866099975 z_k1_s0147
                                 - 0.326998795093142 z_k1_s0150
                                 - 0.333998678525131 z_k1_s0151
                                 - 0.629997218850851 z_k1_s0152
                                 - 0.215999101534376 z_k1_s0153
                                 - 0.445000299576855 z_k1_s0154
                                 - 0.447999478621953 z_k1_s0155
                                 - 0.401997931648759 z_k1_s0156
                                 - 1.05299711378713 z_k1_s0157
                                 - 0.101999950595955 z_k1_s0160
                                 - 0.228000834273039 z_k1_s0161
                                 - 0.0979999167490653 z_k1_s0162
                                 - 0.234999707255438 z_k1_s0163
                                 - 0.417999704501098 z_k1_s0164
                                 - 0.211999238885901 z_k1_s0165
                                 - 0.31400028448154 z_k1_s0166
                                 - 0.217999837156078 z_k1_s0167
                                 - 0.28499984866114 z_k1_s0170
                                 - 0.12000001472574 z_k1_s0171
                                 - 0.0949999947203783 z_k1_s0172
                                 - 0.267000029377272 z_k1_s0173
                                 - 0.143999909270893 z_k1_s0174
                                 - 0.109999979623597 z_k1_s0175
                                 - 0.121000004685955 z_k1_s0176
                                 - 0.213000571525667 z_k1_s0177
                                 - 0.127000078002099 z_k1_s1000
                                 - 0.156000046449835 z_k1_s1001
                                 - 0.167000026761047 z_k1_s1002
                                 - 0.307999000914053 z_k1_s1003
                                 - 0.113999999497965 z_k1_s1004
                                 - 0.565000311757377 z_k1_s1005
                                 - 0.39999948758948 z_k1_s1006
                                 - 0.401998914082572 z_k1_s1007
                                 - 0.525000594696941 z_k1_s1010
                                 - 0.33400067555971 z_k1_s1011
                                 - 0.996999808238195 z_k1_s1012
                                 - 1.01500284533702 z_k1_s1013
                                 - 0.65000106362535 z_k1_s1014
                                 - 0.571000794758643 z_k1_s1015
                                 - 1.23100492046349 z_k1_s1016
                                 - 1.55100392123642 z_k1_s1017
                                 - 0.406000061929071 z_k1_s1020
                                 - 0.507000454124375 z_k1_s1021
                                 - 0.878999456929927 z_k1_s1022
                                 - 0.518000060240531 z_k1_s1023
                                 - 0.74200114566386 z_k1_s1024
                                 - 0.993002303590872 z_k1_s1025
                                 - 1.0060025804557 z_k1_s1026
                                 - 0.393000329464448 z_k1_s1027
                                 - 0.270000790852567 z_k1_s1030
                                 - 0.235000145593944 z_k1_s1031
                                 - 0.241999585045722 z_k1_s1032
                                 - 0.0929999659159083 z_k1_s1033
                                 - 0.270000790852567 z_k1_s1034
                                 - 0.26899914693102 z_k1_s1035
                                 - 0.28599950959949 z_k1_s1036
                                 - 0.0589999301789133 z_k1_s1037
                                 - 0.47900078280597 z_k1_s1040
                                 - 0.480997765374025 z_k1_s1041
                                 - 0.514998286941916 z_k1_s1042
                                 - 0.437000372986409 z_k1_s1043
                                 - 0.581998845797093 z_k1_s1044
                                 - 0.701997169572161 z_k1_s1045
                                 - 0.612997914523175 z_k1_s1046
                                 - 0.176999620214629 z_k1_s1047
                                 - 0.187999775396409 z_k1_s1050
                                 - 0.105999870206488 z_k1_s1051
                                 - 0.0809999190058884 z_k1_s1052
                                 - 0.352000178975826 z_k1_s1053
                                 - 0.187999775396409 z_k1_s1054
                                 - 0.0979999807342284 z_k1_s1055
                                 - 0.114999894162675 z_k1_s1056
                                 - 0.355000285827818 z_k1_s1057
                                 - 0.573999228192018 z_k1_s1060
                                 - 0.205999146971956 z_k1_s1061
                                 - 0.662997339363818 z_k1_s1062
                                 - 0.630997014406655 z_k1_s1063
                                 - 0.760996811300416 z_k1_s1064
                                 - 0.51799662852403 z_k1_s1065
                                 - 0.175999350072854 z_k1_s1066
                                 - 0.532997542470057 z_k1_s1067
                                 - 1.06100058998002 z_k1_s1070
                                 - 0.952003968574393 z_k1_s1071
                                 - 0.504001173025471 z_k1_s1072
                                 - 0.674001881699469 z_k1_s1073
                                 - 1.27000166296793 z_k1_s1074
                                 - 0.87700062988768 z_k1_s1075
                                 - 0.916000994201594 z_k1_s1076
                                 - 0.605001553714047 z_k1_s1077
                                 - 0.536000687075634 z_k1_s1100
                                 - 0.387998667582938 z_k1_s1101
                                 - 0.456001804378887 z_k1_s1102
                                 - 0.452000745743037 z_k1_s1103
                                 - 0.719001753372735 z_k1_s1104
                                 - 0.549998819796046 z_k1_s1105
                                 - 0.380000265314614 z_k1_s1106
                                 - 0.234000261141193 z_k1_s1107
                                 - 0.320999968864974 z_k1_s1110
                                 - 0.6730002315522 z_k1_s1111
                                 - 0.884001815684181 z_k1_s1112
                                 - 0.682997950590915 z_k1_s1113
                                 - 0.231000244310389 z_k1_s1114
                                 - 0.284000123019213 z_k1_s1115
                                 - 1.11399998659 z_k1_s1116
                                 - 0.413000157827581 z_k1_s1117
                                 - 0.379000856135737 z_k1_s1120
                                 - 0.664002232533322 z_k1_s1121
                                 - 0.214000651712099 z_k1_s1122
                                 - 0.616000262900503 z_k1_s1123
                                 - 0.686002093029447 z_k1_s1124
                                 - 0.322000497476041 z_k1_s1125
                                 - 0.745001908405828 z_k1_s1126
                                 - 0.543000915300134 z_k1_s1127
                                 - 0.0880000153713562 z_k1_s1130
                                 - 0.249999979975402 z_k1_s1131
                                 - 0.195000320681035 z_k1_s1132
                                 - 0.220999690838063 z_k1_s1133
                                 - 0.10300001416988 z_k1_s1134
                                 - 0.0770000835124829 z_k1_s1135
                                 - 0.125000550488071 z_k1_s1136
                                 - 0.0620000847139588 z_k1_s1137
                                 - 0.109999928441515 z_k1_s1140
                                 - 0.0930000113550736 z_k1_s1141
                                 - 0.254000420655652 z_k1_s1142
                                 - 0.0680000331456363 z_k1_s1143
                                 - 0.238000298409661 z_k1_s1144
                                 - 0.120999982085645 z_k1_s1145
                                 - 0.260000162498008 z_k1_s1146
                                 - 0.0719999421071778 z_k1_s1147
                                 - 0.214000379199007 z_k1_s1150
                                 - 0.208000418076709 z_k1_s1151
                                 - 0.383001579731894 z_k1_s1152
                                 - 0.126000243070485 z_k1_s1153
                                 - 0.212000038990831 z_k1_s1154
                                 - 0.34100079237358 z_k1_s1155
                                 - 0.255000996453233 z_k1_s1156
                                 - 0.380001069419631 z_k1_s1157
                                 - 0.084999911830659 z_k1_s1160
                                 - 0.159000167852494 z_k1_s1161
                                 - 0.131999732104142 z_k1_s1162
                                 - 0.0769999762606892 z_k1_s1163
                                 - 0.120000169549032 z_k1_s1164
                                 - 0.323999548982799 z_k1_s1165
                                 - 0.155999993213845 z_k1_s1166
                                 - 0.0829999847381408 z_k1_s1167
                                 - 0.188000311447962 z_k1_s1170
                                 - 0.205000162157949 z_k1_s1171
                                 - 0.178000254388982 z_k1_s1172
                                 - 0.305999999582809 z_k1_s1173
                                 - 0.431999867704992 z_k1_s1174
                                 - 0.364001360466064 z_k1_s1175
                                 - 0.264999385010558 z_k1_s1176
                                 - 0.3010003069037 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.14874 w_k1_ADM_s000
                                 + 1.13977 w_k1_AXP_s000
                                 + 0.997618 w_k1_DIS_s000
                                 + 0.903617 w_k1_FDX_s000
                                 + 1.40355 w_k1_IBM_s000
                                 + 0.866287 w_k1_INTC_s000
                                 + 1.22656 w_k1_KO_s000 + 1.73561 w_k1_MAT_s000
                                 + 0.98545 w_k1_MSFT_s000 + 1.0723 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.24637 w_k1_ADM_s000
                                 + 1.38411 w_k1_AXP_s000
                                 + 1.02808 w_k1_DIS_s000
                                 + 1.35493 w_k1_FDX_s000
                                 + 0.731326 w_k1_IBM_s000
                                 + 0.89507 w_k1_INTC_s000
                                 + 0.858921 w_k1_KO_s000
                                 + 1.01864 w_k1_MAT_s000
                                 + 0.961597 w_k1_MSFT_s000
                                 + 1.00685 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.758208 w_k1_ADM_s000
                                 + 0.864032 w_k1_AXP_s000
                                 + 0.710266 w_k1_DIS_s000
                                 + 0.83135 w_k1_FDX_s000
                                 + 0.775729 w_k1_IBM_s000
                                 + 0.791444 w_k1_INTC_s000
                                 + 0.862429 w_k1_KO_s000
                                 + 0.641678 w_k1_MAT_s000
                                 + 0.830694 w_k1_MSFT_s000
                                 + 0.886575 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.61696 w_k1_ADM_s000
                                 + 1.36062 w_k1_AXP_s000
                                 + 1.46554 w_k1_DIS_s000
                                 + 1.26882 w_k1_FDX_s000
                                 + 1.45914 w_k1_IBM_s000
                                 + 1.01399 w_k1_INTC_s000
                                 + 1.44587 w_k1_KO_s000 + 1.71404 w_k1_MAT_s000
                                 + 1.21618 w_k1_MSFT_s000
                                 + 1.21471 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.30009 w_k1_ADM_s000
                                 + 1.42432 w_k1_AXP_s000
                                 + 1.17727 w_k1_DIS_s000
                                 + 1.15052 w_k1_FDX_s000 + 1.0746 w_k1_IBM_s000
                                 + 0.902729 w_k1_INTC_s000
                                 + 1.09165 w_k1_KO_s000 + 1.17926 w_k1_MAT_s000
                                 + 0.99986 w_k1_MSFT_s000
                                 + 1.02588 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.957254 w_k1_ADM_s000
                                 + 0.971926 w_k1_AXP_s000
                                 + 1.02012 w_k1_DIS_s000
                                 + 1.00974 w_k1_FDX_s000
                                 + 0.924147 w_k1_IBM_s000
                                 + 0.977516 w_k1_INTC_s000
                                 + 0.970396 w_k1_KO_s000
                                 + 0.774256 w_k1_MAT_s000
                                 + 0.879399 w_k1_MSFT_s000
                                 + 1.01208 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.836539 w_k1_ADM_s000
                                 + 0.964881 w_k1_AXP_s000
                                 + 0.847286 w_k1_DIS_s000
                                 + 0.842206 w_k1_FDX_s000
                                 + 1.42128 w_k1_IBM_s000
                                 + 0.894823 w_k1_INTC_s000
                                 + 1.0506 w_k1_KO_s000 + 1.17826 w_k1_MAT_s000
                                 + 0.800957 w_k1_MSFT_s000
                                 + 0.921827 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.863494 w_k1_ADM_s000
                                 + 0.885323 w_k1_AXP_s000
                                 + 0.772888 w_k1_DIS_s000
                                 + 0.831812 w_k1_FDX_s000
                                 + 1.00529 w_k1_IBM_s000
                                 + 0.820373 w_k1_INTC_s000
                                 + 0.884177 w_k1_KO_s000
                                 + 0.885332 w_k1_MAT_s000
                                 + 0.779817 w_k1_MSFT_s000
                                 + 0.866212 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.948227 w_k1_ADM_s001
                                 + 1.12542 w_k1_AXP_s001 + 1.0215 w_k1_DIS_s001
                                 + 1.01387 w_k1_FDX_s001
                                 + 1.08729 w_k1_IBM_s001
                                 + 1.07092 w_k1_INTC_s001
                                 + 1.03718 w_k1_KO_s001
                                 + 0.913212 w_k1_MAT_s001
                                 + 1.19418 w_k1_MSFT_s001 + 1.1238 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.720651 w_k1_ADM_s001
                                 + 0.670995 w_k1_AXP_s001
                                 + 0.645853 w_k1_DIS_s001
                                 + 0.776071 w_k1_FDX_s001
                                 + 0.651432 w_k1_IBM_s001
                                 + 0.798937 w_k1_INTC_s001
                                 + 0.738354 w_k1_KO_s001
                                 + 0.588437 w_k1_MAT_s001
                                 + 0.858998 w_k1_MSFT_s001
                                 + 0.81887 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.89182 w_k1_ADM_s001
                                 + 1.15817 w_k1_AXP_s001
                                 + 1.55727 w_k1_DIS_s001
                                 + 1.33088 w_k1_FDX_s001
                                 + 0.882753 w_k1_IBM_s001
                                 + 1.42648 w_k1_INTC_s001 + 1.1618 w_k1_KO_s001
                                 + 0.913358 w_k1_MAT_s001
                                 + 1.50962 w_k1_MSFT_s001
                                 + 1.11938 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.830843 w_k1_ADM_s001
                                 + 1.09351 w_k1_AXP_s001
                                 + 1.21519 w_k1_DIS_s001
                                 + 1.06473 w_k1_FDX_s001
                                 + 1.65905 w_k1_IBM_s001
                                 + 1.18286 w_k1_INTC_s001
                                 + 1.29047 w_k1_KO_s001 + 1.45447 w_k1_MAT_s001
                                 + 1.17025 w_k1_MSFT_s001
                                 + 1.33976 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.941472 w_k1_ADM_s001
                                 + 1.1115 w_k1_AXP_s001 + 1.0698 w_k1_DIS_s001
                                 + 1.08861 w_k1_FDX_s001
                                 + 1.23807 w_k1_IBM_s001
                                 + 1.21029 w_k1_INTC_s001
                                 + 1.29382 w_k1_KO_s001 + 1.16133 w_k1_MAT_s001
                                 + 1.33777 w_k1_MSFT_s001
                                 + 1.21165 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.929976 w_k1_ADM_s001
                                 + 1.23349 w_k1_AXP_s001
                                 + 1.12454 w_k1_DIS_s001
                                 + 1.13995 w_k1_FDX_s001
                                 + 0.891722 w_k1_IBM_s001
                                 + 1.15266 w_k1_INTC_s001
                                 + 1.04518 w_k1_KO_s001
                                 + 0.899342 w_k1_MAT_s001
                                 + 1.24629 w_k1_MSFT_s001
                                 + 0.960989 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.00792 w_k1_ADM_s001
                                 + 1.52171 w_k1_AXP_s001 + 1.7696 w_k1_DIS_s001
                                 + 1.30656 w_k1_FDX_s001
                                 + 1.40674 w_k1_IBM_s001
                                 + 1.38764 w_k1_INTC_s001
                                 + 1.55878 w_k1_KO_s001 + 1.43379 w_k1_MAT_s001
                                 + 1.4122 w_k1_MSFT_s001 + 1.21608 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.893815 w_k1_ADM_s001
                                 + 0.810953 w_k1_AXP_s001
                                 + 0.690254 w_k1_DIS_s001
                                 + 0.845912 w_k1_FDX_s001
                                 + 0.897258 w_k1_IBM_s001
                                 + 1.06499 w_k1_INTC_s001
                                 + 0.887032 w_k1_KO_s001
                                 + 0.712052 w_k1_MAT_s001
                                 + 0.97843 w_k1_MSFT_s001
                                 + 0.864597 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.786323 w_k1_ADM_s002
                                 + 1.21647 w_k1_AXP_s002 + 1.1559 w_k1_DIS_s002
                                 + 1.07629 w_k1_FDX_s002
                                 + 1.23273 w_k1_IBM_s002
                                 + 1.49013 w_k1_INTC_s002
                                 + 1.14146 w_k1_KO_s002 + 1.09274 w_k1_MAT_s002
                                 + 1.1017 w_k1_MSFT_s002 + 1.16535 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.605601 w_k1_ADM_s002
                                 + 0.711347 w_k1_AXP_s002
                                 + 0.541425 w_k1_DIS_s002
                                 + 0.676059 w_k1_FDX_s002
                                 + 0.93931 w_k1_IBM_s002
                                 + 0.886388 w_k1_INTC_s002
                                 + 0.741341 w_k1_KO_s002
                                 + 0.772294 w_k1_MAT_s002
                                 + 0.586945 w_k1_MSFT_s002
                                 + 0.819299 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.03285 w_k1_ADM_s002
                                 + 1.25777 w_k1_AXP_s002
                                 + 1.11203 w_k1_DIS_s002
                                 + 1.14911 w_k1_FDX_s002
                                 + 0.823392 w_k1_IBM_s002
                                 + 1.16036 w_k1_INTC_s002
                                 + 0.87822 w_k1_KO_s002
                                 + 0.745101 w_k1_MAT_s002
                                 + 0.99227 w_k1_MSFT_s002
                                 + 1.00027 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.849657 w_k1_ADM_s002
                                 + 1.07348 w_k1_AXP_s002
                                 + 0.76454 w_k1_DIS_s002
                                 + 0.950598 w_k1_FDX_s002
                                 + 0.844497 w_k1_IBM_s002
                                 + 0.854033 w_k1_INTC_s002
                                 + 0.822601 w_k1_KO_s002 + 1.1801 w_k1_MAT_s002
                                 + 0.845997 w_k1_MSFT_s002
                                 + 0.909531 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.00922 w_k1_ADM_s002
                                 + 0.79117 w_k1_AXP_s002
                                 + 0.897482 w_k1_DIS_s002
                                 + 0.857186 w_k1_FDX_s002
                                 + 0.798138 w_k1_IBM_s002
                                 + 0.867362 w_k1_INTC_s002
                                 + 0.957993 w_k1_KO_s002
                                 + 0.673317 w_k1_MAT_s002
                                 + 0.946089 w_k1_MSFT_s002
                                 + 0.965862 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.09852 w_k1_ADM_s002
                                 + 1.06875 w_k1_AXP_s002
                                 + 1.03625 w_k1_DIS_s002
                                 + 1.00218 w_k1_FDX_s002 + 1.7223 w_k1_IBM_s002
                                 + 1.06386 w_k1_INTC_s002
                                 + 1.33342 w_k1_KO_s002 + 1.82907 w_k1_MAT_s002
                                 + 1.11763 w_k1_MSFT_s002
                                 + 1.14588 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.17503 w_k1_ADM_s002
                                 + 1.5746 w_k1_AXP_s002 + 1.77237 w_k1_DIS_s002
                                 + 1.47315 w_k1_FDX_s002
                                 + 1.15301 w_k1_IBM_s002
                                 + 1.47731 w_k1_INTC_s002
                                 + 1.46248 w_k1_KO_s002 + 1.70564 w_k1_MAT_s002
                                 + 1.43493 w_k1_MSFT_s002
                                 + 1.30477 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.888883 w_k1_ADM_s002
                                 + 0.898619 w_k1_AXP_s002
                                 + 0.705439 w_k1_DIS_s002
                                 + 0.795996 w_k1_FDX_s002
                                 + 1.04416 w_k1_IBM_s002
                                 + 0.832311 w_k1_INTC_s002
                                 + 0.866844 w_k1_KO_s002
                                 + 0.981603 w_k1_MAT_s002
                                 + 0.720476 w_k1_MSFT_s002
                                 + 0.958193 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.772726 w_k1_ADM_s003
                                 + 0.650531 w_k1_AXP_s003
                                 + 0.573724 w_k1_DIS_s003
                                 + 0.69272 w_k1_FDX_s003
                                 + 0.706716 w_k1_IBM_s003
                                 + 0.842718 w_k1_INTC_s003
                                 + 0.636002 w_k1_KO_s003
                                 + 0.500364 w_k1_MAT_s003
                                 + 0.719465 w_k1_MSFT_s003
                                 + 0.763979 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.971996 w_k1_ADM_s003
                                 + 0.938071 w_k1_AXP_s003
                                 + 1.35652 w_k1_DIS_s003
                                 + 1.13067 w_k1_FDX_s003
                                 + 0.913547 w_k1_IBM_s003
                                 + 1.14346 w_k1_INTC_s003
                                 + 1.05402 w_k1_KO_s003 + 0.87784 w_k1_MAT_s003
                                 + 1.25245 w_k1_MSFT_s003 + 1.0765 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.23533 w_k1_ADM_s003
                                 + 1.39275 w_k1_AXP_s003
                                 + 1.27129 w_k1_DIS_s003
                                 + 1.21621 w_k1_FDX_s003
                                 + 0.933847 w_k1_IBM_s003
                                 + 1.29768 w_k1_INTC_s003
                                 + 1.00079 w_k1_KO_s003
                                 + 0.858495 w_k1_MAT_s003
                                 + 1.16346 w_k1_MSFT_s003
                                 + 1.05478 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.975692 w_k1_ADM_s003
                                 + 0.896223 w_k1_AXP_s003
                                 + 0.949035 w_k1_DIS_s003
                                 + 0.845259 w_k1_FDX_s003
                                 + 1.25521 w_k1_IBM_s003
                                 + 0.970765 w_k1_INTC_s003 + 1.017 w_k1_KO_s003
                                 + 0.725586 w_k1_MAT_s003
                                 + 1.00129 w_k1_MSFT_s003
                                 + 1.03622 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.11427 w_k1_ADM_s003
                                 + 0.86794 w_k1_AXP_s003
                                 + 1.05604 w_k1_DIS_s003
                                 + 0.969793 w_k1_FDX_s003
                                 + 1.00506 w_k1_IBM_s003
                                 + 0.955651 w_k1_INTC_s003
                                 + 0.963095 w_k1_KO_s003
                                 + 0.87728 w_k1_MAT_s003
                                 + 0.968195 w_k1_MSFT_s003
                                 + 0.991469 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.05177 w_k1_ADM_s003
                                 + 1.24112 w_k1_AXP_s003
                                 + 0.957826 w_k1_DIS_s003
                                 + 0.99181 w_k1_FDX_s003
                                 + 1.05869 w_k1_IBM_s003
                                 + 0.881586 w_k1_INTC_s003
                                 + 0.962464 w_k1_KO_s003 + 1.3898 w_k1_MAT_s003
                                 + 1.01875 w_k1_MSFT_s003
                                 + 1.04563 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.37857 w_k1_ADM_s003
                                 + 1.44716 w_k1_AXP_s003
                                 + 1.67784 w_k1_DIS_s003
                                 + 1.29485 w_k1_FDX_s003 + 1.4435 w_k1_IBM_s003
                                 + 1.52986 w_k1_INTC_s003
                                 + 1.23172 w_k1_KO_s003 + 1.18624 w_k1_MAT_s003
                                 + 1.36687 w_k1_MSFT_s003
                                 + 1.36688 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.03081 w_k1_ADM_s003
                                 + 0.803467 w_k1_AXP_s003
                                 + 0.837441 w_k1_DIS_s003
                                 + 0.941154 w_k1_FDX_s003
                                 + 0.676602 w_k1_IBM_s003
                                 + 1.02329 w_k1_INTC_s003
                                 + 0.722388 w_k1_KO_s003
                                 + 0.58311 w_k1_MAT_s003
                                 + 0.830873 w_k1_MSFT_s003
                                 + 0.887828 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.14132 w_k1_ADM_s004
                                 + 1.1659 w_k1_AXP_s004 + 1.18716 w_k1_DIS_s004
                                 + 1.20011 w_k1_FDX_s004
                                 + 0.870592 w_k1_IBM_s004
                                 + 1.06672 w_k1_INTC_s004
                                 + 1.05432 w_k1_KO_s004 + 1.25217 w_k1_MAT_s004
                                 + 1.07166 w_k1_MSFT_s004
                                 + 1.04541 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.16171 w_k1_ADM_s004
                                 + 1.07014 w_k1_AXP_s004
                                 + 1.22994 w_k1_DIS_s004
                                 + 1.12395 w_k1_FDX_s004
                                 + 1.11838 w_k1_IBM_s004
                                 + 1.08575 w_k1_INTC_s004
                                 + 1.08309 w_k1_KO_s004 + 1.37389 w_k1_MAT_s004
                                 + 1.19072 w_k1_MSFT_s004
                                 + 1.14437 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.21257 w_k1_ADM_s004
                                 + 1.3327 w_k1_AXP_s004 + 1.57197 w_k1_DIS_s004
                                 + 1.37579 w_k1_FDX_s004
                                 + 1.41753 w_k1_IBM_s004
                                 + 1.22717 w_k1_INTC_s004
                                 + 1.43897 w_k1_KO_s004 + 2.258 w_k1_MAT_s004
                                 + 1.28587 w_k1_MSFT_s004
                                 + 1.17475 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.954074 w_k1_ADM_s004
                                 + 1.01734 w_k1_AXP_s004
                                 + 0.869055 w_k1_DIS_s004
                                 + 0.988558 w_k1_FDX_s004
                                 + 0.925706 w_k1_IBM_s004
                                 + 0.914196 w_k1_INTC_s004
                                 + 0.858518 w_k1_KO_s004
                                 + 1.19957 w_k1_MAT_s004
                                 + 0.917503 w_k1_MSFT_s004
                                 + 1.00642 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.76712 w_k1_ADM_s004
                                 + 0.892563 w_k1_AXP_s004
                                 + 0.790926 w_k1_DIS_s004
                                 + 0.758489 w_k1_FDX_s004
                                 + 1.06457 w_k1_IBM_s004
                                 + 0.850618 w_k1_INTC_s004
                                 + 0.955632 w_k1_KO_s004 + 1.4187 w_k1_MAT_s004
                                 + 0.809279 w_k1_MSFT_s004
                                 + 0.868462 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.69376 w_k1_ADM_s004
                                 + 0.754517 w_k1_AXP_s004
                                 + 0.544812 w_k1_DIS_s004
                                 + 0.751139 w_k1_FDX_s004
                                 + 0.774113 w_k1_IBM_s004
                                 + 0.738308 w_k1_INTC_s004
                                 + 0.619026 w_k1_KO_s004
                                 + 0.819439 w_k1_MAT_s004
                                 + 0.741607 w_k1_MSFT_s004
                                 + 0.781474 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.29771 w_k1_ADM_s004
                                 + 1.34422 w_k1_AXP_s004
                                 + 1.15013 w_k1_DIS_s004
                                 + 1.31538 w_k1_FDX_s004
                                 + 0.810853 w_k1_IBM_s004
                                 + 1.13338 w_k1_INTC_s004
                                 + 1.02885 w_k1_KO_s004 + 1.54354 w_k1_MAT_s004
                                 + 1.24627 w_k1_MSFT_s004
                                 + 1.07133 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.909517 w_k1_ADM_s004
                                 + 1.11566 w_k1_AXP_s004
                                 + 1.06171 w_k1_DIS_s004
                                 + 0.938885 w_k1_FDX_s004
                                 + 1.49192 w_k1_IBM_s004
                                 + 1.03618 w_k1_INTC_s004
                                 + 1.25896 w_k1_KO_s004 + 2.02915 w_k1_MAT_s004
                                 + 0.961623 w_k1_MSFT_s004
                                 + 1.10543 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.874559 w_k1_ADM_s005
                                 + 0.895785 w_k1_AXP_s005
                                 + 1.14934 w_k1_DIS_s005
                                 + 0.998837 w_k1_FDX_s005
                                 + 0.776094 w_k1_IBM_s005
                                 + 0.939755 w_k1_INTC_s005
                                 + 0.934659 w_k1_KO_s005
                                 + 0.768075 w_k1_MAT_s005
                                 + 1.07369 w_k1_MSFT_s005
                                 + 0.982598 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.798711 w_k1_ADM_s005
                                 + 1.12037 w_k1_AXP_s005
                                 + 1.59364 w_k1_DIS_s005
                                 + 1.09861 w_k1_FDX_s005
                                 + 1.59649 w_k1_IBM_s005
                                 + 1.04344 w_k1_INTC_s005 + 1.3112 w_k1_KO_s005
                                 + 1.31787 w_k1_MAT_s005
                                 + 1.17499 w_k1_MSFT_s005
                                 + 1.14335 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.886205 w_k1_ADM_s005
                                 + 0.819521 w_k1_AXP_s005
                                 + 0.740083 w_k1_DIS_s005
                                 + 0.771122 w_k1_FDX_s005
                                 + 0.882979 w_k1_IBM_s005
                                 + 0.641198 w_k1_INTC_s005
                                 + 0.694084 w_k1_KO_s005
                                 + 0.85198 w_k1_MAT_s005
                                 + 0.74443 w_k1_MSFT_s005
                                 + 0.855016 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.991537 w_k1_ADM_s005
                                 + 1.18803 w_k1_AXP_s005
                                 + 1.01948 w_k1_DIS_s005
                                 + 1.01276 w_k1_FDX_s005
                                 + 1.12885 w_k1_IBM_s005
                                 + 1.06862 w_k1_INTC_s005
                                 + 0.94084 w_k1_KO_s005 + 1.1136 w_k1_MAT_s005
                                 + 1.07155 w_k1_MSFT_s005
                                 + 1.10848 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.897853 w_k1_ADM_s005
                                 + 1.11916 w_k1_AXP_s005
                                 + 0.947203 w_k1_DIS_s005
                                 + 1.11244 w_k1_FDX_s005
                                 + 0.686924 w_k1_IBM_s005
                                 + 1.14672 w_k1_INTC_s005
                                 + 0.777977 w_k1_KO_s005
                                 + 0.511718 w_k1_MAT_s005
                                 + 1.06143 w_k1_MSFT_s005
                                 + 0.831758 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.34979 w_k1_ADM_s005
                                 + 1.81222 w_k1_AXP_s005
                                 + 1.37826 w_k1_DIS_s005
                                 + 1.42283 w_k1_FDX_s005
                                 + 1.10246 w_k1_IBM_s005
                                 + 1.55823 w_k1_INTC_s005
                                 + 1.00983 w_k1_KO_s005
                                 + 0.954276 w_k1_MAT_s005
                                 + 1.19886 w_k1_MSFT_s005
                                 + 1.14862 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.19621 w_k1_ADM_s005
                                 + 1.1644 w_k1_AXP_s005 + 1.59027 w_k1_DIS_s005
                                 + 1.2836 w_k1_FDX_s005 + 1.04752 w_k1_IBM_s005
                                 + 1.07805 w_k1_INTC_s005
                                 + 1.02952 w_k1_KO_s005
                                 + 0.761785 w_k1_MAT_s005
                                 + 1.16327 w_k1_MSFT_s005
                                 + 1.11688 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.867901 w_k1_ADM_s005
                                 + 0.793833 w_k1_AXP_s005
                                 + 0.990728 w_k1_DIS_s005
                                 + 0.857867 w_k1_FDX_s005
                                 + 1.30013 w_k1_IBM_s005
                                 + 0.873587 w_k1_INTC_s005
                                 + 1.11497 w_k1_KO_s005
                                 + 0.712034 w_k1_MAT_s005
                                 + 0.885738 w_k1_MSFT_s005
                                 + 0.943951 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.21767 w_k1_ADM_s006
                                 + 1.18577 w_k1_AXP_s006
                                 + 1.07938 w_k1_DIS_s006
                                 + 1.01193 w_k1_FDX_s006
                                 + 1.38865 w_k1_IBM_s006
                                 + 1.02694 w_k1_INTC_s006
                                 + 1.29255 w_k1_KO_s006 + 1.41884 w_k1_MAT_s006
                                 + 1.10921 w_k1_MSFT_s006
                                 + 1.17105 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.22701 w_k1_ADM_s006
                                 + 1.49607 w_k1_AXP_s006
                                 + 1.57372 w_k1_DIS_s006
                                 + 1.33395 w_k1_FDX_s006
                                 + 1.40672 w_k1_IBM_s006
                                 + 1.28508 w_k1_INTC_s006
                                 + 1.45058 w_k1_KO_s006 + 1.59626 w_k1_MAT_s006
                                 + 1.47673 w_k1_MSFT_s006
                                 + 1.36987 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.771956 w_k1_ADM_s006
                                 + 0.814119 w_k1_AXP_s006
                                 + 0.729642 w_k1_DIS_s006
                                 + 0.856864 w_k1_FDX_s006
                                 + 0.745871 w_k1_IBM_s006
                                 + 0.930131 w_k1_INTC_s006
                                 + 0.768277 w_k1_KO_s006
                                 + 0.685734 w_k1_MAT_s006
                                 + 0.895405 w_k1_MSFT_s006
                                 + 0.853279 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.7023 w_k1_ADM_s006
                                 + 0.803404 w_k1_AXP_s006
                                 + 0.718901 w_k1_DIS_s006
                                 + 0.792952 w_k1_FDX_s006
                                 + 0.983631 w_k1_IBM_s006
                                 + 1.03137 w_k1_INTC_s006
                                 + 0.887766 w_k1_KO_s006
                                 + 1.01882 w_k1_MAT_s006
                                 + 0.826453 w_k1_MSFT_s006
                                 + 0.956434 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.791302 w_k1_ADM_s006
                                 + 1.15313 w_k1_AXP_s006
                                 + 0.860554 w_k1_DIS_s006
                                 + 0.935717 w_k1_FDX_s006
                                 + 0.928636 w_k1_IBM_s006
                                 + 1.1084 w_k1_INTC_s006
                                 + 0.892046 w_k1_KO_s006
                                 + 1.09274 w_k1_MAT_s006
                                 + 0.887112 w_k1_MSFT_s006
                                 + 0.904389 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.03416 w_k1_ADM_s006
                                 + 0.84682 w_k1_AXP_s006
                                 + 0.936181 w_k1_DIS_s006
                                 + 0.819266 w_k1_FDX_s006
                                 + 1.24728 w_k1_IBM_s006
                                 + 0.937268 w_k1_INTC_s006 + 1.051 w_k1_KO_s006
                                 + 1.06876 w_k1_MAT_s006
                                 + 0.912942 w_k1_MSFT_s006
                                 + 1.02841 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.14211 w_k1_ADM_s006
                                 + 0.912073 w_k1_AXP_s006
                                 + 1.00251 w_k1_DIS_s006
                                 + 1.00984 w_k1_FDX_s006
                                 + 0.807978 w_k1_IBM_s006
                                 + 1.33697 w_k1_INTC_s006
                                 + 0.99232 w_k1_KO_s006
                                 + 0.655443 w_k1_MAT_s006
                                 + 1.09521 w_k1_MSFT_s006
                                 + 0.987196 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.08858 w_k1_ADM_s006
                                 + 1.51143 w_k1_AXP_s006
                                 + 1.15557 w_k1_DIS_s006
                                 + 1.14664 w_k1_FDX_s006
                                 + 1.00811 w_k1_IBM_s006
                                 + 1.30252 w_k1_INTC_s006 + 1.0468 w_k1_KO_s006
                                 + 0.995876 w_k1_MAT_s006
                                 + 0.963894 w_k1_MSFT_s006
                                 + 1.02632 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.12663 w_k1_ADM_s007
                                 + 1.22275 w_k1_AXP_s007
                                 + 1.11361 w_k1_DIS_s007
                                 + 1.17389 w_k1_FDX_s007
                                 + 1.12335 w_k1_IBM_s007
                                 + 0.986608 w_k1_INTC_s007
                                 + 0.99773 w_k1_KO_s007 + 1.12627 w_k1_MAT_s007
                                 + 1.12706 w_k1_MSFT_s007
                                 + 1.04841 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.894505 w_k1_ADM_s007
                                 + 1.26418 w_k1_AXP_s007
                                 + 0.825631 w_k1_DIS_s007
                                 + 0.917943 w_k1_FDX_s007
                                 + 1.03824 w_k1_IBM_s007
                                 + 0.872926 w_k1_INTC_s007
                                 + 0.825537 w_k1_KO_s007
                                 + 1.42381 w_k1_MAT_s007
                                 + 0.872055 w_k1_MSFT_s007
                                 + 0.903762 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.833574 w_k1_ADM_s007
                                 + 0.961324 w_k1_AXP_s007
                                 + 0.676002 w_k1_DIS_s007
                                 + 0.819215 w_k1_FDX_s007
                                 + 1.03205 w_k1_IBM_s007
                                 + 0.765173 w_k1_INTC_s007
                                 + 0.814877 w_k1_KO_s007
                                 + 0.90228 w_k1_MAT_s007
                                 + 0.793571 w_k1_MSFT_s007
                                 + 0.797665 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.013 w_k1_ADM_s007
                                 + 0.809173 w_k1_AXP_s007
                                 + 0.778466 w_k1_DIS_s007
                                 + 0.90307 w_k1_FDX_s007
                                 + 0.73957 w_k1_IBM_s007
                                 + 0.810555 w_k1_INTC_s007
                                 + 0.834751 w_k1_KO_s007
                                 + 0.774718 w_k1_MAT_s007
                                 + 0.839708 w_k1_MSFT_s007
                                 + 0.898504 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.746284 w_k1_ADM_s007
                                 + 0.770523 w_k1_AXP_s007
                                 + 0.662955 w_k1_DIS_s007
                                 + 0.766716 w_k1_FDX_s007
                                 + 0.77419 w_k1_IBM_s007
                                 + 0.820391 w_k1_INTC_s007
                                 + 0.754154 w_k1_KO_s007
                                 + 0.746759 w_k1_MAT_s007
                                 + 0.842862 w_k1_MSFT_s007
                                 + 0.776981 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.08443 w_k1_ADM_s007
                                 + 0.798038 w_k1_AXP_s007
                                 + 0.983371 w_k1_DIS_s007
                                 + 0.981833 w_k1_FDX_s007
                                 + 0.803409 w_k1_IBM_s007
                                 + 0.904979 w_k1_INTC_s007
                                 + 0.926113 w_k1_KO_s007
                                 + 0.598869 w_k1_MAT_s007
                                 + 0.985368 w_k1_MSFT_s007
                                 + 0.870082 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.07871 w_k1_ADM_s007
                                 + 0.955385 w_k1_AXP_s007
                                 + 1.17743 w_k1_DIS_s007
                                 + 1.01249 w_k1_FDX_s007
                                 + 1.34219 w_k1_IBM_s007
                                 + 0.86148 w_k1_INTC_s007
                                 + 1.20944 w_k1_KO_s007 + 1.19884 w_k1_MAT_s007
                                 + 0.990216 w_k1_MSFT_s007
                                 + 0.98756 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.31756 w_k1_ADM_s007
                                 + 1.14767 w_k1_AXP_s007
                                 + 1.36227 w_k1_DIS_s007
                                 + 1.29569 w_k1_FDX_s007
                                 + 0.946605 w_k1_IBM_s007
                                 + 0.964585 w_k1_INTC_s007
                                 + 1.11247 w_k1_KO_s007 + 1.30386 w_k1_MAT_s007
                                 + 1.20822 w_k1_MSFT_s007 + 1.0748 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.784001 w_k1_ADM_s010
                                 + 1.39077 w_k1_AXP_s010
                                 + 0.86099 w_k1_DIS_s010
                                 + 0.971047 w_k1_FDX_s010
                                 + 1.10205 w_k1_IBM_s010
                                 + 1.20769 w_k1_INTC_s010
                                 + 0.903308 w_k1_KO_s010
                                 + 1.05766 w_k1_MAT_s010
                                 + 0.862055 w_k1_MSFT_s010
                                 + 0.937525 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.806491 w_k1_ADM_s010
                                 + 0.679218 w_k1_AXP_s010
                                 + 0.816965 w_k1_DIS_s010
                                 + 0.794978 w_k1_FDX_s010
                                 + 1.01785 w_k1_IBM_s010
                                 + 0.908931 w_k1_INTC_s010
                                 + 0.96622 w_k1_KO_s010
                                 + 0.594147 w_k1_MAT_s010
                                 + 0.853051 w_k1_MSFT_s010
                                 + 0.820494 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.642113 w_k1_ADM_s010
                                 + 0.772708 w_k1_AXP_s010
                                 + 0.8048 w_k1_DIS_s010
                                 + 0.904752 w_k1_FDX_s010
                                 + 0.72129 w_k1_IBM_s010
                                 + 0.987613 w_k1_INTC_s010
                                 + 0.900744 w_k1_KO_s010
                                 + 0.503181 w_k1_MAT_s010
                                 + 0.934964 w_k1_MSFT_s010
                                 + 0.868301 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.1051 w_k1_ADM_s010
                                 + 1.33037 w_k1_AXP_s010
                                 + 1.68588 w_k1_DIS_s010
                                 + 1.32258 w_k1_FDX_s010
                                 + 1.25635 w_k1_IBM_s010
                                 + 1.41462 w_k1_INTC_s010
                                 + 1.43587 w_k1_KO_s010 + 0.96651 w_k1_MAT_s010
                                 + 1.32796 w_k1_MSFT_s010
                                 + 1.34648 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.08987 w_k1_ADM_s010
                                 + 1.08931 w_k1_AXP_s010
                                 + 1.14488 w_k1_DIS_s010
                                 + 1.18413 w_k1_FDX_s010 + 0.8872 w_k1_IBM_s010
                                 + 1.39944 w_k1_INTC_s010
                                 + 1.13763 w_k1_KO_s010
                                 + 0.631466 w_k1_MAT_s010
                                 + 1.1678 w_k1_MSFT_s010
                                 + 0.976499 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.860784 w_k1_ADM_s010
                                 + 0.919808 w_k1_AXP_s010
                                 + 0.867894 w_k1_DIS_s010
                                 + 0.944832 w_k1_FDX_s010
                                 + 0.954624 w_k1_IBM_s010
                                 + 1.12572 w_k1_INTC_s010
                                 + 1.19666 w_k1_KO_s010 + 0.97603 w_k1_MAT_s010
                                 + 0.999568 w_k1_MSFT_s010
                                 + 0.948454 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.729807 w_k1_ADM_s010
                                 + 0.968563 w_k1_AXP_s010
                                 + 1.12749 w_k1_DIS_s010
                                 + 0.986863 w_k1_FDX_s010
                                 + 1.48769 w_k1_IBM_s010
                                 + 1.25283 w_k1_INTC_s010
                                 + 1.31643 w_k1_KO_s010 + 1.11455 w_k1_MAT_s010
                                 + 0.969778 w_k1_MSFT_s010
                                 + 0.994426 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.639967 w_k1_ADM_s010
                                 + 0.608309 w_k1_AXP_s010
                                 + 0.488547 w_k1_DIS_s010
                                 + 0.646634 w_k1_FDX_s010
                                 + 0.736123 w_k1_IBM_s010
                                 + 0.834889 w_k1_INTC_s010
                                 + 0.738066 w_k1_KO_s010
                                 + 0.736038 w_k1_MAT_s010
                                 + 0.778325 w_k1_MSFT_s010
                                 + 0.69907 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.08936 w_k1_ADM_s011
                                 + 1.42812 w_k1_AXP_s011
                                 + 1.34693 w_k1_DIS_s011
                                 + 1.22978 w_k1_FDX_s011
                                 + 1.12275 w_k1_IBM_s011
                                 + 0.909061 w_k1_INTC_s011
                                 + 1.00933 w_k1_KO_s011 + 1.58107 w_k1_MAT_s011
                                 + 0.947823 w_k1_MSFT_s011
                                 + 1.15235 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.765394 w_k1_ADM_s011
                                 + 0.918822 w_k1_AXP_s011
                                 + 0.626678 w_k1_DIS_s011
                                 + 0.828314 w_k1_FDX_s011
                                 + 0.74916 w_k1_IBM_s011
                                 + 0.858491 w_k1_INTC_s011
                                 + 0.837555 w_k1_KO_s011
                                 + 1.00697 w_k1_MAT_s011
                                 + 0.759343 w_k1_MSFT_s011
                                 + 0.922041 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.11865 w_k1_ADM_s011
                                 + 0.965973 w_k1_AXP_s011
                                 + 1.0661 w_k1_DIS_s011
                                 + 0.899926 w_k1_FDX_s011
                                 + 1.49307 w_k1_IBM_s011
                                 + 1.00347 w_k1_INTC_s011
                                 + 1.32826 w_k1_KO_s011 + 1.74202 w_k1_MAT_s011
                                 + 0.880341 w_k1_MSFT_s011
                                 + 0.966492 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.933225 w_k1_ADM_s011
                                 + 1.40909 w_k1_AXP_s011
                                 + 0.757104 w_k1_DIS_s011
                                 + 1.00196 w_k1_FDX_s011
                                 + 0.882971 w_k1_IBM_s011
                                 + 0.833286 w_k1_INTC_s011
                                 + 0.784416 w_k1_KO_s011
                                 + 1.05526 w_k1_MAT_s011
                                 + 0.818169 w_k1_MSFT_s011
                                 + 1.02088 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.815665 w_k1_ADM_s011
                                 + 0.730086 w_k1_AXP_s011
                                 + 0.822557 w_k1_DIS_s011
                                 + 0.910423 w_k1_FDX_s011
                                 + 0.79606 w_k1_IBM_s011
                                 + 0.858054 w_k1_INTC_s011
                                 + 0.918558 w_k1_KO_s011
                                 + 0.719109 w_k1_MAT_s011
                                 + 0.848254 w_k1_MSFT_s011
                                 + 0.860626 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.931406 w_k1_ADM_s011
                                 + 1.27969 w_k1_AXP_s011 + 1.08 w_k1_DIS_s011
                                 + 1.12784 w_k1_FDX_s011
                                 + 0.875583 w_k1_IBM_s011
                                 + 1.04533 w_k1_INTC_s011
                                 + 0.915993 w_k1_KO_s011
                                 + 1.09332 w_k1_MAT_s011
                                 + 1.06318 w_k1_MSFT_s011
                                 + 1.06407 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.19088 w_k1_ADM_s011
                                 + 1.517 w_k1_AXP_s011 + 1.65689 w_k1_DIS_s011
                                 + 1.36298 w_k1_FDX_s011
                                 + 1.43335 w_k1_IBM_s011
                                 + 1.40395 w_k1_INTC_s011
                                 + 1.49398 w_k1_KO_s011 + 1.46989 w_k1_MAT_s011
                                 + 1.09664 w_k1_MSFT_s011
                                 + 1.20669 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.949203 w_k1_ADM_s011
                                 + 0.929181 w_k1_AXP_s011
                                 + 0.801801 w_k1_DIS_s011
                                 + 0.836852 w_k1_FDX_s011
                                 + 1.16138 w_k1_IBM_s011
                                 + 0.896877 w_k1_INTC_s011
                                 + 1.04399 w_k1_KO_s011 + 1.23856 w_k1_MAT_s011
                                 + 0.76193 w_k1_MSFT_s011
                                 + 1.01032 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.927273 w_k1_ADM_s012
                                 + 1.03848 w_k1_AXP_s012
                                 + 0.970159 w_k1_DIS_s012
                                 + 0.958707 w_k1_FDX_s012
                                 + 1.46253 w_k1_IBM_s012
                                 + 0.941187 w_k1_INTC_s012
                                 + 1.09017 w_k1_KO_s012 + 1.46905 w_k1_MAT_s012
                                 + 0.958784 w_k1_MSFT_s012
                                 + 0.949569 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.25531 w_k1_ADM_s012
                                 + 1.49479 w_k1_AXP_s012 + 1.5883 w_k1_DIS_s012
                                 + 1.31895 w_k1_FDX_s012
                                 + 1.57849 w_k1_IBM_s012
                                 + 1.02002 w_k1_INTC_s012
                                 + 1.51909 w_k1_KO_s012 + 2.17459 w_k1_MAT_s012
                                 + 1.28854 w_k1_MSFT_s012 + 1.0906 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.800364 w_k1_ADM_s012
                                 + 0.853269 w_k1_AXP_s012
                                 + 0.719955 w_k1_DIS_s012
                                 + 0.804304 w_k1_FDX_s012
                                 + 0.92778 w_k1_IBM_s012
                                 + 0.803079 w_k1_INTC_s012
                                 + 0.928954 w_k1_KO_s012
                                 + 1.11878 w_k1_MAT_s012
                                 + 0.921001 w_k1_MSFT_s012
                                 + 0.887953 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.959182 w_k1_ADM_s012
                                 + 0.913413 w_k1_AXP_s012
                                 + 1.09092 w_k1_DIS_s012
                                 + 1.05855 w_k1_FDX_s012
                                 + 1.11758 w_k1_IBM_s012
                                 + 0.858945 w_k1_INTC_s012
                                 + 1.10564 w_k1_KO_s012 + 1.38388 w_k1_MAT_s012
                                 + 1.28471 w_k1_MSFT_s012
                                 + 0.986489 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.17345 w_k1_ADM_s012
                                 + 1.58822 w_k1_AXP_s012
                                 + 1.21606 w_k1_DIS_s012
                                 + 1.19077 w_k1_FDX_s012
                                 + 1.13874 w_k1_IBM_s012
                                 + 0.806822 w_k1_INTC_s012
                                 + 1.04399 w_k1_KO_s012 + 1.76692 w_k1_MAT_s012
                                 + 1.07894 w_k1_MSFT_s012
                                 + 1.00374 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.852178 w_k1_ADM_s012
                                 + 1.02611 w_k1_AXP_s012
                                 + 0.839303 w_k1_DIS_s012
                                 + 0.964901 w_k1_FDX_s012
                                 + 0.672995 w_k1_IBM_s012
                                 + 0.952833 w_k1_INTC_s012
                                 + 0.897963 w_k1_KO_s012
                                 + 1.01681 w_k1_MAT_s012
                                 + 1.13432 w_k1_MSFT_s012
                                 + 0.924675 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.869523 w_k1_ADM_s012
                                 + 0.943158 w_k1_AXP_s012
                                 + 0.785654 w_k1_DIS_s012 + 0.847 w_k1_FDX_s012
                                 + 1.22461 w_k1_IBM_s012
                                 + 0.915694 w_k1_INTC_s012
                                 + 1.14763 w_k1_KO_s012 + 1.69454 w_k1_MAT_s012
                                 + 1.01684 w_k1_MSFT_s012
                                 + 0.892279 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.11619 w_k1_ADM_s012
                                 + 1.11452 w_k1_AXP_s012
                                 + 1.40485 w_k1_DIS_s012
                                 + 1.24012 w_k1_FDX_s012
                                 + 0.872377 w_k1_IBM_s012
                                 + 0.975624 w_k1_INTC_s012
                                 + 1.10338 w_k1_KO_s012 + 1.12008 w_k1_MAT_s012
                                 + 1.41398 w_k1_MSFT_s012
                                 + 1.05612 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.13043 w_k1_ADM_s013
                                 + 1.21589 w_k1_AXP_s013
                                 + 1.18078 w_k1_DIS_s013
                                 + 0.996931 w_k1_FDX_s013
                                 + 1.56044 w_k1_IBM_s013
                                 + 0.850098 w_k1_INTC_s013
                                 + 1.38919 w_k1_KO_s013 + 1.54132 w_k1_MAT_s013
                                 + 1.02654 w_k1_MSFT_s013
                                 + 1.30244 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.03051 w_k1_ADM_s013
                                 + 1.25103 w_k1_AXP_s013
                                 + 0.913721 w_k1_DIS_s013
                                 + 1.10506 w_k1_FDX_s013
                                 + 1.03875 w_k1_IBM_s013
                                 + 1.00918 w_k1_INTC_s013
                                 + 0.858741 w_k1_KO_s013
                                 + 1.15747 w_k1_MAT_s013
                                 + 0.99909 w_k1_MSFT_s013
                                 + 1.12202 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.93259 w_k1_ADM_s013
                                 + 0.768118 w_k1_AXP_s013
                                 + 0.727453 w_k1_DIS_s013
                                 + 0.867367 w_k1_FDX_s013
                                 + 0.639294 w_k1_IBM_s013
                                 + 0.769204 w_k1_INTC_s013
                                 + 0.733249 w_k1_KO_s013
                                 + 0.540289 w_k1_MAT_s013
                                 + 0.901135 w_k1_MSFT_s013
                                 + 0.85748 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.871565 w_k1_ADM_s013
                                 + 0.809931 w_k1_AXP_s013
                                 + 0.793905 w_k1_DIS_s013
                                 + 0.814877 w_k1_FDX_s013
                                 + 1.15905 w_k1_IBM_s013
                                 + 0.718232 w_k1_INTC_s013
                                 + 1.01076 w_k1_KO_s013 + 1.09432 w_k1_MAT_s013
                                 + 0.840112 w_k1_MSFT_s013
                                 + 0.980621 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15052 w_k1_ADM_s013
                                 + 1.66846 w_k1_AXP_s013
                                 + 1.46601 w_k1_DIS_s013
                                 + 1.40468 w_k1_FDX_s013
                                 + 1.43023 w_k1_IBM_s013
                                 + 1.21631 w_k1_INTC_s013
                                 + 1.39144 w_k1_KO_s013 + 1.6657 w_k1_MAT_s013
                                 + 1.26431 w_k1_MSFT_s013 + 1.3842 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.16208 w_k1_ADM_s013
                                 + 1.1612 w_k1_AXP_s013 + 1.36931 w_k1_DIS_s013
                                 + 1.20732 w_k1_FDX_s013
                                 + 1.12955 w_k1_IBM_s013
                                 + 0.949355 w_k1_INTC_s013
                                 + 1.26482 w_k1_KO_s013 + 1.24527 w_k1_MAT_s013
                                 + 1.04184 w_k1_MSFT_s013
                                 + 1.10985 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.791433 w_k1_ADM_s013
                                 + 0.771515 w_k1_AXP_s013
                                 + 0.881253 w_k1_DIS_s013
                                 + 0.860035 w_k1_FDX_s013
                                 + 0.912552 w_k1_IBM_s013
                                 + 0.871141 w_k1_INTC_s013
                                 + 0.917239 w_k1_KO_s013
                                 + 0.889124 w_k1_MAT_s013
                                 + 0.899051 w_k1_MSFT_s013
                                 + 1.00329 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.18511 w_k1_ADM_s013
                                 + 0.887868 w_k1_AXP_s013
                                 + 0.915174 w_k1_DIS_s013
                                 + 1.02581 w_k1_FDX_s013
                                 + 0.799584 w_k1_IBM_s013
                                 + 0.994582 w_k1_INTC_s013
                                 + 0.94332 w_k1_KO_s013
                                 + 0.627205 w_k1_MAT_s013
                                 + 1.0347 w_k1_MSFT_s013 + 1.04553 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.03701 w_k1_ADM_s014
                                 + 1.14851 w_k1_AXP_s014
                                 + 1.42138 w_k1_DIS_s014
                                 + 1.16864 w_k1_FDX_s014
                                 + 1.15314 w_k1_IBM_s014
                                 + 1.03415 w_k1_INTC_s014
                                 + 1.17468 w_k1_KO_s014 + 1.18663 w_k1_MAT_s014
                                 + 1.21231 w_k1_MSFT_s014
                                 + 1.09324 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.767165 w_k1_ADM_s014
                                 + 0.723695 w_k1_AXP_s014
                                 + 0.618811 w_k1_DIS_s014
                                 + 0.756262 w_k1_FDX_s014
                                 + 0.82082 w_k1_IBM_s014
                                 + 0.83556 w_k1_INTC_s014
                                 + 0.804989 w_k1_KO_s014
                                 + 0.831774 w_k1_MAT_s014
                                 + 0.766088 w_k1_MSFT_s014
                                 + 0.944782 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.20609 w_k1_ADM_s014
                                 + 1.08341 w_k1_AXP_s014
                                 + 1.16658 w_k1_DIS_s014
                                 + 1.12085 w_k1_FDX_s014
                                 + 1.00688 w_k1_IBM_s014
                                 + 0.828446 w_k1_INTC_s014
                                 + 1.28547 w_k1_KO_s014 + 1.01731 w_k1_MAT_s014
                                 + 1.13506 w_k1_MSFT_s014
                                 + 1.25937 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.00783 w_k1_ADM_s014
                                 + 1.12044 w_k1_AXP_s014
                                 + 1.18598 w_k1_DIS_s014
                                 + 0.979296 w_k1_FDX_s014
                                 + 1.43962 w_k1_IBM_s014
                                 + 1.03605 w_k1_INTC_s014 + 1.3082 w_k1_KO_s014
                                 + 1.11115 w_k1_MAT_s014
                                 + 0.932338 w_k1_MSFT_s014
                                 + 1.13096 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.784484 w_k1_ADM_s014
                                 + 0.832193 w_k1_AXP_s014
                                 + 0.881802 w_k1_DIS_s014
                                 + 0.951701 w_k1_FDX_s014
                                 + 0.746323 w_k1_IBM_s014
                                 + 1.01383 w_k1_INTC_s014
                                 + 0.935606 w_k1_KO_s014
                                 + 0.624833 w_k1_MAT_s014
                                 + 1.00741 w_k1_MSFT_s014
                                 + 0.93814 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.43942 w_k1_ADM_s014
                                 + 1.13742 w_k1_AXP_s014
                                 + 1.32425 w_k1_DIS_s014 + 1.1764 w_k1_FDX_s014
                                 + 0.790932 w_k1_IBM_s014
                                 + 0.945515 w_k1_INTC_s014
                                 + 1.11602 w_k1_KO_s014
                                 + 0.821809 w_k1_MAT_s014
                                 + 1.13959 w_k1_MSFT_s014
                                 + 1.13737 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.15291 w_k1_ADM_s014
                                 + 1.1868 w_k1_AXP_s014
                                 + 0.840953 w_k1_DIS_s014
                                 + 1.05783 w_k1_FDX_s014
                                 + 0.783107 w_k1_IBM_s014
                                 + 0.743555 w_k1_INTC_s014
                                 + 0.839863 w_k1_KO_s014
                                 + 1.02708 w_k1_MAT_s014
                                 + 0.881713 w_k1_MSFT_s014
                                 + 0.928431 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.872872 w_k1_ADM_s014
                                 + 0.783908 w_k1_AXP_s014
                                 + 0.797657 w_k1_DIS_s014
                                 + 0.835653 w_k1_FDX_s014
                                 + 0.94175 w_k1_IBM_s014
                                 + 0.626344 w_k1_INTC_s014
                                 + 1.04325 w_k1_KO_s014
                                 + 0.925916 w_k1_MAT_s014
                                 + 0.828387 w_k1_MSFT_s014
                                 + 1.00082 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.46355 w_k1_ADM_s015
                                 + 1.61609 w_k1_AXP_s015
                                 + 1.51468 w_k1_DIS_s015
                                 + 1.30076 w_k1_FDX_s015
                                 + 1.39021 w_k1_IBM_s015
                                 + 1.38798 w_k1_INTC_s015
                                 + 1.39831 w_k1_KO_s015 + 1.90275 w_k1_MAT_s015
                                 + 1.45465 w_k1_MSFT_s015
                                 + 1.37511 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.52019 w_k1_ADM_s015
                                 + 1.25016 w_k1_AXP_s015
                                 + 1.98516 w_k1_DIS_s015
                                 + 1.64791 w_k1_FDX_s015
                                 + 1.07106 w_k1_IBM_s015
                                 + 1.30088 w_k1_INTC_s015
                                 + 1.44189 w_k1_KO_s015 + 1.26973 w_k1_MAT_s015
                                 + 1.66829 w_k1_MSFT_s015
                                 + 1.46473 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.984055 w_k1_ADM_s015
                                 + 0.802699 w_k1_AXP_s015
                                 + 0.722261 w_k1_DIS_s015
                                 + 0.79639 w_k1_FDX_s015
                                 + 0.676911 w_k1_IBM_s015
                                 + 0.977893 w_k1_INTC_s015
                                 + 0.689177 w_k1_KO_s015
                                 + 0.543074 w_k1_MAT_s015
                                 + 0.973669 w_k1_MSFT_s015
                                 + 0.88523 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.894348 w_k1_ADM_s015
                                 + 1.14126 w_k1_AXP_s015
                                 + 0.885079 w_k1_DIS_s015
                                 + 0.915375 w_k1_FDX_s015
                                 + 1.19224 w_k1_IBM_s015
                                 + 1.27005 w_k1_INTC_s015
                                 + 0.987506 w_k1_KO_s015
                                 + 1.61329 w_k1_MAT_s015
                                 + 1.10106 w_k1_MSFT_s015
                                 + 1.15971 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.05222 w_k1_ADM_s015
                                 + 0.900583 w_k1_AXP_s015
                                 + 1.03796 w_k1_DIS_s015
                                 + 0.95708 w_k1_FDX_s015
                                 + 1.62082 w_k1_IBM_s015
                                 + 1.05077 w_k1_INTC_s015
                                 + 1.33413 w_k1_KO_s015 + 1.59728 w_k1_MAT_s015
                                 + 1.29756 w_k1_MSFT_s015
                                 + 1.14722 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.66785 w_k1_ADM_s015
                                 + 0.609714 w_k1_AXP_s015
                                 + 0.538702 w_k1_DIS_s015
                                 + 0.639994 w_k1_FDX_s015
                                 + 1.09621 w_k1_IBM_s015
                                 + 1.0097 w_k1_INTC_s015 + 0.9112 w_k1_KO_s015
                                 + 1.03155 w_k1_MAT_s015
                                 + 0.945672 w_k1_MSFT_s015
                                 + 0.902123 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.33311 w_k1_ADM_s015
                                 + 1.23799 w_k1_AXP_s015
                                 + 1.31818 w_k1_DIS_s015
                                 + 1.29288 w_k1_FDX_s015
                                 + 0.721177 w_k1_IBM_s015
                                 + 1.16763 w_k1_INTC_s015
                                 + 0.818534 w_k1_KO_s015
                                 + 1.03169 w_k1_MAT_s015
                                 + 1.37729 w_k1_MSFT_s015
                                 + 0.984485 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.916489 w_k1_ADM_s015
                                 + 0.81698 w_k1_AXP_s015
                                 + 1.02063 w_k1_DIS_s015
                                 + 0.955256 w_k1_FDX_s015
                                 + 0.952236 w_k1_IBM_s015
                                 + 0.90344 w_k1_INTC_s015
                                 + 0.946302 w_k1_KO_s015
                                 + 1.17667 w_k1_MAT_s015
                                 + 0.943264 w_k1_MSFT_s015 + 1.01 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.853185 w_k1_ADM_s016
                                 + 0.751861 w_k1_AXP_s016
                                 + 0.61816 w_k1_DIS_s016
                                 + 0.753915 w_k1_FDX_s016
                                 + 0.891269 w_k1_IBM_s016
                                 + 0.826892 w_k1_INTC_s016
                                 + 0.817021 w_k1_KO_s016
                                 + 0.789285 w_k1_MAT_s016
                                 + 0.807634 w_k1_MSFT_s016
                                 + 0.716118 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.09635 w_k1_ADM_s016
                                 + 1.0121 w_k1_AXP_s016 + 1.143 w_k1_DIS_s016
                                 + 0.977799 w_k1_FDX_s016
                                 + 1.07102 w_k1_IBM_s016
                                 + 0.888236 w_k1_INTC_s016
                                 + 1.24235 w_k1_KO_s016 + 1.12639 w_k1_MAT_s016
                                 + 1.07438 w_k1_MSFT_s016 + 1.0296 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.46147 w_k1_ADM_s016
                                 + 1.00763 w_k1_AXP_s016
                                 + 1.53707 w_k1_DIS_s016
                                 + 1.26129 w_k1_FDX_s016
                                 + 0.967128 w_k1_IBM_s016
                                 + 1.08477 w_k1_INTC_s016
                                 + 1.26058 w_k1_KO_s016
                                 + 0.827493 w_k1_MAT_s016
                                 + 1.21057 w_k1_MSFT_s016
                                 + 1.08307 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.04139 w_k1_ADM_s016
                                 + 1.17437 w_k1_AXP_s016
                                 + 1.21614 w_k1_DIS_s016 + 1.0898 w_k1_FDX_s016
                                 + 1.23858 w_k1_IBM_s016
                                 + 0.879941 w_k1_INTC_s016
                                 + 1.14147 w_k1_KO_s016 + 1.29465 w_k1_MAT_s016
                                 + 1.17863 w_k1_MSFT_s016
                                 + 1.05945 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.10115 w_k1_ADM_s016
                                 + 0.883229 w_k1_AXP_s016
                                 + 0.976691 w_k1_DIS_s016
                                 + 0.994211 w_k1_FDX_s016
                                 + 0.867119 w_k1_IBM_s016
                                 + 0.916775 w_k1_INTC_s016
                                 + 0.905434 w_k1_KO_s016
                                 + 0.932323 w_k1_MAT_s016
                                 + 1.04159 w_k1_MSFT_s016
                                 + 0.939796 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.35883 w_k1_ADM_s016
                                 + 1.5383 w_k1_AXP_s016 + 1.62901 w_k1_DIS_s016
                                 + 1.21793 w_k1_FDX_s016
                                 + 1.23556 w_k1_IBM_s016
                                 + 1.11356 w_k1_INTC_s016
                                 + 1.46741 w_k1_KO_s016 + 1.16413 w_k1_MAT_s016
                                 + 1.34405 w_k1_MSFT_s016
                                 + 1.10439 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.15428 w_k1_ADM_s016
                                 + 1.0538 w_k1_AXP_s016
                                 + 0.850134 w_k1_DIS_s016
                                 + 0.842835 w_k1_FDX_s016
                                 + 1.12753 w_k1_IBM_s016
                                 + 0.958945 w_k1_INTC_s016
                                 + 1.04442 w_k1_KO_s016
                                 + 0.949211 w_k1_MAT_s016
                                 + 0.922291 w_k1_MSFT_s016
                                 + 0.87593 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.10491 w_k1_ADM_s016
                                 + 1.01693 w_k1_AXP_s016
                                 + 0.770048 w_k1_DIS_s016
                                 + 0.941096 w_k1_FDX_s016
                                 + 0.628722 w_k1_IBM_s016
                                 + 0.895724 w_k1_INTC_s016
                                 + 0.779866 w_k1_KO_s016
                                 + 0.802604 w_k1_MAT_s016
                                 + 0.9687 w_k1_MSFT_s016
                                 + 0.920188 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.954292 w_k1_ADM_s017
                                 + 0.982049 w_k1_AXP_s017
                                 + 0.737098 w_k1_DIS_s017
                                 + 0.981484 w_k1_FDX_s017
                                 + 0.743494 w_k1_IBM_s017
                                 + 1.14589 w_k1_INTC_s017
                                 + 0.73062 w_k1_KO_s017
                                 + 0.646973 w_k1_MAT_s017
                                 + 0.963286 w_k1_MSFT_s017
                                 + 0.883038 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.2406 w_k1_ADM_s017
                                 + 1.26884 w_k1_AXP_s017
                                 + 1.14895 w_k1_DIS_s017 + 1.193 w_k1_FDX_s017
                                 + 1.05314 w_k1_IBM_s017
                                 + 1.21608 w_k1_INTC_s017
                                 + 1.06875 w_k1_KO_s017
                                 + 0.777096 w_k1_MAT_s017
                                 + 0.91534 w_k1_MSFT_s017
                                 + 1.06131 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.03745 w_k1_ADM_s017
                                 + 1.12409 w_k1_AXP_s017
                                 + 1.35756 w_k1_DIS_s017
                                 + 1.40493 w_k1_FDX_s017
                                 + 0.962148 w_k1_IBM_s017
                                 + 1.18686 w_k1_INTC_s017
                                 + 0.948111 w_k1_KO_s017
                                 + 0.892579 w_k1_MAT_s017
                                 + 1.28071 w_k1_MSFT_s017
                                 + 1.08591 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.614038 w_k1_ADM_s017
                                 + 0.738342 w_k1_AXP_s017
                                 + 0.732318 w_k1_DIS_s017
                                 + 0.793248 w_k1_FDX_s017
                                 + 0.98082 w_k1_IBM_s017
                                 + 0.826068 w_k1_INTC_s017
                                 + 0.806355 w_k1_KO_s017
                                 + 0.807889 w_k1_MAT_s017
                                 + 0.818284 w_k1_MSFT_s017
                                 + 0.925414 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.13064 w_k1_ADM_s017
                                 + 0.855359 w_k1_AXP_s017
                                 + 1.30824 w_k1_DIS_s017
                                 + 1.06707 w_k1_FDX_s017
                                 + 0.91728 w_k1_IBM_s017
                                 + 1.16375 w_k1_INTC_s017
                                 + 1.24058 w_k1_KO_s017
                                 + 0.776824 w_k1_MAT_s017
                                 + 1.08693 w_k1_MSFT_s017
                                 + 1.04104 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.17871 w_k1_ADM_s017
                                 + 1.28622 w_k1_AXP_s017
                                 + 1.64757 w_k1_DIS_s017
                                 + 1.55685 w_k1_FDX_s017
                                 + 1.08957 w_k1_IBM_s017
                                 + 1.6906 w_k1_INTC_s017 + 1.22688 w_k1_KO_s017
                                 + 1.22817 w_k1_MAT_s017
                                 + 1.47297 w_k1_MSFT_s017
                                 + 1.20059 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.02679 w_k1_ADM_s017
                                 + 1.17895 w_k1_AXP_s017 + 1.4017 w_k1_DIS_s017
                                 + 1.11685 w_k1_FDX_s017
                                 + 1.56989 w_k1_IBM_s017
                                 + 1.09558 w_k1_INTC_s017
                                 + 1.21255 w_k1_KO_s017 + 1.34386 w_k1_MAT_s017
                                 + 1.0286 w_k1_MSFT_s017 + 1.07527 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.706893 w_k1_ADM_s017
                                 + 0.972039 w_k1_AXP_s017
                                 + 1.13856 w_k1_DIS_s017
                                 + 0.986352 w_k1_FDX_s017
                                 + 1.35138 w_k1_IBM_s017
                                 + 1.05727 w_k1_INTC_s017
                                 + 1.14329 w_k1_KO_s017 + 1.29219 w_k1_MAT_s017
                                 + 0.961473 w_k1_MSFT_s017
                                 + 1.06048 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.48941 w_k1_ADM_s100
                                 + 1.49046 w_k1_AXP_s100
                                 + 1.23216 w_k1_DIS_s100 + 1.2813 w_k1_FDX_s100
                                 + 1.21962 w_k1_IBM_s100
                                 + 1.4176 w_k1_INTC_s100 + 1.12049 w_k1_KO_s100
                                 + 1.52032 w_k1_MAT_s100
                                 + 1.30798 w_k1_MSFT_s100
                                 + 1.21672 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.19214 w_k1_ADM_s100
                                 + 1.18946 w_k1_AXP_s100
                                 + 1.17292 w_k1_DIS_s100
                                 + 1.22398 w_k1_FDX_s100
                                 + 0.874922 w_k1_IBM_s100
                                 + 1.36108 w_k1_INTC_s100
                                 + 0.980677 w_k1_KO_s100
                                 + 1.10275 w_k1_MAT_s100
                                 + 1.15211 w_k1_MSFT_s100
                                 + 1.01176 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.15817 w_k1_ADM_s100
                                 + 1.31219 w_k1_AXP_s100
                                 + 1.21961 w_k1_DIS_s100
                                 + 1.12478 w_k1_FDX_s100
                                 + 1.93715 w_k1_IBM_s100
                                 + 1.2964 w_k1_INTC_s100 + 1.46747 w_k1_KO_s100
                                 + 1.17444 w_k1_MAT_s100
                                 + 1.2503 w_k1_MSFT_s100 + 1.24682 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.0575 w_k1_ADM_s100
                                 + 0.871765 w_k1_AXP_s100
                                 + 0.898329 w_k1_DIS_s100
                                 + 0.905565 w_k1_FDX_s100
                                 + 1.38041 w_k1_IBM_s100
                                 + 0.906437 w_k1_INTC_s100
                                 + 1.01263 w_k1_KO_s100 + 1.39684 w_k1_MAT_s100
                                 + 0.975546 w_k1_MSFT_s100
                                 + 1.05205 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.51839 w_k1_ADM_s100
                                 + 1.53256 w_k1_AXP_s100
                                 + 1.69623 w_k1_DIS_s100
                                 + 1.41217 w_k1_FDX_s100
                                 + 1.34364 w_k1_IBM_s100
                                 + 1.44613 w_k1_INTC_s100
                                 + 1.50959 w_k1_KO_s100 + 1.52771 w_k1_MAT_s100
                                 + 1.52647 w_k1_MSFT_s100
                                 + 1.20446 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.966878 w_k1_ADM_s100
                                 + 0.855807 w_k1_AXP_s100
                                 + 0.756414 w_k1_DIS_s100
                                 + 0.884587 w_k1_FDX_s100
                                 + 0.886881 w_k1_IBM_s100
                                 + 0.99799 w_k1_INTC_s100
                                 + 0.799484 w_k1_KO_s100
                                 + 0.620064 w_k1_MAT_s100
                                 + 0.908166 w_k1_MSFT_s100
                                 + 0.874362 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.31869 w_k1_ADM_s100
                                 + 0.908737 w_k1_AXP_s100
                                 + 0.873192 w_k1_DIS_s100
                                 + 1.00461 w_k1_FDX_s100
                                 + 0.729416 w_k1_IBM_s100
                                 + 1.02036 w_k1_INTC_s100
                                 + 0.692335 w_k1_KO_s100
                                 + 0.466101 w_k1_MAT_s100
                                 + 0.933722 w_k1_MSFT_s100
                                 + 0.909869 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.11416 w_k1_ADM_s100
                                 + 0.987286 w_k1_AXP_s100
                                 + 1.21687 w_k1_DIS_s100
                                 + 1.07438 w_k1_FDX_s100
                                 + 1.09252 w_k1_IBM_s100
                                 + 1.15724 w_k1_INTC_s100
                                 + 0.975375 w_k1_KO_s100
                                 + 0.672958 w_k1_MAT_s100
                                 + 1.08499 w_k1_MSFT_s100 + 1.0182 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.32212 w_k1_ADM_s101
                                 + 1.44831 w_k1_AXP_s101
                                 + 1.33823 w_k1_DIS_s101
                                 + 1.37114 w_k1_FDX_s101
                                 + 0.629844 w_k1_IBM_s101
                                 + 1.34486 w_k1_INTC_s101
                                 + 1.07445 w_k1_KO_s101
                                 + 0.861924 w_k1_MAT_s101
                                 + 1.26072 w_k1_MSFT_s101
                                 + 1.06948 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.983983 w_k1_ADM_s101
                                 + 0.913841 w_k1_AXP_s101
                                 + 0.931956 w_k1_DIS_s101
                                 + 1.2129 w_k1_FDX_s101
                                 + 0.481049 w_k1_IBM_s101
                                 + 1.1459 w_k1_INTC_s101
                                 + 0.723062 w_k1_KO_s101
                                 + 0.428713 w_k1_MAT_s101
                                 + 1.132 w_k1_MSFT_s101 + 0.970424 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.968093 w_k1_ADM_s101
                                 + 1.1514 w_k1_AXP_s101 + 1.04009 w_k1_DIS_s101
                                 + 1.1043 w_k1_FDX_s101 + 1.04876 w_k1_IBM_s101
                                 + 0.994921 w_k1_INTC_s101
                                 + 0.914807 w_k1_KO_s101
                                 + 1.14973 w_k1_MAT_s101
                                 + 0.949956 w_k1_MSFT_s101
                                 + 1.0585 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.00373 w_k1_ADM_s101
                                 + 1.03711 w_k1_AXP_s101
                                 + 1.26323 w_k1_DIS_s101
                                 + 1.22559 w_k1_FDX_s101
                                 + 0.812496 w_k1_IBM_s101
                                 + 1.33959 w_k1_INTC_s101
                                 + 0.977357 w_k1_KO_s101
                                 + 0.607874 w_k1_MAT_s101
                                 + 1.13602 w_k1_MSFT_s101
                                 + 1.03072 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.757926 w_k1_ADM_s101
                                 + 0.940987 w_k1_AXP_s101
                                 + 0.710947 w_k1_DIS_s101
                                 + 0.750869 w_k1_FDX_s101
                                 + 1.19204 w_k1_IBM_s101
                                 + 0.889849 w_k1_INTC_s101
                                 + 1.00316 w_k1_KO_s101 + 1.02707 w_k1_MAT_s101
                                 + 0.739242 w_k1_MSFT_s101
                                 + 0.815543 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.815933 w_k1_ADM_s101
                                 + 0.80445 w_k1_AXP_s101
                                 + 0.873332 w_k1_DIS_s101
                                 + 0.957339 w_k1_FDX_s101
                                 + 0.546956 w_k1_IBM_s101
                                 + 0.757672 w_k1_INTC_s101
                                 + 0.76357 w_k1_KO_s101
                                 + 0.522247 w_k1_MAT_s101
                                 + 0.898572 w_k1_MSFT_s101
                                 + 0.871285 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.820882 w_k1_ADM_s101
                                 + 0.822389 w_k1_AXP_s101
                                 + 0.71117 w_k1_DIS_s101
                                 + 0.780842 w_k1_FDX_s101
                                 + 0.780459 w_k1_IBM_s101
                                 + 0.861001 w_k1_INTC_s101
                                 + 0.743119 w_k1_KO_s101
                                 + 0.697839 w_k1_MAT_s101
                                 + 0.693384 w_k1_MSFT_s101
                                 + 0.815978 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.894558 w_k1_ADM_s101
                                 + 1.03897 w_k1_AXP_s101
                                 + 1.18571 w_k1_DIS_s101
                                 + 0.955714 w_k1_FDX_s101
                                 + 1.57311 w_k1_IBM_s101
                                 + 1.00646 w_k1_INTC_s101 + 1.2731 w_k1_KO_s101
                                 + 1.29796 w_k1_MAT_s101
                                 + 0.954379 w_k1_MSFT_s101
                                 + 1.06789 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.996865 w_k1_ADM_s102
                                 + 1.69238 w_k1_AXP_s102
                                 + 1.47434 w_k1_DIS_s102
                                 + 1.11586 w_k1_FDX_s102
                                 + 2.02346 w_k1_IBM_s102
                                 + 1.16551 w_k1_INTC_s102
                                 + 1.50725 w_k1_KO_s102 + 2.38538 w_k1_MAT_s102
                                 + 1.17034 w_k1_MSFT_s102
                                 + 1.25472 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.24455 w_k1_ADM_s102
                                 + 1.35086 w_k1_AXP_s102
                                 + 1.68261 w_k1_DIS_s102
                                 + 1.29929 w_k1_FDX_s102
                                 + 1.46979 w_k1_IBM_s102
                                 + 1.08154 w_k1_INTC_s102
                                 + 1.52467 w_k1_KO_s102 + 1.93386 w_k1_MAT_s102
                                 + 1.25375 w_k1_MSFT_s102
                                 + 1.23402 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.03688 w_k1_ADM_s102
                                 + 1.24227 w_k1_AXP_s102 + 1.2174 w_k1_DIS_s102
                                 + 1.18623 w_k1_FDX_s102
                                 + 0.874815 w_k1_IBM_s102
                                 + 0.993297 w_k1_INTC_s102
                                 + 0.911022 w_k1_KO_s102
                                 + 1.04858 w_k1_MAT_s102
                                 + 1.11057 w_k1_MSFT_s102 + 1.1113 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.996826 w_k1_ADM_s102
                                 + 0.998619 w_k1_AXP_s102
                                 + 0.927355 w_k1_DIS_s102
                                 + 0.921532 w_k1_FDX_s102
                                 + 1.46468 w_k1_IBM_s102
                                 + 0.837932 w_k1_INTC_s102
                                 + 1.04653 w_k1_KO_s102 + 1.39827 w_k1_MAT_s102
                                 + 0.840905 w_k1_MSFT_s102
                                 + 1.11572 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.903884 w_k1_ADM_s102
                                 + 0.867691 w_k1_AXP_s102
                                 + 1.15838 w_k1_DIS_s102
                                 + 1.08665 w_k1_FDX_s102
                                 + 0.83669 w_k1_IBM_s102
                                 + 1.02028 w_k1_INTC_s102
                                 + 0.91132 w_k1_KO_s102 + 0.75564 w_k1_MAT_s102
                                 + 1.10309 w_k1_MSFT_s102
                                 + 0.906268 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.856896 w_k1_ADM_s102
                                 + 0.793274 w_k1_AXP_s102
                                 + 0.729602 w_k1_DIS_s102
                                 + 0.888073 w_k1_FDX_s102
                                 + 0.574311 w_k1_IBM_s102
                                 + 0.784528 w_k1_INTC_s102
                                 + 0.675041 w_k1_KO_s102
                                 + 0.669419 w_k1_MAT_s102
                                 + 0.839477 w_k1_MSFT_s102
                                 + 0.846174 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.881532 w_k1_ADM_s102
                                 + 0.873287 w_k1_AXP_s102
                                 + 0.767646 w_k1_DIS_s102
                                 + 0.827393 w_k1_FDX_s102
                                 + 0.968275 w_k1_IBM_s102
                                 + 0.892179 w_k1_INTC_s102
                                 + 0.753632 w_k1_KO_s102
                                 + 0.617989 w_k1_MAT_s102
                                 + 0.769972 w_k1_MSFT_s102
                                 + 0.902102 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.869828 w_k1_ADM_s102
                                 + 1.11586 w_k1_AXP_s102
                                 + 1.36618 w_k1_DIS_s102
                                 + 1.11611 w_k1_FDX_s102
                                 + 1.33327 w_k1_IBM_s102
                                 + 1.02027 w_k1_INTC_s102
                                 + 1.24709 w_k1_KO_s102 + 1.01697 w_k1_MAT_s102
                                 + 1.03795 w_k1_MSFT_s102
                                 + 1.14511 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.09213 w_k1_ADM_s103
                                 + 1.31762 w_k1_AXP_s103
                                 + 1.36409 w_k1_DIS_s103
                                 + 1.24306 w_k1_FDX_s103
                                 + 1.06515 w_k1_IBM_s103
                                 + 1.0518 w_k1_INTC_s103 + 1.06605 w_k1_KO_s103
                                 + 0.855275 w_k1_MAT_s103
                                 + 1.32622 w_k1_MSFT_s103
                                 + 1.06016 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.07977 w_k1_ADM_s103
                                 + 0.852337 w_k1_AXP_s103
                                 + 0.911534 w_k1_DIS_s103
                                 + 0.956419 w_k1_FDX_s103
                                 + 0.756832 w_k1_IBM_s103
                                 + 0.842953 w_k1_INTC_s103
                                 + 0.918647 w_k1_KO_s103
                                 + 0.883379 w_k1_MAT_s103
                                 + 0.976459 w_k1_MSFT_s103
                                 + 0.94012 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.08749 w_k1_ADM_s103
                                 + 1.06254 w_k1_AXP_s103
                                 + 1.53228 w_k1_DIS_s103
                                 + 1.03485 w_k1_FDX_s103
                                 + 1.72368 w_k1_IBM_s103
                                 + 1.22572 w_k1_INTC_s103
                                 + 1.18957 w_k1_KO_s103 + 1.372 w_k1_MAT_s103
                                 + 1.33965 w_k1_MSFT_s103
                                 + 1.17034 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.21836 w_k1_ADM_s103
                                 + 0.969961 w_k1_AXP_s103
                                 + 1.08951 w_k1_DIS_s103
                                 + 1.20772 w_k1_FDX_s103
                                 + 0.610581 w_k1_IBM_s103
                                 + 1.01633 w_k1_INTC_s103
                                 + 0.854168 w_k1_KO_s103
                                 + 0.71299 w_k1_MAT_s103
                                 + 1.22985 w_k1_MSFT_s103
                                 + 0.952096 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.718919 w_k1_ADM_s103
                                 + 0.700674 w_k1_AXP_s103
                                 + 0.807119 w_k1_DIS_s103
                                 + 0.753131 w_k1_FDX_s103
                                 + 0.814862 w_k1_IBM_s103
                                 + 0.801615 w_k1_INTC_s103
                                 + 0.819532 w_k1_KO_s103
                                 + 0.658241 w_k1_MAT_s103
                                 + 0.793451 w_k1_MSFT_s103
                                 + 0.820902 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.772884 w_k1_ADM_s103
                                 + 0.802171 w_k1_AXP_s103
                                 + 1.04565 w_k1_DIS_s103
                                 + 0.884356 w_k1_FDX_s103
                                 + 0.924351 w_k1_IBM_s103
                                 + 0.963793 w_k1_INTC_s103
                                 + 0.921663 w_k1_KO_s103
                                 + 0.991455 w_k1_MAT_s103
                                 + 1.05507 w_k1_MSFT_s103
                                 + 0.914724 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.938974 w_k1_ADM_s103
                                 + 0.824073 w_k1_AXP_s103
                                 + 1.07863 w_k1_DIS_s103
                                 + 0.944694 w_k1_FDX_s103
                                 + 1.19253 w_k1_IBM_s103
                                 + 0.977005 w_k1_INTC_s103
                                 + 1.10891 w_k1_KO_s103 + 0.99021 w_k1_MAT_s103
                                 + 1.07132 w_k1_MSFT_s103
                                 + 0.977774 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.41934 w_k1_ADM_s103
                                 + 1.21224 w_k1_AXP_s103
                                 + 1.73588 w_k1_DIS_s103
                                 + 1.45278 w_k1_FDX_s103
                                 + 1.32244 w_k1_IBM_s103
                                 + 1.32873 w_k1_INTC_s103
                                 + 1.40899 w_k1_KO_s103 + 1.47311 w_k1_MAT_s103
                                 + 1.64798 w_k1_MSFT_s103
                                 + 1.25575 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.726555 w_k1_ADM_s104
                                 + 0.622704 w_k1_AXP_s104
                                 + 0.572178 w_k1_DIS_s104
                                 + 0.775172 w_k1_FDX_s104
                                 + 0.479406 w_k1_IBM_s104
                                 + 0.894237 w_k1_INTC_s104
                                 + 0.583711 w_k1_KO_s104
                                 + 0.412832 w_k1_MAT_s104
                                 + 0.78097 w_k1_MSFT_s104
                                 + 0.748217 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.26602 w_k1_ADM_s104
                                 + 0.960579 w_k1_AXP_s104
                                 + 0.803404 w_k1_DIS_s104
                                 + 0.953557 w_k1_FDX_s104
                                 + 1.06838 w_k1_IBM_s104
                                 + 0.847819 w_k1_INTC_s104
                                 + 1.02676 w_k1_KO_s104 + 1.01197 w_k1_MAT_s104
                                 + 0.78385 w_k1_MSFT_s104
                                 + 1.01617 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.19158 w_k1_ADM_s104
                                 + 1.47043 w_k1_AXP_s104
                                 + 0.990371 w_k1_DIS_s104
                                 + 1.04557 w_k1_FDX_s104 + 1.5281 w_k1_IBM_s104
                                 + 0.966219 w_k1_INTC_s104
                                 + 1.28729 w_k1_KO_s104 + 1.61897 w_k1_MAT_s104
                                 + 1.0028 w_k1_MSFT_s104 + 1.06918 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.996644 w_k1_ADM_s104
                                 + 0.674746 w_k1_AXP_s104
                                 + 0.54631 w_k1_DIS_s104
                                 + 0.755594 w_k1_FDX_s104
                                 + 0.817415 w_k1_IBM_s104
                                 + 0.67682 w_k1_INTC_s104
                                 + 0.663098 w_k1_KO_s104
                                 + 0.699336 w_k1_MAT_s104
                                 + 0.683658 w_k1_MSFT_s104
                                 + 0.817499 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.3433 w_k1_ADM_s104
                                 + 1.09653 w_k1_AXP_s104
                                 + 1.28019 w_k1_DIS_s104
                                 + 1.25704 w_k1_FDX_s104
                                 + 0.886474 w_k1_IBM_s104
                                 + 1.09686 w_k1_INTC_s104
                                 + 1.02636 w_k1_KO_s104 + 0.91092 w_k1_MAT_s104
                                 + 1.31342 w_k1_MSFT_s104
                                 + 1.15874 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.03096 w_k1_ADM_s104
                                 + 0.82989 w_k1_AXP_s104
                                 + 0.854065 w_k1_DIS_s104
                                 + 0.943434 w_k1_FDX_s104
                                 + 0.695019 w_k1_IBM_s104
                                 + 0.796078 w_k1_INTC_s104
                                 + 0.790678 w_k1_KO_s104
                                 + 0.513249 w_k1_MAT_s104
                                 + 0.960343 w_k1_MSFT_s104
                                 + 0.955497 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.33202 w_k1_ADM_s104
                                 + 1.29566 w_k1_AXP_s104
                                 + 1.41866 w_k1_DIS_s104
                                 + 1.26197 w_k1_FDX_s104
                                 + 1.23663 w_k1_IBM_s104
                                 + 1.18654 w_k1_INTC_s104
                                 + 1.45091 w_k1_KO_s104 + 1.2234 w_k1_MAT_s104
                                 + 1.18349 w_k1_MSFT_s104
                                 + 1.32949 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.87394 w_k1_ADM_s104
                                 + 1.86207 w_k1_AXP_s104
                                 + 1.65318 w_k1_DIS_s104
                                 + 1.31249 w_k1_FDX_s104
                                 + 2.15292 w_k1_IBM_s104
                                 + 1.0836 w_k1_INTC_s104 + 1.58875 w_k1_KO_s104
                                 + 2.32349 w_k1_MAT_s104
                                 + 1.05565 w_k1_MSFT_s104
                                 + 1.28758 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.0202 w_k1_ADM_s105
                                 + 1.29456 w_k1_AXP_s105
                                 + 0.897095 w_k1_DIS_s105
                                 + 1.04101 w_k1_FDX_s105
                                 + 0.91884 w_k1_IBM_s105
                                 + 1.2001 w_k1_INTC_s105 + 0.87397 w_k1_KO_s105
                                 + 0.847124 w_k1_MAT_s105
                                 + 0.867086 w_k1_MSFT_s105
                                 + 0.971083 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.16839 w_k1_ADM_s105
                                 + 0.786077 w_k1_AXP_s105
                                 + 0.817844 w_k1_DIS_s105
                                 + 0.961571 w_k1_FDX_s105
                                 + 0.543324 w_k1_IBM_s105
                                 + 1.00398 w_k1_INTC_s105
                                 + 0.678712 w_k1_KO_s105
                                 + 0.393195 w_k1_MAT_s105
                                 + 0.923842 w_k1_MSFT_s105
                                 + 0.817357 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.18214 w_k1_ADM_s105
                                 + 1.67123 w_k1_AXP_s105
                                 + 1.56589 w_k1_DIS_s105 + 1.2926 w_k1_FDX_s105
                                 + 1.50587 w_k1_IBM_s105
                                 + 1.49415 w_k1_INTC_s105
                                 + 1.36481 w_k1_KO_s105 + 1.13832 w_k1_MAT_s105
                                 + 1.20763 w_k1_MSFT_s105
                                 + 1.22334 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.801859 w_k1_ADM_s105
                                 + 0.766367 w_k1_AXP_s105
                                 + 0.718665 w_k1_DIS_s105
                                 + 0.784058 w_k1_FDX_s105
                                 + 0.820809 w_k1_IBM_s105 + 1.01 w_k1_INTC_s105
                                 + 0.831863 w_k1_KO_s105
                                 + 0.711759 w_k1_MAT_s105
                                 + 0.783113 w_k1_MSFT_s105
                                 + 0.823044 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.751441 w_k1_ADM_s105
                                 + 1.08668 w_k1_AXP_s105
                                 + 0.841103 w_k1_DIS_s105
                                 + 0.813204 w_k1_FDX_s105
                                 + 1.25068 w_k1_IBM_s105
                                 + 1.15509 w_k1_INTC_s105
                                 + 1.10334 w_k1_KO_s105 + 1.60835 w_k1_MAT_s105
                                 + 0.893349 w_k1_MSFT_s105
                                 + 0.905489 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.08167 w_k1_ADM_s105
                                 + 0.986728 w_k1_AXP_s105
                                 + 1.46479 w_k1_DIS_s105
                                 + 1.19095 w_k1_FDX_s105
                                 + 1.10129 w_k1_IBM_s105
                                 + 1.45026 w_k1_INTC_s105
                                 + 1.14394 w_k1_KO_s105
                                 + 0.857114 w_k1_MAT_s105
                                 + 1.19641 w_k1_MSFT_s105
                                 + 0.973483 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.2788 w_k1_ADM_s105
                                 + 1.05048 w_k1_AXP_s105
                                 + 1.29125 w_k1_DIS_s105
                                 + 1.29989 w_k1_FDX_s105 + 1.0257 w_k1_IBM_s105
                                 + 1.42147 w_k1_INTC_s105
                                 + 1.04378 w_k1_KO_s105 + 0.74787 w_k1_MAT_s105
                                 + 1.11118 w_k1_MSFT_s105
                                 + 1.06647 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.983309 w_k1_ADM_s105
                                 + 1.08246 w_k1_AXP_s105
                                 + 1.13174 w_k1_DIS_s105
                                 + 1.00117 w_k1_FDX_s105
                                 + 1.17111 w_k1_IBM_s105
                                 + 1.14109 w_k1_INTC_s105
                                 + 1.16747 w_k1_KO_s105 + 1.31245 w_k1_MAT_s105
                                 + 1.09629 w_k1_MSFT_s105
                                 + 1.04291 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.617968 w_k1_ADM_s106
                                 + 0.725211 w_k1_AXP_s106
                                 + 0.506947 w_k1_DIS_s106
                                 + 0.77955 w_k1_FDX_s106
                                 + 0.617138 w_k1_IBM_s106
                                 + 0.801551 w_k1_INTC_s106
                                 + 0.614309 w_k1_KO_s106
                                 + 0.683773 w_k1_MAT_s106
                                 + 0.686825 w_k1_MSFT_s106
                                 + 0.691672 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.810279 w_k1_ADM_s106
                                 + 1.10702 w_k1_AXP_s106
                                 + 1.01904 w_k1_DIS_s106 + 1.2818 w_k1_FDX_s106
                                 + 0.682543 w_k1_IBM_s106
                                 + 1.16823 w_k1_INTC_s106
                                 + 0.85891 w_k1_KO_s106
                                 + 0.738136 w_k1_MAT_s106
                                 + 0.986685 w_k1_MSFT_s106
                                 + 0.970569 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.856944 w_k1_ADM_s106
                                 + 1.00868 w_k1_AXP_s106
                                 + 0.777163 w_k1_DIS_s106
                                 + 1.01256 w_k1_FDX_s106
                                 + 0.845456 w_k1_IBM_s106
                                 + 1.13659 w_k1_INTC_s106
                                 + 0.903017 w_k1_KO_s106
                                 + 0.840328 w_k1_MAT_s106
                                 + 0.886445 w_k1_MSFT_s106
                                 + 0.956362 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.902181 w_k1_ADM_s106
                                 + 1.24198 w_k1_AXP_s106
                                 + 1.29321 w_k1_DIS_s106
                                 + 1.19055 w_k1_FDX_s106
                                 + 1.08551 w_k1_IBM_s106
                                 + 1.08738 w_k1_INTC_s106
                                 + 1.16086 w_k1_KO_s106 + 1.14338 w_k1_MAT_s106
                                 + 0.944404 w_k1_MSFT_s106
                                 + 1.0158 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.844821 w_k1_ADM_s106
                                 + 1.00994 w_k1_AXP_s106
                                 + 0.840161 w_k1_DIS_s106
                                 + 0.969271 w_k1_FDX_s106
                                 + 1.18917 w_k1_IBM_s106
                                 + 1.00534 w_k1_INTC_s106
                                 + 1.00246 w_k1_KO_s106 + 1.02453 w_k1_MAT_s106
                                 + 0.841786 w_k1_MSFT_s106
                                 + 1.01966 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.848788 w_k1_ADM_s106
                                 + 1.52674 w_k1_AXP_s106
                                 + 1.32377 w_k1_DIS_s106
                                 + 1.20089 w_k1_FDX_s106
                                 + 1.76799 w_k1_IBM_s106
                                 + 0.99291 w_k1_INTC_s106
                                 + 1.41178 w_k1_KO_s106 + 2.33269 w_k1_MAT_s106
                                 + 1.00718 w_k1_MSFT_s106 + 1.2012 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.27293 w_k1_ADM_s106
                                 + 1.33933 w_k1_AXP_s106
                                 + 1.55187 w_k1_DIS_s106
                                 + 1.47184 w_k1_FDX_s106 + 1.1236 w_k1_IBM_s106
                                 + 1.11016 w_k1_INTC_s106
                                 + 1.41505 w_k1_KO_s106 + 1.17009 w_k1_MAT_s106
                                 + 1.22772 w_k1_MSFT_s106 + 1.3211 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.788223 w_k1_ADM_s106
                                 + 0.756755 w_k1_AXP_s106
                                 + 1.0113 w_k1_DIS_s106 + 1.00134 w_k1_FDX_s106
                                 + 0.636466 w_k1_IBM_s106
                                 + 1.07881 w_k1_INTC_s106
                                 + 0.800745 w_k1_KO_s106
                                 + 0.431513 w_k1_MAT_s106
                                 + 0.916032 w_k1_MSFT_s106
                                 + 0.87069 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.15862 w_k1_ADM_s107
                                 + 1.03095 w_k1_AXP_s107
                                 + 1.48408 w_k1_DIS_s107
                                 + 1.25903 w_k1_FDX_s107
                                 + 1.06672 w_k1_IBM_s107
                                 + 1.27768 w_k1_INTC_s107
                                 + 1.13561 w_k1_KO_s107 + 1.10223 w_k1_MAT_s107
                                 + 1.16689 w_k1_MSFT_s107
                                 + 1.08378 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.13239 w_k1_ADM_s107
                                 + 0.946506 w_k1_AXP_s107
                                 + 0.97286 w_k1_DIS_s107
                                 + 1.02222 w_k1_FDX_s107
                                 + 0.842301 w_k1_IBM_s107
                                 + 0.960539 w_k1_INTC_s107
                                 + 0.854726 w_k1_KO_s107
                                 + 0.891397 w_k1_MAT_s107
                                 + 0.838708 w_k1_MSFT_s107
                                 + 0.931906 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.694129 w_k1_ADM_s107
                                 + 0.723895 w_k1_AXP_s107
                                 + 0.601121 w_k1_DIS_s107
                                 + 0.766734 w_k1_FDX_s107
                                 + 0.722536 w_k1_IBM_s107
                                 + 0.975898 w_k1_INTC_s107
                                 + 0.695432 w_k1_KO_s107
                                 + 0.723012 w_k1_MAT_s107
                                 + 0.776833 w_k1_MSFT_s107
                                 + 0.833516 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.901277 w_k1_ADM_s107
                                 + 1.02226 w_k1_AXP_s107
                                 + 0.769965 w_k1_DIS_s107
                                 + 0.927171 w_k1_FDX_s107
                                 + 1.11493 w_k1_IBM_s107
                                 + 1.06617 w_k1_INTC_s107
                                 + 0.998204 w_k1_KO_s107
                                 + 1.24496 w_k1_MAT_s107
                                 + 0.807449 w_k1_MSFT_s107
                                 + 0.88989 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.875119 w_k1_ADM_s107
                                 + 1.09818 w_k1_AXP_s107
                                 + 1.08072 w_k1_DIS_s107
                                 + 0.972863 w_k1_FDX_s107
                                 + 1.61179 w_k1_IBM_s107
                                 + 1.27884 w_k1_INTC_s107
                                 + 1.23231 w_k1_KO_s107 + 1.50907 w_k1_MAT_s107
                                 + 0.927674 w_k1_MSFT_s107
                                 + 1.06959 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.18491 w_k1_ADM_s107
                                 + 1.38658 w_k1_AXP_s107 + 1.1971 w_k1_DIS_s107
                                 + 1.26015 w_k1_FDX_s107
                                 + 1.04551 w_k1_IBM_s107
                                 + 1.36729 w_k1_INTC_s107
                                 + 0.952669 w_k1_KO_s107
                                 + 1.21063 w_k1_MAT_s107
                                 + 1.01161 w_k1_MSFT_s107
                                 + 0.979228 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.816832 w_k1_ADM_s107
                                 + 0.844446 w_k1_AXP_s107
                                 + 0.992769 w_k1_DIS_s107
                                 + 0.888738 w_k1_FDX_s107
                                 + 1.00405 w_k1_IBM_s107
                                 + 0.828103 w_k1_INTC_s107
                                 + 0.91489 w_k1_KO_s107 + 1.20668 w_k1_MAT_s107
                                 + 0.854847 w_k1_MSFT_s107
                                 + 0.882896 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.928219 w_k1_ADM_s107
                                 + 0.750831 w_k1_AXP_s107
                                 + 0.865444 w_k1_DIS_s107
                                 + 0.942136 w_k1_FDX_s107
                                 + 0.623306 w_k1_IBM_s107
                                 + 0.979638 w_k1_INTC_s107
                                 + 0.728562 w_k1_KO_s107
                                 + 0.748415 w_k1_MAT_s107
                                 + 0.897048 w_k1_MSFT_s107
                                 + 0.90801 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.01405 w_k1_ADM_s110
                                 + 1.08115 w_k1_AXP_s110
                                 + 1.00502 w_k1_DIS_s110
                                 + 1.00802 w_k1_FDX_s110
                                 + 1.33443 w_k1_IBM_s110
                                 + 1.02007 w_k1_INTC_s110
                                 + 1.06414 w_k1_KO_s110 + 1.43562 w_k1_MAT_s110
                                 + 0.955212 w_k1_MSFT_s110
                                 + 0.959179 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.00182 w_k1_ADM_s110
                                 + 0.761566 w_k1_AXP_s110
                                 + 1.15729 w_k1_DIS_s110
                                 + 0.909526 w_k1_FDX_s110
                                 + 0.964025 w_k1_IBM_s110
                                 + 1.03534 w_k1_INTC_s110
                                 + 1.04086 w_k1_KO_s110 + 0.56957 w_k1_MAT_s110
                                 + 1.09644 w_k1_MSFT_s110
                                 + 0.981261 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.73403 w_k1_ADM_s110
                                 + 1.4792 w_k1_AXP_s110 + 1.69022 w_k1_DIS_s110
                                 + 1.3639 w_k1_FDX_s110 + 1.23711 w_k1_IBM_s110
                                 + 1.34872 w_k1_INTC_s110
                                 + 1.32551 w_k1_KO_s110
                                 + 0.979699 w_k1_MAT_s110
                                 + 1.48761 w_k1_MSFT_s110
                                 + 1.04641 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.09981 w_k1_ADM_s110
                                 + 0.849039 w_k1_AXP_s110
                                 + 0.856835 w_k1_DIS_s110
                                 + 1.00247 w_k1_FDX_s110
                                 + 0.616233 w_k1_IBM_s110
                                 + 0.954577 w_k1_INTC_s110
                                 + 0.766923 w_k1_KO_s110
                                 + 0.534187 w_k1_MAT_s110
                                 + 1.08164 w_k1_MSFT_s110
                                 + 0.818804 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.05577 w_k1_ADM_s110
                                 + 1.09157 w_k1_AXP_s110
                                 + 0.918989 w_k1_DIS_s110
                                 + 0.987176 w_k1_FDX_s110
                                 + 0.976468 w_k1_IBM_s110
                                 + 0.924653 w_k1_INTC_s110
                                 + 0.936393 w_k1_KO_s110
                                 + 0.892623 w_k1_MAT_s110
                                 + 0.920721 w_k1_MSFT_s110
                                 + 0.863597 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.842085 w_k1_ADM_s110
                                 + 0.735736 w_k1_AXP_s110
                                 + 0.632085 w_k1_DIS_s110
                                 + 0.800654 w_k1_FDX_s110
                                 + 0.811709 w_k1_IBM_s110
                                 + 0.737477 w_k1_INTC_s110
                                 + 0.635199 w_k1_KO_s110
                                 + 0.655873 w_k1_MAT_s110
                                 + 0.807697 w_k1_MSFT_s110
                                 + 0.822274 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.21758 w_k1_ADM_s110
                                 + 1.21305 w_k1_AXP_s110
                                 + 1.17198 w_k1_DIS_s110
                                 + 1.22441 w_k1_FDX_s110
                                 + 0.901956 w_k1_IBM_s110
                                 + 0.992585 w_k1_INTC_s110
                                 + 0.929959 w_k1_KO_s110
                                 + 0.785982 w_k1_MAT_s110
                                 + 1.08934 w_k1_MSFT_s110
                                 + 0.930648 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.23787 w_k1_ADM_s110
                                 + 1.17072 w_k1_AXP_s110
                                 + 1.29859 w_k1_DIS_s110
                                 + 1.05244 w_k1_FDX_s110
                                 + 1.82449 w_k1_IBM_s110
                                 + 1.08752 w_k1_INTC_s110
                                 + 1.42561 w_k1_KO_s110 + 1.30215 w_k1_MAT_s110
                                 + 1.21781 w_k1_MSFT_s110
                                 + 1.12437 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.852993 w_k1_ADM_s111
                                 + 1.39582 w_k1_AXP_s111
                                 + 1.03312 w_k1_DIS_s111
                                 + 1.01447 w_k1_FDX_s111
                                 + 1.59071 w_k1_IBM_s111
                                 + 1.04973 w_k1_INTC_s111
                                 + 1.34659 w_k1_KO_s111 + 2.65838 w_k1_MAT_s111
                                 + 0.985689 w_k1_MSFT_s111
                                 + 1.12443 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.901034 w_k1_ADM_s111
                                 + 1.02708 w_k1_AXP_s111
                                 + 1.06075 w_k1_DIS_s111
                                 + 1.03346 w_k1_FDX_s111
                                 + 1.08748 w_k1_IBM_s111
                                 + 1.12303 w_k1_INTC_s111
                                 + 1.28818 w_k1_KO_s111
                                 + 0.987074 w_k1_MAT_s111
                                 + 1.06043 w_k1_MSFT_s111
                                 + 1.07762 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.799315 w_k1_ADM_s111
                                 + 0.770727 w_k1_AXP_s111
                                 + 0.841851 w_k1_DIS_s111
                                 + 0.877563 w_k1_FDX_s111
                                 + 0.732513 w_k1_IBM_s111
                                 + 0.928565 w_k1_INTC_s111
                                 + 0.797903 w_k1_KO_s111
                                 + 0.646389 w_k1_MAT_s111
                                 + 0.829712 w_k1_MSFT_s111
                                 + 0.863709 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.11899 w_k1_ADM_s111
                                 + 1.24725 w_k1_AXP_s111
                                 + 1.33335 w_k1_DIS_s111
                                 + 1.27223 w_k1_FDX_s111
                                 + 0.969582 w_k1_IBM_s111
                                 + 1.2835 w_k1_INTC_s111 + 1.19826 w_k1_KO_s111
                                 + 1.07662 w_k1_MAT_s111
                                 + 1.17582 w_k1_MSFT_s111 + 1.1089 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.04585 w_k1_ADM_s111
                                 + 1.07639 w_k1_AXP_s111
                                 + 1.10773 w_k1_DIS_s111
                                 + 1.13166 w_k1_FDX_s111
                                 + 0.74871 w_k1_IBM_s111
                                 + 1.1255 w_k1_INTC_s111 + 1.029 w_k1_KO_s111
                                 + 1.0712 w_k1_MAT_s111
                                 + 1.16157 w_k1_MSFT_s111
                                 + 1.10319 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.24125 w_k1_ADM_s111
                                 + 1.18398 w_k1_AXP_s111
                                 + 1.70555 w_k1_DIS_s111
                                 + 1.26526 w_k1_FDX_s111
                                 + 1.44126 w_k1_IBM_s111
                                 + 1.13254 w_k1_INTC_s111
                                 + 1.71476 w_k1_KO_s111 + 1.65934 w_k1_MAT_s111
                                 + 1.39898 w_k1_MSFT_s111
                                 + 1.32988 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.844554 w_k1_ADM_s111
                                 + 0.876814 w_k1_AXP_s111
                                 + 0.725555 w_k1_DIS_s111
                                 + 0.812162 w_k1_FDX_s111
                                 + 1.06241 w_k1_IBM_s111
                                 + 0.774861 w_k1_INTC_s111
                                 + 0.99847 w_k1_KO_s111 + 1.49945 w_k1_MAT_s111
                                 + 0.838764 w_k1_MSFT_s111
                                 + 0.94443 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.9164 w_k1_ADM_s111
                                 + 1.05644 w_k1_AXP_s111
                                 + 0.996819 w_k1_DIS_s111
                                 + 0.997757 w_k1_FDX_s111
                                 + 1.28578 w_k1_IBM_s111
                                 + 1.09163 w_k1_INTC_s111
                                 + 1.19333 w_k1_KO_s111 + 1.59154 w_k1_MAT_s111
                                 + 1.15244 w_k1_MSFT_s111
                                 + 1.09892 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.965449 w_k1_ADM_s112
                                 + 0.815059 w_k1_AXP_s112
                                 + 0.580999 w_k1_DIS_s112
                                 + 0.776587 w_k1_FDX_s112
                                 + 1.0277 w_k1_IBM_s112
                                 + 0.705621 w_k1_INTC_s112
                                 + 0.74705 w_k1_KO_s112
                                 + 0.905167 w_k1_MAT_s112
                                 + 0.746184 w_k1_MSFT_s112
                                 + 0.873725 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.21032 w_k1_ADM_s112
                                 + 1.13123 w_k1_AXP_s112
                                 + 1.39967 w_k1_DIS_s112
                                 + 1.35215 w_k1_FDX_s112
                                 + 0.87171 w_k1_IBM_s112
                                 + 1.18761 w_k1_INTC_s112
                                 + 0.871466 w_k1_KO_s112
                                 + 0.691611 w_k1_MAT_s112
                                 + 1.13375 w_k1_MSFT_s112
                                 + 1.07886 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.39146 w_k1_ADM_s112
                                 + 1.48429 w_k1_AXP_s112
                                 + 1.62005 w_k1_DIS_s112
                                 + 1.44536 w_k1_FDX_s112
                                 + 1.16821 w_k1_IBM_s112
                                 + 1.36262 w_k1_INTC_s112
                                 + 1.18453 w_k1_KO_s112 + 1.16808 w_k1_MAT_s112
                                 + 1.28853 w_k1_MSFT_s112
                                 + 1.11373 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.18913 w_k1_ADM_s112
                                 + 1.31202 w_k1_AXP_s112
                                 + 1.19819 w_k1_DIS_s112 + 1.2077 w_k1_FDX_s112
                                 + 1.36055 w_k1_IBM_s112
                                 + 1.17068 w_k1_INTC_s112
                                 + 1.16738 w_k1_KO_s112 + 1.30654 w_k1_MAT_s112
                                 + 1.11218 w_k1_MSFT_s112
                                 + 1.08589 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.867868 w_k1_ADM_s112
                                 + 0.792703 w_k1_AXP_s112
                                 + 0.72589 w_k1_DIS_s112
                                 + 0.839746 w_k1_FDX_s112
                                 + 0.897234 w_k1_IBM_s112
                                 + 0.777823 w_k1_INTC_s112
                                 + 0.797707 w_k1_KO_s112
                                 + 0.824154 w_k1_MAT_s112
                                 + 0.779521 w_k1_MSFT_s112
                                 + 0.879754 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.914781 w_k1_ADM_s112
                                 + 1.03894 w_k1_AXP_s112
                                 + 0.882582 w_k1_DIS_s112
                                 + 0.898971 w_k1_FDX_s112
                                 + 1.56912 w_k1_IBM_s112
                                 + 1.07219 w_k1_INTC_s112
                                 + 1.05292 w_k1_KO_s112 + 1.69461 w_k1_MAT_s112
                                 + 0.901845 w_k1_MSFT_s112
                                 + 0.971888 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.866848 w_k1_ADM_s112
                                 + 1.04378 w_k1_AXP_s112
                                 + 0.940333 w_k1_DIS_s112
                                 + 1.02461 w_k1_FDX_s112
                                 + 1.10129 w_k1_IBM_s112
                                 + 1.06165 w_k1_INTC_s112
                                 + 0.962403 w_k1_KO_s112
                                 + 1.00278 w_k1_MAT_s112
                                 + 0.936237 w_k1_MSFT_s112
                                 + 0.946407 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.748718 w_k1_ADM_s112
                                 + 0.808484 w_k1_AXP_s112
                                 + 0.82694 w_k1_DIS_s112
                                 + 0.929422 w_k1_FDX_s112
                                 + 0.686982 w_k1_IBM_s112
                                 + 0.851526 w_k1_INTC_s112
                                 + 0.760069 w_k1_KO_s112
                                 + 0.683489 w_k1_MAT_s112
                                 + 0.95635 w_k1_MSFT_s112
                                 + 0.905558 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.620794 w_k1_ADM_s113
                                 + 0.653855 w_k1_AXP_s113
                                 + 0.567318 w_k1_DIS_s113
                                 + 0.755121 w_k1_FDX_s113
                                 + 0.620993 w_k1_IBM_s113
                                 + 0.861836 w_k1_INTC_s113
                                 + 0.772475 w_k1_KO_s113
                                 + 0.569517 w_k1_MAT_s113
                                 + 0.827596 w_k1_MSFT_s113
                                 + 0.865595 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.11098 w_k1_ADM_s113
                                 + 1.27855 w_k1_AXP_s113
                                 + 1.43856 w_k1_DIS_s113
                                 + 1.29506 w_k1_FDX_s113
                                 + 1.14479 w_k1_IBM_s113
                                 + 1.06475 w_k1_INTC_s113
                                 + 1.26996 w_k1_KO_s113
                                 + 0.860486 w_k1_MAT_s113
                                 + 1.31992 w_k1_MSFT_s113
                                 + 1.24912 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.0056 w_k1_ADM_s113
                                 + 0.894167 w_k1_AXP_s113
                                 + 1.02758 w_k1_DIS_s113
                                 + 1.06818 w_k1_FDX_s113
                                 + 0.891394 w_k1_IBM_s113
                                 + 0.930298 w_k1_INTC_s113
                                 + 1.03511 w_k1_KO_s113
                                 + 0.699936 w_k1_MAT_s113
                                 + 1.01598 w_k1_MSFT_s113
                                 + 1.04652 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.8759 w_k1_ADM_s113
                                 + 0.934033 w_k1_AXP_s113
                                 + 0.731976 w_k1_DIS_s113
                                 + 0.8031 w_k1_FDX_s113
                                 + 0.956023 w_k1_IBM_s113
                                 + 0.610663 w_k1_INTC_s113
                                 + 0.950202 w_k1_KO_s113
                                 + 1.23155 w_k1_MAT_s113
                                 + 0.750359 w_k1_MSFT_s113
                                 + 1.05832 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.927192 w_k1_ADM_s113
                                 + 1.45412 w_k1_AXP_s113
                                 + 1.20936 w_k1_DIS_s113
                                 + 1.05028 w_k1_FDX_s113
                                 + 1.52954 w_k1_IBM_s113
                                 + 1.29653 w_k1_INTC_s113
                                 + 1.35027 w_k1_KO_s113 + 1.60149 w_k1_MAT_s113
                                 + 1.06989 w_k1_MSFT_s113
                                 + 1.29621 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.765844 w_k1_ADM_s113
                                 + 1.02793 w_k1_AXP_s113
                                 + 0.805397 w_k1_DIS_s113
                                 + 0.846614 w_k1_FDX_s113
                                 + 1.2637 w_k1_IBM_s113
                                 + 1.00269 w_k1_INTC_s113
                                 + 1.21034 w_k1_KO_s113 + 1.65384 w_k1_MAT_s113
                                 + 1.07867 w_k1_MSFT_s113
                                 + 1.05866 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.02862 w_k1_ADM_s113
                                 + 1.07879 w_k1_AXP_s113
                                 + 0.739338 w_k1_DIS_s113
                                 + 0.932746 w_k1_FDX_s113
                                 + 0.728952 w_k1_IBM_s113
                                 + 0.875228 w_k1_INTC_s113
                                 + 0.944532 w_k1_KO_s113
                                 + 0.867462 w_k1_MAT_s113
                                 + 0.969887 w_k1_MSFT_s113
                                 + 0.980291 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.2912 w_k1_ADM_s113
                                 + 1.24638 w_k1_AXP_s113
                                 + 1.02531 w_k1_DIS_s113
                                 + 1.25403 w_k1_FDX_s113
                                 + 0.823095 w_k1_IBM_s113
                                 + 0.97633 w_k1_INTC_s113 + 1.1411 w_k1_KO_s113
                                 + 0.894347 w_k1_MAT_s113
                                 + 1.11665 w_k1_MSFT_s113 + 1.2071 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.855132 w_k1_ADM_s114
                                 + 0.799 w_k1_AXP_s114 + 0.792631 w_k1_DIS_s114
                                 + 0.83137 w_k1_FDX_s114
                                 + 0.992982 w_k1_IBM_s114
                                 + 0.774286 w_k1_INTC_s114
                                 + 0.874415 w_k1_KO_s114
                                 + 0.762652 w_k1_MAT_s114
                                 + 0.918431 w_k1_MSFT_s114
                                 + 1.0571 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.812623 w_k1_ADM_s114
                                 + 0.477422 w_k1_AXP_s114
                                 + 0.516726 w_k1_DIS_s114
                                 + 0.683781 w_k1_FDX_s114
                                 + 0.656681 w_k1_IBM_s114
                                 + 0.648927 w_k1_INTC_s114
                                 + 0.605811 w_k1_KO_s114
                                 + 0.42772 w_k1_MAT_s114
                                 + 0.775376 w_k1_MSFT_s114
                                 + 0.822275 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.992548 w_k1_ADM_s114
                                 + 0.748591 w_k1_AXP_s114
                                 + 0.857174 w_k1_DIS_s114
                                 + 0.914351 w_k1_FDX_s114
                                 + 0.772137 w_k1_IBM_s114
                                 + 0.922806 w_k1_INTC_s114
                                 + 0.853267 w_k1_KO_s114
                                 + 0.731713 w_k1_MAT_s114
                                 + 0.942741 w_k1_MSFT_s114
                                 + 0.879097 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.862071 w_k1_ADM_s114
                                 + 0.644079 w_k1_AXP_s114
                                 + 0.540353 w_k1_DIS_s114
                                 + 0.714408 w_k1_FDX_s114
                                 + 0.954235 w_k1_IBM_s114
                                 + 0.635241 w_k1_INTC_s114
                                 + 0.781883 w_k1_KO_s114
                                 + 0.902157 w_k1_MAT_s114
                                 + 0.667256 w_k1_MSFT_s114
                                 + 0.878467 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.990793 w_k1_ADM_s114
                                 + 0.9019 w_k1_AXP_s114
                                 + 0.703198 w_k1_DIS_s114
                                 + 0.853497 w_k1_FDX_s114
                                 + 1.33204 w_k1_IBM_s114
                                 + 0.835388 w_k1_INTC_s114
                                 + 0.932078 w_k1_KO_s114
                                 + 1.28012 w_k1_MAT_s114
                                 + 0.794037 w_k1_MSFT_s114
                                 + 1.00536 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.1229 w_k1_ADM_s114
                                 + 1.36842 w_k1_AXP_s114
                                 + 1.09949 w_k1_DIS_s114
                                 + 1.20449 w_k1_FDX_s114
                                 + 0.92932 w_k1_IBM_s114
                                 + 1.08432 w_k1_INTC_s114
                                 + 1.00533 w_k1_KO_s114 + 1.15468 w_k1_MAT_s114
                                 + 1.06608 w_k1_MSFT_s114
                                 + 1.18581 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.20918 w_k1_ADM_s114
                                 + 1.22001 w_k1_AXP_s114
                                 + 1.41134 w_k1_DIS_s114
                                 + 1.24659 w_k1_FDX_s114
                                 + 1.09695 w_k1_IBM_s114
                                 + 1.10972 w_k1_INTC_s114
                                 + 1.28048 w_k1_KO_s114 + 1.09697 w_k1_MAT_s114
                                 + 1.32812 w_k1_MSFT_s114
                                 + 1.14063 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.47401 w_k1_ADM_s114
                                 + 1.94254 w_k1_AXP_s114
                                 + 1.98184 w_k1_DIS_s114 + 1.5993 w_k1_FDX_s114
                                 + 1.52282 w_k1_IBM_s114
                                 + 1.12401 w_k1_INTC_s114
                                 + 1.18278 w_k1_KO_s114 + 1.72125 w_k1_MAT_s114
                                 + 1.24795 w_k1_MSFT_s114
                                 + 1.23871 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.03164 w_k1_ADM_s115
                                 + 0.73424 w_k1_AXP_s115
                                 + 0.958135 w_k1_DIS_s115
                                 + 0.783553 w_k1_FDX_s115
                                 + 1.23883 w_k1_IBM_s115
                                 + 0.908867 w_k1_INTC_s115
                                 + 1.0344 w_k1_KO_s115 + 1.08009 w_k1_MAT_s115
                                 + 0.829617 w_k1_MSFT_s115
                                 + 0.91561 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.04969 w_k1_ADM_s115
                                 + 1.54609 w_k1_AXP_s115
                                 + 1.30644 w_k1_DIS_s115
                                 + 1.39615 w_k1_FDX_s115
                                 + 0.898001 w_k1_IBM_s115
                                 + 1.25332 w_k1_INTC_s115
                                 + 1.08599 w_k1_KO_s115 + 1.32992 w_k1_MAT_s115
                                 + 1.24308 w_k1_MSFT_s115 + 1.13 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.87672 w_k1_ADM_s115
                                 + 0.565662 w_k1_AXP_s115
                                 + 0.755867 w_k1_DIS_s115
                                 + 0.803975 w_k1_FDX_s115
                                 + 0.730016 w_k1_IBM_s115
                                 + 0.933991 w_k1_INTC_s115
                                 + 0.774606 w_k1_KO_s115
                                 + 0.700143 w_k1_MAT_s115
                                 + 0.904311 w_k1_MSFT_s115
                                 + 0.892748 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.05798 w_k1_ADM_s115
                                 + 1.08924 w_k1_AXP_s115
                                 + 1.26173 w_k1_DIS_s115
                                 + 0.93725 w_k1_FDX_s115
                                 + 1.76356 w_k1_IBM_s115
                                 + 1.05341 w_k1_INTC_s115
                                 + 1.57236 w_k1_KO_s115 + 1.77805 w_k1_MAT_s115
                                 + 1.009 w_k1_MSFT_s115 + 1.19478 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.26734 w_k1_ADM_s115
                                 + 1.05835 w_k1_AXP_s115
                                 + 1.00265 w_k1_DIS_s115
                                 + 0.967867 w_k1_FDX_s115
                                 + 1.36256 w_k1_IBM_s115
                                 + 0.997818 w_k1_INTC_s115
                                 + 1.29124 w_k1_KO_s115 + 1.79114 w_k1_MAT_s115
                                 + 0.982418 w_k1_MSFT_s115
                                 + 1.17493 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.908745 w_k1_ADM_s115
                                 + 0.859807 w_k1_AXP_s115
                                 + 0.804233 w_k1_DIS_s115
                                 + 0.81702 w_k1_FDX_s115
                                 + 1.07178 w_k1_IBM_s115
                                 + 0.757227 w_k1_INTC_s115
                                 + 0.94361 w_k1_KO_s115
                                 + 0.924616 w_k1_MAT_s115
                                 + 0.815156 w_k1_MSFT_s115
                                 + 0.824604 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.24097 w_k1_ADM_s115
                                 + 0.949923 w_k1_AXP_s115
                                 + 1.09767 w_k1_DIS_s115
                                 + 1.14218 w_k1_FDX_s115
                                 + 0.658877 w_k1_IBM_s115
                                 + 0.800192 w_k1_INTC_s115
                                 + 0.859962 w_k1_KO_s115
                                 + 1.01638 w_k1_MAT_s115
                                 + 1.12431 w_k1_MSFT_s115
                                 + 1.02958 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.15595 w_k1_ADM_s115
                                 + 1.19104 w_k1_AXP_s115
                                 + 1.46422 w_k1_DIS_s115
                                 + 1.26494 w_k1_FDX_s115 + 1.1123 w_k1_IBM_s115
                                 + 1.04539 w_k1_INTC_s115
                                 + 1.33492 w_k1_KO_s115 + 1.32706 w_k1_MAT_s115
                                 + 1.29768 w_k1_MSFT_s115
                                 + 1.17196 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.0774 w_k1_ADM_s116
                                 + 1.79722 w_k1_AXP_s116
                                 + 0.981492 w_k1_DIS_s116
                                 + 1.17679 w_k1_FDX_s116
                                 + 1.10985 w_k1_IBM_s116
                                 + 1.0843 w_k1_INTC_s116
                                 + 0.989549 w_k1_KO_s116
                                 + 1.51597 w_k1_MAT_s116
                                 + 0.923344 w_k1_MSFT_s116
                                 + 1.14193 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.771796 w_k1_ADM_s116
                                 + 0.858844 w_k1_AXP_s116
                                 + 0.887389 w_k1_DIS_s116
                                 + 0.876577 w_k1_FDX_s116
                                 + 1.14577 w_k1_IBM_s116
                                 + 0.964286 w_k1_INTC_s116
                                 + 0.900811 w_k1_KO_s116
                                 + 1.30508 w_k1_MAT_s116
                                 + 0.871283 w_k1_MSFT_s116
                                 + 0.983161 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.30414 w_k1_ADM_s116
                                 + 1.26325 w_k1_AXP_s116
                                 + 1.41347 w_k1_DIS_s116
                                 + 1.31046 w_k1_FDX_s116
                                 + 0.904837 w_k1_IBM_s116
                                 + 1.09039 w_k1_INTC_s116
                                 + 1.06655 w_k1_KO_s116 + 1.17064 w_k1_MAT_s116
                                 + 1.09078 w_k1_MSFT_s116
                                 + 1.27123 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.12532 w_k1_ADM_s116
                                 + 1.07325 w_k1_AXP_s116
                                 + 1.65596 w_k1_DIS_s116
                                 + 1.22557 w_k1_FDX_s116
                                 + 1.19825 w_k1_IBM_s116
                                 + 1.22746 w_k1_INTC_s116
                                 + 1.27739 w_k1_KO_s116 + 1.17175 w_k1_MAT_s116
                                 + 1.23762 w_k1_MSFT_s116
                                 + 1.28386 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.10451 w_k1_ADM_s116
                                 + 1.26696 w_k1_AXP_s116
                                 + 1.06151 w_k1_DIS_s116
                                 + 0.947497 w_k1_FDX_s116
                                 + 1.36787 w_k1_IBM_s116
                                 + 0.940611 w_k1_INTC_s116
                                 + 1.11617 w_k1_KO_s116 + 1.48661 w_k1_MAT_s116
                                 + 0.99684 w_k1_MSFT_s116
                                 + 1.15333 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.873899 w_k1_ADM_s116
                                 + 0.97304 w_k1_AXP_s116
                                 + 0.859204 w_k1_DIS_s116
                                 + 0.897072 w_k1_FDX_s116
                                 + 0.904881 w_k1_IBM_s116
                                 + 0.752024 w_k1_INTC_s116
                                 + 0.845716 w_k1_KO_s116
                                 + 1.24135 w_k1_MAT_s116
                                 + 0.801937 w_k1_MSFT_s116
                                 + 0.999933 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.904907 w_k1_ADM_s116
                                 + 0.771144 w_k1_AXP_s116
                                 + 0.795752 w_k1_DIS_s116
                                 + 0.875147 w_k1_FDX_s116
                                 + 0.647608 w_k1_IBM_s116
                                 + 0.65027 w_k1_INTC_s116
                                 + 0.744413 w_k1_KO_s116
                                 + 0.798587 w_k1_MAT_s116
                                 + 0.757452 w_k1_MSFT_s116
                                 + 0.859419 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.43737 w_k1_ADM_s116
                                 + 1.6692 w_k1_AXP_s116 + 1.73659 w_k1_DIS_s116
                                 + 1.34514 w_k1_FDX_s116
                                 + 1.53961 w_k1_IBM_s116
                                 + 1.17032 w_k1_INTC_s116 + 1.4778 w_k1_KO_s116
                                 + 1.48583 w_k1_MAT_s116
                                 + 1.26947 w_k1_MSFT_s116
                                 + 1.47333 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.965664 w_k1_ADM_s117
                                 + 0.988227 w_k1_AXP_s117
                                 + 0.965908 w_k1_DIS_s117
                                 + 0.858129 w_k1_FDX_s117
                                 + 1.53038 w_k1_IBM_s117
                                 + 0.747912 w_k1_INTC_s117
                                 + 1.14556 w_k1_KO_s117 + 1.42174 w_k1_MAT_s117
                                 + 0.789236 w_k1_MSFT_s117
                                 + 1.08076 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.16912 w_k1_ADM_s117
                                 + 1.30685 w_k1_AXP_s117
                                 + 0.959397 w_k1_DIS_s117
                                 + 0.997916 w_k1_FDX_s117
                                 + 1.16529 w_k1_IBM_s117
                                 + 0.997226 w_k1_INTC_s117
                                 + 1.29501 w_k1_KO_s117 + 1.43592 w_k1_MAT_s117
                                 + 0.991919 w_k1_MSFT_s117
                                 + 1.15015 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.00008 w_k1_ADM_s117
                                 + 0.995429 w_k1_AXP_s117
                                 + 1.0952 w_k1_DIS_s117 + 1.12923 w_k1_FDX_s117
                                 + 0.736696 w_k1_IBM_s117
                                 + 0.961237 w_k1_INTC_s117
                                 + 0.817556 w_k1_KO_s117
                                 + 1.04902 w_k1_MAT_s117
                                 + 1.12817 w_k1_MSFT_s117
                                 + 0.94583 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.36972 w_k1_ADM_s117
                                 + 1.49222 w_k1_AXP_s117
                                 + 1.58204 w_k1_DIS_s117
                                 + 1.37247 w_k1_FDX_s117
                                 + 0.838012 w_k1_IBM_s117
                                 + 1.07431 w_k1_INTC_s117
                                 + 1.02966 w_k1_KO_s117 + 1.02463 w_k1_MAT_s117
                                 + 1.17394 w_k1_MSFT_s117 + 1.074 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.958006 w_k1_ADM_s117
                                 + 0.743638 w_k1_AXP_s117
                                 + 0.624816 w_k1_DIS_s117
                                 + 0.787415 w_k1_FDX_s117
                                 + 0.80442 w_k1_IBM_s117
                                 + 0.733719 w_k1_INTC_s117
                                 + 0.703152 w_k1_KO_s117
                                 + 0.857449 w_k1_MAT_s117
                                 + 0.782259 w_k1_MSFT_s117
                                 + 0.827131 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.01779 w_k1_ADM_s117
                                 + 0.911442 w_k1_AXP_s117
                                 + 1.05047 w_k1_DIS_s117
                                 + 1.00964 w_k1_FDX_s117
                                 + 1.35818 w_k1_IBM_s117
                                 + 1.02531 w_k1_INTC_s117
                                 + 1.23157 w_k1_KO_s117 + 1.4291 w_k1_MAT_s117
                                 + 1.1081 w_k1_MSFT_s117 + 1.07299 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.15482 w_k1_ADM_s117
                                 + 0.954287 w_k1_AXP_s117
                                 + 1.38155 w_k1_DIS_s117
                                 + 1.14563 w_k1_FDX_s117
                                 + 0.988278 w_k1_IBM_s117
                                 + 1.02233 w_k1_INTC_s117
                                 + 1.11946 w_k1_KO_s117 + 1.11961 w_k1_MAT_s117
                                 + 1.2885 w_k1_MSFT_s117 + 1.10934 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.806567 w_k1_ADM_s117
                                 + 0.785593 w_k1_AXP_s117
                                 + 0.975146 w_k1_DIS_s117
                                 + 0.876452 w_k1_FDX_s117
                                 + 0.879274 w_k1_IBM_s117
                                 + 0.992361 w_k1_INTC_s117
                                 + 1.11846 w_k1_KO_s117 + 1.03366 w_k1_MAT_s117
                                 + 1.07474 w_k1_MSFT_s117
                                 + 0.999643 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0343744752149598 w_k1_ADM_s000
                                 + 0.0318698136661689 w_k1_ADM_s001
                                 + 0.0113446055356416 w_k1_ADM_s002
                                 + 0.0248645940541159 w_k1_ADM_s003
                                 + 0.0395302792982021 w_k1_ADM_s004
                                 + 0.0235051318457236 w_k1_ADM_s005
                                 + 0.0281882413772295 w_k1_ADM_s006
                                 + 0.0404401171283414 w_k1_ADM_s007
                                 + 0.0456238325879285 w_k1_ADM_s010
                                 + 0.0184482472480754 w_k1_ADM_s011
                                 + 0.0188980201817686 w_k1_ADM_s012
                                 + 0.0394764658509043 w_k1_ADM_s013
                                 + 0.034991491307814 w_k1_ADM_s014
                                 + 0.0406785321681739 w_k1_ADM_s015
                                 + 0.0207660957480464 w_k1_ADM_s016
                                 + 0.0125063134362787 w_k1_ADM_s017
                                 + 0.0258900171108494 w_k1_ADM_s100
                                 + 0.0636326595881929 w_k1_ADM_s101
                                 + 0.0521373637468296 w_k1_ADM_s102
                                 + 0.0167940253779885 w_k1_ADM_s103
                                 + 0.0465991086506885 w_k1_ADM_s104
                                 + 0.0143706553407371 w_k1_ADM_s105
                                 + 0.0338578623135275 w_k1_ADM_s106
                                 + 0.0672500276463768 w_k1_ADM_s107
                                 + 0.0419466093453666 w_k1_ADM_s110
                                 + 0.0426448385263112 w_k1_ADM_s111
                                 + 0.0414211609424369 w_k1_ADM_s112
                                 + 0.0108514422769886 w_k1_ADM_s113
                                 + 0.0127256241383118 w_k1_ADM_s114
                                 + 0.0224246790943842 w_k1_ADM_s115
                                 + 0.0114924178551355 w_k1_ADM_s116
                                 + 0.0235150265043972 w_k1_ADM_s117
                                 + 0.035159600512471 w_k1_AXP_s000
                                 + 0.0373405594054316 w_k1_AXP_s001
                                 + 0.0137966561945703 w_k1_AXP_s002
                                 + 0.0231008864439186 w_k1_AXP_s003
                                 + 0.0424893971939945 w_k1_AXP_s004
                                 + 0.025756930823378 w_k1_AXP_s005
                                 + 0.0301151119055365 w_k1_AXP_s006
                                 + 0.0387639811429586 w_k1_AXP_s007
                                 + 0.0533931970721322 w_k1_AXP_s010
                                 + 0.0218268653791639 w_k1_AXP_s011
                                 + 0.0209449326047997 w_k1_AXP_s012
                                 + 0.0396421556579113 w_k1_AXP_s013
                                 + 0.034505705370484 w_k1_AXP_s014
                                 + 0.0373007880322452 w_k1_AXP_s015
                                 + 0.0193006565591696 w_k1_AXP_s016
                                 + 0.0135042412507339 w_k1_AXP_s017
                                 + 0.0228110816093392 w_k1_AXP_s100
                                 + 0.0696099789257589 w_k1_AXP_s101
                                 + 0.0572984532435924 w_k1_AXP_s102
                                 + 0.0161557073822941 w_k1_AXP_s103
                                 + 0.04157027426843 w_k1_AXP_s104
                                 + 0.0153790548414574 w_k1_AXP_s105
                                 + 0.0429523061520206 w_k1_AXP_s106
                                 + 0.0688725437600677 w_k1_AXP_s107
                                 + 0.0385767064948412 w_k1_AXP_s110
                                 + 0.0467045400523793 w_k1_AXP_s111
                                 + 0.0428085175453011 w_k1_AXP_s112
                                 + 0.0119901066631964 w_k1_AXP_s113
                                 + 0.0120352684690476 w_k1_AXP_s114
                                 + 0.020450018439934 w_k1_AXP_s115
                                 + 0.0126485073704761 w_k1_AXP_s116
                                 + 0.0222986422813232 w_k1_AXP_s117
                                 + 0.0320572674736845 w_k1_DIS_s000
                                 + 0.0365649508640559 w_k1_DIS_s001
                                 + 0.0126388897372436 w_k1_DIS_s002
                                 + 0.0242119031545858 w_k1_DIS_s003
                                 + 0.0402028035497443 w_k1_DIS_s004
                                 + 0.0260366770134738 w_k1_DIS_s005
                                 + 0.0274091491787065 w_k1_DIS_s006
                                 + 0.0398780175036346 w_k1_DIS_s007
                                 + 0.0554589610469851 w_k1_DIS_s010
                                 + 0.0189823730641186 w_k1_DIS_s011
                                 + 0.0199882364727133 w_k1_DIS_s012
                                 + 0.0387516272296302 w_k1_DIS_s013
                                 + 0.0364359344998265 w_k1_DIS_s014
                                 + 0.0411239484555073 w_k1_DIS_s015
                                 + 0.0194850060502979 w_k1_DIS_s016
                                 + 0.0145418270073124 w_k1_DIS_s017
                                 + 0.0227902046383614 w_k1_DIS_s100
                                 + 0.0700828125530342 w_k1_DIS_s101
                                 + 0.0589530885095242 w_k1_DIS_s102
                                 + 0.0196475769468563 w_k1_DIS_s103
                                 + 0.0391615444939557 w_k1_DIS_s104
                                 + 0.0148709039183945 w_k1_DIS_s105
                                 + 0.0396939397273515 w_k1_DIS_s106
                                 + 0.0717804589132408 w_k1_DIS_s107
                                 + 0.0390337563370764 w_k1_DIS_s110
                                 + 0.0466060658844993 w_k1_DIS_s111
                                 + 0.0416609319284268 w_k1_DIS_s112
                                 + 0.0111427556039217 w_k1_DIS_s113
                                 + 0.0119975205438347 w_k1_DIS_s114
                                 + 0.0224837590026208 w_k1_DIS_s115
                                 + 0.0121264561399737 w_k1_DIS_s116
                                 + 0.0236923974745673 w_k1_DIS_s117
                                 + 0.0323445384673923 w_k1_FDX_s000
                                 + 0.0363177156778477 w_k1_FDX_s001
                                 + 0.0126397078285707 w_k1_FDX_s002
                                 + 0.0232782332824966 w_k1_FDX_s003
                                 + 0.0411051088848012 w_k1_FDX_s004
                                 + 0.0245971747056755 w_k1_FDX_s005
                                 + 0.0270426236391042 w_k1_FDX_s006
                                 + 0.0395392986449567 w_k1_FDX_s007
                                 + 0.0539163373445492 w_k1_FDX_s010
                                 + 0.0194827651999597 w_k1_FDX_s011
                                 + 0.0200394474932373 w_k1_FDX_s012
                                 + 0.0390465466250352 w_k1_FDX_s013
                                 + 0.0342291033491379 w_k1_FDX_s014
                                 + 0.0391342513955945 w_k1_FDX_s015
                                 + 0.0182313661821558 w_k1_FDX_s016
                                 + 0.0143829223733878 w_k1_FDX_s017
                                 + 0.0231493993076765 w_k1_FDX_s100
                                 + 0.0694816644443502 w_k1_FDX_s101
                                 + 0.0559097417691304 w_k1_FDX_s102
                                 + 0.0172026989123235 w_k1_FDX_s103
                                 + 0.0409840743619541 w_k1_FDX_s104
                                 + 0.0145282319400643 w_k1_FDX_s105
                                 + 0.042864926404762 w_k1_FDX_s106
                                 + 0.0704510751060196 w_k1_FDX_s107
                                 + 0.0382993972693718 w_k1_FDX_s110
                                 + 0.0460343141348199 w_k1_FDX_s111
                                 + 0.0437895928094877 w_k1_FDX_s112
                                 + 0.0114370786121587 w_k1_FDX_s113
                                 + 0.0122400650667372 w_k1_FDX_s114
                                 + 0.021398257457067 w_k1_FDX_s115
                                 + 0.0115927241580192 w_k1_FDX_s116
                                 + 0.0226195758361952 w_k1_FDX_s117
                                 + 0.033923508799575 w_k1_IBM_s000
                                 + 0.0368120200995817 w_k1_IBM_s001
                                 + 0.012648620205461 w_k1_IBM_s002
                                 + 0.0216218233224458 w_k1_IBM_s003
                                 + 0.0420004467656111 w_k1_IBM_s004
                                 + 0.0250343237950819 w_k1_IBM_s005
                                 + 0.0294016226841812 w_k1_IBM_s006
                                 + 0.039940237994395 w_k1_IBM_s007
                                 + 0.0559210777244495 w_k1_IBM_s010
                                 + 0.0198388615713115 w_k1_IBM_s011
                                 + 0.02022240918502 w_k1_IBM_s012
                                 + 0.0409956140133389 w_k1_IBM_s013
                                 + 0.0358757769223373 w_k1_IBM_s014
                                 + 0.0400128475178765 w_k1_IBM_s015
                                 + 0.0183644654813201 w_k1_IBM_s016
                                 + 0.0142129784100205 w_k1_IBM_s017
                                 + 0.0242526959113503 w_k1_IBM_s100
                                 + 0.0684942666259467 w_k1_IBM_s101
                                 + 0.057835551033246 w_k1_IBM_s102
                                 + 0.018271107072563 w_k1_IBM_s103
                                 + 0.0403064963303009 w_k1_IBM_s104
                                 + 0.0151798798409772 w_k1_IBM_s105
                                 + 0.0409808378008331 w_k1_IBM_s106
                                 + 0.0730998915917035 w_k1_IBM_s107
                                 + 0.0385015383201942 w_k1_IBM_s110
                                 + 0.0484907977212863 w_k1_IBM_s111
                                 + 0.0437067025403795 w_k1_IBM_s112
                                 + 0.0112294678212391 w_k1_IBM_s113
                                 + 0.0125563417477017 w_k1_IBM_s114
                                 + 0.0219873102652843 w_k1_IBM_s115
                                 + 0.0117435835552996 w_k1_IBM_s116
                                 + 0.0228260381006146 w_k1_IBM_s117
                                 + 0.0288065334616384 w_k1_INTC_s000
                                 + 0.0397807401096915 w_k1_INTC_s001
                                 + 0.0137849857132104 w_k1_INTC_s002
                                 + 0.0248803000267136 w_k1_INTC_s003
                                 + 0.0395403591356615 w_k1_INTC_s004
                                 + 0.0238323287672553 w_k1_INTC_s005
                                 + 0.0306357348935725 w_k1_INTC_s006
                                 + 0.0340820998919911 w_k1_INTC_s007
                                 + 0.0629159197900795 w_k1_INTC_s010
                                 + 0.0184286848789032 w_k1_INTC_s011
                                 + 0.0179330748508736 w_k1_INTC_s012
                                 + 0.0349029051931231 w_k1_INTC_s013
                                 + 0.0309293954828525 w_k1_INTC_s014
                                 + 0.0411940211353245 w_k1_INTC_s015
                                 + 0.017156169797602 w_k1_INTC_s016
                                 + 0.0151713012784788 w_k1_INTC_s017
                                 + 0.0249012055977672 w_k1_INTC_s100
                                 + 0.0707236136642909 w_k1_INTC_s101
                                 + 0.0516328261588077 w_k1_INTC_s102
                                 + 0.0170673740316939 w_k1_INTC_s103
                                 + 0.0374587008043367 w_k1_INTC_s104
                                 + 0.0173642477822432 w_k1_INTC_s105
                                 + 0.0419020880226618 w_k1_INTC_s106
                                 + 0.0765497929802857 w_k1_INTC_s107
                                 + 0.0369705816606327 w_k1_INTC_s110
                                 + 0.0468591454311488 w_k1_INTC_s111
                                 + 0.0420089200315075 w_k1_INTC_s112
                                 + 0.0103908014908173 w_k1_INTC_s113
                                 + 0.0112259247331905 w_k1_INTC_s114
                                 + 0.0201669044836053 w_k1_INTC_s115
                                 + 0.0103904019303561 w_k1_INTC_s116
                                 + 0.0210467700339291 w_k1_INTC_s117
                                 + 0.0332461672303841 w_k1_KO_s000
                                 + 0.0378094029379331 w_k1_KO_s001
                                 + 0.0123292474120692 w_k1_KO_s002
                                 + 0.021070363460067 w_k1_KO_s003
                                 + 0.040412209318984 w_k1_KO_s004
                                 + 0.0224847485429456 w_k1_KO_s005
                                 + 0.0289292954087421 w_k1_KO_s006
                                 + 0.0384201432320308 w_k1_KO_s007
                                 + 0.0602697805351205 w_k1_KO_s010
                                 + 0.0193605911336522 w_k1_KO_s011
                                 + 0.0208717406440965 w_k1_KO_s012
                                 + 0.0401999181467301 w_k1_KO_s013
                                 + 0.0382035402878543 w_k1_KO_s014
                                 + 0.0391372266615104 w_k1_KO_s015
                                 + 0.0194589457469195 w_k1_KO_s016
                                 + 0.013456841153952 w_k1_KO_s017
                                 + 0.0214507983707021 w_k1_KO_s100
                                 + 0.0668709300638495 w_k1_KO_s101
                                 + 0.0532262269169763 w_k1_KO_s102
                                 + 0.0174050843980675 w_k1_KO_s103
                                 + 0.0404920051382399 w_k1_KO_s104
                                 + 0.0149363551532409 w_k1_KO_s105
                                 + 0.0403075382288115 w_k1_KO_s106
                                 + 0.0672122369603891 w_k1_KO_s107
                                 + 0.0363492147213359 w_k1_KO_s110
                                 + 0.0515278995866445 w_k1_KO_s111
                                 + 0.0385036420780388 w_k1_KO_s112
                                 + 0.0121839834344977 w_k1_KO_s113
                                 + 0.0118398911788703 w_k1_KO_s114
                                 + 0.0226411831267442 w_k1_KO_s115
                                 + 0.0111322198411421 w_k1_KO_s116
                                 + 0.0232681177361529 w_k1_KO_s117
                                 + 0.0342125551866458 w_k1_MAT_s000
                                 + 0.0333712194957217 w_k1_MAT_s001
                                 + 0.0131417009712844 w_k1_MAT_s002
                                 + 0.0185841153816499 w_k1_MAT_s003
                                 + 0.0573720042077018 w_k1_MAT_s004
                                 + 0.0212219669925318 w_k1_MAT_s005
                                 + 0.0289320121079016 w_k1_MAT_s006
                                 + 0.0412673308735396 w_k1_MAT_s007
                                 + 0.0451991162095038 w_k1_MAT_s010
                                 + 0.0236107558446284 w_k1_MAT_s011
                                 + 0.0262480093903154 w_k1_MAT_s012
                                 + 0.0409617647107315 w_k1_MAT_s013
                                 + 0.0338076535862833 w_k1_MAT_s014
                                 + 0.04563580700255 w_k1_MAT_s015
                                 + 0.0183218698462927 w_k1_MAT_s016
                                 + 0.0126294566086304 w_k1_MAT_s017
                                 + 0.0197293048460742 w_k1_MAT_s100
                                 + 0.0619695981011424 w_k1_MAT_s101
                                 + 0.0584172594924308 w_k1_MAT_s102
                                 + 0.0165138969773063 w_k1_MAT_s103
                                 + 0.0387553125934546 w_k1_MAT_s104
                                 + 0.014819725541172 w_k1_MAT_s105
                                 + 0.04247065221677 w_k1_MAT_s106
                                 + 0.0775794144022307 w_k1_MAT_s107
                                 + 0.032845868282036 w_k1_MAT_s110
                                 + 0.0587077300791118 w_k1_MAT_s111
                                 + 0.0408633397904755 w_k1_MAT_s112
                                 + 0.0113008102933154 w_k1_MAT_s113
                                 + 0.012243992259066 w_k1_MAT_s114
                                 + 0.0245843286324582 w_k1_MAT_s115
                                 + 0.0140960787005993 w_k1_MAT_s116
                                 + 0.0256035413956711 w_k1_MAT_s117
                                 + 0.0297057856616116 w_k1_MSFT_s000
                                 + 0.0412598569389057 w_k1_MSFT_s001
                                 + 0.0119496553072988 w_k1_MSFT_s002
                                 + 0.0231793627135426 w_k1_MSFT_s003
                                 + 0.0399906561493441 w_k1_MSFT_s004
                                 + 0.0243421182242405 w_k1_MSFT_s005
                                 + 0.0277433114042842 w_k1_MSFT_s006
                                 + 0.0384445639089857 w_k1_MSFT_s007
                                 + 0.0547360105181917 w_k1_MSFT_s010
                                 + 0.0172173278703925 w_k1_MSFT_s011
                                 + 0.0223371068660894 w_k1_MSFT_s012
                                 + 0.0379328654107245 w_k1_MSFT_s013
                                 + 0.033909674184415 w_k1_MSFT_s014
                                 + 0.0443210933798181 w_k1_MSFT_s015
                                 + 0.0194404959647103 w_k1_MSFT_s016
                                 + 0.0137232586767019 w_k1_MSFT_s017
                                 + 0.0235189487804813 w_k1_MSFT_s100
                                 + 0.0646753509621515 w_k1_MSFT_s101
                                 + 0.053571943488039 w_k1_MSFT_s102
                                 + 0.0192779223521483 w_k1_MSFT_s103
                                 + 0.0391720796568077 w_k1_MSFT_s104
                                 + 0.0143657936773476 w_k1_MSFT_s105
                                 + 0.0364775310212861 w_k1_MSFT_s106
                                 + 0.0636335747707725 w_k1_MSFT_s107
                                 + 0.0390981623538319 w_k1_MSFT_s110
                                 + 0.0464259977113824 w_k1_MSFT_s111
                                 + 0.0403841119138684 w_k1_MSFT_s112
                                 + 0.0115047913586317 w_k1_MSFT_s113
                                 + 0.0121110780918856 w_k1_MSFT_s114
                                 + 0.0217564775341996 w_k1_MSFT_s115
                                 + 0.0105927437305192 w_k1_MSFT_s116
                                 + 0.0231799482562903 w_k1_MSFT_s117
                                 + 0.03198449669976 w_k1_PG_s000
                                 + 0.0366917064714432 w_k1_PG_s001
                                 + 0.0127191883232061 w_k1_PG_s002
                                 + 0.0230628087187729 w_k1_PG_s003
                                 + 0.0405301246783058 w_k1_PG_s004
                                 + 0.0242138796402756 w_k1_PG_s005
                                 + 0.0283730573349237 w_k1_PG_s006
                                 + 0.0365986938019156 w_k1_PG_s007
                                 + 0.0528004010269912 w_k1_PG_s010
                                 + 0.0198270620556671 w_k1_PG_s011
                                 + 0.0191341975213558 w_k1_PG_s012
                                 + 0.0418125084086922 w_k1_PG_s013
                                 + 0.0370638814650209 w_k1_PG_s014
                                 + 0.0412101784959808 w_k1_PG_s015
                                 + 0.0176551226233699 w_k1_PG_s016
                                 + 0.0136723498365743 w_k1_PG_s017
                                 + 0.0224919313425719 w_k1_PG_s100
                                 + 0.0667549069269435 w_k1_PG_s101
                                 + 0.0556009039442448 w_k1_PG_s102
                                 + 0.0170037527692821 w_k1_PG_s103
                                 + 0.0414306256478642 w_k1_PG_s104
                                 + 0.0141651552711466 w_k1_PG_s105
                                 + 0.0396675173937054 w_k1_PG_s106
                                 + 0.0663219573204991 w_k1_PG_s107
                                 + 0.0343204027760736 w_k1_PG_s110
                                 + 0.0474554964004581 w_k1_PG_s111
                                 + 0.0406801079175846 w_k1_PG_s112
                                 + 0.0123881625037429 w_k1_PG_s113
                                 + 0.0124428930401079 w_k1_PG_s114
                                 + 0.021616136267946 w_k1_PG_s115
                                 + 0.012387792845487 w_k1_PG_s116
                                 + 0.0227872370317006 w_k1_PG_s117 >= 1.1
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
