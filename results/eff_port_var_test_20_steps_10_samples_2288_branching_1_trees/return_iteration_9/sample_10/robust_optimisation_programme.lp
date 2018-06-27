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
 w_k1_ADM_s0_return_constraint:  1.18121 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.937981 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.18299 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.868177 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.09366 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.939191 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.01678 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.74942 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.13797 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.859845 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.0888 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.03097 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.947587 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.733508 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.924092 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.1627 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.16176 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.46093 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.10937 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.21396 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.974943 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.883053 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.54768 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.3103 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.8295 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.14629 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.0658 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.77625 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.19783 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.757919 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.860339 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.919913 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.837895 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.06933 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.675218 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.952738 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.32525 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.99309 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.22976 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.960485 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.27181 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.853036 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.16877 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.765161 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.03199 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.738819 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.66585 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.0373 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.40948 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.06203 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.961448 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.715305 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.680625 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.16976 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.21796 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.92851 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.16061 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.19491 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.08149 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.938347 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.69508 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.52575 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.699765 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.03587 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.960001 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.800278 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.06909 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.58532 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.963021 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.897363 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.708647 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.20186 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.731517 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.37804 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.39397 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.963547 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28875 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.912564 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.42217 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.831584 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.29932 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.759915 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.09234 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.788119 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.88466 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.922625 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.39637 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.11178 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.839993 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.600178 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.701582 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.39472 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.48321 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.77094 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.918608 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.03665 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.876175 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.930175 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.75463 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.61981 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.735233 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.01261 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.858716 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.732936 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.12286 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.599469 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.890055 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.978751 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.653561 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.34626 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.567357 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.2694 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.54999 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.921365 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.2228 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.919516 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.22154 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.883845 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.12566 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.855992 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.02878 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.860428 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.47723 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.845598 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.08144 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.15289 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.922689 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.745573 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.822933 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.23487 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.12291 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.4748 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.03835 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.12753 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.891872 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.860881 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.59581 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.28728 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.874346 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.17508 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.830614 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.848812 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.12212 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.710219 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.995662 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.03267 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.832274 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.1488 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.696622 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.05769 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.44279 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.9084 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.997599 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.02845 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.32781 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.890105 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.24097 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.792668 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.04506 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.952401 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.14406 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.50546 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.53438 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.643824 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.810332 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.764737 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.724842 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.00405 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.66395 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.40736 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.802072 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.14164 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.6318 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.21833 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.01505 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.60307 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.573537 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.740393 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.45311 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.921 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.0834 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.751179 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.886695 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.664782 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.623389 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.0481 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.923503 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.84603 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.898448 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.31613 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.14221 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.936626 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.07186 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.911538 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13779 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.931022 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.26289 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.06522 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.41459 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.952234 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.33158 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.14954 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.894534 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.914301 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.918689 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.21153 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.09339 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.27784 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.10825 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.03135 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.932817 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.05598 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.10162 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.16016 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.942144 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.976924 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.773259 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.83669 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.951222 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.841069 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.927702 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.932443 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.843025 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.17829 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.769056 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.24538 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.22714 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.946749 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13072 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.999634 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.21427 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.80989 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.14711 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.823418 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.01339 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.07055 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.40011 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.06338 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.41104 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.943868 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.961114 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.82146 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.831765 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.1921 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.54085 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.66768 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.866968 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.11746 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.26621 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.11739 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.25714 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.49505 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.770845 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.906268 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.12487 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.863932 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.07807 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.750719 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.861452 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.768848 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.700691 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.10901 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.755904 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.45234 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.14405 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.99355 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.06345 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.0671 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.56268 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.834721 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.13631 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.703778 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.764009 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.801165 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.39296 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.2599 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.4279 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.616282 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.750398 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.645907 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.638256 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.962639 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.41142 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.92157 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.843334 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.1912 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.66956 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.05417 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.28061 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.79383 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.632852 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.945933 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.59338 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.14936 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.14746 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.661153 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.869921 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.701793 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.495672 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.96406 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.863664 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.94094 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.275 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.21935 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14552 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.9594 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.20256 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.924222 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.109 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.895563 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.15111 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.955843 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.41007 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.90326 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.28617 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.15526 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.855648 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.811988 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.780495 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.23322 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.26879 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.45749 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.959248 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.07954 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.881302 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.983734 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.44127 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.3641 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.840182 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.0616 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.845528 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.825391 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.1197 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.768159 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.860668 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.949879 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.826938 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.13997 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.655642 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.01535 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.3258 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.88606 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11016 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.00336 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.18653 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.861095 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.09199 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.80712 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.06599 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.936382 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.23526 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.980302 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.20596 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.974163 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.90294 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.792387 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.87036 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.20189 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.29247 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.51527 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.01207 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11849 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.09745 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.992309 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.18862 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.15882 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.86716 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.976872 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.06027 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.842123 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.04836 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.784172 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.00162 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.938 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.877074 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.18346 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.874546 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.19127 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.17664 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.04561 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.313001192826791 z_k1_s0000
                                 - 0.359000574091185 z_k1_s0001
                                 - 0.294999916765119 z_k1_s0002
                                 - 0.172000097686944 z_k1_s0003
                                 - 0.201000145206289 z_k1_s0004
                                 - 0.237001054127802 z_k1_s0005
                                 - 0.301999956566372 z_k1_s0006
                                 - 0.175000173430403 z_k1_s0007
                                 - 0.180999936109832 z_k1_s0010
                                 - 0.116000419971347 z_k1_s0011
                                 - 0.0430000605996399 z_k1_s0012
                                 - 0.0410000265414897 z_k1_s0013
                                 - 0.0969997782686401 z_k1_s0014
                                 - 0.11500001409193 z_k1_s0015
                                 - 0.0770001446877591 z_k1_s0016
                                 - 0.0370000291252513 z_k1_s0017
                                 - 0.0700000658225937 z_k1_s0020
                                 - 0.209999996067574 z_k1_s0021
                                 - 0.151999757391521 z_k1_s0022
                                 - 0.1250000456117 z_k1_s0023
                                 - 0.0830000463981857 z_k1_s0024
                                 - 0.0960001276738813 z_k1_s0025
                                 - 0.130999757784764 z_k1_s0026
                                 - 0.140000333045395 z_k1_s0027
                                 - 0.169000227343714 z_k1_s0030
                                 - 0.0950000706736215 z_k1_s0031
                                 - 0.0580000950386899 z_k1_s0032
                                 - 0.111000337705253 z_k1_s0033
                                 - 0.119999948734944 z_k1_s0034
                                 - 0.132000457109011 z_k1_s0035
                                 - 0.215000635609252 z_k1_s0036
                                 - 0.126999988536197 z_k1_s0037
                                 - 0.202000310394662 z_k1_s0040
                                 - 0.402001155846691 z_k1_s0041
                                 - 0.149000489866275 z_k1_s0042
                                 - 0.2900014729563 z_k1_s0043
                                 - 0.184999917365124 z_k1_s0044
                                 - 0.122000466315566 z_k1_s0045
                                 - 0.135000447154691 z_k1_s0046
                                 - 0.162000635405972 z_k1_s0047
                                 - 0.174999821256847 z_k1_s0050
                                 - 0.261001162791734 z_k1_s0051
                                 - 0.207001131318014 z_k1_s0052
                                 - 0.148000260320926 z_k1_s0053
                                 - 0.226001198540365 z_k1_s0054
                                 - 0.137000293213397 z_k1_s0055
                                 - 0.131000256027359 z_k1_s0056
                                 - 0.231000976861542 z_k1_s0057
                                 - 0.388001376676144 z_k1_s0060
                                 - 0.138000449642404 z_k1_s0061
                                 - 0.171000548730775 z_k1_s0062
                                 - 0.344002020561069 z_k1_s0063
                                 - 0.206001570918665 z_k1_s0064
                                 - 0.444001615371811 z_k1_s0065
                                 - 0.0960002706206531 z_k1_s0066
                                 - 0.153000582864704 z_k1_s0067
                                 - 0.287000707717617 z_k1_s0070
                                 - 0.18300017539329 z_k1_s0071
                                 - 0.495000437965435 z_k1_s0072
                                 - 0.124000087247893 z_k1_s0073
                                 - 0.142000042519325 z_k1_s0074
                                 - 0.439000082898447 z_k1_s0075
                                 - 0.246999818669371 z_k1_s0076
                                 - 0.30700115224174 z_k1_s0077
                                 - 0.267999799136921 z_k1_s0100
                                 - 0.885999700518143 z_k1_s0101
                                 - 0.960997569385874 z_k1_s0102
                                 - 0.433999690235155 z_k1_s0103
                                 - 0.522999829083506 z_k1_s0104
                                 - 0.514999571351195 z_k1_s0105
                                 - 0.504999509035673 z_k1_s0106
                                 - 1.10500116916801 z_k1_s0107
                                 - 0.732999394083305 z_k1_s0110
                                 - 0.179999666178589 z_k1_s0111
                                 - 0.233999466672394 z_k1_s0112
                                 - 0.360000325954899 z_k1_s0113
                                 - 0.36599917219014 z_k1_s0114
                                 - 0.201999658498119 z_k1_s0115
                                 - 0.190999538138639 z_k1_s0116
                                 - 0.217999562585256 z_k1_s0117
                                 - 0.169999800861334 z_k1_s0120
                                 - 0.285999404908008 z_k1_s0121
                                 - 0.182999702039003 z_k1_s0122
                                 - 0.262999701209001 z_k1_s0123
                                 - 0.217000293810987 z_k1_s0124
                                 - 0.191000491455649 z_k1_s0125
                                 - 0.211000096498326 z_k1_s0126
                                 - 0.0579999599232662 z_k1_s0127
                                 - 0.085999993402792 z_k1_s0130
                                 - 0.142999525563095 z_k1_s0131
                                 - 0.0829998946260172 z_k1_s0132
                                 - 0.161999410749834 z_k1_s0133
                                 - 0.107999977366307 z_k1_s0134
                                 - 0.113999928120023 z_k1_s0135
                                 - 0.326000368149853 z_k1_s0136
                                 - 0.211999823030247 z_k1_s0137
                                 - 0.712001861413329 z_k1_s0140
                                 - 0.259000270013156 z_k1_s0141
                                 - 0.771999772975593 z_k1_s0142
                                 - 0.44700099284371 z_k1_s0143
                                 - 0.488000776844944 z_k1_s0144
                                 - 0.260000235205861 z_k1_s0145
                                 - 0.198999936250184 z_k1_s0146
                                 - 0.900000969443409 z_k1_s0147
                                 - 0.142999497023621 z_k1_s0150
                                 - 0.280998829518279 z_k1_s0151
                                 - 0.21599933903399 z_k1_s0152
                                 - 0.212999372736224 z_k1_s0153
                                 - 0.167999436871025 z_k1_s0154
                                 - 0.31299802862908 z_k1_s0155
                                 - 0.540998115651546 z_k1_s0156
                                 - 0.331998035880953 z_k1_s0157
                                 - 0.287000510964015 z_k1_s0160
                                 - 0.0979999090048634 z_k1_s0161
                                 - 0.465999670551667 z_k1_s0162
                                 - 0.125000016899022 z_k1_s0163
                                 - 0.227999957239837 z_k1_s0164
                                 - 0.123999887991867 z_k1_s0165
                                 - 0.109999922891166 z_k1_s0166
                                 - 0.0949999821832654 z_k1_s0167
                                 - 0.293999132968886 z_k1_s0170
                                 - 0.560997697290121 z_k1_s0171
                                 - 0.222999049430615 z_k1_s0172
                                 - 0.456997859934348 z_k1_s0173
                                 - 0.149999394368442 z_k1_s0174
                                 - 0.275998753870452 z_k1_s0175
                                 - 0.152999457551514 z_k1_s0176
                                 - 0.238997974612561 z_k1_s0177
                                 - 0.112999975134541 z_k1_s1000
                                 - 0.264999863032301 z_k1_s1001
                                 - 0.312999929694791 z_k1_s1002
                                 - 0.12199996743375 z_k1_s1003
                                 - 0.195999868205011 z_k1_s1004
                                 - 0.33600046663743 z_k1_s1005
                                 - 0.139000132443437 z_k1_s1006
                                 - 0.132000102521816 z_k1_s1007
                                 - 0.682998601562112 z_k1_s1010
                                 - 0.600997749477146 z_k1_s1011
                                 - 0.372999380095781 z_k1_s1012
                                 - 0.43999897873273 z_k1_s1013
                                 - 0.485999245658764 z_k1_s1014
                                 - 0.951999127783324 z_k1_s1015
                                 - 0.215999616659524 z_k1_s1016
                                 - 0.576000708556094 z_k1_s1017
                                 - 0.248000220043516 z_k1_s1020
                                 - 0.339000447395149 z_k1_s1021
                                 - 0.292000321470669 z_k1_s1022
                                 - 0.323000445676228 z_k1_s1023
                                 - 0.835001661083135 z_k1_s1024
                                 - 0.897001909494253 z_k1_s1025
                                 - 0.460002466921963 z_k1_s1026
                                 - 0.474000776173933 z_k1_s1027
                                 - 1.02000350627711 z_k1_s1030
                                 - 0.409000373287867 z_k1_s1031
                                 - 1.54400359740262 z_k1_s1032
                                 - 0.708001249826838 z_k1_s1033
                                 - 0.741002843192689 z_k1_s1034
                                 - 0.635000571352795 z_k1_s1035
                                 - 0.411999969220754 z_k1_s1036
                                 - 1.23999847437217 z_k1_s1037
                                 - 1.07100158389096 z_k1_s1040
                                 - 1.58700078787084 z_k1_s1041
                                 - 0.400000268361827 z_k1_s1042
                                 - 0.3870005625858 z_k1_s1043
                                 - 1.0019995329385 z_k1_s1044
                                 - 0.321000138536807 z_k1_s1045
                                 - 0.492000103962874 z_k1_s1046
                                 - 0.53800031166362 z_k1_s1047
                                 - 0.853996656777998 z_k1_s1050
                                 - 0.821000100168934 z_k1_s1051
                                 - 0.860000051061369 z_k1_s1052
                                 - 0.836998095616518 z_k1_s1053
                                 - 0.880000309364535 z_k1_s1054
                                 - 0.569996305665472 z_k1_s1055
                                 - 0.431998945763527 z_k1_s1056
                                 - 0.273999116363469 z_k1_s1057
                                 - 0.54899950643421 z_k1_s1060
                                 - 0.532999568549192 z_k1_s1061
                                 - 0.818998206281654 z_k1_s1062
                                 - 1.23899573588926 z_k1_s1063
                                 - 0.974999784333284 z_k1_s1064
                                 - 0.864002020985324 z_k1_s1065
                                 - 0.631999226206447 z_k1_s1066
                                 - 1.10799553056267 z_k1_s1067
                                 - 0.48900065888041 z_k1_s1070
                                 - 0.195999955759141 z_k1_s1071
                                 - 0.461999325653537 z_k1_s1072
                                 - 0.205000040668002 z_k1_s1073
                                 - 0.16399998939439 z_k1_s1074
                                 - 0.22700058375642 z_k1_s1075
                                 - 0.30900111770376 z_k1_s1076
                                 - 0.104999984003113 z_k1_s1077
                                 - 0.559999922390208 z_k1_s1100
                                 - 1.21899980543158 z_k1_s1101
                                 - 0.604998164870387 z_k1_s1102
                                 - 0.750001006628225 z_k1_s1103
                                 - 0.256999477440458 z_k1_s1104
                                 - 0.605998019503425 z_k1_s1105
                                 - 0.524000371607373 z_k1_s1106
                                 - 0.262999562037379 z_k1_s1107
                                 - 0.171999757602951 z_k1_s1110
                                 - 0.161999769689991 z_k1_s1111
                                 - 0.085999965051393 z_k1_s1112
                                 - 0.246999063200733 z_k1_s1113
                                 - 0.169999794520326 z_k1_s1114
                                 - 0.223999660250378 z_k1_s1115
                                 - 0.244999790117446 z_k1_s1116
                                 - 0.419000373302196 z_k1_s1117
                                 - 0.920000005373336 z_k1_s1120
                                 - 0.439000366320059 z_k1_s1121
                                 - 0.544998530837229 z_k1_s1122
                                 - 0.184999585359316 z_k1_s1123
                                 - 0.314999724692322 z_k1_s1124
                                 - 0.837997441280391 z_k1_s1125
                                 - 0.211999914629622 z_k1_s1126
                                 - 0.529998790575365 z_k1_s1127
                                 - 0.49199874208105 z_k1_s1130
                                 - 1.18899555560027 z_k1_s1131
                                 - 0.405998937493789 z_k1_s1132
                                 - 0.858998895001505 z_k1_s1133
                                 - 0.597998417237705 z_k1_s1134
                                 - 0.598998170006928 z_k1_s1135
                                 - 1.19599563067918 z_k1_s1136
                                 - 0.680000414544302 z_k1_s1137
                                 - 0.155999714486198 z_k1_s1140
                                 - 0.309001001815198 z_k1_s1141
                                 - 0.502000621085973 z_k1_s1142
                                 - 0.401999429237333 z_k1_s1143
                                 - 0.350998821394526 z_k1_s1144
                                 - 0.403999453074305 z_k1_s1145
                                 - 0.305000954141252 z_k1_s1146
                                 - 0.251999786262053 z_k1_s1147
                                 - 0.338000087262523 z_k1_s1150
                                 - 0.209999801189253 z_k1_s1151
                                 - 0.243998976627861 z_k1_s1152
                                 - 0.154000024632044 z_k1_s1153
                                 - 0.225999836948412 z_k1_s1154
                                 - 0.300999699544602 z_k1_s1155
                                 - 0.283999647025503 z_k1_s1156
                                 - 0.566999974490758 z_k1_s1157
                                 - 0.415000109648141 z_k1_s1160
                                 - 0.911998433743587 z_k1_s1161
                                 - 0.32199911123865 z_k1_s1162
                                 - 0.336999021563592 z_k1_s1163
                                 - 0.280999226817572 z_k1_s1164
                                 - 0.680995928752354 z_k1_s1165
                                 - 0.368998959789705 z_k1_s1166
                                 - 0.568997893655166 z_k1_s1167
                                 - 0.84499840041252 z_k1_s1170
                                 - 0.419000193178834 z_k1_s1171
                                 - 0.392000599374856 z_k1_s1172
                                 - 0.38800048858522 z_k1_s1173
                                 - 0.764001157702005 z_k1_s1174
                                 - 1.21800014046314 z_k1_s1175
                                 - 0.615998597186896 z_k1_s1176
                                 - 0.487000537901225 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.03232 w_k1_ADM_s000
                                 + 0.789026 w_k1_AXP_s000
                                 + 0.676013 w_k1_DIS_s000
                                 + 0.905654 w_k1_FDX_s000
                                 + 0.653099 w_k1_IBM_s000
                                 + 0.793072 w_k1_INTC_s000
                                 + 0.828939 w_k1_KO_s000
                                 + 0.659683 w_k1_MAT_s000
                                 + 0.819207 w_k1_MSFT_s000
                                 + 0.793142 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.00966 w_k1_ADM_s000
                                 + 1.09199 w_k1_AXP_s000
                                 + 1.23503 w_k1_DIS_s000
                                 + 1.12434 w_k1_FDX_s000
                                 + 0.876134 w_k1_IBM_s000
                                 + 0.962987 w_k1_INTC_s000
                                 + 1.05897 w_k1_KO_s000
                                 + 0.802596 w_k1_MAT_s000
                                 + 1.00667 w_k1_MSFT_s000
                                 + 1.02253 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.03861 w_k1_ADM_s000
                                 + 1.05599 w_k1_AXP_s000 + 1.0429 w_k1_DIS_s000
                                 + 1.01904 w_k1_FDX_s000
                                 + 1.24872 w_k1_IBM_s000
                                 + 0.91129 w_k1_INTC_s000
                                 + 1.28322 w_k1_KO_s000 + 1.60654 w_k1_MAT_s000
                                 + 0.916334 w_k1_MSFT_s000
                                 + 0.949884 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.13663 w_k1_ADM_s000
                                 + 1.39672 w_k1_AXP_s000
                                 + 0.914641 w_k1_DIS_s000
                                 + 1.07827 w_k1_FDX_s000
                                 + 1.05431 w_k1_IBM_s000
                                 + 0.828214 w_k1_INTC_s000
                                 + 1.10353 w_k1_KO_s000 + 1.5996 w_k1_MAT_s000
                                 + 0.842916 w_k1_MSFT_s000
                                 + 0.95194 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.03841 w_k1_ADM_s000
                                 + 0.829338 w_k1_AXP_s000
                                 + 0.912193 w_k1_DIS_s000
                                 + 0.984486 w_k1_FDX_s000
                                 + 0.859252 w_k1_IBM_s000
                                 + 0.772345 w_k1_INTC_s000
                                 + 0.987643 w_k1_KO_s000
                                 + 1.02757 w_k1_MAT_s000
                                 + 0.925496 w_k1_MSFT_s000
                                 + 0.911597 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.27031 w_k1_ADM_s000
                                 + 1.47837 w_k1_AXP_s000
                                 + 1.62826 w_k1_DIS_s000 + 1.2932 w_k1_FDX_s000
                                 + 1.09345 w_k1_IBM_s000
                                 + 1.1861 w_k1_INTC_s000 + 1.41659 w_k1_KO_s000
                                 + 1.5092 w_k1_MAT_s000
                                 + 1.25636 w_k1_MSFT_s000
                                 + 1.12458 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.98558 w_k1_ADM_s000
                                 + 0.816358 w_k1_AXP_s000
                                 + 0.671741 w_k1_DIS_s000
                                 + 0.822616 w_k1_FDX_s000
                                 + 0.963877 w_k1_IBM_s000
                                 + 0.767592 w_k1_INTC_s000
                                 + 0.946176 w_k1_KO_s000
                                 + 1.01433 w_k1_MAT_s000
                                 + 0.80334 w_k1_MSFT_s000
                                 + 0.830091 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.854902 w_k1_ADM_s000
                                 + 1.28049 w_k1_AXP_s000 + 1.1006 w_k1_DIS_s000
                                 + 1.02311 w_k1_FDX_s000
                                 + 1.52366 w_k1_IBM_s000
                                 + 0.949944 w_k1_INTC_s000
                                 + 1.43767 w_k1_KO_s000 + 2.16097 w_k1_MAT_s000
                                 + 0.936122 w_k1_MSFT_s000
                                 + 0.995889 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.27711 w_k1_ADM_s001
                                 + 1.05808 w_k1_AXP_s001
                                 + 0.839722 w_k1_DIS_s001
                                 + 1.0292 w_k1_FDX_s001
                                 + 0.658008 w_k1_IBM_s001
                                 + 0.902719 w_k1_INTC_s001
                                 + 0.771803 w_k1_KO_s001
                                 + 0.716476 w_k1_MAT_s001
                                 + 1.0185 w_k1_MSFT_s001 + 0.90757 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.09235 w_k1_ADM_s001
                                 + 0.876535 w_k1_AXP_s001
                                 + 0.765374 w_k1_DIS_s001
                                 + 0.749312 w_k1_FDX_s001
                                 + 0.9007 w_k1_IBM_s001
                                 + 0.687643 w_k1_INTC_s001
                                 + 0.760352 w_k1_KO_s001
                                 + 0.764353 w_k1_MAT_s001
                                 + 0.834168 w_k1_MSFT_s001
                                 + 0.834951 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.49313 w_k1_ADM_s001
                                 + 1.34814 w_k1_AXP_s001 + 1.5012 w_k1_DIS_s001
                                 + 1.30772 w_k1_FDX_s001
                                 + 0.975758 w_k1_IBM_s001
                                 + 1.25958 w_k1_INTC_s001
                                 + 1.07134 w_k1_KO_s001
                                 + 0.771227 w_k1_MAT_s001
                                 + 1.39791 w_k1_MSFT_s001
                                 + 1.17726 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.8694 w_k1_ADM_s001
                                 + 1.50007 w_k1_AXP_s001
                                 + 1.48251 w_k1_DIS_s001
                                 + 1.23157 w_k1_FDX_s001
                                 + 1.37697 w_k1_IBM_s001
                                 + 1.03874 w_k1_INTC_s001
                                 + 1.21276 w_k1_KO_s001
                                 + 0.907557 w_k1_MAT_s001
                                 + 1.20029 w_k1_MSFT_s001 + 1.2343 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.325 w_k1_ADM_s001
                                 + 1.41725 w_k1_AXP_s001
                                 + 0.845527 w_k1_DIS_s001
                                 + 1.02348 w_k1_FDX_s001
                                 + 1.01538 w_k1_IBM_s001
                                 + 1.08501 w_k1_INTC_s001
                                 + 0.973962 w_k1_KO_s001
                                 + 1.11311 w_k1_MAT_s001
                                 + 0.967382 w_k1_MSFT_s001
                                 + 1.02495 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.04287 w_k1_ADM_s001
                                 + 1.15734 w_k1_AXP_s001
                                 + 0.99292 w_k1_DIS_s001
                                 + 0.900806 w_k1_FDX_s001
                                 + 1.27598 w_k1_IBM_s001
                                 + 0.831611 w_k1_INTC_s001
                                 + 1.06513 w_k1_KO_s001 + 1.6901 w_k1_MAT_s001
                                 + 1.00245 w_k1_MSFT_s001
                                 + 1.01968 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.991781 w_k1_ADM_s001
                                 + 1.59885 w_k1_AXP_s001
                                 + 1.11079 w_k1_DIS_s001
                                 + 0.970514 w_k1_FDX_s001
                                 + 1.56335 w_k1_IBM_s001
                                 + 1.00613 w_k1_INTC_s001
                                 + 1.05456 w_k1_KO_s001 + 2.02602 w_k1_MAT_s001
                                 + 0.935144 w_k1_MSFT_s001
                                 + 1.12504 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.50205 w_k1_ADM_s001
                                 + 1.62764 w_k1_AXP_s001 + 1.5242 w_k1_DIS_s001
                                 + 1.37219 w_k1_FDX_s001
                                 + 1.26123 w_k1_IBM_s001
                                 + 1.12037 w_k1_INTC_s001
                                 + 1.22703 w_k1_KO_s001 + 1.97427 w_k1_MAT_s001
                                 + 1.32641 w_k1_MSFT_s001 + 1.2069 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.02183 w_k1_ADM_s002
                                 + 0.795938 w_k1_AXP_s002
                                 + 1.09719 w_k1_DIS_s002
                                 + 0.93273 w_k1_FDX_s002 + 1.1002 w_k1_IBM_s002
                                 + 0.823609 w_k1_INTC_s002
                                 + 1.10569 w_k1_KO_s002
                                 + 0.961972 w_k1_MAT_s002
                                 + 1.06672 w_k1_MSFT_s002
                                 + 0.907278 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.16288 w_k1_ADM_s002
                                 + 1.12695 w_k1_AXP_s002
                                 + 0.941896 w_k1_DIS_s002
                                 + 1.02051 w_k1_FDX_s002
                                 + 1.14853 w_k1_IBM_s002
                                 + 1.0027 w_k1_INTC_s002
                                 + 0.963127 w_k1_KO_s002
                                 + 0.820274 w_k1_MAT_s002
                                 + 0.81506 w_k1_MSFT_s002
                                 + 0.942795 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.37616 w_k1_ADM_s002
                                 + 1.16543 w_k1_AXP_s002 + 1.6477 w_k1_DIS_s002
                                 + 1.32004 w_k1_FDX_s002
                                 + 1.31441 w_k1_IBM_s002
                                 + 1.39987 w_k1_INTC_s002
                                 + 1.65069 w_k1_KO_s002 + 1.45738 w_k1_MAT_s002
                                 + 1.64284 w_k1_MSFT_s002
                                 + 1.30971 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.81569 w_k1_ADM_s002
                                 + 0.637158 w_k1_AXP_s002
                                 + 0.544771 w_k1_DIS_s002
                                 + 0.723456 w_k1_FDX_s002
                                 + 0.853643 w_k1_IBM_s002
                                 + 0.810886 w_k1_INTC_s002
                                 + 0.651815 w_k1_KO_s002
                                 + 0.459131 w_k1_MAT_s002
                                 + 0.726083 w_k1_MSFT_s002
                                 + 0.800809 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.01184 w_k1_ADM_s002
                                 + 0.787263 w_k1_AXP_s002
                                 + 0.895074 w_k1_DIS_s002
                                 + 0.959909 w_k1_FDX_s002
                                 + 0.785724 w_k1_IBM_s002
                                 + 0.854658 w_k1_INTC_s002
                                 + 0.836612 w_k1_KO_s002
                                 + 0.877044 w_k1_MAT_s002
                                 + 1.02256 w_k1_MSFT_s002
                                 + 0.953213 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.16326 w_k1_ADM_s002
                                 + 0.834756 w_k1_AXP_s002
                                 + 0.780742 w_k1_DIS_s002
                                 + 0.80166 w_k1_FDX_s002
                                 + 1.08035 w_k1_IBM_s002
                                 + 0.970325 w_k1_INTC_s002
                                 + 0.946522 w_k1_KO_s002
                                 + 0.66857 w_k1_MAT_s002
                                 + 0.826908 w_k1_MSFT_s002
                                 + 0.910185 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.27149 w_k1_ADM_s002
                                 + 1.13814 w_k1_AXP_s002
                                 + 1.20837 w_k1_DIS_s002
                                 + 1.22592 w_k1_FDX_s002
                                 + 1.03085 w_k1_IBM_s002
                                 + 1.19876 w_k1_INTC_s002
                                 + 1.04265 w_k1_KO_s002
                                 + 0.817906 w_k1_MAT_s002
                                 + 1.16071 w_k1_MSFT_s002
                                 + 1.06406 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.17827 w_k1_ADM_s002
                                 + 0.735653 w_k1_AXP_s002
                                 + 0.750794 w_k1_DIS_s002
                                 + 0.913198 w_k1_FDX_s002
                                 + 0.675444 w_k1_IBM_s002
                                 + 0.9656 w_k1_INTC_s002
                                 + 0.835148 w_k1_KO_s002
                                 + 0.293656 w_k1_MAT_s002
                                 + 0.859811 w_k1_MSFT_s002
                                 + 0.830871 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.19319 w_k1_ADM_s003
                                 + 0.819019 w_k1_AXP_s003
                                 + 1.02558 w_k1_DIS_s003
                                 + 1.05793 w_k1_FDX_s003
                                 + 0.693693 w_k1_IBM_s003
                                 + 1.09121 w_k1_INTC_s003
                                 + 1.11993 w_k1_KO_s003
                                 + 0.780968 w_k1_MAT_s003
                                 + 1.16773 w_k1_MSFT_s003
                                 + 1.06563 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.14718 w_k1_ADM_s003
                                 + 0.839199 w_k1_AXP_s003
                                 + 0.724252 w_k1_DIS_s003
                                 + 0.847303 w_k1_FDX_s003
                                 + 1.02312 w_k1_IBM_s003
                                 + 0.988776 w_k1_INTC_s003
                                 + 0.97327 w_k1_KO_s003 + 1.0722 w_k1_MAT_s003
                                 + 0.969972 w_k1_MSFT_s003
                                 + 0.926673 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.740032 w_k1_ADM_s003
                                 + 0.672486 w_k1_AXP_s003
                                 + 0.696412 w_k1_DIS_s003
                                 + 0.739021 w_k1_FDX_s003
                                 + 1.25637 w_k1_IBM_s003
                                 + 1.04161 w_k1_INTC_s003
                                 + 1.03055 w_k1_KO_s003 + 1.05277 w_k1_MAT_s003
                                 + 1.00818 w_k1_MSFT_s003
                                 + 0.978016 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.826637 w_k1_ADM_s003
                                 + 0.621026 w_k1_AXP_s003
                                 + 0.570491 w_k1_DIS_s003
                                 + 0.735408 w_k1_FDX_s003
                                 + 0.722554 w_k1_IBM_s003
                                 + 0.914925 w_k1_INTC_s003
                                 + 0.797641 w_k1_KO_s003
                                 + 0.568399 w_k1_MAT_s003
                                 + 0.738154 w_k1_MSFT_s003
                                 + 0.732693 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.30769 w_k1_ADM_s003
                                 + 1.27825 w_k1_AXP_s003
                                 + 0.993959 w_k1_DIS_s003
                                 + 1.08684 w_k1_FDX_s003
                                 + 0.97875 w_k1_IBM_s003
                                 + 1.29305 w_k1_INTC_s003
                                 + 1.32266 w_k1_KO_s003 + 1.03862 w_k1_MAT_s003
                                 + 1.08415 w_k1_MSFT_s003
                                 + 1.10673 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.10463 w_k1_ADM_s003
                                 + 0.708432 w_k1_AXP_s003
                                 + 0.89146 w_k1_DIS_s003
                                 + 0.882129 w_k1_FDX_s003
                                 + 0.851562 w_k1_IBM_s003
                                 + 0.941013 w_k1_INTC_s003
                                 + 1.02927 w_k1_KO_s003
                                 + 0.721336 w_k1_MAT_s003
                                 + 0.942797 w_k1_MSFT_s003
                                 + 0.948182 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.14389 w_k1_ADM_s003
                                 + 0.971119 w_k1_AXP_s003
                                 + 1.07005 w_k1_DIS_s003
                                 + 1.00712 w_k1_FDX_s003
                                 + 1.39468 w_k1_IBM_s003
                                 + 1.20511 w_k1_INTC_s003 + 1.3174 w_k1_KO_s003
                                 + 1.31177 w_k1_MAT_s003
                                 + 1.12519 w_k1_MSFT_s003
                                 + 1.08488 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.28675 w_k1_ADM_s003
                                 + 0.96512 w_k1_AXP_s003
                                 + 1.51349 w_k1_DIS_s003 + 1.141 w_k1_FDX_s003
                                 + 1.02629 w_k1_IBM_s003
                                 + 1.2997 w_k1_INTC_s003 + 1.41258 w_k1_KO_s003
                                 + 0.941318 w_k1_MAT_s003
                                 + 1.316 w_k1_MSFT_s003 + 1.19137 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.871022 w_k1_ADM_s004
                                 + 0.759545 w_k1_AXP_s004
                                 + 0.563369 w_k1_DIS_s004
                                 + 0.802796 w_k1_FDX_s004
                                 + 0.718507 w_k1_IBM_s004
                                 + 0.835757 w_k1_INTC_s004
                                 + 0.722627 w_k1_KO_s004
                                 + 0.618444 w_k1_MAT_s004
                                 + 0.686885 w_k1_MSFT_s004
                                 + 0.811107 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.16054 w_k1_ADM_s004
                                 + 0.8734 w_k1_AXP_s004
                                 + 0.965544 w_k1_DIS_s004
                                 + 1.08589 w_k1_FDX_s004
                                 + 0.826299 w_k1_IBM_s004
                                 + 0.887062 w_k1_INTC_s004
                                 + 0.946596 w_k1_KO_s004
                                 + 0.902268 w_k1_MAT_s004
                                 + 0.950902 w_k1_MSFT_s004
                                 + 0.869752 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.34113 w_k1_ADM_s004
                                 + 1.13324 w_k1_AXP_s004
                                 + 1.64701 w_k1_DIS_s004
                                 + 1.36548 w_k1_FDX_s004
                                 + 1.39606 w_k1_IBM_s004
                                 + 0.983981 w_k1_INTC_s004
                                 + 1.3084 w_k1_KO_s004 + 1.59185 w_k1_MAT_s004
                                 + 1.20426 w_k1_MSFT_s004
                                 + 1.15009 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.791335 w_k1_ADM_s004
                                 + 0.765805 w_k1_AXP_s004
                                 + 0.744428 w_k1_DIS_s004
                                 + 0.850302 w_k1_FDX_s004
                                 + 1.16345 w_k1_IBM_s004
                                 + 1.05051 w_k1_INTC_s004 + 1.0298 w_k1_KO_s004
                                 + 1.14714 w_k1_MAT_s004
                                 + 0.894032 w_k1_MSFT_s004
                                 + 0.931979 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.41793 w_k1_ADM_s004
                                 + 1.49607 w_k1_AXP_s004
                                 + 0.963127 w_k1_DIS_s004
                                 + 1.17227 w_k1_FDX_s004
                                 + 0.989452 w_k1_IBM_s004
                                 + 0.898505 w_k1_INTC_s004
                                 + 0.911515 w_k1_KO_s004
                                 + 1.30899 w_k1_MAT_s004
                                 + 0.803743 w_k1_MSFT_s004
                                 + 0.95581 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.05282 w_k1_ADM_s004
                                 + 1.35492 w_k1_AXP_s004
                                 + 1.20685 w_k1_DIS_s004 + 1.0541 w_k1_FDX_s004
                                 + 1.83337 w_k1_IBM_s004
                                 + 1.05243 w_k1_INTC_s004
                                 + 1.53147 w_k1_KO_s004 + 1.93561 w_k1_MAT_s004
                                 + 1.04531 w_k1_MSFT_s004
                                 + 1.05084 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.1903 w_k1_ADM_s004
                                 + 1.0035 w_k1_AXP_s004 + 1.36096 w_k1_DIS_s004
                                 + 1.43897 w_k1_FDX_s004
                                 + 0.880143 w_k1_IBM_s004
                                 + 1.29019 w_k1_INTC_s004
                                 + 1.02196 w_k1_KO_s004
                                 + 0.690508 w_k1_MAT_s004
                                 + 1.27251 w_k1_MSFT_s004
                                 + 1.02995 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.938358 w_k1_ADM_s004
                                 + 1.04213 w_k1_AXP_s004
                                 + 1.25576 w_k1_DIS_s004
                                 + 1.05837 w_k1_FDX_s004
                                 + 1.38884 w_k1_IBM_s004
                                 + 0.999811 w_k1_INTC_s004
                                 + 1.26509 w_k1_KO_s004 + 1.24728 w_k1_MAT_s004
                                 + 0.932004 w_k1_MSFT_s004
                                 + 1.00822 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.22153 w_k1_ADM_s005
                                 + 0.923119 w_k1_AXP_s005
                                 + 1.16622 w_k1_DIS_s005
                                 + 1.21599 w_k1_FDX_s005
                                 + 0.717332 w_k1_IBM_s005
                                 + 1.02427 w_k1_INTC_s005
                                 + 0.98842 w_k1_KO_s005
                                 + 0.826101 w_k1_MAT_s005
                                 + 1.15484 w_k1_MSFT_s005
                                 + 1.14282 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.18047 w_k1_ADM_s005
                                 + 1.21582 w_k1_AXP_s005
                                 + 1.25979 w_k1_DIS_s005
                                 + 1.10659 w_k1_FDX_s005
                                 + 1.41298 w_k1_IBM_s005
                                 + 1.28576 w_k1_INTC_s005 + 1.1856 w_k1_KO_s005
                                 + 0.977386 w_k1_MAT_s005
                                 + 1.01071 w_k1_MSFT_s005
                                 + 1.07647 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.01108 w_k1_ADM_s005
                                 + 0.732633 w_k1_AXP_s005
                                 + 0.791613 w_k1_DIS_s005
                                 + 0.768545 w_k1_FDX_s005
                                 + 0.922136 w_k1_IBM_s005
                                 + 0.772005 w_k1_INTC_s005
                                 + 0.888646 w_k1_KO_s005
                                 + 0.854634 w_k1_MAT_s005
                                 + 0.844541 w_k1_MSFT_s005
                                 + 1.02118 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.805859 w_k1_ADM_s005
                                 + 0.932804 w_k1_AXP_s005
                                 + 0.650001 w_k1_DIS_s005
                                 + 0.808199 w_k1_FDX_s005
                                 + 1.01881 w_k1_IBM_s005
                                 + 0.678645 w_k1_INTC_s005
                                 + 0.775998 w_k1_KO_s005
                                 + 1.36541 w_k1_MAT_s005
                                 + 0.735347 w_k1_MSFT_s005
                                 + 0.869626 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.14704 w_k1_ADM_s005
                                 + 1.51921 w_k1_AXP_s005
                                 + 1.44062 w_k1_DIS_s005 + 1.3781 w_k1_FDX_s005
                                 + 1.38518 w_k1_IBM_s005
                                 + 1.25758 w_k1_INTC_s005 + 1.0598 w_k1_KO_s005
                                 + 1.16179 w_k1_MAT_s005
                                 + 1.14052 w_k1_MSFT_s005
                                 + 1.08354 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.05518 w_k1_ADM_s005
                                 + 1.3146 w_k1_AXP_s005 + 1.00409 w_k1_DIS_s005
                                 + 1.15379 w_k1_FDX_s005 + 1.1424 w_k1_IBM_s005
                                 + 1.28215 w_k1_INTC_s005
                                 + 0.942602 w_k1_KO_s005
                                 + 1.17102 w_k1_MAT_s005
                                 + 0.945137 w_k1_MSFT_s005
                                 + 0.980561 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.02676 w_k1_ADM_s005
                                 + 0.798216 w_k1_AXP_s005
                                 + 0.591233 w_k1_DIS_s005
                                 + 0.815011 w_k1_FDX_s005
                                 + 0.67469 w_k1_IBM_s005
                                 + 0.883134 w_k1_INTC_s005
                                 + 0.707877 w_k1_KO_s005
                                 + 0.637883 w_k1_MAT_s005
                                 + 0.809503 w_k1_MSFT_s005
                                 + 0.875427 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.981188 w_k1_ADM_s005
                                 + 0.95437 w_k1_AXP_s005
                                 + 0.922563 w_k1_DIS_s005
                                 + 0.966679 w_k1_FDX_s005
                                 + 0.909427 w_k1_IBM_s005
                                 + 0.953521 w_k1_INTC_s005
                                 + 0.922042 w_k1_KO_s005
                                 + 0.861228 w_k1_MAT_s005
                                 + 0.92776 w_k1_MSFT_s005
                                 + 0.956212 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.968062 w_k1_ADM_s006
                                 + 0.677404 w_k1_AXP_s006
                                 + 0.704857 w_k1_DIS_s006
                                 + 0.803147 w_k1_FDX_s006
                                 + 0.670559 w_k1_IBM_s006
                                 + 1.09136 w_k1_INTC_s006
                                 + 0.707011 w_k1_KO_s006
                                 + 0.499211 w_k1_MAT_s006
                                 + 0.877825 w_k1_MSFT_s006
                                 + 0.884692 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.00242 w_k1_ADM_s006
                                 + 0.998784 w_k1_AXP_s006
                                 + 1.20909 w_k1_DIS_s006
                                 + 1.08259 w_k1_FDX_s006 + 1.0617 w_k1_IBM_s006
                                 + 1.15938 w_k1_INTC_s006
                                 + 0.956526 w_k1_KO_s006
                                 + 1.12853 w_k1_MAT_s006
                                 + 1.21922 w_k1_MSFT_s006
                                 + 1.01602 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.972532 w_k1_ADM_s006
                                 + 0.784898 w_k1_AXP_s006
                                 + 1.09771 w_k1_DIS_s006 + 1.0351 w_k1_FDX_s006
                                 + 0.809403 w_k1_IBM_s006
                                 + 1.28594 w_k1_INTC_s006
                                 + 1.02674 w_k1_KO_s006
                                 + 0.718065 w_k1_MAT_s006
                                 + 1.26044 w_k1_MSFT_s006
                                 + 0.974395 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.14046 w_k1_ADM_s006
                                 + 1.44104 w_k1_AXP_s006
                                 + 1.45536 w_k1_DIS_s006
                                 + 1.17325 w_k1_FDX_s006
                                 + 1.36619 w_k1_IBM_s006
                                 + 1.50888 w_k1_INTC_s006
                                 + 1.15416 w_k1_KO_s006 + 1.26644 w_k1_MAT_s006
                                 + 1.45867 w_k1_MSFT_s006
                                 + 1.15925 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.05104 w_k1_ADM_s006
                                 + 1.22924 w_k1_AXP_s006
                                 + 0.907728 w_k1_DIS_s006
                                 + 1.1059 w_k1_FDX_s006
                                 + 0.840917 w_k1_IBM_s006
                                 + 1.18957 w_k1_INTC_s006
                                 + 0.757415 w_k1_KO_s006
                                 + 0.862265 w_k1_MAT_s006
                                 + 1.10359 w_k1_MSFT_s006
                                 + 0.929992 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.732623 w_k1_ADM_s006
                                 + 0.98479 w_k1_AXP_s006
                                 + 0.875607 w_k1_DIS_s006
                                 + 0.858879 w_k1_FDX_s006
                                 + 1.17041 w_k1_IBM_s006
                                 + 0.919105 w_k1_INTC_s006
                                 + 0.83471 w_k1_KO_s006 + 1.32854 w_k1_MAT_s006
                                 + 0.90234 w_k1_MSFT_s006
                                 + 1.03614 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.21464 w_k1_ADM_s006
                                 + 0.997958 w_k1_AXP_s006
                                 + 1.68191 w_k1_DIS_s006
                                 + 1.21307 w_k1_FDX_s006
                                 + 1.02948 w_k1_IBM_s006
                                 + 1.14825 w_k1_INTC_s006
                                 + 1.21079 w_k1_KO_s006
                                 + 0.782536 w_k1_MAT_s006
                                 + 1.44913 w_k1_MSFT_s006
                                 + 1.18265 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.11813 w_k1_ADM_s006
                                 + 0.987449 w_k1_AXP_s006
                                 + 1.04799 w_k1_DIS_s006 + 1.2198 w_k1_FDX_s006
                                 + 0.68918 w_k1_IBM_s006
                                 + 1.39118 w_k1_INTC_s006
                                 + 0.82968 w_k1_KO_s006
                                 + 0.723349 w_k1_MAT_s006
                                 + 1.30285 w_k1_MSFT_s006
                                 + 0.868692 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.915385 w_k1_ADM_s007
                                 + 1.00777 w_k1_AXP_s007
                                 + 1.05853 w_k1_DIS_s007
                                 + 0.896882 w_k1_FDX_s007
                                 + 1.36057 w_k1_IBM_s007
                                 + 0.903747 w_k1_INTC_s007
                                 + 0.952657 w_k1_KO_s007
                                 + 1.15066 w_k1_MAT_s007
                                 + 0.951281 w_k1_MSFT_s007
                                 + 0.979853 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.967717 w_k1_ADM_s007
                                 + 0.826878 w_k1_AXP_s007
                                 + 0.638335 w_k1_DIS_s007
                                 + 0.783364 w_k1_FDX_s007
                                 + 1.09079 w_k1_IBM_s007
                                 + 0.783564 w_k1_INTC_s007
                                 + 0.811213 w_k1_KO_s007
                                 + 0.831711 w_k1_MAT_s007
                                 + 0.797932 w_k1_MSFT_s007
                                 + 0.984544 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.960555 w_k1_ADM_s007
                                 + 0.949418 w_k1_AXP_s007
                                 + 0.966438 w_k1_DIS_s007
                                 + 1.05779 w_k1_FDX_s007
                                 + 0.682575 w_k1_IBM_s007
                                 + 0.996475 w_k1_INTC_s007
                                 + 0.687542 w_k1_KO_s007
                                 + 0.750455 w_k1_MAT_s007
                                 + 0.86242 w_k1_MSFT_s007
                                 + 0.979814 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.976266 w_k1_ADM_s007
                                 + 0.861711 w_k1_AXP_s007
                                 + 1.1963 w_k1_DIS_s007
                                 + 0.804222 w_k1_FDX_s007
                                 + 1.9312 w_k1_IBM_s007
                                 + 0.657726 w_k1_INTC_s007
                                 + 1.2348 w_k1_KO_s007 + 1.83236 w_k1_MAT_s007
                                 + 0.889913 w_k1_MSFT_s007
                                 + 1.09446 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.994801 w_k1_ADM_s007
                                 + 0.93335 w_k1_AXP_s007
                                 + 1.03935 w_k1_DIS_s007
                                 + 0.914446 w_k1_FDX_s007 + 1.082 w_k1_IBM_s007
                                 + 1.14716 w_k1_INTC_s007
                                 + 1.00056 w_k1_KO_s007
                                 + 0.885977 w_k1_MAT_s007
                                 + 0.954858 w_k1_MSFT_s007
                                 + 1.02965 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.28791 w_k1_ADM_s007
                                 + 1.10559 w_k1_AXP_s007
                                 + 1.53389 w_k1_DIS_s007
                                 + 1.24682 w_k1_FDX_s007
                                 + 1.01761 w_k1_IBM_s007
                                 + 1.18235 w_k1_INTC_s007
                                 + 1.11442 w_k1_KO_s007
                                 + 0.917581 w_k1_MAT_s007
                                 + 1.28067 w_k1_MSFT_s007
                                 + 1.15361 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.860644 w_k1_ADM_s007
                                 + 0.780143 w_k1_AXP_s007
                                 + 0.67225 w_k1_DIS_s007
                                 + 0.735493 w_k1_FDX_s007
                                 + 0.707862 w_k1_IBM_s007
                                 + 0.856258 w_k1_INTC_s007
                                 + 0.714647 w_k1_KO_s007
                                 + 0.793464 w_k1_MAT_s007
                                 + 0.716915 w_k1_MSFT_s007
                                 + 0.915235 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.12544 w_k1_ADM_s007
                                 + 1.40257 w_k1_AXP_s007
                                 + 1.23436 w_k1_DIS_s007
                                 + 1.11149 w_k1_FDX_s007
                                 + 1.56513 w_k1_IBM_s007
                                 + 1.03944 w_k1_INTC_s007
                                 + 1.19265 w_k1_KO_s007 + 1.41648 w_k1_MAT_s007
                                 + 0.903294 w_k1_MSFT_s007
                                 + 1.07179 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.780874 w_k1_ADM_s010
                                 + 0.646301 w_k1_AXP_s010
                                 + 0.780443 w_k1_DIS_s010
                                 + 0.731722 w_k1_FDX_s010
                                 + 0.873298 w_k1_IBM_s010
                                 + 0.788427 w_k1_INTC_s010
                                 + 0.993992 w_k1_KO_s010 + 0.7791 w_k1_MAT_s010
                                 + 0.962823 w_k1_MSFT_s010
                                 + 0.918826 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.781312 w_k1_ADM_s010
                                 + 0.932641 w_k1_AXP_s010
                                 + 0.8742 w_k1_DIS_s010
                                 + 0.894923 w_k1_FDX_s010
                                 + 0.934426 w_k1_IBM_s010
                                 + 0.880378 w_k1_INTC_s010
                                 + 0.876708 w_k1_KO_s010
                                 + 1.25474 w_k1_MAT_s010
                                 + 0.947192 w_k1_MSFT_s010
                                 + 0.95388 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.642802 w_k1_ADM_s010
                                 + 0.755905 w_k1_AXP_s010
                                 + 0.576462 w_k1_DIS_s010
                                 + 0.731705 w_k1_FDX_s010
                                 + 0.718931 w_k1_IBM_s010
                                 + 0.652083 w_k1_INTC_s010
                                 + 0.622757 w_k1_KO_s010
                                 + 0.933379 w_k1_MAT_s010
                                 + 0.754823 w_k1_MSFT_s010
                                 + 0.832915 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.22658 w_k1_ADM_s010
                                 + 1.7103 w_k1_AXP_s010 + 1.58731 w_k1_DIS_s010
                                 + 1.32171 w_k1_FDX_s010
                                 + 1.26324 w_k1_IBM_s010
                                 + 1.16831 w_k1_INTC_s010
                                 + 1.16058 w_k1_KO_s010 + 1.40326 w_k1_MAT_s010
                                 + 1.38841 w_k1_MSFT_s010
                                 + 1.23425 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.965358 w_k1_ADM_s010
                                 + 1.06172 w_k1_AXP_s010
                                 + 1.28023 w_k1_DIS_s010
                                 + 0.991882 w_k1_FDX_s010
                                 + 1.58964 w_k1_IBM_s010
                                 + 0.890236 w_k1_INTC_s010
                                 + 1.37853 w_k1_KO_s010 + 1.58386 w_k1_MAT_s010
                                 + 1.10037 w_k1_MSFT_s010
                                 + 1.14249 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.950498 w_k1_ADM_s010
                                 + 1.05106 w_k1_AXP_s010
                                 + 0.839605 w_k1_DIS_s010
                                 + 0.883566 w_k1_FDX_s010
                                 + 1.34798 w_k1_IBM_s010
                                 + 0.829421 w_k1_INTC_s010
                                 + 1.08053 w_k1_KO_s010 + 1.37315 w_k1_MAT_s010
                                 + 0.912742 w_k1_MSFT_s010
                                 + 0.993408 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.19884 w_k1_ADM_s010
                                 + 1.31989 w_k1_AXP_s010
                                 + 1.21118 w_k1_DIS_s010
                                 + 1.25452 w_k1_FDX_s010
                                 + 0.793234 w_k1_IBM_s010
                                 + 0.865918 w_k1_INTC_s010
                                 + 0.881733 w_k1_KO_s010
                                 + 1.00092 w_k1_MAT_s010
                                 + 1.16467 w_k1_MSFT_s010
                                 + 1.09727 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.02058 w_k1_ADM_s010
                                 + 0.972539 w_k1_AXP_s010
                                 + 1.0629 w_k1_DIS_s010 + 1.08834 w_k1_FDX_s010
                                 + 0.946304 w_k1_IBM_s010
                                 + 0.943249 w_k1_INTC_s010
                                 + 0.943101 w_k1_KO_s010 + 1.154 w_k1_MAT_s010
                                 + 1.18006 w_k1_MSFT_s010
                                 + 1.08673 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.10167 w_k1_ADM_s011
                                 + 1.02118 w_k1_AXP_s011 + 1.2869 w_k1_DIS_s011
                                 + 1.15374 w_k1_FDX_s011
                                 + 1.00045 w_k1_IBM_s011
                                 + 1.07542 w_k1_INTC_s011
                                 + 1.19007 w_k1_KO_s011 + 1.05437 w_k1_MAT_s011
                                 + 1.20385 w_k1_MSFT_s011
                                 + 1.03002 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.11324 w_k1_ADM_s011
                                 + 1.05778 w_k1_AXP_s011
                                 + 1.01925 w_k1_DIS_s011
                                 + 1.12667 w_k1_FDX_s011
                                 + 0.643941 w_k1_IBM_s011
                                 + 0.90692 w_k1_INTC_s011
                                 + 0.868557 w_k1_KO_s011
                                 + 0.816945 w_k1_MAT_s011
                                 + 0.919671 w_k1_MSFT_s011
                                 + 0.892264 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.08454 w_k1_ADM_s011
                                 + 0.733123 w_k1_AXP_s011
                                 + 0.839617 w_k1_DIS_s011
                                 + 0.918005 w_k1_FDX_s011
                                 + 0.64537 w_k1_IBM_s011
                                 + 0.848936 w_k1_INTC_s011
                                 + 0.80824 w_k1_KO_s011
                                 + 0.709669 w_k1_MAT_s011
                                 + 0.823863 w_k1_MSFT_s011
                                 + 0.878647 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.10584 w_k1_ADM_s011
                                 + 1.03838 w_k1_AXP_s011
                                 + 0.960457 w_k1_DIS_s011
                                 + 0.937273 w_k1_FDX_s011
                                 + 1.15591 w_k1_IBM_s011
                                 + 0.968891 w_k1_INTC_s011
                                 + 1.1571 w_k1_KO_s011 + 1.26567 w_k1_MAT_s011
                                 + 0.961129 w_k1_MSFT_s011
                                 + 0.96832 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.884457 w_k1_ADM_s011
                                 + 0.659204 w_k1_AXP_s011
                                 + 0.692217 w_k1_DIS_s011
                                 + 0.738017 w_k1_FDX_s011
                                 + 0.84913 w_k1_IBM_s011
                                 + 0.850994 w_k1_INTC_s011
                                 + 0.837357 w_k1_KO_s011
                                 + 0.598889 w_k1_MAT_s011
                                 + 0.786433 w_k1_MSFT_s011
                                 + 0.773982 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.00122 w_k1_ADM_s011
                                 + 0.989881 w_k1_AXP_s011
                                 + 1.04188 w_k1_DIS_s011
                                 + 1.03411 w_k1_FDX_s011
                                 + 0.879401 w_k1_IBM_s011
                                 + 1.05279 w_k1_INTC_s011
                                 + 1.15133 w_k1_KO_s011 + 1.13808 w_k1_MAT_s011
                                 + 1.13471 w_k1_MSFT_s011
                                 + 1.06944 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.33507 w_k1_ADM_s011
                                 + 1.4401 w_k1_AXP_s011 + 1.57434 w_k1_DIS_s011
                                 + 1.23855 w_k1_FDX_s011
                                 + 1.76727 w_k1_IBM_s011
                                 + 1.24662 w_k1_INTC_s011
                                 + 1.52848 w_k1_KO_s011 + 1.63631 w_k1_MAT_s011
                                 + 1.2945 w_k1_MSFT_s011 + 1.1519 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.11717 w_k1_ADM_s011
                                 + 1.11351 w_k1_AXP_s011
                                 + 0.850459 w_k1_DIS_s011
                                 + 0.975701 w_k1_FDX_s011
                                 + 0.855902 w_k1_IBM_s011
                                 + 0.918359 w_k1_INTC_s011
                                 + 0.822732 w_k1_KO_s011
                                 + 1.20122 w_k1_MAT_s011
                                 + 0.817858 w_k1_MSFT_s011
                                 + 0.901506 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.20867 w_k1_ADM_s012
                                 + 1.08003 w_k1_AXP_s012
                                 + 1.12071 w_k1_DIS_s012
                                 + 1.02419 w_k1_FDX_s012
                                 + 1.57548 w_k1_IBM_s012
                                 + 0.839597 w_k1_INTC_s012
                                 + 1.25066 w_k1_KO_s012 + 2.17073 w_k1_MAT_s012
                                 + 0.887719 w_k1_MSFT_s012
                                 + 0.962441 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.06672 w_k1_ADM_s012
                                 + 1.02232 w_k1_AXP_s012 + 1.1387 w_k1_DIS_s012
                                 + 1.17595 w_k1_FDX_s012
                                 + 0.776573 w_k1_IBM_s012
                                 + 1.24473 w_k1_INTC_s012
                                 + 1.08962 w_k1_KO_s012 + 0.89703 w_k1_MAT_s012
                                 + 1.23746 w_k1_MSFT_s012
                                 + 1.04895 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.12636 w_k1_ADM_s012
                                 + 1.56794 w_k1_AXP_s012
                                 + 1.60311 w_k1_DIS_s012 + 1.3986 w_k1_FDX_s012
                                 + 1.21228 w_k1_IBM_s012
                                 + 1.37484 w_k1_INTC_s012
                                 + 1.41024 w_k1_KO_s012 + 2.41063 w_k1_MAT_s012
                                 + 1.2064 w_k1_MSFT_s012 + 1.3084 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.870174 w_k1_ADM_s012
                                 + 0.944379 w_k1_AXP_s012
                                 + 0.881225 w_k1_DIS_s012
                                 + 1.00119 w_k1_FDX_s012
                                 + 0.801024 w_k1_IBM_s012
                                 + 0.989512 w_k1_INTC_s012
                                 + 0.910301 w_k1_KO_s012
                                 + 1.23604 w_k1_MAT_s012
                                 + 0.84495 w_k1_MSFT_s012
                                 + 0.889089 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.949862 w_k1_ADM_s012
                                 + 1.01891 w_k1_AXP_s012
                                 + 0.978996 w_k1_DIS_s012
                                 + 1.00384 w_k1_FDX_s012
                                 + 1.09215 w_k1_IBM_s012
                                 + 1.13477 w_k1_INTC_s012
                                 + 1.00016 w_k1_KO_s012 + 1.02319 w_k1_MAT_s012
                                 + 0.964834 w_k1_MSFT_s012
                                 + 0.928407 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.26959 w_k1_ADM_s012
                                 + 1.32982 w_k1_AXP_s012
                                 + 1.07776 w_k1_DIS_s012
                                 + 1.14169 w_k1_FDX_s012
                                 + 1.16599 w_k1_IBM_s012
                                 + 1.16552 w_k1_INTC_s012
                                 + 0.947701 w_k1_KO_s012
                                 + 1.53475 w_k1_MAT_s012
                                 + 1.03835 w_k1_MSFT_s012
                                 + 1.05081 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.806775 w_k1_ADM_s012
                                 + 0.802799 w_k1_AXP_s012
                                 + 0.686275 w_k1_DIS_s012
                                 + 0.791726 w_k1_FDX_s012
                                 + 0.834089 w_k1_IBM_s012
                                 + 0.970326 w_k1_INTC_s012
                                 + 0.815542 w_k1_KO_s012
                                 + 0.842217 w_k1_MAT_s012
                                 + 0.811202 w_k1_MSFT_s012
                                 + 0.797065 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.908726 w_k1_ADM_s012
                                 + 0.84227 w_k1_AXP_s012
                                 + 0.667997 w_k1_DIS_s012
                                 + 0.822215 w_k1_FDX_s012
                                 + 1.16558 w_k1_IBM_s012
                                 + 0.937816 w_k1_INTC_s012
                                 + 0.979157 w_k1_KO_s012
                                 + 1.43013 w_k1_MAT_s012
                                 + 0.848079 w_k1_MSFT_s012
                                 + 0.937766 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.40157 w_k1_ADM_s013
                                 + 1.17298 w_k1_AXP_s013
                                 + 1.30603 w_k1_DIS_s013
                                 + 1.45679 w_k1_FDX_s013
                                 + 0.825187 w_k1_IBM_s013
                                 + 1.50189 w_k1_INTC_s013
                                 + 0.876985 w_k1_KO_s013
                                 + 0.590589 w_k1_MAT_s013
                                 + 1.13096 w_k1_MSFT_s013
                                 + 1.01139 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.11445 w_k1_ADM_s013
                                 + 1.05233 w_k1_AXP_s013
                                 + 1.16636 w_k1_DIS_s013 + 1.0362 w_k1_FDX_s013
                                 + 1.45134 w_k1_IBM_s013
                                 + 1.09989 w_k1_INTC_s013
                                 + 1.10549 w_k1_KO_s013 + 1.2828 w_k1_MAT_s013
                                 + 1.04706 w_k1_MSFT_s013
                                 + 1.11885 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.05829 w_k1_ADM_s013
                                 + 1.35422 w_k1_AXP_s013
                                 + 1.82881 w_k1_DIS_s013 + 1.3109 w_k1_FDX_s013
                                 + 1.59745 w_k1_IBM_s013
                                 + 1.33162 w_k1_INTC_s013
                                 + 1.49401 w_k1_KO_s013 + 1.54634 w_k1_MAT_s013
                                 + 1.27073 w_k1_MSFT_s013
                                 + 1.26274 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.884972 w_k1_ADM_s013
                                 + 0.811855 w_k1_AXP_s013
                                 + 0.875901 w_k1_DIS_s013
                                 + 0.858847 w_k1_FDX_s013
                                 + 1.29294 w_k1_IBM_s013
                                 + 0.89479 w_k1_INTC_s013
                                 + 1.06206 w_k1_KO_s013 + 1.05287 w_k1_MAT_s013
                                 + 0.789735 w_k1_MSFT_s013
                                 + 0.941073 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.900456 w_k1_ADM_s013
                                 + 0.856925 w_k1_AXP_s013
                                 + 0.586148 w_k1_DIS_s013
                                 + 0.827469 w_k1_FDX_s013
                                 + 0.8419 w_k1_IBM_s013
                                 + 0.962089 w_k1_INTC_s013
                                 + 0.676599 w_k1_KO_s013
                                 + 0.610567 w_k1_MAT_s013
                                 + 0.724214 w_k1_MSFT_s013
                                 + 0.747877 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.03662 w_k1_ADM_s013
                                 + 0.868174 w_k1_AXP_s013
                                 + 1.36616 w_k1_DIS_s013
                                 + 1.04405 w_k1_FDX_s013
                                 + 1.04516 w_k1_IBM_s013
                                 + 1.09258 w_k1_INTC_s013
                                 + 1.12311 w_k1_KO_s013
                                 + 0.917115 w_k1_MAT_s013
                                 + 0.980593 w_k1_MSFT_s013
                                 + 1.0105 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.11256 w_k1_ADM_s013
                                 + 0.925855 w_k1_AXP_s013
                                 + 1.0381 w_k1_DIS_s013 + 1.15502 w_k1_FDX_s013
                                 + 0.788454 w_k1_IBM_s013
                                 + 1.10758 w_k1_INTC_s013
                                 + 0.902419 w_k1_KO_s013
                                 + 0.879352 w_k1_MAT_s013
                                 + 0.971522 w_k1_MSFT_s013
                                 + 0.944952 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.973965 w_k1_ADM_s013
                                 + 0.673899 w_k1_AXP_s013
                                 + 0.794915 w_k1_DIS_s013
                                 + 0.913968 w_k1_FDX_s013
                                 + 0.755365 w_k1_IBM_s013
                                 + 0.914481 w_k1_INTC_s013
                                 + 0.703874 w_k1_KO_s013
                                 + 0.358859 w_k1_MAT_s013
                                 + 0.804278 w_k1_MSFT_s013
                                 + 0.847974 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.07927 w_k1_ADM_s014
                                 + 1.30593 w_k1_AXP_s014
                                 + 1.29552 w_k1_DIS_s014
                                 + 1.35531 w_k1_FDX_s014
                                 + 0.860306 w_k1_IBM_s014
                                 + 0.993035 w_k1_INTC_s014
                                 + 1.11814 w_k1_KO_s014 + 1.36478 w_k1_MAT_s014
                                 + 1.20495 w_k1_MSFT_s014
                                 + 1.18988 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.14311 w_k1_ADM_s014
                                 + 1.43236 w_k1_AXP_s014
                                 + 1.28277 w_k1_DIS_s014
                                 + 1.12775 w_k1_FDX_s014
                                 + 1.54083 w_k1_IBM_s014
                                 + 0.903268 w_k1_INTC_s014
                                 + 1.46554 w_k1_KO_s014 + 2.27667 w_k1_MAT_s014
                                 + 1.06213 w_k1_MSFT_s014
                                 + 1.21511 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.02651 w_k1_ADM_s014
                                 + 0.744871 w_k1_AXP_s014
                                 + 0.957786 w_k1_DIS_s014
                                 + 1.03131 w_k1_FDX_s014
                                 + 0.753707 w_k1_IBM_s014
                                 + 1.01105 w_k1_INTC_s014
                                 + 0.879796 w_k1_KO_s014
                                 + 0.59497 w_k1_MAT_s014
                                 + 1.05748 w_k1_MSFT_s014
                                 + 0.920248 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.796313 w_k1_ADM_s014
                                 + 0.731807 w_k1_AXP_s014
                                 + 0.769297 w_k1_DIS_s014
                                 + 0.827207 w_k1_FDX_s014
                                 + 0.728674 w_k1_IBM_s014
                                 + 0.754955 w_k1_INTC_s014
                                 + 0.944326 w_k1_KO_s014
                                 + 0.812388 w_k1_MAT_s014
                                 + 0.866167 w_k1_MSFT_s014
                                 + 0.914452 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.11332 w_k1_ADM_s014
                                 + 1.02918 w_k1_AXP_s014
                                 + 1.52416 w_k1_DIS_s014
                                 + 1.18032 w_k1_FDX_s014
                                 + 1.18224 w_k1_IBM_s014
                                 + 0.940825 w_k1_INTC_s014
                                 + 1.23498 w_k1_KO_s014 + 1.03077 w_k1_MAT_s014
                                 + 1.19128 w_k1_MSFT_s014
                                 + 1.13385 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.19258 w_k1_ADM_s014
                                 + 0.83161 w_k1_AXP_s014
                                 + 0.797812 w_k1_DIS_s014
                                 + 1.00288 w_k1_FDX_s014
                                 + 0.627238 w_k1_IBM_s014
                                 + 0.753648 w_k1_INTC_s014
                                 + 0.830972 w_k1_KO_s014
                                 + 0.701961 w_k1_MAT_s014
                                 + 0.864454 w_k1_MSFT_s014
                                 + 0.845822 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.846054 w_k1_ADM_s014
                                 + 0.623011 w_k1_AXP_s014
                                 + 0.683434 w_k1_DIS_s014
                                 + 0.746442 w_k1_FDX_s014
                                 + 0.934794 w_k1_IBM_s014
                                 + 0.663705 w_k1_INTC_s014
                                 + 0.848973 w_k1_KO_s014
                                 + 1.04839 w_k1_MAT_s014
                                 + 0.76435 w_k1_MSFT_s014
                                 + 0.904495 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.846432 w_k1_ADM_s014
                                 + 0.93113 w_k1_AXP_s014
                                 + 0.970051 w_k1_DIS_s014
                                 + 0.917289 w_k1_FDX_s014
                                 + 1.17139 w_k1_IBM_s014
                                 + 0.895139 w_k1_INTC_s014
                                 + 1.09013 w_k1_KO_s014 + 1.58604 w_k1_MAT_s014
                                 + 0.899481 w_k1_MSFT_s014
                                 + 0.951433 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.19229 w_k1_ADM_s015
                                 + 1.92847 w_k1_AXP_s015
                                 + 1.43532 w_k1_DIS_s015
                                 + 1.11907 w_k1_FDX_s015
                                 + 1.65085 w_k1_IBM_s015
                                 + 1.06194 w_k1_INTC_s015
                                 + 1.19399 w_k1_KO_s015 + 1.88434 w_k1_MAT_s015
                                 + 1.04469 w_k1_MSFT_s015
                                 + 1.14176 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.07285 w_k1_ADM_s015
                                 + 1.3336 w_k1_AXP_s015 + 1.60365 w_k1_DIS_s015
                                 + 1.26689 w_k1_FDX_s015
                                 + 1.16467 w_k1_IBM_s015
                                 + 1.59118 w_k1_INTC_s015
                                 + 1.40887 w_k1_KO_s015 + 1.1221 w_k1_MAT_s015
                                 + 1.29464 w_k1_MSFT_s015
                                 + 1.10366 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.966891 w_k1_ADM_s015
                                 + 1.13004 w_k1_AXP_s015 + 1.2709 w_k1_DIS_s015
                                 + 1.07318 w_k1_FDX_s015
                                 + 1.38776 w_k1_IBM_s015
                                 + 1.36956 w_k1_INTC_s015
                                 + 1.16058 w_k1_KO_s015 + 1.37305 w_k1_MAT_s015
                                 + 1.01453 w_k1_MSFT_s015
                                 + 1.18788 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.01492 w_k1_ADM_s015
                                 + 1.26474 w_k1_AXP_s015
                                 + 1.12124 w_k1_DIS_s015
                                 + 1.08339 w_k1_FDX_s015
                                 + 1.05681 w_k1_IBM_s015
                                 + 0.965723 w_k1_INTC_s015
                                 + 1.15488 w_k1_KO_s015 + 1.26589 w_k1_MAT_s015
                                 + 1.01004 w_k1_MSFT_s015
                                 + 1.14202 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.906467 w_k1_ADM_s015
                                 + 0.982171 w_k1_AXP_s015
                                 + 1.20209 w_k1_DIS_s015
                                 + 1.19461 w_k1_FDX_s015
                                 + 0.867012 w_k1_IBM_s015
                                 + 1.13311 w_k1_INTC_s015
                                 + 0.966081 w_k1_KO_s015
                                 + 0.909987 w_k1_MAT_s015
                                 + 1.12244 w_k1_MSFT_s015
                                 + 1.02298 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.939408 w_k1_ADM_s015
                                 + 1.1046 w_k1_AXP_s015 + 1.07987 w_k1_DIS_s015
                                 + 0.966583 w_k1_FDX_s015
                                 + 0.823049 w_k1_IBM_s015
                                 + 1.06312 w_k1_INTC_s015
                                 + 1.00902 w_k1_KO_s015
                                 + 0.788878 w_k1_MAT_s015
                                 + 0.944814 w_k1_MSFT_s015
                                 + 0.934435 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.852041 w_k1_ADM_s015
                                 + 0.931816 w_k1_AXP_s015
                                 + 0.849867 w_k1_DIS_s015
                                 + 0.8384 w_k1_FDX_s015 + 1.14444 w_k1_IBM_s015
                                 + 0.9624 w_k1_INTC_s015
                                 + 0.939972 w_k1_KO_s015
                                 + 1.09393 w_k1_MAT_s015
                                 + 0.784998 w_k1_MSFT_s015
                                 + 0.905963 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.809367 w_k1_ADM_s015
                                 + 0.800118 w_k1_AXP_s015
                                 + 0.650247 w_k1_DIS_s015
                                 + 0.857216 w_k1_FDX_s015
                                 + 0.746849 w_k1_IBM_s015
                                 + 0.954773 w_k1_INTC_s015
                                 + 0.778353 w_k1_KO_s015
                                 + 0.774017 w_k1_MAT_s015
                                 + 0.81664 w_k1_MSFT_s015
                                 + 0.89998 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.828424 w_k1_ADM_s016
                                 + 0.805074 w_k1_AXP_s016
                                 + 0.657572 w_k1_DIS_s016
                                 + 0.783099 w_k1_FDX_s016
                                 + 0.940193 w_k1_IBM_s016
                                 + 1.02362 w_k1_INTC_s016
                                 + 0.795677 w_k1_KO_s016
                                 + 0.74297 w_k1_MAT_s016
                                 + 0.949561 w_k1_MSFT_s016
                                 + 0.855136 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.17254 w_k1_ADM_s016
                                 + 1.61995 w_k1_AXP_s016
                                 + 1.92069 w_k1_DIS_s016 + 1.3856 w_k1_FDX_s016
                                 + 1.20125 w_k1_IBM_s016
                                 + 1.30707 w_k1_INTC_s016
                                 + 1.33757 w_k1_KO_s016 + 1.06849 w_k1_MAT_s016
                                 + 1.2627 w_k1_MSFT_s016 + 1.34675 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.1135 w_k1_ADM_s016
                                 + 0.889699 w_k1_AXP_s016
                                 + 0.881241 w_k1_DIS_s016
                                 + 1.00133 w_k1_FDX_s016
                                 + 0.75831 w_k1_IBM_s016
                                 + 0.820522 w_k1_INTC_s016
                                 + 0.812342 w_k1_KO_s016
                                 + 0.599461 w_k1_MAT_s016
                                 + 0.849267 w_k1_MSFT_s016
                                 + 0.872733 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.06429 w_k1_ADM_s016
                                 + 1.34678 w_k1_AXP_s016
                                 + 1.60373 w_k1_DIS_s016
                                 + 1.20521 w_k1_FDX_s016
                                 + 0.894327 w_k1_IBM_s016
                                 + 1.12946 w_k1_INTC_s016
                                 + 1.09674 w_k1_KO_s016 + 1.11219 w_k1_MAT_s016
                                 + 1.32316 w_k1_MSFT_s016
                                 + 1.09535 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.05976 w_k1_ADM_s016
                                 + 0.970952 w_k1_AXP_s016
                                 + 0.911943 w_k1_DIS_s016
                                 + 0.908114 w_k1_FDX_s016
                                 + 1.20187 w_k1_IBM_s016
                                 + 0.930871 w_k1_INTC_s016
                                 + 1.04644 w_k1_KO_s016 + 1.22589 w_k1_MAT_s016
                                 + 0.876422 w_k1_MSFT_s016
                                 + 1.00176 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.22079 w_k1_ADM_s016
                                 + 1.20669 w_k1_AXP_s016
                                 + 1.30603 w_k1_DIS_s016
                                 + 1.18052 w_k1_FDX_s016
                                 + 1.05689 w_k1_IBM_s016
                                 + 1.24718 w_k1_INTC_s016
                                 + 1.02556 w_k1_KO_s016 + 1.06967 w_k1_MAT_s016
                                 + 1.17597 w_k1_MSFT_s016
                                 + 1.03696 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.944318 w_k1_ADM_s016
                                 + 1.31526 w_k1_AXP_s016
                                 + 1.35941 w_k1_DIS_s016
                                 + 1.11319 w_k1_FDX_s016
                                 + 1.37609 w_k1_IBM_s016
                                 + 1.09811 w_k1_INTC_s016
                                 + 1.27045 w_k1_KO_s016 + 1.63168 w_k1_MAT_s016
                                 + 1.11863 w_k1_MSFT_s016
                                 + 1.17401 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.01621 w_k1_ADM_s016
                                 + 1.13385 w_k1_AXP_s016
                                 + 1.21167 w_k1_DIS_s016
                                 + 1.01093 w_k1_FDX_s016
                                 + 1.77986 w_k1_IBM_s016
                                 + 0.953208 w_k1_INTC_s016
                                 + 1.21413 w_k1_KO_s016 + 1.59987 w_k1_MAT_s016
                                 + 0.947445 w_k1_MSFT_s016
                                 + 1.07758 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.18491 w_k1_ADM_s017
                                 + 0.796285 w_k1_AXP_s017
                                 + 0.940455 w_k1_DIS_s017
                                 + 0.903543 w_k1_FDX_s017
                                 + 1.33265 w_k1_IBM_s017
                                 + 0.839655 w_k1_INTC_s017
                                 + 1.26108 w_k1_KO_s017 + 1.2037 w_k1_MAT_s017
                                 + 1.02376 w_k1_MSFT_s017
                                 + 0.997087 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.10671 w_k1_ADM_s017
                                 + 0.922649 w_k1_AXP_s017
                                 + 1.05652 w_k1_DIS_s017
                                 + 1.06268 w_k1_FDX_s017
                                 + 0.974407 w_k1_IBM_s017
                                 + 0.808822 w_k1_INTC_s017
                                 + 0.981027 w_k1_KO_s017
                                 + 0.762509 w_k1_MAT_s017
                                 + 0.904963 w_k1_MSFT_s017
                                 + 1.07445 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.80723 w_k1_ADM_s017
                                 + 0.563762 w_k1_AXP_s017
                                 + 0.585142 w_k1_DIS_s017
                                 + 0.671975 w_k1_FDX_s017
                                 + 0.849333 w_k1_IBM_s017
                                 + 0.734707 w_k1_INTC_s017
                                 + 0.865835 w_k1_KO_s017
                                 + 0.934127 w_k1_MAT_s017
                                 + 0.691314 w_k1_MSFT_s017
                                 + 0.861438 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.32086 w_k1_ADM_s017
                                 + 1.31995 w_k1_AXP_s017
                                 + 1.31151 w_k1_DIS_s017
                                 + 1.27421 w_k1_FDX_s017
                                 + 1.11024 w_k1_IBM_s017
                                 + 1.07014 w_k1_INTC_s017
                                 + 1.10326 w_k1_KO_s017
                                 + 0.987057 w_k1_MAT_s017
                                 + 1.08065 w_k1_MSFT_s017
                                 + 1.24639 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.932763 w_k1_ADM_s017
                                 + 0.619084 w_k1_AXP_s017
                                 + 0.907153 w_k1_DIS_s017
                                 + 0.991696 w_k1_FDX_s017
                                 + 0.61876 w_k1_IBM_s017
                                 + 0.836711 w_k1_INTC_s017
                                 + 0.897089 w_k1_KO_s017
                                 + 0.59822 w_k1_MAT_s017
                                 + 0.905662 w_k1_MSFT_s017
                                 + 0.975104 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.942334 w_k1_ADM_s017
                                 + 0.810845 w_k1_AXP_s017
                                 + 0.8154 w_k1_DIS_s017 + 0.91735 w_k1_FDX_s017
                                 + 0.878366 w_k1_IBM_s017
                                 + 0.865615 w_k1_INTC_s017
                                 + 0.82738 w_k1_KO_s017
                                 + 0.972909 w_k1_MAT_s017
                                 + 0.785798 w_k1_MSFT_s017
                                 + 0.926716 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.862575 w_k1_ADM_s017
                                 + 0.687385 w_k1_AXP_s017
                                 + 0.698912 w_k1_DIS_s017
                                 + 0.783257 w_k1_FDX_s017
                                 + 0.730069 w_k1_IBM_s017
                                 + 0.728952 w_k1_INTC_s017
                                 + 0.889345 w_k1_KO_s017
                                 + 0.962789 w_k1_MAT_s017
                                 + 0.827675 w_k1_MSFT_s017
                                 + 0.909764 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.34691 w_k1_ADM_s017
                                 + 1.29108 w_k1_AXP_s017
                                 + 1.69208 w_k1_DIS_s017
                                 + 1.36673 w_k1_FDX_s017
                                 + 1.50542 w_k1_IBM_s017
                                 + 0.956432 w_k1_INTC_s017
                                 + 1.43338 w_k1_KO_s017 + 1.8555 w_k1_MAT_s017
                                 + 1.21141 w_k1_MSFT_s017
                                 + 1.25627 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.13895 w_k1_ADM_s100
                                 + 1.10293 w_k1_AXP_s100
                                 + 1.05424 w_k1_DIS_s100
                                 + 0.932223 w_k1_FDX_s100
                                 + 1.05426 w_k1_IBM_s100
                                 + 1.14042 w_k1_INTC_s100
                                 + 0.998869 w_k1_KO_s100
                                 + 0.794356 w_k1_MAT_s100
                                 + 0.827936 w_k1_MSFT_s100
                                 + 1.08734 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.963618 w_k1_ADM_s100
                                 + 0.977452 w_k1_AXP_s100
                                 + 1.42864 w_k1_DIS_s100
                                 + 1.19415 w_k1_FDX_s100
                                 + 1.06558 w_k1_IBM_s100
                                 + 1.50204 w_k1_INTC_s100
                                 + 1.23674 w_k1_KO_s100
                                 + 0.948183 w_k1_MAT_s100
                                 + 1.20764 w_k1_MSFT_s100
                                 + 1.07619 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.94167 w_k1_ADM_s100
                                 + 1.08172 w_k1_AXP_s100
                                 + 1.11182 w_k1_DIS_s100
                                 + 0.991932 w_k1_FDX_s100
                                 + 1.33392 w_k1_IBM_s100
                                 + 1.15432 w_k1_INTC_s100
                                 + 1.29175 w_k1_KO_s100 + 1.42247 w_k1_MAT_s100
                                 + 0.982874 w_k1_MSFT_s100
                                 + 1.09202 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.755661 w_k1_ADM_s100
                                 + 1.10195 w_k1_AXP_s100
                                 + 0.737508 w_k1_DIS_s100
                                 + 0.925563 w_k1_FDX_s100
                                 + 0.897011 w_k1_IBM_s100
                                 + 1.05933 w_k1_INTC_s100
                                 + 0.734006 w_k1_KO_s100
                                 + 1.34316 w_k1_MAT_s100
                                 + 0.796555 w_k1_MSFT_s100
                                 + 1.01595 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.832201 w_k1_ADM_s100
                                 + 0.978467 w_k1_AXP_s100
                                 + 0.902126 w_k1_DIS_s100
                                 + 1.05082 w_k1_FDX_s100
                                 + 0.708495 w_k1_IBM_s100
                                 + 1.09844 w_k1_INTC_s100
                                 + 0.870815 w_k1_KO_s100
                                 + 0.807095 w_k1_MAT_s100
                                 + 0.87047 w_k1_MSFT_s100
                                 + 0.915714 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.809314 w_k1_ADM_s100
                                 + 0.666263 w_k1_AXP_s100
                                 + 0.704591 w_k1_DIS_s100
                                 + 0.77718 w_k1_FDX_s100
                                 + 0.789094 w_k1_IBM_s100
                                 + 0.856696 w_k1_INTC_s100
                                 + 0.813421 w_k1_KO_s100
                                 + 0.86852 w_k1_MAT_s100
                                 + 0.806657 w_k1_MSFT_s100
                                 + 0.919908 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.934358 w_k1_ADM_s100
                                 + 1.06271 w_k1_AXP_s100
                                 + 0.940561 w_k1_DIS_s100
                                 + 1.04456 w_k1_FDX_s100
                                 + 1.18033 w_k1_IBM_s100
                                 + 1.21115 w_k1_INTC_s100
                                 + 1.10525 w_k1_KO_s100 + 1.18183 w_k1_MAT_s100
                                 + 0.93117 w_k1_MSFT_s100
                                 + 1.04552 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.2124 w_k1_ADM_s100
                                 + 1.57674 w_k1_AXP_s100 + 2.0101 w_k1_DIS_s100
                                 + 1.52825 w_k1_FDX_s100
                                 + 1.31563 w_k1_IBM_s100
                                 + 1.70241 w_k1_INTC_s100
                                 + 1.38221 w_k1_KO_s100 + 1.08121 w_k1_MAT_s100
                                 + 1.49783 w_k1_MSFT_s100
                                 + 1.13919 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.817257 w_k1_ADM_s101
                                 + 0.631666 w_k1_AXP_s101
                                 + 0.6619 w_k1_DIS_s101
                                 + 0.791052 w_k1_FDX_s101 + 0.758 w_k1_IBM_s101
                                 + 0.733469 w_k1_INTC_s101
                                 + 0.838146 w_k1_KO_s101
                                 + 0.66524 w_k1_MAT_s101
                                 + 0.77927 w_k1_MSFT_s101
                                 + 0.89968 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.40014 w_k1_ADM_s101
                                 + 1.47467 w_k1_AXP_s101
                                 + 1.62921 w_k1_DIS_s101
                                 + 1.40211 w_k1_FDX_s101
                                 + 1.45642 w_k1_IBM_s101
                                 + 1.38951 w_k1_INTC_s101
                                 + 1.55067 w_k1_KO_s101 + 1.68787 w_k1_MAT_s101
                                 + 1.4448 w_k1_MSFT_s101 + 1.25659 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.06734 w_k1_ADM_s101
                                 + 0.949852 w_k1_AXP_s101
                                 + 1.05148 w_k1_DIS_s101
                                 + 0.979375 w_k1_FDX_s101
                                 + 1.25737 w_k1_IBM_s101
                                 + 1.06951 w_k1_INTC_s101
                                 + 1.23871 w_k1_KO_s101 + 0.91831 w_k1_MAT_s101
                                 + 0.858769 w_k1_MSFT_s101
                                 + 1.1044 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.810957 w_k1_ADM_s101
                                 + 0.865268 w_k1_AXP_s101
                                 + 0.926316 w_k1_DIS_s101
                                 + 0.780752 w_k1_FDX_s101
                                 + 1.50555 w_k1_IBM_s101
                                 + 0.968551 w_k1_INTC_s101
                                 + 1.17533 w_k1_KO_s101 + 1.36059 w_k1_MAT_s101
                                 + 0.774994 w_k1_MSFT_s101
                                 + 1.09981 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.937879 w_k1_ADM_s101
                                 + 1.22671 w_k1_AXP_s101
                                 + 0.700748 w_k1_DIS_s101
                                 + 0.982444 w_k1_FDX_s101
                                 + 1.03238 w_k1_IBM_s101
                                 + 1.05548 w_k1_INTC_s101
                                 + 0.85956 w_k1_KO_s101 + 1.30012 w_k1_MAT_s101
                                 + 0.778493 w_k1_MSFT_s101
                                 + 1.02727 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.783166 w_k1_ADM_s101
                                 + 0.870126 w_k1_AXP_s101
                                 + 0.928956 w_k1_DIS_s101
                                 + 1.00495 w_k1_FDX_s101
                                 + 0.930939 w_k1_IBM_s101
                                 + 1.19198 w_k1_INTC_s101
                                 + 0.996114 w_k1_KO_s101
                                 + 1.06657 w_k1_MAT_s101
                                 + 1.02659 w_k1_MSFT_s101
                                 + 1.09747 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.858486 w_k1_ADM_s101
                                 + 0.962491 w_k1_AXP_s101
                                 + 1.44119 w_k1_DIS_s101
                                 + 1.22914 w_k1_FDX_s101
                                 + 0.940745 w_k1_IBM_s101
                                 + 1.30807 w_k1_INTC_s101
                                 + 1.06345 w_k1_KO_s101
                                 + 0.924752 w_k1_MAT_s101
                                 + 1.1461 w_k1_MSFT_s101 + 1.05199 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.893377 w_k1_ADM_s101
                                 + 1.07101 w_k1_AXP_s101
                                 + 1.15914 w_k1_DIS_s101
                                 + 1.10038 w_k1_FDX_s101
                                 + 0.763116 w_k1_IBM_s101
                                 + 1.06978 w_k1_INTC_s101
                                 + 1.00376 w_k1_KO_s101
                                 + 0.882666 w_k1_MAT_s101
                                 + 1.03829 w_k1_MSFT_s101
                                 + 1.06627 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.25863 w_k1_ADM_s102
                                 + 1.20492 w_k1_AXP_s102
                                 + 1.09546 w_k1_DIS_s102
                                 + 1.30264 w_k1_FDX_s102
                                 + 0.740733 w_k1_IBM_s102
                                 + 1.01183 w_k1_INTC_s102
                                 + 0.814449 w_k1_KO_s102
                                 + 0.962973 w_k1_MAT_s102
                                 + 1.11391 w_k1_MSFT_s102
                                 + 0.949635 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.897883 w_k1_ADM_s102
                                 + 0.725092 w_k1_AXP_s102
                                 + 0.497246 w_k1_DIS_s102
                                 + 0.671509 w_k1_FDX_s102
                                 + 0.753185 w_k1_IBM_s102
                                 + 0.766425 w_k1_INTC_s102 + 0.696 w_k1_KO_s102
                                 + 0.52155 w_k1_MAT_s102
                                 + 0.808979 w_k1_MSFT_s102
                                 + 0.783634 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.893581 w_k1_ADM_s102
                                 + 1.06599 w_k1_AXP_s102
                                 + 0.826496 w_k1_DIS_s102
                                 + 0.959582 w_k1_FDX_s102
                                 + 1.1411 w_k1_IBM_s102
                                 + 0.79304 w_k1_INTC_s102
                                 + 0.832842 w_k1_KO_s102
                                 + 0.902997 w_k1_MAT_s102
                                 + 0.862271 w_k1_MSFT_s102
                                 + 0.887501 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.960351 w_k1_ADM_s102
                                 + 0.942598 w_k1_AXP_s102
                                 + 0.656938 w_k1_DIS_s102
                                 + 0.919998 w_k1_FDX_s102
                                 + 0.619792 w_k1_IBM_s102
                                 + 0.830539 w_k1_INTC_s102
                                 + 0.648127 w_k1_KO_s102
                                 + 0.565379 w_k1_MAT_s102
                                 + 0.777743 w_k1_MSFT_s102
                                 + 0.797788 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.09558 w_k1_ADM_s102
                                 + 1.0553 w_k1_AXP_s102
                                 + 0.879214 w_k1_DIS_s102
                                 + 1.0101 w_k1_FDX_s102
                                 + 0.971577 w_k1_IBM_s102
                                 + 0.772326 w_k1_INTC_s102
                                 + 0.853175 w_k1_KO_s102
                                 + 0.818666 w_k1_MAT_s102
                                 + 1.00503 w_k1_MSFT_s102
                                 + 0.937898 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.22838 w_k1_ADM_s102
                                 + 1.26509 w_k1_AXP_s102
                                 + 1.41294 w_k1_DIS_s102
                                 + 1.24634 w_k1_FDX_s102
                                 + 1.15461 w_k1_IBM_s102
                                 + 1.05267 w_k1_INTC_s102
                                 + 1.17541 w_k1_KO_s102 + 1.06734 w_k1_MAT_s102
                                 + 1.44543 w_k1_MSFT_s102
                                 + 1.12386 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.08797 w_k1_ADM_s102
                                 + 0.815514 w_k1_AXP_s102
                                 + 0.79277 w_k1_DIS_s102
                                 + 0.793778 w_k1_FDX_s102
                                 + 1.1037 w_k1_IBM_s102
                                 + 0.76504 w_k1_INTC_s102
                                 + 0.824587 w_k1_KO_s102
                                 + 0.67185 w_k1_MAT_s102
                                 + 0.915838 w_k1_MSFT_s102
                                 + 0.857895 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.06625 w_k1_ADM_s102
                                 + 1.20662 w_k1_AXP_s102
                                 + 1.12874 w_k1_DIS_s102
                                 + 0.964603 w_k1_FDX_s102
                                 + 1.5355 w_k1_IBM_s102
                                 + 1.02258 w_k1_INTC_s102
                                 + 1.19373 w_k1_KO_s102 + 1.53454 w_k1_MAT_s102
                                 + 1.03941 w_k1_MSFT_s102
                                 + 1.12793 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.11865 w_k1_ADM_s103
                                 + 1.05408 w_k1_AXP_s103
                                 + 0.953681 w_k1_DIS_s103
                                 + 1.06998 w_k1_FDX_s103
                                 + 0.696646 w_k1_IBM_s103
                                 + 1.29311 w_k1_INTC_s103
                                 + 0.881357 w_k1_KO_s103
                                 + 0.43629 w_k1_MAT_s103
                                 + 1.03733 w_k1_MSFT_s103
                                 + 0.880068 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.19338 w_k1_ADM_s103
                                 + 1.27912 w_k1_AXP_s103
                                 + 1.03192 w_k1_DIS_s103
                                 + 1.02807 w_k1_FDX_s103
                                 + 1.19217 w_k1_IBM_s103
                                 + 1.0747 w_k1_INTC_s103 + 1.21882 w_k1_KO_s103
                                 + 1.21982 w_k1_MAT_s103
                                 + 0.910918 w_k1_MSFT_s103
                                 + 1.1385 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.849803 w_k1_ADM_s103
                                 + 0.812223 w_k1_AXP_s103
                                 + 0.714944 w_k1_DIS_s103
                                 + 0.798738 w_k1_FDX_s103
                                 + 0.798789 w_k1_IBM_s103
                                 + 0.935577 w_k1_INTC_s103
                                 + 0.828497 w_k1_KO_s103
                                 + 0.801477 w_k1_MAT_s103
                                 + 0.902313 w_k1_MSFT_s103
                                 + 0.8459 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.09023 w_k1_ADM_s103
                                 + 1.3118 w_k1_AXP_s103 + 1.51121 w_k1_DIS_s103
                                 + 1.25189 w_k1_FDX_s103
                                 + 1.31799 w_k1_IBM_s103
                                 + 1.25956 w_k1_INTC_s103
                                 + 1.38179 w_k1_KO_s103 + 1.57424 w_k1_MAT_s103
                                 + 1.22645 w_k1_MSFT_s103
                                 + 1.25789 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.28571 w_k1_ADM_s103
                                 + 1.08333 w_k1_AXP_s103
                                 + 1.39972 w_k1_DIS_s103
                                 + 1.15169 w_k1_FDX_s103
                                 + 0.836212 w_k1_IBM_s103
                                 + 1.32064 w_k1_INTC_s103
                                 + 1.06254 w_k1_KO_s103
                                 + 0.691664 w_k1_MAT_s103
                                 + 1.16759 w_k1_MSFT_s103
                                 + 1.03365 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.878444 w_k1_ADM_s103
                                 + 1.11052 w_k1_AXP_s103
                                 + 0.896738 w_k1_DIS_s103
                                 + 0.865553 w_k1_FDX_s103
                                 + 1.39727 w_k1_IBM_s103
                                 + 1.06355 w_k1_INTC_s103
                                 + 1.18919 w_k1_KO_s103 + 1.3889 w_k1_MAT_s103
                                 + 1.02503 w_k1_MSFT_s103
                                 + 1.04418 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.38685 w_k1_ADM_s103
                                 + 1.96786 w_k1_AXP_s103
                                 + 1.87999 w_k1_DIS_s103
                                 + 1.51226 w_k1_FDX_s103
                                 + 1.12722 w_k1_IBM_s103
                                 + 1.40926 w_k1_INTC_s103
                                 + 1.30991 w_k1_KO_s103 + 1.08023 w_k1_MAT_s103
                                 + 1.26549 w_k1_MSFT_s103
                                 + 1.19214 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.911378 w_k1_ADM_s103
                                 + 0.975117 w_k1_AXP_s103
                                 + 1.11596 w_k1_DIS_s103 + 1.0019 w_k1_FDX_s103
                                 + 1.09201 w_k1_IBM_s103
                                 + 1.17988 w_k1_INTC_s103
                                 + 1.04173 w_k1_KO_s103
                                 + 0.994965 w_k1_MAT_s103
                                 + 1.15367 w_k1_MSFT_s103
                                 + 1.06642 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.7677 w_k1_ADM_s104
                                 + 0.847708 w_k1_AXP_s104
                                 + 0.704986 w_k1_DIS_s104
                                 + 0.790111 w_k1_FDX_s104
                                 + 0.92296 w_k1_IBM_s104
                                 + 0.915547 w_k1_INTC_s104
                                 + 0.79669 w_k1_KO_s104
                                 + 0.806902 w_k1_MAT_s104
                                 + 0.736656 w_k1_MSFT_s104
                                 + 0.746539 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.979675 w_k1_ADM_s104
                                 + 0.943873 w_k1_AXP_s104
                                 + 0.929203 w_k1_DIS_s104
                                 + 1.08158 w_k1_FDX_s104
                                 + 0.739942 w_k1_IBM_s104
                                 + 1.10941 w_k1_INTC_s104
                                 + 0.852971 w_k1_KO_s104
                                 + 0.683037 w_k1_MAT_s104
                                 + 1.05087 w_k1_MSFT_s104
                                 + 0.861113 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.13814 w_k1_ADM_s104
                                 + 1.76382 w_k1_AXP_s104 + 1.3888 w_k1_DIS_s104
                                 + 1.39398 w_k1_FDX_s104
                                 + 1.14369 w_k1_IBM_s104
                                 + 1.29252 w_k1_INTC_s104
                                 + 1.05957 w_k1_KO_s104 + 1.44983 w_k1_MAT_s104
                                 + 1.12769 w_k1_MSFT_s104
                                 + 0.998922 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.977143 w_k1_ADM_s104
                                 + 1.42997 w_k1_AXP_s104
                                 + 1.50369 w_k1_DIS_s104
                                 + 1.20341 w_k1_FDX_s104
                                 + 1.24729 w_k1_IBM_s104
                                 + 1.57265 w_k1_INTC_s104
                                 + 1.32265 w_k1_KO_s104 + 1.15856 w_k1_MAT_s104
                                 + 1.30671 w_k1_MSFT_s104
                                 + 1.09683 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.90333 w_k1_ADM_s104
                                 + 1.01858 w_k1_AXP_s104
                                 + 0.987381 w_k1_DIS_s104
                                 + 1.04097 w_k1_FDX_s104
                                 + 0.954671 w_k1_IBM_s104
                                 + 1.17608 w_k1_INTC_s104
                                 + 1.00343 w_k1_KO_s104
                                 + 0.885527 w_k1_MAT_s104
                                 + 1.01977 w_k1_MSFT_s104
                                 + 1.00361 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.05657 w_k1_ADM_s104
                                 + 1.1788 w_k1_AXP_s104 + 1.52226 w_k1_DIS_s104
                                 + 1.3164 w_k1_FDX_s104 + 1.00173 w_k1_IBM_s104
                                 + 1.46592 w_k1_INTC_s104
                                 + 1.22089 w_k1_KO_s104 + 1.11699 w_k1_MAT_s104
                                 + 1.42885 w_k1_MSFT_s104
                                 + 1.05082 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.614079 w_k1_ADM_s104
                                 + 0.925399 w_k1_AXP_s104
                                 + 1.02414 w_k1_DIS_s104
                                 + 0.954801 w_k1_FDX_s104
                                 + 1.34297 w_k1_IBM_s104
                                 + 1.28474 w_k1_INTC_s104
                                 + 1.19749 w_k1_KO_s104 + 1.15125 w_k1_MAT_s104
                                 + 1.13484 w_k1_MSFT_s104
                                 + 0.876129 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.19761 w_k1_ADM_s104
                                 + 1.54539 w_k1_AXP_s104
                                 + 1.37908 w_k1_DIS_s104
                                 + 1.24331 w_k1_FDX_s104
                                 + 1.76474 w_k1_IBM_s104
                                 + 1.19636 w_k1_INTC_s104
                                 + 1.55488 w_k1_KO_s104 + 2.10443 w_k1_MAT_s104
                                 + 1.11701 w_k1_MSFT_s104
                                 + 1.22934 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.08519 w_k1_ADM_s105
                                 + 0.869333 w_k1_AXP_s105
                                 + 0.984307 w_k1_DIS_s105
                                 + 0.927508 w_k1_FDX_s105
                                 + 1.14081 w_k1_IBM_s105
                                 + 1.14561 w_k1_INTC_s105
                                 + 1.01996 w_k1_KO_s105
                                 + 0.846022 w_k1_MAT_s105
                                 + 1.10272 w_k1_MSFT_s105 + 1.0637 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.06233 w_k1_ADM_s105
                                 + 0.762972 w_k1_AXP_s105
                                 + 0.714962 w_k1_DIS_s105
                                 + 0.810906 w_k1_FDX_s105
                                 + 0.94165 w_k1_IBM_s105
                                 + 0.839606 w_k1_INTC_s105
                                 + 0.796889 w_k1_KO_s105
                                 + 0.723927 w_k1_MAT_s105
                                 + 0.848575 w_k1_MSFT_s105
                                 + 0.888195 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.11511 w_k1_ADM_s105
                                 + 0.938376 w_k1_AXP_s105
                                 + 0.886336 w_k1_DIS_s105
                                 + 0.966521 w_k1_FDX_s105
                                 + 0.725283 w_k1_IBM_s105
                                 + 1.05573 w_k1_INTC_s105
                                 + 0.833981 w_k1_KO_s105
                                 + 0.686765 w_k1_MAT_s105
                                 + 0.954143 w_k1_MSFT_s105
                                 + 1.01962 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.26624 w_k1_ADM_s105
                                 + 1.32283 w_k1_AXP_s105
                                 + 0.925778 w_k1_DIS_s105
                                 + 1.04702 w_k1_FDX_s105
                                 + 0.972484 w_k1_IBM_s105
                                 + 1.03804 w_k1_INTC_s105
                                 + 1.05466 w_k1_KO_s105
                                 + 0.990606 w_k1_MAT_s105
                                 + 1.00566 w_k1_MSFT_s105 + 1.0693 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.02313 w_k1_ADM_s105
                                 + 1.34092 w_k1_AXP_s105
                                 + 1.11289 w_k1_DIS_s105
                                 + 1.04545 w_k1_FDX_s105
                                 + 1.46374 w_k1_IBM_s105
                                 + 0.998062 w_k1_INTC_s105
                                 + 1.2677 w_k1_KO_s105 + 1.38179 w_k1_MAT_s105
                                 + 0.96598 w_k1_MSFT_s105
                                 + 1.03126 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.5788 w_k1_ADM_s105
                                 + 1.58347 w_k1_AXP_s105
                                 + 1.43759 w_k1_DIS_s105
                                 + 1.36678 w_k1_FDX_s105
                                 + 1.26695 w_k1_IBM_s105
                                 + 1.24327 w_k1_INTC_s105
                                 + 1.41793 w_k1_KO_s105 + 1.25805 w_k1_MAT_s105
                                 + 1.07471 w_k1_MSFT_s105
                                 + 1.28038 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.39144 w_k1_ADM_s105
                                 + 1.1595 w_k1_AXP_s105 + 1.31929 w_k1_DIS_s105
                                 + 1.27317 w_k1_FDX_s105
                                 + 0.854239 w_k1_IBM_s105
                                 + 1.39849 w_k1_INTC_s105
                                 + 1.02914 w_k1_KO_s105
                                 + 0.658497 w_k1_MAT_s105
                                 + 1.24522 w_k1_MSFT_s105 + 1.1323 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.899932 w_k1_ADM_s105
                                 + 0.701721 w_k1_AXP_s105
                                 + 0.641078 w_k1_DIS_s105
                                 + 0.663664 w_k1_FDX_s105
                                 + 0.995429 w_k1_IBM_s105
                                 + 0.788334 w_k1_INTC_s105
                                 + 0.785225 w_k1_KO_s105
                                 + 0.63508 w_k1_MAT_s105
                                 + 0.687741 w_k1_MSFT_s105
                                 + 0.771104 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.12307 w_k1_ADM_s106
                                 + 1.04012 w_k1_AXP_s106
                                 + 1.05843 w_k1_DIS_s106 + 1.0659 w_k1_FDX_s106
                                 + 0.823759 w_k1_IBM_s106
                                 + 1.03571 w_k1_INTC_s106
                                 + 0.992334 w_k1_KO_s106
                                 + 1.02433 w_k1_MAT_s106
                                 + 0.895862 w_k1_MSFT_s106
                                 + 1.06617 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.775037 w_k1_ADM_s106
                                 + 1.00378 w_k1_AXP_s106
                                 + 0.958296 w_k1_DIS_s106
                                 + 1.03733 w_k1_FDX_s106
                                 + 0.663164 w_k1_IBM_s106
                                 + 1.02831 w_k1_INTC_s106
                                 + 0.785445 w_k1_KO_s106
                                 + 0.889221 w_k1_MAT_s106
                                 + 1.02311 w_k1_MSFT_s106
                                 + 0.881062 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.859219 w_k1_ADM_s106
                                 + 1.0227 w_k1_AXP_s106
                                 + 0.862402 w_k1_DIS_s106
                                 + 1.02959 w_k1_FDX_s106
                                 + 0.938267 w_k1_IBM_s106
                                 + 1.07275 w_k1_INTC_s106
                                 + 0.936899 w_k1_KO_s106
                                 + 0.842557 w_k1_MAT_s106
                                 + 0.973347 w_k1_MSFT_s106
                                 + 1.03414 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.872376 w_k1_ADM_s106
                                 + 1.20706 w_k1_AXP_s106
                                 + 1.01357 w_k1_DIS_s106
                                 + 0.923682 w_k1_FDX_s106
                                 + 1.40293 w_k1_IBM_s106
                                 + 1.3655 w_k1_INTC_s106 + 1.32614 w_k1_KO_s106
                                 + 1.70072 w_k1_MAT_s106
                                 + 0.884539 w_k1_MSFT_s106
                                 + 1.00206 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.795306 w_k1_ADM_s106
                                 + 0.661636 w_k1_AXP_s106
                                 + 0.660496 w_k1_DIS_s106
                                 + 0.732087 w_k1_FDX_s106
                                 + 0.737181 w_k1_IBM_s106
                                 + 0.985825 w_k1_INTC_s106
                                 + 0.790156 w_k1_KO_s106
                                 + 0.692666 w_k1_MAT_s106
                                 + 0.798287 w_k1_MSFT_s106
                                 + 0.895715 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.06299 w_k1_ADM_s106
                                 + 1.20581 w_k1_AXP_s106
                                 + 1.24485 w_k1_DIS_s106
                                 + 1.10306 w_k1_FDX_s106
                                 + 1.22932 w_k1_IBM_s106
                                 + 1.2777 w_k1_INTC_s106 + 1.28479 w_k1_KO_s106
                                 + 1.14768 w_k1_MAT_s106
                                 + 1.13041 w_k1_MSFT_s106
                                 + 1.08785 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.76334 w_k1_ADM_s106
                                 + 0.880887 w_k1_AXP_s106
                                 + 0.988782 w_k1_DIS_s106
                                 + 0.842846 w_k1_FDX_s106
                                 + 0.956681 w_k1_IBM_s106
                                 + 1.11126 w_k1_INTC_s106
                                 + 0.888248 w_k1_KO_s106
                                 + 1.10565 w_k1_MAT_s106
                                 + 0.935819 w_k1_MSFT_s106
                                 + 0.929015 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.27929 w_k1_ADM_s106
                                 + 1.36073 w_k1_AXP_s106
                                 + 1.59149 w_k1_DIS_s106
                                 + 1.21646 w_k1_FDX_s106 + 1.0472 w_k1_IBM_s106
                                 + 1.30799 w_k1_INTC_s106
                                 + 1.33469 w_k1_KO_s106
                                 + 0.904225 w_k1_MAT_s106
                                 + 1.2323 w_k1_MSFT_s106 + 1.15497 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.11313 w_k1_ADM_s107
                                 + 1.22463 w_k1_AXP_s107
                                 + 0.944248 w_k1_DIS_s107
                                 + 1.00988 w_k1_FDX_s107
                                 + 1.05516 w_k1_IBM_s107
                                 + 0.915531 w_k1_INTC_s107
                                 + 0.985047 w_k1_KO_s107
                                 + 0.955804 w_k1_MAT_s107
                                 + 0.964359 w_k1_MSFT_s107 + 1.048 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.91749 w_k1_ADM_s107
                                 + 0.935001 w_k1_AXP_s107
                                 + 1.02471 w_k1_DIS_s107
                                 + 1.03898 w_k1_FDX_s107
                                 + 0.858896 w_k1_IBM_s107
                                 + 0.873368 w_k1_INTC_s107
                                 + 1.02632 w_k1_KO_s107
                                 + 0.848839 w_k1_MAT_s107
                                 + 1.02703 w_k1_MSFT_s107
                                 + 0.905767 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.25734 w_k1_ADM_s107
                                 + 1.22623 w_k1_AXP_s107
                                 + 1.20958 w_k1_DIS_s107
                                 + 1.10355 w_k1_FDX_s107
                                 + 1.50106 w_k1_IBM_s107
                                 + 1.10471 w_k1_INTC_s107 + 1.1399 w_k1_KO_s107
                                 + 1.50932 w_k1_MAT_s107
                                 + 1.13853 w_k1_MSFT_s107
                                 + 1.13241 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.51854 w_k1_ADM_s107
                                 + 1.59798 w_k1_AXP_s107
                                 + 1.35331 w_k1_DIS_s107
                                 + 1.32008 w_k1_FDX_s107
                                 + 0.983938 w_k1_IBM_s107
                                 + 1.12961 w_k1_INTC_s107
                                 + 1.17006 w_k1_KO_s107 + 1.10717 w_k1_MAT_s107
                                 + 1.19708 w_k1_MSFT_s107
                                 + 1.14308 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.46085 w_k1_ADM_s107
                                 + 1.63883 w_k1_AXP_s107
                                 + 1.90653 w_k1_DIS_s107
                                 + 1.42997 w_k1_FDX_s107
                                 + 1.06805 w_k1_IBM_s107
                                 + 1.29789 w_k1_INTC_s107
                                 + 1.42501 w_k1_KO_s107 + 1.3309 w_k1_MAT_s107
                                 + 1.60297 w_k1_MSFT_s107 + 1.2994 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.943902 w_k1_ADM_s107
                                 + 1.04852 w_k1_AXP_s107
                                 + 0.72856 w_k1_DIS_s107
                                 + 0.898757 w_k1_FDX_s107
                                 + 0.710926 w_k1_IBM_s107
                                 + 0.956566 w_k1_INTC_s107
                                 + 0.835636 w_k1_KO_s107
                                 + 0.624573 w_k1_MAT_s107
                                 + 0.86755 w_k1_MSFT_s107
                                 + 0.97399 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.841065 w_k1_ADM_s107
                                 + 0.881561 w_k1_AXP_s107
                                 + 0.708706 w_k1_DIS_s107
                                 + 0.808301 w_k1_FDX_s107
                                 + 1.02017 w_k1_IBM_s107
                                 + 0.592472 w_k1_INTC_s107
                                 + 0.819807 w_k1_KO_s107
                                 + 1.23229 w_k1_MAT_s107
                                 + 0.756467 w_k1_MSFT_s107
                                 + 0.838745 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.743467 w_k1_ADM_s107
                                 + 0.78708 w_k1_AXP_s107
                                 + 0.593216 w_k1_DIS_s107
                                 + 0.687051 w_k1_FDX_s107
                                 + 0.728908 w_k1_IBM_s107
                                 + 0.570311 w_k1_INTC_s107
                                 + 0.613484 w_k1_KO_s107
                                 + 0.728139 w_k1_MAT_s107
                                 + 0.615731 w_k1_MSFT_s107
                                 + 0.714749 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.44448 w_k1_ADM_s110
                                 + 1.40759 w_k1_AXP_s110 + 1.5204 w_k1_DIS_s110
                                 + 1.38727 w_k1_FDX_s110
                                 + 1.15516 w_k1_IBM_s110
                                 + 1.15851 w_k1_INTC_s110
                                 + 1.19321 w_k1_KO_s110 + 1.21916 w_k1_MAT_s110
                                 + 1.42561 w_k1_MSFT_s110
                                 + 1.20112 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.14379 w_k1_ADM_s110
                                 + 1.25986 w_k1_AXP_s110
                                 + 1.20304 w_k1_DIS_s110
                                 + 1.03328 w_k1_FDX_s110
                                 + 1.41906 w_k1_IBM_s110
                                 + 0.987347 w_k1_INTC_s110
                                 + 1.20235 w_k1_KO_s110 + 1.42411 w_k1_MAT_s110
                                 + 1.05498 w_k1_MSFT_s110
                                 + 1.13873 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.16215 w_k1_ADM_s110
                                 + 0.995464 w_k1_AXP_s110
                                 + 1.20542 w_k1_DIS_s110
                                 + 1.00899 w_k1_FDX_s110
                                 + 0.959996 w_k1_IBM_s110
                                 + 0.960928 w_k1_INTC_s110
                                 + 1.10906 w_k1_KO_s110 + 1.0338 w_k1_MAT_s110
                                 + 1.00013 w_k1_MSFT_s110
                                 + 0.986883 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.987327 w_k1_ADM_s110
                                 + 0.877626 w_k1_AXP_s110
                                 + 0.834328 w_k1_DIS_s110
                                 + 0.84768 w_k1_FDX_s110
                                 + 0.808501 w_k1_IBM_s110
                                 + 0.933256 w_k1_INTC_s110
                                 + 0.875229 w_k1_KO_s110
                                 + 0.758514 w_k1_MAT_s110
                                 + 0.990763 w_k1_MSFT_s110
                                 + 0.939156 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.651124 w_k1_ADM_s110
                                 + 0.597191 w_k1_AXP_s110
                                 + 0.598051 w_k1_DIS_s110
                                 + 0.649208 w_k1_FDX_s110
                                 + 0.755477 w_k1_IBM_s110
                                 + 0.792168 w_k1_INTC_s110
                                 + 0.779441 w_k1_KO_s110
                                 + 0.882136 w_k1_MAT_s110
                                 + 0.804 w_k1_MSFT_s110 + 0.818583 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.869562 w_k1_ADM_s110
                                 + 0.851069 w_k1_AXP_s110
                                 + 0.851388 w_k1_DIS_s110
                                 + 0.83535 w_k1_FDX_s110
                                 + 1.07845 w_k1_IBM_s110
                                 + 0.929166 w_k1_INTC_s110
                                 + 1.01397 w_k1_KO_s110 + 1.12499 w_k1_MAT_s110
                                 + 0.797223 w_k1_MSFT_s110
                                 + 0.83934 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.990861 w_k1_ADM_s110
                                 + 1.08463 w_k1_AXP_s110
                                 + 0.972121 w_k1_DIS_s110
                                 + 1.07155 w_k1_FDX_s110
                                 + 0.715178 w_k1_IBM_s110
                                 + 0.992098 w_k1_INTC_s110
                                 + 0.882151 w_k1_KO_s110
                                 + 0.946583 w_k1_MAT_s110
                                 + 1.18297 w_k1_MSFT_s110
                                 + 1.02383 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.92999 w_k1_ADM_s110
                                 + 0.760095 w_k1_AXP_s110
                                 + 0.750408 w_k1_DIS_s110
                                 + 0.78736 w_k1_FDX_s110
                                 + 1.06847 w_k1_IBM_s110
                                 + 0.779432 w_k1_INTC_s110
                                 + 0.97754 w_k1_KO_s110 + 1.2886 w_k1_MAT_s110
                                 + 0.94512 w_k1_MSFT_s110
                                 + 0.979756 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.08406 w_k1_ADM_s111
                                 + 1.33929 w_k1_AXP_s111
                                 + 1.01338 w_k1_DIS_s111
                                 + 1.00976 w_k1_FDX_s111
                                 + 1.30306 w_k1_IBM_s111
                                 + 0.881268 w_k1_INTC_s111
                                 + 1.16081 w_k1_KO_s111 + 1.68348 w_k1_MAT_s111
                                 + 0.901034 w_k1_MSFT_s111
                                 + 1.13751 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.2155 w_k1_ADM_s111
                                 + 1.06704 w_k1_AXP_s111
                                 + 0.911755 w_k1_DIS_s111
                                 + 0.996705 w_k1_FDX_s111
                                 + 1.03479 w_k1_IBM_s111
                                 + 0.967142 w_k1_INTC_s111
                                 + 0.970132 w_k1_KO_s111
                                 + 0.952873 w_k1_MAT_s111
                                 + 1.10695 w_k1_MSFT_s111
                                 + 1.05754 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.03405 w_k1_ADM_s111
                                 + 1.31383 w_k1_AXP_s111
                                 + 1.40692 w_k1_DIS_s111
                                 + 1.31577 w_k1_FDX_s111
                                 + 0.952415 w_k1_IBM_s111
                                 + 1.51072 w_k1_INTC_s111
                                 + 1.24994 w_k1_KO_s111 + 1.10439 w_k1_MAT_s111
                                 + 1.30573 w_k1_MSFT_s111
                                 + 1.12992 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.914902 w_k1_ADM_s111
                                 + 0.691287 w_k1_AXP_s111
                                 + 0.693536 w_k1_DIS_s111
                                 + 0.836554 w_k1_FDX_s111
                                 + 0.860608 w_k1_IBM_s111
                                 + 0.827328 w_k1_INTC_s111
                                 + 0.862379 w_k1_KO_s111
                                 + 0.813896 w_k1_MAT_s111
                                 + 0.806842 w_k1_MSFT_s111
                                 + 0.96876 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.833998 w_k1_ADM_s111
                                 + 0.786877 w_k1_AXP_s111
                                 + 0.613541 w_k1_DIS_s111
                                 + 0.831659 w_k1_FDX_s111
                                 + 0.715014 w_k1_IBM_s111
                                 + 0.955659 w_k1_INTC_s111
                                 + 0.813663 w_k1_KO_s111
                                 + 0.524624 w_k1_MAT_s111
                                 + 0.800416 w_k1_MSFT_s111
                                 + 0.90751 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.17108 w_k1_ADM_s111
                                 + 1.21567 w_k1_AXP_s111
                                 + 1.05625 w_k1_DIS_s111
                                 + 1.22609 w_k1_FDX_s111
                                 + 0.872749 w_k1_IBM_s111
                                 + 1.1879 w_k1_INTC_s111
                                 + 0.989538 w_k1_KO_s111
                                 + 0.620208 w_k1_MAT_s111
                                 + 1.14534 w_k1_MSFT_s111
                                 + 1.03898 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.29579 w_k1_ADM_s111
                                 + 1.53068 w_k1_AXP_s111
                                 + 1.47895 w_k1_DIS_s111
                                 + 1.23733 w_k1_FDX_s111
                                 + 1.29692 w_k1_IBM_s111
                                 + 1.22601 w_k1_INTC_s111
                                 + 1.24596 w_k1_KO_s111 + 1.17077 w_k1_MAT_s111
                                 + 1.23711 w_k1_MSFT_s111
                                 + 1.23133 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.856337 w_k1_ADM_s111
                                 + 1.02325 w_k1_AXP_s111
                                 + 1.06531 w_k1_DIS_s111
                                 + 0.980824 w_k1_FDX_s111
                                 + 0.937486 w_k1_IBM_s111
                                 + 1.17991 w_k1_INTC_s111
                                 + 1.02874 w_k1_KO_s111
                                 + 0.702001 w_k1_MAT_s111
                                 + 0.961381 w_k1_MSFT_s111
                                 + 1.06463 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.800215 w_k1_ADM_s112
                                 + 0.747442 w_k1_AXP_s112
                                 + 0.724265 w_k1_DIS_s112
                                 + 0.801692 w_k1_FDX_s112
                                 + 0.723494 w_k1_IBM_s112
                                 + 0.682257 w_k1_INTC_s112
                                 + 0.717725 w_k1_KO_s112 + 1.0804 w_k1_MAT_s112
                                 + 0.79763 w_k1_MSFT_s112
                                 + 0.831872 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17958 w_k1_ADM_s112
                                 + 1.49344 w_k1_AXP_s112
                                 + 1.34766 w_k1_DIS_s112
                                 + 1.29882 w_k1_FDX_s112
                                 + 1.28348 w_k1_IBM_s112
                                 + 1.11194 w_k1_INTC_s112
                                 + 1.17694 w_k1_KO_s112 + 1.69291 w_k1_MAT_s112
                                 + 1.20342 w_k1_MSFT_s112
                                 + 1.12889 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.968194 w_k1_ADM_s112
                                 + 0.817199 w_k1_AXP_s112
                                 + 0.987694 w_k1_DIS_s112
                                 + 1.01818 w_k1_FDX_s112
                                 + 0.743014 w_k1_IBM_s112
                                 + 0.999535 w_k1_INTC_s112
                                 + 0.913343 w_k1_KO_s112
                                 + 0.833612 w_k1_MAT_s112
                                 + 1.00896 w_k1_MSFT_s112
                                 + 0.985569 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.692155 w_k1_ADM_s112
                                 + 0.852462 w_k1_AXP_s112
                                 + 1.05758 w_k1_DIS_s112
                                 + 0.914597 w_k1_FDX_s112
                                 + 1.19106 w_k1_IBM_s112
                                 + 0.957458 w_k1_INTC_s112
                                 + 0.937061 w_k1_KO_s112 + 1.2963 w_k1_MAT_s112
                                 + 1.06603 w_k1_MSFT_s112 + 1.0221 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.20854 w_k1_ADM_s112
                                 + 1.31249 w_k1_AXP_s112
                                 + 1.86025 w_k1_DIS_s112
                                 + 1.42994 w_k1_FDX_s112
                                 + 0.966991 w_k1_IBM_s112
                                 + 1.13336 w_k1_INTC_s112
                                 + 1.20868 w_k1_KO_s112 + 1.47601 w_k1_MAT_s112
                                 + 1.41496 w_k1_MSFT_s112
                                 + 1.22272 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.888103 w_k1_ADM_s112
                                 + 1.10985 w_k1_AXP_s112
                                 + 0.886478 w_k1_DIS_s112
                                 + 0.929646 w_k1_FDX_s112
                                 + 1.10382 w_k1_IBM_s112
                                 + 0.958267 w_k1_INTC_s112
                                 + 0.873588 w_k1_KO_s112
                                 + 2.04562 w_k1_MAT_s112
                                 + 0.818038 w_k1_MSFT_s112
                                 + 0.99687 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.777364 w_k1_ADM_s112
                                 + 0.943782 w_k1_AXP_s112
                                 + 1.32686 w_k1_DIS_s112
                                 + 0.947088 w_k1_FDX_s112
                                 + 1.55517 w_k1_IBM_s112
                                 + 1.22017 w_k1_INTC_s112
                                 + 1.42707 w_k1_KO_s112 + 2.10807 w_k1_MAT_s112
                                 + 1.09092 w_k1_MSFT_s112
                                 + 1.12836 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.04307 w_k1_ADM_s112
                                 + 1.15723 w_k1_AXP_s112 + 1.2179 w_k1_DIS_s112
                                 + 1.19866 w_k1_FDX_s112
                                 + 0.863638 w_k1_IBM_s112
                                 + 1.05164 w_k1_INTC_s112
                                 + 0.95734 w_k1_KO_s112 + 1.11917 w_k1_MAT_s112
                                 + 1.14387 w_k1_MSFT_s112
                                 + 1.03239 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.876093 w_k1_ADM_s113
                                 + 0.497981 w_k1_AXP_s113
                                 + 0.643297 w_k1_DIS_s113
                                 + 0.791427 w_k1_FDX_s113
                                 + 0.586961 w_k1_IBM_s113
                                 + 0.680405 w_k1_INTC_s113
                                 + 0.719723 w_k1_KO_s113
                                 + 0.635026 w_k1_MAT_s113
                                 + 0.820494 w_k1_MSFT_s113
                                 + 0.775163 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.991869 w_k1_ADM_s113
                                 + 0.882826 w_k1_AXP_s113
                                 + 0.899203 w_k1_DIS_s113
                                 + 0.871482 w_k1_FDX_s113
                                 + 1.15859 w_k1_IBM_s113
                                 + 0.817615 w_k1_INTC_s113
                                 + 1.12052 w_k1_KO_s113 + 1.39377 w_k1_MAT_s113
                                 + 0.863758 w_k1_MSFT_s113
                                 + 0.935688 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.05508 w_k1_ADM_s113
                                 + 1.135 w_k1_AXP_s113 + 1.10985 w_k1_DIS_s113
                                 + 1.06073 w_k1_FDX_s113
                                 + 1.04327 w_k1_IBM_s113
                                 + 0.908326 w_k1_INTC_s113
                                 + 1.07344 w_k1_KO_s113 + 1.44112 w_k1_MAT_s113
                                 + 0.924659 w_k1_MSFT_s113
                                 + 0.972785 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.23092 w_k1_ADM_s113
                                 + 1.25635 w_k1_AXP_s113
                                 + 1.53158 w_k1_DIS_s113
                                 + 1.26388 w_k1_FDX_s113
                                 + 1.25731 w_k1_IBM_s113
                                 + 1.0692 w_k1_INTC_s113 + 1.29633 w_k1_KO_s113
                                 + 1.669 w_k1_MAT_s113 + 1.17107 w_k1_MSFT_s113
                                 + 1.14239 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.879287 w_k1_ADM_s113
                                 + 0.898696 w_k1_AXP_s113
                                 + 0.82583 w_k1_DIS_s113
                                 + 0.956602 w_k1_FDX_s113
                                 + 0.806405 w_k1_IBM_s113
                                 + 0.936049 w_k1_INTC_s113
                                 + 0.922306 w_k1_KO_s113
                                 + 1.20915 w_k1_MAT_s113
                                 + 0.919254 w_k1_MSFT_s113
                                 + 0.948882 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.879603 w_k1_ADM_s113
                                 + 1.27646 w_k1_AXP_s113
                                 + 1.19084 w_k1_DIS_s113
                                 + 1.04011 w_k1_FDX_s113
                                 + 1.65634 w_k1_IBM_s113
                                 + 0.862263 w_k1_INTC_s113
                                 + 1.28626 w_k1_KO_s113 + 2.17855 w_k1_MAT_s113
                                 + 0.956543 w_k1_MSFT_s113
                                 + 1.01936 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.03135 w_k1_ADM_s113
                                 + 0.944212 w_k1_AXP_s113
                                 + 1.15075 w_k1_DIS_s113
                                 + 1.09634 w_k1_FDX_s113
                                 + 0.882358 w_k1_IBM_s113
                                 + 1.10156 w_k1_INTC_s113
                                 + 1.11369 w_k1_KO_s113
                                 + 0.963011 w_k1_MAT_s113
                                 + 1.15799 w_k1_MSFT_s113
                                 + 1.10494 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.813565 w_k1_ADM_s113
                                 + 0.726632 w_k1_AXP_s113
                                 + 0.648849 w_k1_DIS_s113
                                 + 0.831219 w_k1_FDX_s113
                                 + 0.851952 w_k1_IBM_s113
                                 + 0.830021 w_k1_INTC_s113
                                 + 0.817724 w_k1_KO_s113
                                 + 0.80118 w_k1_MAT_s113
                                 + 0.815055 w_k1_MSFT_s113
                                 + 0.831844 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.27023 w_k1_ADM_s114
                                 + 1.02624 w_k1_AXP_s114
                                 + 1.54619 w_k1_DIS_s114
                                 + 1.19555 w_k1_FDX_s114
                                 + 1.09056 w_k1_IBM_s114
                                 + 1.24359 w_k1_INTC_s114
                                 + 1.08264 w_k1_KO_s114
                                 + 0.712348 w_k1_MAT_s114
                                 + 1.30881 w_k1_MSFT_s114
                                 + 1.04897 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.98831 w_k1_ADM_s114
                                 + 0.954347 w_k1_AXP_s114
                                 + 1.12256 w_k1_DIS_s114
                                 + 0.915298 w_k1_FDX_s114
                                 + 1.2967 w_k1_IBM_s114
                                 + 1.14074 w_k1_INTC_s114
                                 + 1.24714 w_k1_KO_s114 + 1.06851 w_k1_MAT_s114
                                 + 1.02135 w_k1_MSFT_s114
                                 + 1.02831 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.07422 w_k1_ADM_s114
                                 + 0.798175 w_k1_AXP_s114
                                 + 0.940644 w_k1_DIS_s114
                                 + 0.867559 w_k1_FDX_s114
                                 + 0.925302 w_k1_IBM_s114
                                 + 0.912475 w_k1_INTC_s114
                                 + 0.980491 w_k1_KO_s114
                                 + 1.01847 w_k1_MAT_s114
                                 + 1.02661 w_k1_MSFT_s114
                                 + 0.925629 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.789251 w_k1_ADM_s114
                                 + 0.777646 w_k1_AXP_s114
                                 + 0.583898 w_k1_DIS_s114
                                 + 0.714935 w_k1_FDX_s114
                                 + 0.751012 w_k1_IBM_s114
                                 + 0.776194 w_k1_INTC_s114
                                 + 0.68131 w_k1_KO_s114
                                 + 0.819463 w_k1_MAT_s114
                                 + 0.693669 w_k1_MSFT_s114
                                 + 0.736322 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.943745 w_k1_ADM_s114
                                 + 0.824943 w_k1_AXP_s114
                                 + 1.1301 w_k1_DIS_s114 + 0.90935 w_k1_FDX_s114
                                 + 1.0344 w_k1_IBM_s114
                                 + 0.927547 w_k1_INTC_s114
                                 + 1.14631 w_k1_KO_s114
                                 + 0.992073 w_k1_MAT_s114
                                 + 1.07985 w_k1_MSFT_s114
                                 + 1.07634 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.24202 w_k1_ADM_s114
                                 + 1.26168 w_k1_AXP_s114
                                 + 1.43905 w_k1_DIS_s114
                                 + 1.12071 w_k1_FDX_s114
                                 + 1.58038 w_k1_IBM_s114
                                 + 1.0348 w_k1_INTC_s114 + 1.1804 w_k1_KO_s114
                                 + 1.72394 w_k1_MAT_s114
                                 + 1.14412 w_k1_MSFT_s114
                                 + 1.02454 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.16151 w_k1_ADM_s114
                                 + 0.795717 w_k1_AXP_s114
                                 + 1.14481 w_k1_DIS_s114
                                 + 0.943757 w_k1_FDX_s114
                                 + 0.720704 w_k1_IBM_s114
                                 + 0.833244 w_k1_INTC_s114
                                 + 0.932192 w_k1_KO_s114
                                 + 0.429233 w_k1_MAT_s114
                                 + 1.20291 w_k1_MSFT_s114
                                 + 0.976471 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.40958 w_k1_ADM_s114
                                 + 1.55584 w_k1_AXP_s114
                                 + 1.37362 w_k1_DIS_s114
                                 + 1.31268 w_k1_FDX_s114
                                 + 1.09155 w_k1_IBM_s114
                                 + 1.04559 w_k1_INTC_s114
                                 + 1.27795 w_k1_KO_s114 + 1.16704 w_k1_MAT_s114
                                 + 1.24902 w_k1_MSFT_s114
                                 + 1.10891 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.02542 w_k1_ADM_s115
                                 + 0.827222 w_k1_AXP_s115
                                 + 1.00836 w_k1_DIS_s115
                                 + 0.94606 w_k1_FDX_s115
                                 + 0.752519 w_k1_IBM_s115
                                 + 0.72128 w_k1_INTC_s115
                                 + 0.785201 w_k1_KO_s115
                                 + 0.694221 w_k1_MAT_s115
                                 + 0.894219 w_k1_MSFT_s115
                                 + 0.832353 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.14147 w_k1_ADM_s115
                                 + 1.01022 w_k1_AXP_s115
                                 + 1.21535 w_k1_DIS_s115
                                 + 1.23006 w_k1_FDX_s115
                                 + 0.682537 w_k1_IBM_s115
                                 + 0.925421 w_k1_INTC_s115
                                 + 0.94887 w_k1_KO_s115 + 1.08156 w_k1_MAT_s115
                                 + 1.17148 w_k1_MSFT_s115
                                 + 1.05384 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.583179 w_k1_ADM_s115
                                 + 0.62423 w_k1_AXP_s115
                                 + 0.744162 w_k1_DIS_s115
                                 + 0.725282 w_k1_FDX_s115
                                 + 0.979531 w_k1_IBM_s115
                                 + 0.764729 w_k1_INTC_s115
                                 + 0.743235 w_k1_KO_s115
                                 + 0.845045 w_k1_MAT_s115
                                 + 0.822397 w_k1_MSFT_s115
                                 + 0.877249 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.912168 w_k1_ADM_s115
                                 + 0.897135 w_k1_AXP_s115
                                 + 0.839767 w_k1_DIS_s115
                                 + 0.839706 w_k1_FDX_s115
                                 + 1.00911 w_k1_IBM_s115
                                 + 0.666033 w_k1_INTC_s115
                                 + 0.858846 w_k1_KO_s115
                                 + 0.934224 w_k1_MAT_s115
                                 + 0.826645 w_k1_MSFT_s115
                                 + 0.952026 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.730317 w_k1_ADM_s115
                                 + 0.707269 w_k1_AXP_s115
                                 + 0.500977 w_k1_DIS_s115
                                 + 0.664286 w_k1_FDX_s115
                                 + 0.769455 w_k1_IBM_s115
                                 + 0.618747 w_k1_INTC_s115
                                 + 0.757414 w_k1_KO_s115
                                 + 0.66094 w_k1_MAT_s115
                                 + 0.705663 w_k1_MSFT_s115
                                 + 0.887643 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.26185 w_k1_ADM_s115
                                 + 1.31001 w_k1_AXP_s115
                                 + 1.16818 w_k1_DIS_s115
                                 + 1.16084 w_k1_FDX_s115 + 1.1626 w_k1_IBM_s115
                                 + 0.845302 w_k1_INTC_s115
                                 + 1.06538 w_k1_KO_s115 + 1.53284 w_k1_MAT_s115
                                 + 1.07076 w_k1_MSFT_s115
                                 + 1.06468 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.9346 w_k1_ADM_s115
                                 + 0.848106 w_k1_AXP_s115
                                 + 0.826541 w_k1_DIS_s115
                                 + 0.841439 w_k1_FDX_s115
                                 + 1.53662 w_k1_IBM_s115
                                 + 0.728065 w_k1_INTC_s115
                                 + 1.10752 w_k1_KO_s115 + 1.38112 w_k1_MAT_s115
                                 + 0.86746 w_k1_MSFT_s115
                                 + 0.983531 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.29425 w_k1_ADM_s115
                                 + 1.06908 w_k1_AXP_s115
                                 + 1.52046 w_k1_DIS_s115
                                 + 1.22027 w_k1_FDX_s115
                                 + 1.13083 w_k1_IBM_s115
                                 + 1.08701 w_k1_INTC_s115
                                 + 1.03358 w_k1_KO_s115
                                 + 0.738527 w_k1_MAT_s115
                                 + 1.08651 w_k1_MSFT_s115
                                 + 1.11303 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.828564 w_k1_ADM_s116
                                 + 1.41397 w_k1_AXP_s116
                                 + 1.16115 w_k1_DIS_s116
                                 + 1.20581 w_k1_FDX_s116
                                 + 1.41928 w_k1_IBM_s116
                                 + 1.15247 w_k1_INTC_s116
                                 + 1.23837 w_k1_KO_s116 + 1.23705 w_k1_MAT_s116
                                 + 1.0494 w_k1_MSFT_s116 + 1.20045 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.968437 w_k1_ADM_s116
                                 + 0.945428 w_k1_AXP_s116
                                 + 1.08258 w_k1_DIS_s116 + 1.0326 w_k1_FDX_s116
                                 + 0.972082 w_k1_IBM_s116
                                 + 1.24885 w_k1_INTC_s116
                                 + 0.932509 w_k1_KO_s116
                                 + 0.812397 w_k1_MAT_s116
                                 + 1.10645 w_k1_MSFT_s116
                                 + 0.989478 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.0438 w_k1_ADM_s116
                                 + 1.16702 w_k1_AXP_s116
                                 + 0.662906 w_k1_DIS_s116
                                 + 0.94659 w_k1_FDX_s116
                                 + 0.837023 w_k1_IBM_s116
                                 + 0.998895 w_k1_INTC_s116
                                 + 0.721516 w_k1_KO_s116
                                 + 0.741995 w_k1_MAT_s116
                                 + 0.823269 w_k1_MSFT_s116
                                 + 0.960194 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.16655 w_k1_ADM_s116
                                 + 1.12346 w_k1_AXP_s116
                                 + 1.02295 w_k1_DIS_s116
                                 + 1.24434 w_k1_FDX_s116
                                 + 0.715617 w_k1_IBM_s116
                                 + 0.929827 w_k1_INTC_s116
                                 + 0.779907 w_k1_KO_s116
                                 + 0.933051 w_k1_MAT_s116
                                 + 1.09288 w_k1_MSFT_s116
                                 + 1.09482 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.703997 w_k1_ADM_s116
                                 + 0.700157 w_k1_AXP_s116
                                 + 0.634082 w_k1_DIS_s116
                                 + 0.823095 w_k1_FDX_s116
                                 + 0.553527 w_k1_IBM_s116
                                 + 0.837688 w_k1_INTC_s116
                                 + 0.580472 w_k1_KO_s116
                                 + 0.471216 w_k1_MAT_s116
                                 + 0.739763 w_k1_MSFT_s116
                                 + 0.814985 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.790591 w_k1_ADM_s116
                                 + 0.93248 w_k1_AXP_s116
                                 + 0.875393 w_k1_DIS_s116
                                 + 0.833845 w_k1_FDX_s116
                                 + 1.21885 w_k1_IBM_s116
                                 + 0.878712 w_k1_INTC_s116
                                 + 0.986225 w_k1_KO_s116
                                 + 0.93004 w_k1_MAT_s116
                                 + 0.820365 w_k1_MSFT_s116
                                 + 0.968605 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.747774 w_k1_ADM_s116
                                 + 0.760409 w_k1_AXP_s116
                                 + 0.619449 w_k1_DIS_s116
                                 + 0.768741 w_k1_FDX_s116
                                 + 0.792069 w_k1_IBM_s116
                                 + 0.693732 w_k1_INTC_s116
                                 + 0.758901 w_k1_KO_s116
                                 + 0.576889 w_k1_MAT_s116
                                 + 0.693214 w_k1_MSFT_s116
                                 + 0.817645 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.12704 w_k1_ADM_s116
                                 + 1.45359 w_k1_AXP_s116
                                 + 1.73209 w_k1_DIS_s116
                                 + 1.34087 w_k1_FDX_s116
                                 + 1.63206 w_k1_IBM_s116
                                 + 1.16042 w_k1_INTC_s116
                                 + 1.52211 w_k1_KO_s116 + 1.34591 w_k1_MAT_s116
                                 + 1.28192 w_k1_MSFT_s116
                                 + 1.37023 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.82614 w_k1_ADM_s117
                                 + 1.32857 w_k1_AXP_s117
                                 + 1.42561 w_k1_DIS_s117
                                 + 1.18749 w_k1_FDX_s117
                                 + 1.48678 w_k1_IBM_s117
                                 + 1.34434 w_k1_INTC_s117
                                 + 1.59972 w_k1_KO_s117 + 1.61327 w_k1_MAT_s117
                                 + 1.16427 w_k1_MSFT_s117
                                 + 1.22305 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.791529 w_k1_ADM_s117
                                 + 1.06657 w_k1_AXP_s117
                                 + 0.922056 w_k1_DIS_s117
                                 + 0.924104 w_k1_FDX_s117
                                 + 1.08978 w_k1_IBM_s117
                                 + 1.12483 w_k1_INTC_s117
                                 + 1.01831 w_k1_KO_s117 + 1.03959 w_k1_MAT_s117
                                 + 0.871451 w_k1_MSFT_s117
                                 + 0.917629 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.0159 w_k1_ADM_s117
                                 + 1.62489 w_k1_AXP_s117
                                 + 1.70736 w_k1_DIS_s117 + 1.4748 w_k1_FDX_s117
                                 + 1.12258 w_k1_IBM_s117
                                 + 1.26962 w_k1_INTC_s117
                                 + 1.22901 w_k1_KO_s117 + 1.02929 w_k1_MAT_s117
                                 + 1.2931 w_k1_MSFT_s117 + 1.1292 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.14966 w_k1_ADM_s117
                                 + 1.27709 w_k1_AXP_s117
                                 + 1.12742 w_k1_DIS_s117
                                 + 1.15653 w_k1_FDX_s117 + 1.1219 w_k1_IBM_s117
                                 + 0.942172 w_k1_INTC_s117
                                 + 1.05873 w_k1_KO_s117 + 1.08044 w_k1_MAT_s117
                                 + 1.00364 w_k1_MSFT_s117
                                 + 1.08641 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.0047 w_k1_ADM_s117
                                 + 1.15586 w_k1_AXP_s117
                                 + 1.29462 w_k1_DIS_s117
                                 + 1.16591 w_k1_FDX_s117
                                 + 0.880269 w_k1_IBM_s117
                                 + 1.03028 w_k1_INTC_s117
                                 + 1.09937 w_k1_KO_s117 + 1.14098 w_k1_MAT_s117
                                 + 1.12103 w_k1_MSFT_s117
                                 + 1.15605 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.776085 w_k1_ADM_s117
                                 + 0.712637 w_k1_AXP_s117
                                 + 0.672712 w_k1_DIS_s117
                                 + 0.79649 w_k1_FDX_s117
                                 + 0.778989 w_k1_IBM_s117
                                 + 0.666764 w_k1_INTC_s117
                                 + 0.853843 w_k1_KO_s117
                                 + 0.82935 w_k1_MAT_s117
                                 + 0.810872 w_k1_MSFT_s117
                                 + 0.903074 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.900616 w_k1_ADM_s117
                                 + 0.854462 w_k1_AXP_s117
                                 + 1.02455 w_k1_DIS_s117
                                 + 0.960165 w_k1_FDX_s117
                                 + 0.919007 w_k1_IBM_s117
                                 + 0.780195 w_k1_INTC_s117
                                 + 1.01328 w_k1_KO_s117 + 1.14622 w_k1_MAT_s117
                                 + 1.0128 w_k1_MSFT_s117 + 1.06482 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.668494 w_k1_ADM_s117
                                 + 0.766789 w_k1_AXP_s117
                                 + 0.70713 w_k1_DIS_s117
                                 + 0.753799 w_k1_FDX_s117
                                 + 1.26568 w_k1_IBM_s117
                                 + 0.884276 w_k1_INTC_s117
                                 + 1.03646 w_k1_KO_s117 + 1.07982 w_k1_MAT_s117
                                 + 0.848609 w_k1_MSFT_s117
                                 + 0.91957 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0214451473091263 w_k1_ADM_s000
                                 + 0.00879117962408884 w_k1_ADM_s001
                                 + 0.0115404918593209 w_k1_ADM_s002
                                 + 0.0115739918504449 w_k1_ADM_s003
                                 + 0.0177526956660552 w_k1_ADM_s004
                                 + 0.0161538721346574 w_k1_ADM_s005
                                 + 0.0190204984644779 w_k1_ADM_s006
                                 + 0.0230108540821403 w_k1_ADM_s007
                                 + 0.0477912660902387 w_k1_ADM_s010
                                 + 0.0268428776553064 w_k1_ADM_s011
                                 + 0.0161708231235977 w_k1_ADM_s012
                                 + 0.012957029433648 w_k1_ADM_s013
                                 + 0.0399645265420479 w_k1_ADM_s014
                                 + 0.0207297123223561 w_k1_ADM_s015
                                 + 0.0159801172276944 w_k1_ADM_s016
                                 + 0.0260674515406082 w_k1_ADM_s017
                                 + 0.0149594719890491 w_k1_ADM_s100
                                 + 0.0405600405456141 w_k1_ADM_s101
                                 + 0.0421018553289412 w_k1_ADM_s102
                                 + 0.0692511957376629 w_k1_ADM_s103
                                 + 0.0540110094852662 w_k1_ADM_s104
                                 + 0.0646570286369908 w_k1_ADM_s105
                                 + 0.0640795480458809 w_k1_ADM_s106
                                 + 0.0240814003438622 w_k1_ADM_s107
                                 + 0.0510487320901891 w_k1_ADM_s110
                                 + 0.0177865178953226 w_k1_ADM_s111
                                 + 0.0375229130096813 w_k1_ADM_s112
                                 + 0.059354922180799 w_k1_ADM_s113
                                 + 0.0290258615651195 w_k1_ADM_s114
                                 + 0.0241320327708936 w_k1_ADM_s115
                                 + 0.036097191381588 w_k1_ADM_s116
                                 + 0.0446723856537761 w_k1_ADM_s117
                                 + 0.0217844241405381 w_k1_AXP_s000
                                 + 0.008665653128294 w_k1_AXP_s001
                                 + 0.00946732341712825 w_k1_AXP_s002
                                 + 0.00904341422633354 w_k1_AXP_s003
                                 + 0.0164184639749666 w_k1_AXP_s004
                                 + 0.0161705795735888 w_k1_AXP_s005
                                 + 0.0196796321852386 w_k1_AXP_s006
                                 + 0.0225853971421316 w_k1_AXP_s007
                                 + 0.0530599608481026 w_k1_AXP_s010
                                 + 0.0244331876767666 w_k1_AXP_s011
                                 + 0.0170463407982075 w_k1_AXP_s012
                                 + 0.0113149430570611 w_k1_AXP_s013
                                 + 0.0388342140317147 w_k1_AXP_s014
                                 + 0.0244447710076879 w_k1_AXP_s015
                                 + 0.0159615923330886 w_k1_AXP_s016
                                 + 0.0221103325116572 w_k1_AXP_s017
                                 + 0.0162816261110428 w_k1_AXP_s100
                                 + 0.0430205112300121 w_k1_AXP_s101
                                 + 0.0412339860129503 w_k1_AXP_s102
                                 + 0.0730899042317232 w_k1_AXP_s103
                                 + 0.0635047982656649 w_k1_AXP_s104
                                 + 0.0605877293930374 w_k1_AXP_s105
                                 + 0.0719049122698672 w_k1_AXP_s106
                                 + 0.0253803790603497 w_k1_AXP_s107
                                 + 0.0502196874082211 w_k1_AXP_s110
                                 + 0.0189679137110418 w_k1_AXP_s111
                                 + 0.0410324237474249 w_k1_AXP_s112
                                 + 0.0576009374906013 w_k1_AXP_s113
                                 + 0.0260232276231446 w_k1_AXP_s114
                                 + 0.0218340283306945 w_k1_AXP_s115
                                 + 0.0414284932136693 w_k1_AXP_s116
                                 + 0.0535284635180725 w_k1_AXP_s117
                                 + 0.0208466431448073 w_k1_DIS_s000
                                 + 0.00704236154728106 w_k1_DIS_s001
                                 + 0.0100579803119093 w_k1_DIS_s002
                                 + 0.0100506669951093 w_k1_DIS_s003
                                 + 0.0167581983487425 w_k1_DIS_s004
                                 + 0.0154666714983647 w_k1_DIS_s005
                                 + 0.0202626745540266 w_k1_DIS_s006
                                 + 0.0241330396547226 w_k1_DIS_s007
                                 + 0.0511467746893566 w_k1_DIS_s010
                                 + 0.0261890449931492 w_k1_DIS_s011
                                 + 0.0164316156298998 w_k1_DIS_s012
                                 + 0.0129878291522003 w_k1_DIS_s013
                                 + 0.0429821025974067 w_k1_DIS_s014
                                 + 0.0238481558432589 w_k1_DIS_s015
                                 + 0.0162258940030894 w_k1_DIS_s016
                                 + 0.0247150024906579 w_k1_DIS_s017
                                 + 0.0174532502346678 w_k1_DIS_s100
                                 + 0.0443489804996175 w_k1_DIS_s101
                                 + 0.0379502412223884 w_k1_DIS_s102
                                 + 0.0733359846738844 w_k1_DIS_s103
                                 + 0.0609095969237144 w_k1_DIS_s104
                                 + 0.0550905341710811 w_k1_DIS_s105
                                 + 0.071617731404552 w_k1_DIS_s106
                                 + 0.022581678198883 w_k1_DIS_s107
                                 + 0.0504934011259306 w_k1_DIS_s110
                                 + 0.0176391223984549 w_k1_DIS_s111
                                 + 0.0424751494534511 w_k1_DIS_s112
                                 + 0.0617653870841002 w_k1_DIS_s113
                                 + 0.0296836716781296 w_k1_DIS_s114
                                 + 0.0246862861429316 w_k1_DIS_s115
                                 + 0.0401582415272452 w_k1_DIS_s116
                                 + 0.0548165400831254 w_k1_DIS_s117
                                 + 0.0210503476413608 w_k1_FDX_s000
                                 + 0.00708302907012231 w_k1_FDX_s001
                                 + 0.0101575125121576 w_k1_FDX_s002
                                 + 0.00992078682549232 w_k1_FDX_s003
                                 + 0.017599288482759 w_k1_FDX_s004
                                 + 0.0157991491923555 w_k1_FDX_s005
                                 + 0.0195386904701422 w_k1_FDX_s006
                                 + 0.0222419243215695 w_k1_FDX_s007
                                 + 0.050757305003583 w_k1_FDX_s010
                                 + 0.0252899055782177 w_k1_FDX_s011
                                 + 0.016803287758487 w_k1_FDX_s012
                                 + 0.0130008356626802 w_k1_FDX_s013
                                 + 0.0423385324593728 w_k1_FDX_s014
                                 + 0.0221998784463069 w_k1_FDX_s015
                                 + 0.0154973265462572 w_k1_FDX_s016
                                 + 0.0244238777513077 w_k1_FDX_s017
                                 + 0.0165920062357232 w_k1_FDX_s100
                                 + 0.0442528070967683 w_k1_FDX_s101
                                 + 0.0391182182119329 w_k1_FDX_s102
                                 + 0.0689989615438031 w_k1_FDX_s103
                                 + 0.0619027056063861 w_k1_FDX_s104
                                 + 0.0559631181520934 w_k1_FDX_s105
                                 + 0.066730864381753 w_k1_FDX_s106
                                 + 0.0223836751387462 w_k1_FDX_s107
                                 + 0.0472427165866578 w_k1_FDX_s110
                                 + 0.0178506549901566 w_k1_FDX_s111
                                 + 0.040973883710712 w_k1_FDX_s112
                                 + 0.0601341367246515 w_k1_FDX_s113
                                 + 0.0258284094658192 w_k1_FDX_s114
                                 + 0.0231476598522852 w_k1_FDX_s115
                                 + 0.0401203430991931 w_k1_FDX_s116
                                 + 0.052369259272359 w_k1_FDX_s117
                                 + 0.0205825724985919 w_k1_IBM_s000
                                 + 0.00734267519676052 w_k1_IBM_s001
                                 + 0.0102323313460519 w_k1_IBM_s002
                                 + 0.0102755649221569 w_k1_IBM_s003
                                 + 0.0177326082189245 w_k1_IBM_s004
                                 + 0.016040142985144 w_k1_IBM_s005
                                 + 0.0191224282409936 w_k1_IBM_s006
                                 + 0.0242315662606925 w_k1_IBM_s007
                                 + 0.0527292193442491 w_k1_IBM_s010
                                 + 0.0242893621557708 w_k1_IBM_s011
                                 + 0.0162574464027132 w_k1_IBM_s012
                                 + 0.0124779759420312 w_k1_IBM_s013
                                 + 0.0389948664172953 w_k1_IBM_s014
                                 + 0.0235855926307715 w_k1_IBM_s015
                                 + 0.0157825837479375 w_k1_IBM_s016
                                 + 0.0244194982205593 w_k1_IBM_s017
                                 + 0.0167019228689394 w_k1_IBM_s100
                                 + 0.0455520170859722 w_k1_IBM_s101
                                 + 0.0405491535616973 w_k1_IBM_s102
                                 + 0.0669005905163158 w_k1_IBM_s103
                                 + 0.0599126581198779 w_k1_IBM_s104
                                 + 0.0583707882951924 w_k1_IBM_s105
                                 + 0.068581735398464 w_k1_IBM_s106
                                 + 0.0230782335527413 w_k1_IBM_s107
                                 + 0.050673521945541 w_k1_IBM_s110
                                 + 0.017138385966448 w_k1_IBM_s111
                                 + 0.0387137305572516 w_k1_IBM_s112
                                 + 0.0627893441503556 w_k1_IBM_s113
                                 + 0.0283364883257415 w_k1_IBM_s114
                                 + 0.0239351072239813 w_k1_IBM_s115
                                 + 0.0419270539417531 w_k1_IBM_s116
                                 + 0.0539212364340193 w_k1_IBM_s117
                                 + 0.0183963025375466 w_k1_INTC_s000
                                 + 0.00659716226359305 w_k1_INTC_s001
                                 + 0.0103866993640503 w_k1_INTC_s002
                                 + 0.0114386037186515 w_k1_INTC_s003
                                 + 0.0160745276681365 w_k1_INTC_s004
                                 + 0.0157090204280692 w_k1_INTC_s005
                                 + 0.0229861941319864 w_k1_INTC_s006
                                 + 0.0219013538604816 w_k1_INTC_s007
                                 + 0.0449733342610874 w_k1_INTC_s010
                                 + 0.0246141117722915 w_k1_INTC_s011
                                 + 0.0173855251902846 w_k1_INTC_s012
                                 + 0.0132532698029974 w_k1_INTC_s013
                                 + 0.0365176063797523 w_k1_INTC_s014
                                 + 0.0246125470475079 w_k1_INTC_s015
                                 + 0.015236531917076 w_k1_INTC_s016
                                 + 0.0205802514997946 w_k1_INTC_s017
                                 + 0.0191366078619942 w_k1_INTC_s100
                                 + 0.04707601762728 w_k1_INTC_s101
                                 + 0.0343634937659098 w_k1_INTC_s102
                                 + 0.0779245107211605 w_k1_INTC_s103
                                 + 0.0679153801302133 w_k1_INTC_s104
                                 + 0.0585153227792449 w_k1_INTC_s105
                                 + 0.0790379914702738 w_k1_INTC_s106
                                 + 0.020337728652902 w_k1_INTC_s107
                                 + 0.0462515344013996 w_k1_INTC_s110
                                 + 0.0186583172499074 w_k1_INTC_s111
                                 + 0.0381377019650631 w_k1_INTC_s112
                                 + 0.0555224536526442 w_k1_INTC_s113
                                 + 0.0257783725883096 w_k1_INTC_s114
                                 + 0.0194467033518007 w_k1_INTC_s115
                                 + 0.0400227040877937 w_k1_INTC_s116
                                 + 0.049810198058119 w_k1_INTC_s117
                                 + 0.0227957411463651 w_k1_KO_s000
                                 + 0.00667253803498005 w_k1_KO_s001
                                 + 0.0102584938651085 w_k1_KO_s002
                                 + 0.0118726228554368 w_k1_KO_s003
                                 + 0.0171848001685617 w_k1_KO_s004
                                 + 0.0145559242682218 w_k1_KO_s005
                                 + 0.0174874769433476 w_k1_KO_s006
                                 + 0.0208928689190733 w_k1_KO_s007
                                 + 0.0491015788655939 w_k1_KO_s010
                                 + 0.0264467909090481 w_k1_KO_s011
                                 + 0.0164864189140287 w_k1_KO_s012
                                 + 0.0117407824209824 w_k1_KO_s013
                                 + 0.0424579773340367 w_k1_KO_s014
                                 + 0.023083611016827 w_k1_KO_s015
                                 + 0.0149593439754731 w_k1_KO_s016
                                 + 0.0245994103059699 w_k1_KO_s017
                                 + 0.0171454568894208 w_k1_KO_s100
                                 + 0.0465750022220703 w_k1_KO_s101
                                 + 0.0360235184155745 w_k1_KO_s102
                                 + 0.0702890444153078 w_k1_KO_s103
                                 + 0.0596564985892203 w_k1_KO_s104
                                 + 0.0570879009804119 w_k1_KO_s105
                                 + 0.0729449881749377 w_k1_KO_s106
                                 + 0.0219047125970168 w_k1_KO_s107
                                 + 0.0499538360443664 w_k1_KO_s110
                                 + 0.0177360290177156 w_k1_KO_s111
                                 + 0.0377083672091445 w_k1_KO_s112
                                 + 0.0644578359490452 w_k1_KO_s113
                                 + 0.0280594935205614 w_k1_KO_s114
                                 + 0.0217070102553332 w_k1_KO_s115
                                 + 0.0384036429640971 w_k1_KO_s116
                                 + 0.0567982982448835 w_k1_KO_s117
                                 + 0.0249239596401189 w_k1_MAT_s000
                                 + 0.00820104875701362 w_k1_MAT_s001
                                 + 0.00803743366206487 w_k1_MAT_s002
                                 + 0.0097942587379597 w_k1_MAT_s003
                                 + 0.0183108478405582 w_k1_MAT_s004
                                 + 0.0148416101157107 w_k1_MAT_s005
                                 + 0.0186117796764979 w_k1_MAT_s006
                                 + 0.0224060520974702 w_k1_MAT_s007
                                 + 0.0614265289186788 w_k1_MAT_s010
                                 + 0.0256509071688877 w_k1_MAT_s011
                                 + 0.0216762232216062 w_k1_MAT_s012
                                 + 0.010663802046328 w_k1_MAT_s013
                                 + 0.047054326592473 w_k1_MAT_s014
                                 + 0.023995608820196 w_k1_MAT_s015
                                 + 0.0147993121315464 w_k1_MAT_s016
                                 + 0.0239006616515694 w_k1_MAT_s017
                                 + 0.017071370559149 w_k1_MAT_s100
                                 + 0.0476534470023213 w_k1_MAT_s101
                                 + 0.0353933557892129 w_k1_MAT_s102
                                 + 0.0636925851675148 w_k1_MAT_s103
                                 + 0.0592092003876834 w_k1_MAT_s104
                                 + 0.0512813821807692 w_k1_MAT_s105
                                 + 0.072011428165399 w_k1_MAT_s106
                                 + 0.0237531550159966 w_k1_MAT_s107
                                 + 0.0535641396707018 w_k1_MAT_s110
                                 + 0.0154902184823848 w_k1_MAT_s111
                                 + 0.0565052486969842 w_k1_MAT_s112
                                 + 0.0771295760316847 w_k1_MAT_s113
                                 + 0.0275169055188135 w_k1_MAT_s114
                                 + 0.0223357387211329 w_k1_MAT_s115
                                 + 0.0355209586303514 w_k1_MAT_s116
                                 + 0.0573529381538986 w_k1_MAT_s117
                                 + 0.0192332159637526 w_k1_MSFT_s000
                                 + 0.00720635445388294 w_k1_MSFT_s001
                                 + 0.0102298711279401 w_k1_MSFT_s002
                                 + 0.010935000525989 w_k1_MSFT_s003
                                 + 0.0155871596833564 w_k1_MSFT_s004
                                 + 0.0145714726833022 w_k1_MSFT_s005
                                 + 0.021926069484823 w_k1_MSFT_s006
                                 + 0.021084818701233 w_k1_MSFT_s007
                                 + 0.0536288129573337 w_k1_MSFT_s010
                                 + 0.0252933849731187 w_k1_MSFT_s011
                                 + 0.0157586418496789 w_k1_MSFT_s012
                                 + 0.0115762464697772 w_k1_MSFT_s013
                                 + 0.04104313610627 w_k1_MSFT_s014
                                 + 0.0212755768721653 w_k1_MSFT_s015
                                 + 0.0151612294053167 w_k1_MSFT_s016
                                 + 0.0222557019644787 w_k1_MSFT_s017
                                 + 0.0158719595954372 w_k1_MSFT_s100
                                 + 0.0426315147353526 w_k1_MSFT_s101
                                 + 0.0410321298840085 w_k1_MSFT_s102
                                 + 0.0716016171634958 w_k1_MSFT_s103
                                 + 0.0605322830626367 w_k1_MSFT_s104
                                 + 0.0548971546895293 w_k1_MSFT_s105
                                 + 0.0664208043428867 w_k1_MSFT_s106
                                 + 0.0220249387497338 w_k1_MSFT_s107
                                 + 0.0499069797069258 w_k1_MSFT_s110
                                 + 0.017444222418077 w_k1_MSFT_s111
                                 + 0.0397796867554091 w_k1_MSFT_s112
                                 + 0.0587391350833167 w_k1_MSFT_s113
                                 + 0.0283687696153067 w_k1_MSFT_s114
                                 + 0.0222041168119037 w_k1_MSFT_s115
                                 + 0.0382971623994237 w_k1_MSFT_s116
                                 + 0.0512652005682449 w_k1_MSFT_s117
                                 + 0.0193401869151934 w_k1_PG_s000
                                 + 0.00710320077808874 w_k1_PG_s001
                                 + 0.00982881648211851 w_k1_PG_s002
                                 + 0.0104870177072735 w_k1_PG_s003
                                 + 0.0156252816343369 w_k1_PG_s004
                                 + 0.0153582905750038 w_k1_PG_s005
                                 + 0.0194695210186855 w_k1_PG_s006
                                 + 0.0228986059716917 w_k1_PG_s007
                                 + 0.0529156148670579 w_k1_PG_s010
                                 + 0.0238565327078299 w_k1_PG_s011
                                 + 0.0156162180615061 w_k1_PG_s012
                                 + 0.0118802777517281 w_k1_PG_s013
                                 + 0.0409062146141033 w_k1_PG_s014
                                 + 0.0222648135740625 w_k1_PG_s015
                                 + 0.0150951315667706 w_k1_PG_s016
                                 + 0.0249908096183937 w_k1_PG_s017
                                 + 0.0165807760154023 w_k1_PG_s100
                                 + 0.0465098837614472 w_k1_PG_s101
                                 + 0.0373852288817641 w_k1_PG_s102
                                 + 0.0680249300774617 w_k1_PG_s103
                                 + 0.0542554533507081 w_k1_PG_s104
                                 + 0.0574723495328585 w_k1_PG_s105
                                 + 0.0682350423121048 w_k1_PG_s106
                                 + 0.0221592674611153 w_k1_PG_s107
                                 + 0.0487534558484107 w_k1_PG_s110
                                 + 0.0183819221321221 w_k1_PG_s111
                                 + 0.0399403574897802 w_k1_PG_s112
                                 + 0.059353519041294 w_k1_PG_s113
                                 + 0.026110325380589 w_k1_PG_s114
                                 + 0.022947874346423 w_k1_PG_s115
                                 + 0.0404871850599325 w_k1_PG_s116
                                 + 0.0536906321761339 w_k1_PG_s117 >= 1.12
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
