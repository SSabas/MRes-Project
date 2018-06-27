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
 w_k1_ADM_s0_return_constraint:  1.15848 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.933486 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.37553 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.00281 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.19054 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.999419 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.850133 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.14562 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.842622 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.875757 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.09164 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.626533 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.98 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.796952 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.18765 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.916615 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.18822 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.11658 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.73595 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.85551 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.01079 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.19184 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.909966 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.00441 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.1343 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.64117 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.97435 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.729086 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.767773 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.15418 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.11572 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.973625 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.691961 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.00057 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.853436 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.91215 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.716826 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.20345 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24681 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.921932 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.51807 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.946853 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.31486 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.845429 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.931886 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.22052 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.918947 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.845557 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.4983 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.590178 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.02286 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.694548 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.48149 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.940292 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.05215 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.36183 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.720116 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.05714 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.19412 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.66824 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.33937 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.937041 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.3473 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.3352 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.835932 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.72719 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.925203 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.994338 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.52881 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.16244 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.671376 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.08388 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.935195 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.18425 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.885014 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.42861 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38352 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.880926 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.61184 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.878856 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.31821 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.799056 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.07417 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.28851 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.817709 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.901834 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.63907 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.58681 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.02033 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.747336 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.49071 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.949041 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.15807 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.23074 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.695221 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.902976 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.06508 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.74897 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.04717 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.03937 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.77381 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.45989 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.935234 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.721967 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.814176 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.20688 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.77436 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.53244 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.540155 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.10553 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.891968 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.11575 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.80588 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.73377 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21072 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.935047 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.30433 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.932307 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.1854 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.865908 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.05578 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.1947 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.944525 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.798771 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.4555 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.741218 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.19036 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.941873 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.31802 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.862753 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.28842 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.24085 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.8165 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.955873 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.938306 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.34838 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.08965 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.996313 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.29882 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.51175 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.07109 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.794601 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.788131 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.17822 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.29055 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.22281 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.696027 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.17248 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.00595 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.95444 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.811026 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.5276 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20627 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.938119 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.26992 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.994784 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.22061 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.887605 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.82913 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.12231 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.974992 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.42652 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.47191 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.709369 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.571176 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.601586 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.1275 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.13539 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.643803 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.87886 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.675574 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.838864 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.74099 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.78257 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.24421 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.988207 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.61092 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.810657 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.605056 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.713845 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.35237 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.785341 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.52523 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.921694 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.679634 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.0109 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.736799 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.60109 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.10933 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.945609 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.07003 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.949107 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.32129 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.11384 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.16719 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.899979 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.31924 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.35361 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.00508 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.8842 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.29889 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.789279 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.04965 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.02107 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.10462 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.753181 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.01859 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.04692 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.803045 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.861649 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.870231 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.19739 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.04251 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.963779 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.3791 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.18924 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.891619 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.860208 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.01238 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.1237 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.25973 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.28577 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.759953 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.28345 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.988833 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.04293 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.867776 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.18279 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21236 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.912041 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.30746 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.967989 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.22119 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.914371 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.0612 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.25826 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.901189 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.17984 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.41896 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.717609 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.80378 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.717401 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.12613 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.00326 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.874954 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.962935 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.746677 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.853661 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.28905 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.5374 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.16597 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.03372 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.61827 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.12778 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.793815 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.719093 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.03414 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.00746 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.5006 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.06986 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.691397 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.995867 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.79732 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.09941 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.849427 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.22377 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.30037 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.929166 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.252 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.907033 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.23497 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.816395 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.694365 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.24504 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.993779 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.55403 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.82545 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.728055 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.703788 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.633026 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.35894 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.10768 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.782464 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.06467 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.630441 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.23782 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.89634 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.88278 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.45769 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.888505 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.78472 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.896122 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.527804 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.774655 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.33764 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.70956 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.34584 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.13776 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.790578 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.09526 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.863062 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.78149 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.01984 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.05602 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15496 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.9719 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.27896 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.992145 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.08888 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.8642 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.09462 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.2258 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.999787 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.88301 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.2756 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.831532 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.05524 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.912162 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.2542 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.871097 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.11768 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.10178 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.831085 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.874082 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.941974 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.21121 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.04814 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.03764 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.38673 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.33353 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.00522 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.844124 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.874232 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.20228 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.21022 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.18698 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.777553 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.16491 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.988636 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.960574 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.802965 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.3877 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10935 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.941283 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.25403 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.99383 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.03132 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.957659 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.0183 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.14481 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.894393 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.00288 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.24867 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.766795 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.852705 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.90325 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.0981 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.962648 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.0638 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.0767 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.781339 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.909602 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.07909 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.22389 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.18087 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.02489 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.51591 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.33961 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.02151 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.929876 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.04497 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.14144 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.19916 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.06171 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.77231 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.02083 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.853386 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.08404 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.856593 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.25664 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0740002962178261 z_k1_s0000
                                 - 0.0780000025803818 z_k1_s0001
                                 - 0.096999964555019 z_k1_s0002
                                 - 0.0580000838650501 z_k1_s0003
                                 - 0.053000119261245 z_k1_s0004
                                 - 0.0950004128742962 z_k1_s0005
                                 - 0.090000327670269 z_k1_s0006
                                 - 0.0580000838650501 z_k1_s0007
                                 - 0.0660001230652019 z_k1_s0010
                                 - 0.275000617938617 z_k1_s0011
                                 - 0.1980007477966 z_k1_s0012
                                 - 0.172000955541284 z_k1_s0013
                                 - 0.161000072661486 z_k1_s0014
                                 - 0.191999922043079 z_k1_s0015
                                 - 0.112000270025958 z_k1_s0016
                                 - 0.0860002253699792 z_k1_s0017
                                 - 0.116000073016265 z_k1_s0020
                                 - 0.0750001083985334 z_k1_s0021
                                 - 0.0780001161144795 z_k1_s0022
                                 - 0.0650001108454193 z_k1_s0023
                                 - 0.189000570604722 z_k1_s0024
                                 - 0.183999853577156 z_k1_s0025
                                 - 0.0480000389550186 z_k1_s0026
                                 - 0.0900002314783826 z_k1_s0027
                                 - 0.117000039921861 z_k1_s0030
                                 - 0.256999819488966 z_k1_s0031
                                 - 0.132999906583784 z_k1_s0032
                                 - 0.101000051159902 z_k1_s0033
                                 - 0.236999833536518 z_k1_s0034
                                 - 0.12699991079805 z_k1_s0035
                                 - 0.127999910095672 z_k1_s0036
                                 - 0.121000037112351 z_k1_s0037
                                 - 0.086000234120112 z_k1_s0040
                                 - 0.0330000861949177 z_k1_s0041
                                 - 0.086000234120112 z_k1_s0042
                                 - 0.0530003567257093 z_k1_s0043
                                 - 0.0760002554551025 z_k1_s0044
                                 - 0.0310000904619158 z_k1_s0045
                                 - 0.117000324582028 z_k1_s0046
                                 - 0.0400001234605531 z_k1_s0047
                                 - 0.0270000602744163 z_k1_s0050
                                 - 0.0410000833240979 z_k1_s0051
                                 - 0.0580003296491315 z_k1_s0052
                                 - 0.0840000496979188 z_k1_s0053
                                 - 0.0350000797742465 z_k1_s0054
                                 - 0.0430000992740768 z_k1_s0055
                                 - 0.102000060347473 z_k1_s0056
                                 - 0.0530001790239711 z_k1_s0057
                                 - 0.0370000576259503 z_k1_s0060
                                 - 0.0790000704498904 z_k1_s0061
                                 - 0.0600001391773136 z_k1_s0062
                                 - 0.0280000790494429 z_k1_s0063
                                 - 0.044000130263423 z_k1_s0064
                                 - 0.0509999491003579 z_k1_s0065
                                 - 0.0819999928085768 z_k1_s0066
                                 - 0.0420000974241195 z_k1_s0067
                                 - 0.0790001787487412 z_k1_s0070
                                 - 0.0880001510109767 z_k1_s0071
                                 - 0.0580003448041135 z_k1_s0072
                                 - 0.198000909778008 z_k1_s0073
                                 - 0.0500003103484498 z_k1_s0074
                                 - 0.114000661994201 z_k1_s0075
                                 - 0.104000523924069 z_k1_s0076
                                 - 0.0690004206808166 z_k1_s0077
                                 - 0.373001189369885 z_k1_s0100
                                 - 0.15800000254436 z_k1_s0101
                                 - 0.497001266500899 z_k1_s0102
                                 - 0.175999978927895 z_k1_s0103
                                 - 0.206999998210648 z_k1_s0104
                                 - 0.665000326613911 z_k1_s0105
                                 - 0.222999947240458 z_k1_s0106
                                 - 0.398999116768372 z_k1_s0107
                                 - 0.920003046437523 z_k1_s0110
                                 - 1.02400179067791 z_k1_s0111
                                 - 0.357000027210813 z_k1_s0112
                                 - 0.328999908684448 z_k1_s0113
                                 - 0.700002963159135 z_k1_s0114
                                 - 0.901001906008671 z_k1_s0115
                                 - 0.484000183212453 z_k1_s0116
                                 - 1.22099986859491 z_k1_s0117
                                 - 0.621000954818474 z_k1_s0120
                                 - 0.404000353402057 z_k1_s0121
                                 - 0.554999622470722 z_k1_s0122
                                 - 0.14900010643987 z_k1_s0123
                                 - 0.130999864981382 z_k1_s0124
                                 - 0.38899992868667 z_k1_s0125
                                 - 0.167999929212755 z_k1_s0126
                                 - 0.265000083105818 z_k1_s0127
                                 - 0.375001426778615 z_k1_s0130
                                 - 0.259000156627929 z_k1_s0131
                                 - 0.22000005304298 z_k1_s0132
                                 - 0.512999510231669 z_k1_s0133
                                 - 0.28099976993199 z_k1_s0134
                                 - 0.392999837662987 z_k1_s0135
                                 - 0.269000082675291 z_k1_s0136
                                 - 0.490000296323109 z_k1_s0137
                                 - 1.48699664884847 z_k1_s0140
                                 - 1.05299914133176 z_k1_s0141
                                 - 1.24399684700268 z_k1_s0142
                                 - 0.741998197332717 z_k1_s0143
                                 - 1.05500162045162 z_k1_s0144
                                 - 0.426999916986799 z_k1_s0145
                                 - 0.505999450513586 z_k1_s0146
                                 - 0.414999593153987 z_k1_s0147
                                 - 0.298999167631227 z_k1_s0150
                                 - 0.370000025830217 z_k1_s0151
                                 - 0.0999999033731363 z_k1_s0152
                                 - 0.374999355774546 z_k1_s0153
                                 - 0.387999417627979 z_k1_s0154
                                 - 0.354999780999509 z_k1_s0155
                                 - 0.165999853129393 z_k1_s0156
                                 - 0.201999764223776 z_k1_s0157
                                 - 0.194999905818794 z_k1_s0160
                                 - 0.264999716717982 z_k1_s0161
                                 - 0.123999924581195 z_k1_s0162
                                 - 0.436999165550347 z_k1_s0163
                                 - 0.532999511851569 z_k1_s0164
                                 - 0.150999765122385 z_k1_s0165
                                 - 0.255000176332011 z_k1_s0166
                                 - 0.202999754427384 z_k1_s0167
                                 - 0.274999605188943 z_k1_s0170
                                 - 0.310998900086606 z_k1_s0171
                                 - 0.399000464946546 z_k1_s0172
                                 - 0.152999857214188 z_k1_s0173
                                 - 0.302999526553647 z_k1_s0174
                                 - 0.348000230675237 z_k1_s0175
                                 - 0.0929999049184307 z_k1_s0176
                                 - 0.232000858449941 z_k1_s0177
                                 - 0.463000094460582 z_k1_s1000
                                 - 0.282999797187157 z_k1_s1001
                                 - 0.377998579116038 z_k1_s1002
                                 - 0.189999935055624 z_k1_s1003
                                 - 0.247999731150665 z_k1_s1004
                                 - 0.729999937676702 z_k1_s1005
                                 - 0.17999970447396 z_k1_s1006
                                 - 0.491999844308236 z_k1_s1007
                                 - 0.509000904534668 z_k1_s1010
                                 - 0.498000839888418 z_k1_s1011
                                 - 0.292000756125221 z_k1_s1012
                                 - 0.602001033724601 z_k1_s1013
                                 - 0.424001072779022 z_k1_s1014
                                 - 0.616999869784409 z_k1_s1015
                                 - 1.29600135603276 z_k1_s1016
                                 - 0.353001072881588 z_k1_s1017
                                 - 0.358001318862962 z_k1_s1020
                                 - 0.393000706349194 z_k1_s1021
                                 - 0.259000167900465 z_k1_s1022
                                 - 0.248999987404221 z_k1_s1023
                                 - 0.565000382282876 z_k1_s1024
                                 - 0.548999319288499 z_k1_s1025
                                 - 0.217000073416572 z_k1_s1026
                                 - 0.174999978932679 z_k1_s1027
                                 - 0.0740000544419011 z_k1_s1030
                                 - 0.3319997642525 z_k1_s1031
                                 - 0.334999726459575 z_k1_s1032
                                 - 0.120000116284184 z_k1_s1033
                                 - 0.195999504196997 z_k1_s1034
                                 - 0.0980000480987162 z_k1_s1035
                                 - 0.0920001236845669 z_k1_s1036
                                 - 0.232999531417947 z_k1_s1037
                                 - 0.128000040800832 z_k1_s1040
                                 - 0.240000504939239 z_k1_s1041
                                 - 0.422999376142278 z_k1_s1042
                                 - 0.120999940385902 z_k1_s1043
                                 - 0.554999311108717 z_k1_s1044
                                 - 0.153000007711134 z_k1_s1045
                                 - 0.334000874082179 z_k1_s1046
                                 - 0.331000667832855 z_k1_s1047
                                 - 0.38500056849356 z_k1_s1050
                                 - 1.22800446182593 z_k1_s1051
                                 - 0.730002373410717 z_k1_s1052
                                 - 0.305000522336576 z_k1_s1053
                                 - 0.634003124103674 z_k1_s1054
                                 - 0.451000997018719 z_k1_s1055
                                 - 0.426000888132005 z_k1_s1056
                                 - 0.879001470668959 z_k1_s1057
                                 - 1.1560025835494 z_k1_s1060
                                 - 0.388000633256061 z_k1_s1061
                                 - 0.347000641430134 z_k1_s1062
                                 - 1.00200059611762 z_k1_s1063
                                 - 0.940001233815832 z_k1_s1064
                                 - 1.27100440820994 z_k1_s1065
                                 - 0.362000524741908 z_k1_s1066
                                 - 0.361000610214559 z_k1_s1067
                                 - 0.535001109574585 z_k1_s1070
                                 - 0.310000989432133 z_k1_s1071
                                 - 0.813002379140705 z_k1_s1072
                                 - 0.259000883659602 z_k1_s1073
                                 - 0.300000968692421 z_k1_s1074
                                 - 0.630001306601848 z_k1_s1075
                                 - 0.212000786182957 z_k1_s1076
                                 - 0.406002228036562 z_k1_s1077
                                 - 0.133999923762822 z_k1_s1100
                                 - 0.36499870548904 z_k1_s1101
                                 - 0.174999754273936 z_k1_s1102
                                 - 0.207999795463246 z_k1_s1103
                                 - 0.162999944741316 z_k1_s1104
                                 - 0.504999412867559 z_k1_s1105
                                 - 0.326999768985424 z_k1_s1106
                                 - 0.633999436916774 z_k1_s1107
                                 - 0.407999170236319 z_k1_s1110
                                 - 0.363999788131887 z_k1_s1111
                                 - 0.341999060580682 z_k1_s1112
                                 - 0.316998704953852 z_k1_s1113
                                 - 0.303998934627497 z_k1_s1114
                                 - 0.110999920584738 z_k1_s1115
                                 - 0.086999993782578 z_k1_s1116
                                 - 0.139999918512264 z_k1_s1117
                                 - 0.319000880968111 z_k1_s1120
                                 - 0.870003858644866 z_k1_s1121
                                 - 0.212000852823532 z_k1_s1122
                                 - 0.548001169468906 z_k1_s1123
                                 - 0.525002121388494 z_k1_s1124
                                 - 0.402999859026004 z_k1_s1125
                                 - 0.393001038889095 z_k1_s1126
                                 - 0.734003568411707 z_k1_s1127
                                 - 0.79899844817576 z_k1_s1130
                                 - 0.25500034129738 z_k1_s1131
                                 - 1.23200287074076 z_k1_s1132
                                 - 1.13800217501329 z_k1_s1133
                                 - 0.46700035374708 z_k1_s1134
                                 - 0.388000351455449 z_k1_s1135
                                 - 0.423000642271026 z_k1_s1136
                                 - 0.275000436377637 z_k1_s1137
                                 - 0.943996547701451 z_k1_s1140
                                 - 1.26299743247005 z_k1_s1141
                                 - 1.25799854800647 z_k1_s1142
                                 - 1.99199691324244 z_k1_s1143
                                 - 0.746999968340795 z_k1_s1144
                                 - 0.953003546734927 z_k1_s1145
                                 - 1.34399737477809 z_k1_s1146
                                 - 0.505999695202095 z_k1_s1147
                                 - 0.349000286307575 z_k1_s1150
                                 - 0.930000863994544 z_k1_s1151
                                 - 0.515000535332544 z_k1_s1152
                                 - 1.28999949192546 z_k1_s1153
                                 - 0.433000553958968 z_k1_s1154
                                 - 1.13700080980588 z_k1_s1155
                                 - 0.817002029422841 z_k1_s1156
                                 - 0.407000359152424 z_k1_s1157
                                 - 1.60699900967497 z_k1_s1160
                                 - 0.609999086006977 z_k1_s1161
                                 - 0.47999914249753 z_k1_s1162
                                 - 0.992999860513968 z_k1_s1163
                                 - 0.779998255059088 z_k1_s1164
                                 - 1.21499985056841 z_k1_s1165
                                 - 0.579999315350581 z_k1_s1166
                                 - 0.764996260734499 z_k1_s1167
                                 - 0.251000408759443 z_k1_s1170
                                 - 0.588999076105205 z_k1_s1171
                                 - 0.348000381343807 z_k1_s1172
                                 - 0.412999277333546 z_k1_s1173
                                 - 0.210999920165991 z_k1_s1174
                                 - 0.119000069880797 z_k1_s1175
                                 - 0.363000503341363 z_k1_s1176
                                 - 0.154999872875031 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.41623 w_k1_ADM_s000
                                 + 1.32388 w_k1_AXP_s000 + 1.4597 w_k1_DIS_s000
                                 + 1.27461 w_k1_FDX_s000
                                 + 1.53351 w_k1_IBM_s000
                                 + 1.06021 w_k1_INTC_s000
                                 + 1.38869 w_k1_KO_s000 + 1.66712 w_k1_MAT_s000
                                 + 1.22165 w_k1_MSFT_s000
                                 + 1.09092 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.904529 w_k1_ADM_s000
                                 + 0.786835 w_k1_AXP_s000
                                 + 0.769747 w_k1_DIS_s000
                                 + 0.879731 w_k1_FDX_s000
                                 + 1.20693 w_k1_IBM_s000
                                 + 0.703605 w_k1_INTC_s000
                                 + 0.815055 w_k1_KO_s000
                                 + 1.42883 w_k1_MAT_s000
                                 + 0.81343 w_k1_MSFT_s000
                                 + 0.858294 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.18704 w_k1_ADM_s000
                                 + 1.20129 w_k1_AXP_s000
                                 + 1.31231 w_k1_DIS_s000
                                 + 1.25126 w_k1_FDX_s000
                                 + 1.14287 w_k1_IBM_s000
                                 + 1.02954 w_k1_INTC_s000
                                 + 1.03785 w_k1_KO_s000 + 1.07562 w_k1_MAT_s000
                                 + 1.20287 w_k1_MSFT_s000
                                 + 1.06172 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.20634 w_k1_ADM_s000
                                 + 0.725654 w_k1_AXP_s000
                                 + 0.87468 w_k1_DIS_s000
                                 + 0.829395 w_k1_FDX_s000
                                 + 0.838236 w_k1_IBM_s000
                                 + 0.634206 w_k1_INTC_s000
                                 + 0.787135 w_k1_KO_s000 + 0.8958 w_k1_MAT_s000
                                 + 0.986798 w_k1_MSFT_s000
                                 + 0.898312 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.880796 w_k1_ADM_s000
                                 + 0.979885 w_k1_AXP_s000
                                 + 1.08288 w_k1_DIS_s000
                                 + 0.935037 w_k1_FDX_s000
                                 + 1.63969 w_k1_IBM_s000
                                 + 0.928543 w_k1_INTC_s000
                                 + 1.1901 w_k1_KO_s000 + 2.09475 w_k1_MAT_s000
                                 + 0.994504 w_k1_MSFT_s000
                                 + 0.967495 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.770813 w_k1_ADM_s000
                                 + 0.77097 w_k1_AXP_s000
                                 + 0.614493 w_k1_DIS_s000
                                 + 0.783045 w_k1_FDX_s000
                                 + 0.788939 w_k1_IBM_s000
                                 + 0.597215 w_k1_INTC_s000
                                 + 0.757589 w_k1_KO_s000
                                 + 1.00522 w_k1_MAT_s000
                                 + 0.838847 w_k1_MSFT_s000
                                 + 0.850016 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.993468 w_k1_ADM_s000
                                 + 1.01639 w_k1_AXP_s000
                                 + 0.933842 w_k1_DIS_s000
                                 + 1.12016 w_k1_FDX_s000
                                 + 1.00904 w_k1_IBM_s000
                                 + 0.771186 w_k1_INTC_s000
                                 + 0.888642 w_k1_KO_s000
                                 + 1.29685 w_k1_MAT_s000
                                 + 0.993272 w_k1_MSFT_s000
                                 + 0.949409 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.956893 w_k1_ADM_s000
                                 + 0.772527 w_k1_AXP_s000
                                 + 1.16432 w_k1_DIS_s000
                                 + 1.07917 w_k1_FDX_s000
                                 + 0.993535 w_k1_IBM_s000
                                 + 1.03233 w_k1_INTC_s000
                                 + 1.06874 w_k1_KO_s000
                                 + 0.629554 w_k1_MAT_s000
                                 + 1.21597 w_k1_MSFT_s000
                                 + 0.925476 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.75534 w_k1_ADM_s001
                                 + 0.791989 w_k1_AXP_s001
                                 + 1.20144 w_k1_DIS_s001
                                 + 0.854887 w_k1_FDX_s001
                                 + 1.28843 w_k1_IBM_s001
                                 + 0.771759 w_k1_INTC_s001
                                 + 1.04528 w_k1_KO_s001 + 1.17395 w_k1_MAT_s001
                                 + 0.895271 w_k1_MSFT_s001
                                 + 0.914265 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.786903 w_k1_ADM_s001
                                 + 0.797443 w_k1_AXP_s001
                                 + 0.643678 w_k1_DIS_s001
                                 + 0.772489 w_k1_FDX_s001
                                 + 0.885321 w_k1_IBM_s001
                                 + 0.604365 w_k1_INTC_s001
                                 + 0.772266 w_k1_KO_s001
                                 + 0.843579 w_k1_MAT_s001
                                 + 0.710807 w_k1_MSFT_s001
                                 + 0.847586 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.15469 w_k1_ADM_s001
                                 + 1.3927 w_k1_AXP_s001 + 1.12984 w_k1_DIS_s001
                                 + 1.08198 w_k1_FDX_s001
                                 + 1.71552 w_k1_IBM_s001
                                 + 0.978994 w_k1_INTC_s001
                                 + 1.31756 w_k1_KO_s001 + 1.62171 w_k1_MAT_s001
                                 + 0.964119 w_k1_MSFT_s001
                                 + 1.10615 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.856538 w_k1_ADM_s001
                                 + 0.829565 w_k1_AXP_s001
                                 + 1.04064 w_k1_DIS_s001
                                 + 1.00582 w_k1_FDX_s001
                                 + 0.88009 w_k1_IBM_s001
                                 + 0.867163 w_k1_INTC_s001
                                 + 0.868151 w_k1_KO_s001
                                 + 0.586145 w_k1_MAT_s001
                                 + 0.972013 w_k1_MSFT_s001
                                 + 0.825501 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.07899 w_k1_ADM_s001
                                 + 1.11907 w_k1_AXP_s001
                                 + 1.06525 w_k1_DIS_s001
                                 + 1.20125 w_k1_FDX_s001
                                 + 0.693662 w_k1_IBM_s001
                                 + 1.06015 w_k1_INTC_s001
                                 + 0.962484 w_k1_KO_s001
                                 + 0.986661 w_k1_MAT_s001
                                 + 1.06854 w_k1_MSFT_s001
                                 + 1.06054 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.904596 w_k1_ADM_s001
                                 + 1.01787 w_k1_AXP_s001
                                 + 0.772998 w_k1_DIS_s001
                                 + 0.896883 w_k1_FDX_s001
                                 + 1.01887 w_k1_IBM_s001
                                 + 0.739049 w_k1_INTC_s001
                                 + 0.901767 w_k1_KO_s001
                                 + 1.31943 w_k1_MAT_s001
                                 + 0.804754 w_k1_MSFT_s001
                                 + 0.9327 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.07391 w_k1_ADM_s001
                                 + 1.25692 w_k1_AXP_s001
                                 + 1.47174 w_k1_DIS_s001
                                 + 1.37172 w_k1_FDX_s001
                                 + 0.987374 w_k1_IBM_s001
                                 + 1.08643 w_k1_INTC_s001
                                 + 1.01309 w_k1_KO_s001 + 1.24091 w_k1_MAT_s001
                                 + 1.16553 w_k1_MSFT_s001 + 1.011 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.00392 w_k1_ADM_s001
                                 + 1.13224 w_k1_AXP_s001
                                 + 1.39402 w_k1_DIS_s001
                                 + 1.51109 w_k1_FDX_s001
                                 + 0.676283 w_k1_IBM_s001
                                 + 1.09989 w_k1_INTC_s001
                                 + 0.953769 w_k1_KO_s001
                                 + 1.01487 w_k1_MAT_s001
                                 + 1.42421 w_k1_MSFT_s001
                                 + 1.16883 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.2343 w_k1_ADM_s002
                                 + 1.48039 w_k1_AXP_s002
                                 + 1.20482 w_k1_DIS_s002
                                 + 1.31255 w_k1_FDX_s002
                                 + 1.06934 w_k1_IBM_s002
                                 + 1.17635 w_k1_INTC_s002 + 1.0454 w_k1_KO_s002
                                 + 1.14361 w_k1_MAT_s002
                                 + 1.07689 w_k1_MSFT_s002
                                 + 1.09075 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.965458 w_k1_ADM_s002
                                 + 0.960946 w_k1_AXP_s002
                                 + 0.733564 w_k1_DIS_s002
                                 + 0.85981 w_k1_FDX_s002
                                 + 0.639039 w_k1_IBM_s002
                                 + 0.854606 w_k1_INTC_s002
                                 + 0.793612 w_k1_KO_s002
                                 + 0.630546 w_k1_MAT_s002
                                 + 0.804603 w_k1_MSFT_s002
                                 + 0.932262 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.10941 w_k1_ADM_s002
                                 + 0.963004 w_k1_AXP_s002
                                 + 1.14899 w_k1_DIS_s002
                                 + 1.13184 w_k1_FDX_s002
                                 + 0.724136 w_k1_IBM_s002
                                 + 1.07876 w_k1_INTC_s002
                                 + 0.960659 w_k1_KO_s002
                                 + 0.722725 w_k1_MAT_s002
                                 + 0.9331 w_k1_MSFT_s002
                                 + 0.978419 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.1419 w_k1_ADM_s002
                                 + 0.882162 w_k1_AXP_s002
                                 + 1.3881 w_k1_DIS_s002 + 1.09251 w_k1_FDX_s002
                                 + 1.03302 w_k1_IBM_s002
                                 + 1.00049 w_k1_INTC_s002
                                 + 1.29712 w_k1_KO_s002 + 1.04505 w_k1_MAT_s002
                                 + 1.16425 w_k1_MSFT_s002
                                 + 1.34443 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.73803 w_k1_ADM_s002
                                 + 0.742587 w_k1_AXP_s002
                                 + 0.82954 w_k1_DIS_s002
                                 + 0.870992 w_k1_FDX_s002
                                 + 0.857576 w_k1_IBM_s002
                                 + 0.867133 w_k1_INTC_s002
                                 + 0.938469 w_k1_KO_s002
                                 + 0.927781 w_k1_MAT_s002
                                 + 0.836423 w_k1_MSFT_s002
                                 + 0.89184 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.12322 w_k1_ADM_s002
                                 + 0.9979 w_k1_AXP_s002 + 1.04853 w_k1_DIS_s002
                                 + 0.983634 w_k1_FDX_s002
                                 + 1.15703 w_k1_IBM_s002
                                 + 1.04091 w_k1_INTC_s002
                                 + 1.24934 w_k1_KO_s002 + 1.45091 w_k1_MAT_s002
                                 + 0.994317 w_k1_MSFT_s002
                                 + 1.20353 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.04422 w_k1_ADM_s002
                                 + 1.12851 w_k1_AXP_s002
                                 + 0.80003 w_k1_DIS_s002
                                 + 0.912356 w_k1_FDX_s002
                                 + 0.92403 w_k1_IBM_s002
                                 + 0.661427 w_k1_INTC_s002
                                 + 0.861835 w_k1_KO_s002 + 1.1115 w_k1_MAT_s002
                                 + 0.694849 w_k1_MSFT_s002
                                 + 1.03231 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.2858 w_k1_ADM_s002
                                 + 1.65937 w_k1_AXP_s002
                                 + 1.59467 w_k1_DIS_s002
                                 + 1.29345 w_k1_FDX_s002
                                 + 1.46803 w_k1_IBM_s002
                                 + 1.21261 w_k1_INTC_s002
                                 + 1.49861 w_k1_KO_s002 + 1.6298 w_k1_MAT_s002
                                 + 1.10395 w_k1_MSFT_s002
                                 + 1.32909 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.971917 w_k1_ADM_s003
                                 + 1.3201 w_k1_AXP_s003
                                 + 0.808412 w_k1_DIS_s003
                                 + 1.06441 w_k1_FDX_s003
                                 + 1.08393 w_k1_IBM_s003
                                 + 0.930672 w_k1_INTC_s003
                                 + 0.872471 w_k1_KO_s003
                                 + 1.28739 w_k1_MAT_s003
                                 + 0.931866 w_k1_MSFT_s003
                                 + 0.917514 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.888062 w_k1_ADM_s003
                                 + 1.05855 w_k1_AXP_s003
                                 + 0.816616 w_k1_DIS_s003
                                 + 1.13992 w_k1_FDX_s003
                                 + 0.75129 w_k1_IBM_s003
                                 + 1.01371 w_k1_INTC_s003
                                 + 0.795914 w_k1_KO_s003
                                 + 0.511496 w_k1_MAT_s003
                                 + 0.939385 w_k1_MSFT_s003
                                 + 0.821515 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.699418 w_k1_ADM_s003
                                 + 0.755456 w_k1_AXP_s003
                                 + 0.526563 w_k1_DIS_s003
                                 + 0.808742 w_k1_FDX_s003
                                 + 0.628186 w_k1_IBM_s003
                                 + 1.00713 w_k1_INTC_s003
                                 + 0.667251 w_k1_KO_s003
                                 + 0.583831 w_k1_MAT_s003
                                 + 0.796416 w_k1_MSFT_s003
                                 + 0.848554 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.975485 w_k1_ADM_s003
                                 + 1.10737 w_k1_AXP_s003
                                 + 1.05655 w_k1_DIS_s003
                                 + 1.04399 w_k1_FDX_s003 + 1.4936 w_k1_IBM_s003
                                 + 1.17142 w_k1_INTC_s003
                                 + 1.25826 w_k1_KO_s003 + 1.64226 w_k1_MAT_s003
                                 + 1.17713 w_k1_MSFT_s003
                                 + 1.13733 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.24638 w_k1_ADM_s003
                                 + 1.24435 w_k1_AXP_s003
                                 + 1.11901 w_k1_DIS_s003
                                 + 1.30368 w_k1_FDX_s003
                                 + 0.991145 w_k1_IBM_s003
                                 + 1.15719 w_k1_INTC_s003
                                 + 1.02599 w_k1_KO_s003 + 1.12222 w_k1_MAT_s003
                                 + 1.14783 w_k1_MSFT_s003
                                 + 0.928396 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.43385 w_k1_ADM_s003
                                 + 1.4429 w_k1_AXP_s003 + 1.54421 w_k1_DIS_s003
                                 + 1.36847 w_k1_FDX_s003
                                 + 1.69131 w_k1_IBM_s003
                                 + 1.21641 w_k1_INTC_s003
                                 + 1.30775 w_k1_KO_s003 + 1.32724 w_k1_MAT_s003
                                 + 1.26139 w_k1_MSFT_s003
                                 + 1.21341 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.754576 w_k1_ADM_s003
                                 + 0.920068 w_k1_AXP_s003
                                 + 0.943081 w_k1_DIS_s003
                                 + 0.987675 w_k1_FDX_s003
                                 + 1.15716 w_k1_IBM_s003
                                 + 0.81471 w_k1_INTC_s003
                                 + 0.960007 w_k1_KO_s003
                                 + 0.947692 w_k1_MAT_s003
                                 + 0.837591 w_k1_MSFT_s003
                                 + 0.982841 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.650193 w_k1_ADM_s003
                                 + 0.674272 w_k1_AXP_s003
                                 + 0.701645 w_k1_DIS_s003
                                 + 0.75641 w_k1_FDX_s003
                                 + 0.971725 w_k1_IBM_s003
                                 + 0.747879 w_k1_INTC_s003
                                 + 0.840644 w_k1_KO_s003
                                 + 0.761909 w_k1_MAT_s003
                                 + 0.772618 w_k1_MSFT_s003
                                 + 0.816676 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.17316 w_k1_ADM_s004
                                 + 0.89687 w_k1_AXP_s004
                                 + 0.780304 w_k1_DIS_s004
                                 + 0.924515 w_k1_FDX_s004
                                 + 0.601648 w_k1_IBM_s004
                                 + 0.853647 w_k1_INTC_s004
                                 + 0.827553 w_k1_KO_s004
                                 + 0.654379 w_k1_MAT_s004
                                 + 0.844708 w_k1_MSFT_s004
                                 + 0.858956 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.915533 w_k1_ADM_s004
                                 + 0.627213 w_k1_AXP_s004
                                 + 0.452528 w_k1_DIS_s004
                                 + 0.675647 w_k1_FDX_s004
                                 + 0.869842 w_k1_IBM_s004
                                 + 0.875983 w_k1_INTC_s004
                                 + 0.654846 w_k1_KO_s004
                                 + 0.589416 w_k1_MAT_s004
                                 + 0.774519 w_k1_MSFT_s004
                                 + 0.819586 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.21366 w_k1_ADM_s004
                                 + 1.16419 w_k1_AXP_s004 + 1.0381 w_k1_DIS_s004
                                 + 1.01687 w_k1_FDX_s004
                                 + 1.27032 w_k1_IBM_s004
                                 + 0.989566 w_k1_INTC_s004
                                 + 1.26844 w_k1_KO_s004 + 1.5104 w_k1_MAT_s004
                                 + 1.07271 w_k1_MSFT_s004
                                 + 1.05006 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.20881 w_k1_ADM_s004
                                 + 1.20146 w_k1_AXP_s004 + 1.0759 w_k1_DIS_s004
                                 + 0.9043 w_k1_FDX_s004 + 1.90021 w_k1_IBM_s004
                                 + 1.03923 w_k1_INTC_s004
                                 + 1.26632 w_k1_KO_s004 + 1.51573 w_k1_MAT_s004
                                 + 1.11663 w_k1_MSFT_s004
                                 + 1.10935 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.763539 w_k1_ADM_s004
                                 + 0.814206 w_k1_AXP_s004
                                 + 0.850199 w_k1_DIS_s004
                                 + 0.729241 w_k1_FDX_s004
                                 + 1.30125 w_k1_IBM_s004
                                 + 0.828356 w_k1_INTC_s004
                                 + 0.989094 w_k1_KO_s004
                                 + 0.933076 w_k1_MAT_s004
                                 + 0.849444 w_k1_MSFT_s004
                                 + 0.920858 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.12323 w_k1_ADM_s004
                                 + 0.822399 w_k1_AXP_s004
                                 + 1.32146 w_k1_DIS_s004
                                 + 1.18195 w_k1_FDX_s004
                                 + 0.604744 w_k1_IBM_s004
                                 + 1.14057 w_k1_INTC_s004
                                 + 1.01685 w_k1_KO_s004
                                 + 0.702168 w_k1_MAT_s004
                                 + 1.51488 w_k1_MSFT_s004
                                 + 1.00676 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.3017 w_k1_ADM_s004
                                 + 1.61059 w_k1_AXP_s004
                                 + 1.59943 w_k1_DIS_s004
                                 + 1.26508 w_k1_FDX_s004
                                 + 1.11212 w_k1_IBM_s004
                                 + 1.14486 w_k1_INTC_s004
                                 + 1.34837 w_k1_KO_s004 + 1.1731 w_k1_MAT_s004
                                 + 1.34099 w_k1_MSFT_s004
                                 + 1.27753 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.55773 w_k1_ADM_s004
                                 + 1.20308 w_k1_AXP_s004
                                 + 0.91026 w_k1_DIS_s004
                                 + 0.962594 w_k1_FDX_s004
                                 + 0.963941 w_k1_IBM_s004
                                 + 0.739675 w_k1_INTC_s004
                                 + 0.91032 w_k1_KO_s004
                                 + 0.766459 w_k1_MAT_s004
                                 + 0.773659 w_k1_MSFT_s004
                                 + 1.03642 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.847965 w_k1_ADM_s005
                                 + 0.873416 w_k1_AXP_s005
                                 + 0.747881 w_k1_DIS_s005
                                 + 0.763773 w_k1_FDX_s005
                                 + 1.30435 w_k1_IBM_s005
                                 + 0.750119 w_k1_INTC_s005
                                 + 0.908427 w_k1_KO_s005
                                 + 1.19188 w_k1_MAT_s005
                                 + 0.844264 w_k1_MSFT_s005
                                 + 0.948943 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.24742 w_k1_ADM_s005
                                 + 1.17242 w_k1_AXP_s005
                                 + 0.876892 w_k1_DIS_s005
                                 + 0.954132 w_k1_FDX_s005
                                 + 0.973425 w_k1_IBM_s005
                                 + 0.949246 w_k1_INTC_s005
                                 + 0.836608 w_k1_KO_s005
                                 + 0.880209 w_k1_MAT_s005
                                 + 0.779843 w_k1_MSFT_s005
                                 + 0.855551 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.09116 w_k1_ADM_s005
                                 + 1.02941 w_k1_AXP_s005
                                 + 1.11927 w_k1_DIS_s005
                                 + 1.21676 w_k1_FDX_s005
                                 + 0.632738 w_k1_IBM_s005
                                 + 1.16081 w_k1_INTC_s005
                                 + 0.877594 w_k1_KO_s005
                                 + 0.59824 w_k1_MAT_s005
                                 + 1.07468 w_k1_MSFT_s005
                                 + 0.939119 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.49262 w_k1_ADM_s005
                                 + 1.37454 w_k1_AXP_s005
                                 + 1.54477 w_k1_DIS_s005 + 1.2347 w_k1_FDX_s005
                                 + 1.22092 w_k1_IBM_s005
                                 + 1.14298 w_k1_INTC_s005 + 1.3712 w_k1_KO_s005
                                 + 1.33009 w_k1_MAT_s005
                                 + 1.08786 w_k1_MSFT_s005 + 1.1815 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.890829 w_k1_ADM_s005
                                 + 0.910825 w_k1_AXP_s005
                                 + 0.963477 w_k1_DIS_s005
                                 + 0.787429 w_k1_FDX_s005
                                 + 1.00662 w_k1_IBM_s005
                                 + 1.02184 w_k1_INTC_s005
                                 + 1.03959 w_k1_KO_s005 + 0.92363 w_k1_MAT_s005
                                 + 0.829708 w_k1_MSFT_s005
                                 + 0.996461 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.1355 w_k1_ADM_s005
                                 + 0.960593 w_k1_AXP_s005
                                 + 0.946735 w_k1_DIS_s005
                                 + 0.941073 w_k1_FDX_s005
                                 + 1.19815 w_k1_IBM_s005
                                 + 0.935498 w_k1_INTC_s005
                                 + 1.07773 w_k1_KO_s005 + 1.17431 w_k1_MAT_s005
                                 + 1.06893 w_k1_MSFT_s005
                                 + 1.08258 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.934465 w_k1_ADM_s005
                                 + 0.799954 w_k1_AXP_s005
                                 + 0.748244 w_k1_DIS_s005
                                 + 0.825536 w_k1_FDX_s005
                                 + 0.693946 w_k1_IBM_s005
                                 + 0.867673 w_k1_INTC_s005
                                 + 0.758074 w_k1_KO_s005
                                 + 0.422148 w_k1_MAT_s005
                                 + 0.712173 w_k1_MSFT_s005
                                 + 0.832687 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.1543 w_k1_ADM_s005
                                 + 1.2188 w_k1_AXP_s005 + 1.09763 w_k1_DIS_s005
                                 + 1.01388 w_k1_FDX_s005
                                 + 1.66236 w_k1_IBM_s005
                                 + 1.01346 w_k1_INTC_s005
                                 + 1.34024 w_k1_KO_s005 + 2.03286 w_k1_MAT_s005
                                 + 0.91442 w_k1_MSFT_s005
                                 + 1.16976 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.1468 w_k1_ADM_s006
                                 + 1.1015 w_k1_AXP_s006 + 1.11586 w_k1_DIS_s006
                                 + 1.0892 w_k1_FDX_s006
                                 + 0.988832 w_k1_IBM_s006
                                 + 1.0329 w_k1_INTC_s006 + 1.04837 w_k1_KO_s006
                                 + 1.00581 w_k1_MAT_s006
                                 + 1.04825 w_k1_MSFT_s006
                                 + 1.03471 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.910953 w_k1_ADM_s006
                                 + 0.712364 w_k1_AXP_s006
                                 + 0.626541 w_k1_DIS_s006
                                 + 0.744588 w_k1_FDX_s006
                                 + 0.739423 w_k1_IBM_s006
                                 + 0.862746 w_k1_INTC_s006
                                 + 0.8072 w_k1_KO_s006 + 0.824759 w_k1_MAT_s006
                                 + 0.89811 w_k1_MSFT_s006
                                 + 1.04671 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.956829 w_k1_ADM_s006
                                 + 0.899724 w_k1_AXP_s006
                                 + 0.762791 w_k1_DIS_s006
                                 + 0.946879 w_k1_FDX_s006
                                 + 0.856105 w_k1_IBM_s006
                                 + 1.12463 w_k1_INTC_s006
                                 + 0.917745 w_k1_KO_s006
                                 + 0.792661 w_k1_MAT_s006
                                 + 0.956275 w_k1_MSFT_s006
                                 + 1.09347 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.944944 w_k1_ADM_s006
                                 + 0.805641 w_k1_AXP_s006
                                 + 0.625534 w_k1_DIS_s006
                                 + 0.906038 w_k1_FDX_s006
                                 + 0.684532 w_k1_IBM_s006
                                 + 0.783731 w_k1_INTC_s006
                                 + 0.739662 w_k1_KO_s006
                                 + 0.936374 w_k1_MAT_s006
                                 + 0.915526 w_k1_MSFT_s006
                                 + 1.03907 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.02699 w_k1_ADM_s006
                                 + 1.09477 w_k1_AXP_s006
                                 + 1.47909 w_k1_DIS_s006 + 1.2382 w_k1_FDX_s006
                                 + 1.16563 w_k1_IBM_s006
                                 + 1.20536 w_k1_INTC_s006
                                 + 1.19832 w_k1_KO_s006
                                 + 0.968623 w_k1_MAT_s006
                                 + 1.17814 w_k1_MSFT_s006
                                 + 1.14709 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.08311 w_k1_ADM_s006
                                 + 0.809083 w_k1_AXP_s006
                                 + 0.931572 w_k1_DIS_s006
                                 + 0.966178 w_k1_FDX_s006
                                 + 0.786512 w_k1_IBM_s006
                                 + 0.98925 w_k1_INTC_s006
                                 + 0.946633 w_k1_KO_s006
                                 + 0.735921 w_k1_MAT_s006
                                 + 1.02109 w_k1_MSFT_s006
                                 + 1.03712 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.890485 w_k1_ADM_s006
                                 + 1.08709 w_k1_AXP_s006
                                 + 0.963583 w_k1_DIS_s006
                                 + 1.01507 w_k1_FDX_s006
                                 + 1.38249 w_k1_IBM_s006
                                 + 1.10699 w_k1_INTC_s006
                                 + 1.16265 w_k1_KO_s006 + 1.46371 w_k1_MAT_s006
                                 + 1.03232 w_k1_MSFT_s006
                                 + 1.20141 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.25217 w_k1_ADM_s006
                                 + 1.63726 w_k1_AXP_s006
                                 + 1.54932 w_k1_DIS_s006 + 1.4273 w_k1_FDX_s006
                                 + 1.48879 w_k1_IBM_s006
                                 + 1.08095 w_k1_INTC_s006
                                 + 1.22357 w_k1_KO_s006 + 1.92992 w_k1_MAT_s006
                                 + 1.22382 w_k1_MSFT_s006
                                 + 1.31995 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.32948 w_k1_ADM_s007
                                 + 1.68996 w_k1_AXP_s007
                                 + 1.48678 w_k1_DIS_s007
                                 + 1.34967 w_k1_FDX_s007
                                 + 1.11603 w_k1_IBM_s007
                                 + 1.12601 w_k1_INTC_s007
                                 + 1.25421 w_k1_KO_s007 + 1.54469 w_k1_MAT_s007
                                 + 1.35645 w_k1_MSFT_s007
                                 + 1.08664 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.20014 w_k1_ADM_s007
                                 + 0.954825 w_k1_AXP_s007
                                 + 1.14423 w_k1_DIS_s007
                                 + 1.14258 w_k1_FDX_s007
                                 + 0.897681 w_k1_IBM_s007
                                 + 0.85244 w_k1_INTC_s007
                                 + 1.04469 w_k1_KO_s007
                                 + 0.694385 w_k1_MAT_s007
                                 + 1.0322 w_k1_MSFT_s007
                                 + 0.878492 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.995841 w_k1_ADM_s007
                                 + 1.25579 w_k1_AXP_s007
                                 + 1.14705 w_k1_DIS_s007
                                 + 0.968824 w_k1_FDX_s007
                                 + 1.56605 w_k1_IBM_s007
                                 + 0.958271 w_k1_INTC_s007
                                 + 1.22756 w_k1_KO_s007 + 1.38624 w_k1_MAT_s007
                                 + 0.898231 w_k1_MSFT_s007
                                 + 0.886119 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.875561 w_k1_ADM_s007
                                 + 0.957064 w_k1_AXP_s007
                                 + 0.791379 w_k1_DIS_s007
                                 + 0.920152 w_k1_FDX_s007
                                 + 0.881869 w_k1_IBM_s007
                                 + 0.789756 w_k1_INTC_s007
                                 + 0.898766 w_k1_KO_s007
                                 + 0.998822 w_k1_MAT_s007
                                 + 0.947859 w_k1_MSFT_s007
                                 + 0.932888 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.1314 w_k1_ADM_s007
                                 + 0.979436 w_k1_AXP_s007
                                 + 1.19025 w_k1_DIS_s007
                                 + 1.16513 w_k1_FDX_s007
                                 + 1.18656 w_k1_IBM_s007 + 1.021 w_k1_INTC_s007
                                 + 1.43066 w_k1_KO_s007 + 1.05876 w_k1_MAT_s007
                                 + 1.21896 w_k1_MSFT_s007
                                 + 1.04604 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.663707 w_k1_ADM_s007
                                 + 0.73064 w_k1_AXP_s007
                                 + 0.752723 w_k1_DIS_s007
                                 + 0.761724 w_k1_FDX_s007
                                 + 1.01833 w_k1_IBM_s007
                                 + 0.781172 w_k1_INTC_s007
                                 + 0.960658 w_k1_KO_s007
                                 + 0.842781 w_k1_MAT_s007
                                 + 0.871485 w_k1_MSFT_s007
                                 + 0.864727 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.19759 w_k1_ADM_s007
                                 + 1.25955 w_k1_AXP_s007
                                 + 0.860625 w_k1_DIS_s007
                                 + 1.02558 w_k1_FDX_s007
                                 + 1.31654 w_k1_IBM_s007
                                 + 0.900641 w_k1_INTC_s007
                                 + 1.02489 w_k1_KO_s007 + 1.30123 w_k1_MAT_s007
                                 + 1.06817 w_k1_MSFT_s007
                                 + 0.875657 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.927936 w_k1_ADM_s007
                                 + 1.47854 w_k1_AXP_s007
                                 + 1.50787 w_k1_DIS_s007
                                 + 1.20877 w_k1_FDX_s007
                                 + 1.60371 w_k1_IBM_s007
                                 + 1.19186 w_k1_INTC_s007
                                 + 1.42668 w_k1_KO_s007 + 1.98031 w_k1_MAT_s007
                                 + 1.3442 w_k1_MSFT_s007 + 1.07718 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.9913 w_k1_ADM_s010
                                 + 1.1689 w_k1_AXP_s010
                                 + 0.852193 w_k1_DIS_s010
                                 + 0.988045 w_k1_FDX_s010
                                 + 1.17935 w_k1_IBM_s010
                                 + 0.904373 w_k1_INTC_s010
                                 + 1.04052 w_k1_KO_s010 + 1.55044 w_k1_MAT_s010
                                 + 0.920775 w_k1_MSFT_s010
                                 + 1.13703 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.33802 w_k1_ADM_s010
                                 + 1.09444 w_k1_AXP_s010
                                 + 1.63198 w_k1_DIS_s010
                                 + 1.35693 w_k1_FDX_s010
                                 + 0.854835 w_k1_IBM_s010
                                 + 1.40119 w_k1_INTC_s010
                                 + 1.26232 w_k1_KO_s010
                                 + 0.842128 w_k1_MAT_s010
                                 + 1.2869 w_k1_MSFT_s010 + 1.3029 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.994879 w_k1_ADM_s010
                                 + 1.04784 w_k1_AXP_s010
                                 + 1.27047 w_k1_DIS_s010
                                 + 1.02123 w_k1_FDX_s010
                                 + 1.16222 w_k1_IBM_s010
                                 + 1.03525 w_k1_INTC_s010
                                 + 1.34566 w_k1_KO_s010 + 1.19147 w_k1_MAT_s010
                                 + 1.04313 w_k1_MSFT_s010
                                 + 1.15441 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.34101 w_k1_ADM_s010
                                 + 1.38841 w_k1_AXP_s010
                                 + 1.28804 w_k1_DIS_s010
                                 + 1.29748 w_k1_FDX_s010
                                 + 1.02754 w_k1_IBM_s010
                                 + 1.2408 w_k1_INTC_s010 + 1.1126 w_k1_KO_s010
                                 + 1.29475 w_k1_MAT_s010
                                 + 1.12811 w_k1_MSFT_s010
                                 + 1.28455 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.857733 w_k1_ADM_s010
                                 + 0.840558 w_k1_AXP_s010
                                 + 1.20792 w_k1_DIS_s010
                                 + 1.10191 w_k1_FDX_s010
                                 + 0.815782 w_k1_IBM_s010
                                 + 1.19932 w_k1_INTC_s010
                                 + 1.18509 w_k1_KO_s010
                                 + 0.808064 w_k1_MAT_s010
                                 + 1.11388 w_k1_MSFT_s010
                                 + 0.981556 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.906434 w_k1_ADM_s010
                                 + 0.711862 w_k1_AXP_s010
                                 + 0.758904 w_k1_DIS_s010
                                 + 0.854114 w_k1_FDX_s010
                                 + 0.774166 w_k1_IBM_s010
                                 + 0.892226 w_k1_INTC_s010
                                 + 0.965066 w_k1_KO_s010
                                 + 0.773176 w_k1_MAT_s010
                                 + 0.776382 w_k1_MSFT_s010
                                 + 0.886655 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.50058 w_k1_ADM_s010
                                 + 1.60816 w_k1_AXP_s010 + 1.8304 w_k1_DIS_s010
                                 + 1.45354 w_k1_FDX_s010 + 1.411 w_k1_IBM_s010
                                 + 1.45238 w_k1_INTC_s010
                                 + 1.71936 w_k1_KO_s010 + 1.18076 w_k1_MAT_s010
                                 + 1.31713 w_k1_MSFT_s010
                                 + 1.39037 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.715038 w_k1_ADM_s010
                                 + 0.742027 w_k1_AXP_s010
                                 + 0.799072 w_k1_DIS_s010
                                 + 0.787502 w_k1_FDX_s010
                                 + 1.17291 w_k1_IBM_s010
                                 + 0.821862 w_k1_INTC_s010
                                 + 1.03924 w_k1_KO_s010
                                 + 0.932354 w_k1_MAT_s010
                                 + 0.752412 w_k1_MSFT_s010
                                 + 0.873165 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.789698 w_k1_ADM_s011
                                 + 0.740498 w_k1_AXP_s011
                                 + 0.617325 w_k1_DIS_s011
                                 + 0.785195 w_k1_FDX_s011
                                 + 0.860858 w_k1_IBM_s011
                                 + 0.930646 w_k1_INTC_s011
                                 + 0.811007 w_k1_KO_s011
                                 + 0.985685 w_k1_MAT_s011
                                 + 0.83307 w_k1_MSFT_s011
                                 + 0.847394 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.948856 w_k1_ADM_s011
                                 + 1.18323 w_k1_AXP_s011
                                 + 1.15928 w_k1_DIS_s011
                                 + 1.14878 w_k1_FDX_s011
                                 + 1.08582 w_k1_IBM_s011
                                 + 1.33085 w_k1_INTC_s011
                                 + 1.08135 w_k1_KO_s011 + 1.11719 w_k1_MAT_s011
                                 + 1.2552 w_k1_MSFT_s011 + 1.05133 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.862712 w_k1_ADM_s011
                                 + 0.598654 w_k1_AXP_s011
                                 + 0.749425 w_k1_DIS_s011
                                 + 0.861223 w_k1_FDX_s011
                                 + 0.632458 w_k1_IBM_s011
                                 + 0.958464 w_k1_INTC_s011
                                 + 0.820668 w_k1_KO_s011
                                 + 0.530217 w_k1_MAT_s011
                                 + 0.9381 w_k1_MSFT_s011
                                 + 0.850229 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.13538 w_k1_ADM_s011
                                 + 1.35102 w_k1_AXP_s011
                                 + 1.83217 w_k1_DIS_s011
                                 + 1.55427 w_k1_FDX_s011
                                 + 1.20224 w_k1_IBM_s011
                                 + 1.70538 w_k1_INTC_s011
                                 + 1.73768 w_k1_KO_s011 + 1.55041 w_k1_MAT_s011
                                 + 1.74582 w_k1_MSFT_s011
                                 + 1.36129 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.00391 w_k1_ADM_s011
                                 + 0.741366 w_k1_AXP_s011
                                 + 1.20001 w_k1_DIS_s011 + 1.0947 w_k1_FDX_s011
                                 + 0.941312 w_k1_IBM_s011
                                 + 1.06126 w_k1_INTC_s011 + 1.13 w_k1_KO_s011
                                 + 0.694145 w_k1_MAT_s011
                                 + 1.10418 w_k1_MSFT_s011
                                 + 1.08361 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.85539 w_k1_ADM_s011
                                 + 0.940528 w_k1_AXP_s011
                                 + 0.832173 w_k1_DIS_s011
                                 + 0.82488 w_k1_FDX_s011
                                 + 1.10516 w_k1_IBM_s011
                                 + 0.890087 w_k1_INTC_s011
                                 + 0.957695 w_k1_KO_s011
                                 + 1.18341 w_k1_MAT_s011
                                 + 0.902931 w_k1_MSFT_s011
                                 + 1.01518 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.08512 w_k1_ADM_s011
                                 + 1.10048 w_k1_AXP_s011
                                 + 1.10198 w_k1_DIS_s011
                                 + 1.14391 w_k1_FDX_s011
                                 + 0.75567 w_k1_IBM_s011
                                 + 1.02428 w_k1_INTC_s011
                                 + 1.00409 w_k1_KO_s011 + 1.07414 w_k1_MAT_s011
                                 + 1.07541 w_k1_MSFT_s011
                                 + 1.03372 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.965108 w_k1_ADM_s011
                                 + 1.16359 w_k1_AXP_s011
                                 + 1.10398 w_k1_DIS_s011
                                 + 1.15198 w_k1_FDX_s011
                                 + 1.47299 w_k1_IBM_s011
                                 + 1.28961 w_k1_INTC_s011
                                 + 1.33978 w_k1_KO_s011 + 1.60274 w_k1_MAT_s011
                                 + 1.06662 w_k1_MSFT_s011 + 1.1559 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.10912 w_k1_ADM_s012
                                 + 1.41109 w_k1_AXP_s012
                                 + 1.39981 w_k1_DIS_s012
                                 + 1.22292 w_k1_FDX_s012
                                 + 1.10061 w_k1_IBM_s012
                                 + 1.2121 w_k1_INTC_s012 + 1.16318 w_k1_KO_s012
                                 + 1.67153 w_k1_MAT_s012
                                 + 1.28288 w_k1_MSFT_s012
                                 + 1.13633 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.829607 w_k1_ADM_s012
                                 + 0.881713 w_k1_AXP_s012
                                 + 1.14961 w_k1_DIS_s012
                                 + 1.03426 w_k1_FDX_s012
                                 + 0.869213 w_k1_IBM_s012
                                 + 0.922046 w_k1_INTC_s012
                                 + 1.04958 w_k1_KO_s012
                                 + 0.694013 w_k1_MAT_s012
                                 + 1.02004 w_k1_MSFT_s012
                                 + 1.04847 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.714177 w_k1_ADM_s012
                                 + 0.994356 w_k1_AXP_s012
                                 + 0.965436 w_k1_DIS_s012
                                 + 0.847749 w_k1_FDX_s012
                                 + 1.2047 w_k1_IBM_s012
                                 + 0.84155 w_k1_INTC_s012
                                 + 1.24011 w_k1_KO_s012 + 1.36733 w_k1_MAT_s012
                                 + 0.932454 w_k1_MSFT_s012
                                 + 0.927517 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.15857 w_k1_ADM_s012
                                 + 1.30285 w_k1_AXP_s012
                                 + 1.69923 w_k1_DIS_s012
                                 + 1.11495 w_k1_FDX_s012
                                 + 1.63797 w_k1_IBM_s012
                                 + 1.24748 w_k1_INTC_s012
                                 + 1.42283 w_k1_KO_s012 + 1.68694 w_k1_MAT_s012
                                 + 1.29913 w_k1_MSFT_s012
                                 + 1.37263 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.719521 w_k1_ADM_s012
                                 + 0.877775 w_k1_AXP_s012
                                 + 0.74592 w_k1_DIS_s012
                                 + 0.823493 w_k1_FDX_s012
                                 + 0.849315 w_k1_IBM_s012
                                 + 0.798461 w_k1_INTC_s012
                                 + 0.687167 w_k1_KO_s012
                                 + 0.692553 w_k1_MAT_s012
                                 + 0.76294 w_k1_MSFT_s012
                                 + 0.747864 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.08716 w_k1_ADM_s012
                                 + 1.02441 w_k1_AXP_s012
                                 + 0.813304 w_k1_DIS_s012
                                 + 0.933659 w_k1_FDX_s012
                                 + 0.843514 w_k1_IBM_s012
                                 + 0.966158 w_k1_INTC_s012
                                 + 0.814429 w_k1_KO_s012
                                 + 0.783055 w_k1_MAT_s012
                                 + 0.828456 w_k1_MSFT_s012
                                 + 0.968274 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.883094 w_k1_ADM_s012
                                 + 0.596045 w_k1_AXP_s012
                                 + 0.669189 w_k1_DIS_s012
                                 + 0.813844 w_k1_FDX_s012
                                 + 0.58889 w_k1_IBM_s012
                                 + 0.860318 w_k1_INTC_s012
                                 + 0.762616 w_k1_KO_s012
                                 + 0.596947 w_k1_MAT_s012
                                 + 0.947669 w_k1_MSFT_s012
                                 + 0.804735 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.665775 w_k1_ADM_s012
                                 + 0.733921 w_k1_AXP_s012
                                 + 0.693283 w_k1_DIS_s012
                                 + 0.698956 w_k1_FDX_s012
                                 + 0.930594 w_k1_IBM_s012
                                 + 1.01446 w_k1_INTC_s012
                                 + 0.845577 w_k1_KO_s012
                                 + 0.791168 w_k1_MAT_s012
                                 + 0.884488 w_k1_MSFT_s012
                                 + 0.822684 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.834062 w_k1_ADM_s013
                                 + 0.818409 w_k1_AXP_s013
                                 + 0.851167 w_k1_DIS_s013
                                 + 0.882439 w_k1_FDX_s013
                                 + 0.704895 w_k1_IBM_s013
                                 + 0.836918 w_k1_INTC_s013
                                 + 0.818252 w_k1_KO_s013
                                 + 0.678592 w_k1_MAT_s013
                                 + 0.87517 w_k1_MSFT_s013
                                 + 0.840723 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.16773 w_k1_ADM_s013
                                 + 1.27595 w_k1_AXP_s013
                                 + 1.32498 w_k1_DIS_s013
                                 + 1.34131 w_k1_FDX_s013
                                 + 0.965353 w_k1_IBM_s013
                                 + 1.38041 w_k1_INTC_s013
                                 + 1.11007 w_k1_KO_s013 + 1.00014 w_k1_MAT_s013
                                 + 1.25046 w_k1_MSFT_s013
                                 + 0.978789 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.939943 w_k1_ADM_s013
                                 + 1.16516 w_k1_AXP_s013 + 0.9862 w_k1_DIS_s013
                                 + 0.971183 w_k1_FDX_s013
                                 + 1.20188 w_k1_IBM_s013
                                 + 0.992362 w_k1_INTC_s013
                                 + 1.14029 w_k1_KO_s013 + 1.51081 w_k1_MAT_s013
                                 + 0.927026 w_k1_MSFT_s013
                                 + 1.06884 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.03173 w_k1_ADM_s013
                                 + 0.867946 w_k1_AXP_s013
                                 + 1.0792 w_k1_DIS_s013 + 1.06155 w_k1_FDX_s013
                                 + 0.95897 w_k1_IBM_s013
                                 + 0.877217 w_k1_INTC_s013
                                 + 1.02094 w_k1_KO_s013 + 0.89149 w_k1_MAT_s013
                                 + 1.06266 w_k1_MSFT_s013
                                 + 1.05806 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.02021 w_k1_ADM_s013
                                 + 1.06551 w_k1_AXP_s013
                                 + 1.29383 w_k1_DIS_s013
                                 + 1.13484 w_k1_FDX_s013
                                 + 1.21764 w_k1_IBM_s013
                                 + 1.04579 w_k1_INTC_s013
                                 + 1.35174 w_k1_KO_s013
                                 + 0.812673 w_k1_MAT_s013
                                 + 1.17141 w_k1_MSFT_s013
                                 + 1.18353 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.960794 w_k1_ADM_s013
                                 + 0.895751 w_k1_AXP_s013
                                 + 0.921284 w_k1_DIS_s013
                                 + 0.826456 w_k1_FDX_s013
                                 + 1.40578 w_k1_IBM_s013
                                 + 1.03864 w_k1_INTC_s013
                                 + 1.25156 w_k1_KO_s013 + 1.46517 w_k1_MAT_s013
                                 + 0.98453 w_k1_MSFT_s013
                                 + 1.05073 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.35419 w_k1_ADM_s013
                                 + 1.56527 w_k1_AXP_s013 + 1.7364 w_k1_DIS_s013
                                 + 1.35328 w_k1_FDX_s013
                                 + 1.26808 w_k1_IBM_s013
                                 + 1.26866 w_k1_INTC_s013
                                 + 1.45072 w_k1_KO_s013 + 1.42869 w_k1_MAT_s013
                                 + 1.37939 w_k1_MSFT_s013
                                 + 1.37171 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.866011 w_k1_ADM_s013
                                 + 0.659781 w_k1_AXP_s013
                                 + 0.659704 w_k1_DIS_s013
                                 + 0.787021 w_k1_FDX_s013
                                 + 0.907079 w_k1_IBM_s013
                                 + 0.83348 w_k1_INTC_s013
                                 + 0.886607 w_k1_KO_s013
                                 + 0.742677 w_k1_MAT_s013
                                 + 0.843705 w_k1_MSFT_s013
                                 + 0.936826 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.07533 w_k1_ADM_s014
                                 + 1.0888 w_k1_AXP_s014
                                 + 0.869354 w_k1_DIS_s014
                                 + 0.910954 w_k1_FDX_s014
                                 + 0.928692 w_k1_IBM_s014
                                 + 1.02785 w_k1_INTC_s014
                                 + 0.891816 w_k1_KO_s014
                                 + 1.02229 w_k1_MAT_s014
                                 + 0.910066 w_k1_MSFT_s014
                                 + 0.982878 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.27336 w_k1_ADM_s014
                                 + 1.15688 w_k1_AXP_s014
                                 + 1.30806 w_k1_DIS_s014
                                 + 1.08442 w_k1_FDX_s014
                                 + 1.53972 w_k1_IBM_s014
                                 + 1.00344 w_k1_INTC_s014
                                 + 1.20449 w_k1_KO_s014 + 1.42124 w_k1_MAT_s014
                                 + 1.05924 w_k1_MSFT_s014
                                 + 1.21526 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.653933 w_k1_ADM_s014
                                 + 0.667708 w_k1_AXP_s014
                                 + 0.573313 w_k1_DIS_s014
                                 + 0.714164 w_k1_FDX_s014
                                 + 0.852997 w_k1_IBM_s014
                                 + 0.814189 w_k1_INTC_s014
                                 + 0.716744 w_k1_KO_s014
                                 + 0.868028 w_k1_MAT_s014
                                 + 0.775936 w_k1_MSFT_s014
                                 + 0.754746 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.52827 w_k1_ADM_s014
                                 + 1.43618 w_k1_AXP_s014
                                 + 1.66614 w_k1_DIS_s014
                                 + 1.49498 w_k1_FDX_s014
                                 + 0.960381 w_k1_IBM_s014
                                 + 1.148 w_k1_INTC_s014 + 1.26271 w_k1_KO_s014
                                 + 0.921153 w_k1_MAT_s014
                                 + 1.28092 w_k1_MSFT_s014
                                 + 1.17769 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.30903 w_k1_ADM_s014
                                 + 0.912747 w_k1_AXP_s014
                                 + 1.22116 w_k1_DIS_s014
                                 + 1.11249 w_k1_FDX_s014
                                 + 0.961881 w_k1_IBM_s014
                                 + 1.13513 w_k1_INTC_s014
                                 + 1.16713 w_k1_KO_s014
                                 + 0.915158 w_k1_MAT_s014
                                 + 1.12656 w_k1_MSFT_s014
                                 + 1.16588 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.50706 w_k1_ADM_s014
                                 + 1.16199 w_k1_AXP_s014
                                 + 0.994161 w_k1_DIS_s014
                                 + 1.16569 w_k1_FDX_s014
                                 + 0.727492 w_k1_IBM_s014
                                 + 1.09563 w_k1_INTC_s014
                                 + 0.961783 w_k1_KO_s014
                                 + 0.822875 w_k1_MAT_s014
                                 + 1.11211 w_k1_MSFT_s014
                                 + 1.05319 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.96768 w_k1_ADM_s014
                                 + 0.773646 w_k1_AXP_s014
                                 + 0.971551 w_k1_DIS_s014
                                 + 0.977484 w_k1_FDX_s014
                                 + 0.536571 w_k1_IBM_s014
                                 + 0.781 w_k1_INTC_s014 + 0.720759 w_k1_KO_s014
                                 + 0.501469 w_k1_MAT_s014
                                 + 0.91742 w_k1_MSFT_s014
                                 + 1.02522 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.981208 w_k1_ADM_s014
                                 + 0.715595 w_k1_AXP_s014
                                 + 0.817399 w_k1_DIS_s014
                                 + 0.73019 w_k1_FDX_s014
                                 + 1.52257 w_k1_IBM_s014
                                 + 0.754235 w_k1_INTC_s014
                                 + 1.10307 w_k1_KO_s014 + 1.33289 w_k1_MAT_s014
                                 + 0.831587 w_k1_MSFT_s014
                                 + 1.01453 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.25478 w_k1_ADM_s015
                                 + 1.37287 w_k1_AXP_s015
                                 + 0.994306 w_k1_DIS_s015
                                 + 1.14578 w_k1_FDX_s015
                                 + 1.23649 w_k1_IBM_s015
                                 + 1.30222 w_k1_INTC_s015
                                 + 1.05296 w_k1_KO_s015 + 1.10322 w_k1_MAT_s015
                                 + 1.10327 w_k1_MSFT_s015 + 1.0923 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.25665 w_k1_ADM_s015
                                 + 1.20867 w_k1_AXP_s015
                                 + 1.37959 w_k1_DIS_s015 + 1.3188 w_k1_FDX_s015
                                 + 1.03628 w_k1_IBM_s015
                                 + 1.28599 w_k1_INTC_s015
                                 + 1.24636 w_k1_KO_s015
                                 + 0.785934 w_k1_MAT_s015
                                 + 1.26931 w_k1_MSFT_s015
                                 + 1.27723 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.842431 w_k1_ADM_s015
                                 + 0.739127 w_k1_AXP_s015
                                 + 0.477582 w_k1_DIS_s015
                                 + 0.739431 w_k1_FDX_s015
                                 + 0.785359 w_k1_IBM_s015
                                 + 0.713003 w_k1_INTC_s015
                                 + 0.568343 w_k1_KO_s015
                                 + 0.693499 w_k1_MAT_s015
                                 + 0.660817 w_k1_MSFT_s015
                                 + 0.731831 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.14068 w_k1_ADM_s015
                                 + 0.888352 w_k1_AXP_s015
                                 + 1.05498 w_k1_DIS_s015
                                 + 1.04529 w_k1_FDX_s015
                                 + 1.04776 w_k1_IBM_s015
                                 + 1.08405 w_k1_INTC_s015
                                 + 1.17255 w_k1_KO_s015
                                 + 0.737686 w_k1_MAT_s015
                                 + 1.04949 w_k1_MSFT_s015
                                 + 1.04502 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.866382 w_k1_ADM_s015
                                 + 0.774614 w_k1_AXP_s015
                                 + 0.725565 w_k1_DIS_s015
                                 + 0.881066 w_k1_FDX_s015
                                 + 0.910702 w_k1_IBM_s015
                                 + 0.829195 w_k1_INTC_s015
                                 + 0.853857 w_k1_KO_s015
                                 + 0.710314 w_k1_MAT_s015
                                 + 0.891166 w_k1_MSFT_s015
                                 + 0.887917 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.07132 w_k1_ADM_s015
                                 + 0.960508 w_k1_AXP_s015
                                 + 0.878687 w_k1_DIS_s015
                                 + 1.00304 w_k1_FDX_s015
                                 + 0.695467 w_k1_IBM_s015
                                 + 0.960966 w_k1_INTC_s015
                                 + 0.732502 w_k1_KO_s015
                                 + 0.520084 w_k1_MAT_s015
                                 + 0.914073 w_k1_MSFT_s015
                                 + 0.866432 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.924869 w_k1_ADM_s015
                                 + 0.803417 w_k1_AXP_s015
                                 + 0.72454 w_k1_DIS_s015
                                 + 0.854232 w_k1_FDX_s015
                                 + 1.23595 w_k1_IBM_s015
                                 + 1.04641 w_k1_INTC_s015
                                 + 0.968979 w_k1_KO_s015
                                 + 0.848029 w_k1_MAT_s015
                                 + 0.906845 w_k1_MSFT_s015
                                 + 0.954428 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.15866 w_k1_ADM_s015
                                 + 1.11808 w_k1_AXP_s015
                                 + 1.07929 w_k1_DIS_s015 + 1.0678 w_k1_FDX_s015
                                 + 1.66893 w_k1_IBM_s015
                                 + 1.03989 w_k1_INTC_s015
                                 + 1.34898 w_k1_KO_s015 + 1.36163 w_k1_MAT_s015
                                 + 1.09715 w_k1_MSFT_s015
                                 + 1.07235 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.871149 w_k1_ADM_s016
                                 + 1.01416 w_k1_AXP_s016
                                 + 1.16907 w_k1_DIS_s016
                                 + 1.11848 w_k1_FDX_s016
                                 + 0.710117 w_k1_IBM_s016
                                 + 1.116 w_k1_INTC_s016 + 0.89084 w_k1_KO_s016
                                 + 0.70262 w_k1_MAT_s016
                                 + 0.969523 w_k1_MSFT_s016
                                 + 1.01156 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.585604 w_k1_ADM_s016
                                 + 0.689936 w_k1_AXP_s016
                                 + 0.702136 w_k1_DIS_s016
                                 + 0.71754 w_k1_FDX_s016
                                 + 0.820596 w_k1_IBM_s016
                                 + 0.916557 w_k1_INTC_s016
                                 + 0.753483 w_k1_KO_s016
                                 + 0.717526 w_k1_MAT_s016
                                 + 0.80374 w_k1_MSFT_s016
                                 + 0.867411 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.86511 w_k1_ADM_s016
                                 + 0.698642 w_k1_AXP_s016
                                 + 0.802445 w_k1_DIS_s016
                                 + 0.926545 w_k1_FDX_s016
                                 + 0.587302 w_k1_IBM_s016
                                 + 0.844979 w_k1_INTC_s016
                                 + 0.68901 w_k1_KO_s016
                                 + 0.475754 w_k1_MAT_s016
                                 + 0.751471 w_k1_MSFT_s016
                                 + 0.853477 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.91388 w_k1_ADM_s016
                                 + 1.32032 w_k1_AXP_s016
                                 + 1.58964 w_k1_DIS_s016
                                 + 1.22814 w_k1_FDX_s016
                                 + 1.13555 w_k1_IBM_s016
                                 + 1.38155 w_k1_INTC_s016
                                 + 1.30462 w_k1_KO_s016 + 1.23625 w_k1_MAT_s016
                                 + 1.19894 w_k1_MSFT_s016
                                 + 1.27059 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.693278 w_k1_ADM_s016
                                 + 0.835954 w_k1_AXP_s016
                                 + 0.968641 w_k1_DIS_s016
                                 + 0.817866 w_k1_FDX_s016
                                 + 1.10183 w_k1_IBM_s016
                                 + 1.08769 w_k1_INTC_s016
                                 + 1.01822 w_k1_KO_s016
                                 + 0.877155 w_k1_MAT_s016
                                 + 0.873902 w_k1_MSFT_s016
                                 + 0.968323 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.17278 w_k1_ADM_s016
                                 + 1.14622 w_k1_AXP_s016
                                 + 1.00775 w_k1_DIS_s016 + 1.1283 w_k1_FDX_s016
                                 + 0.911399 w_k1_IBM_s016
                                 + 1.13597 w_k1_INTC_s016
                                 + 0.927829 w_k1_KO_s016
                                 + 0.846013 w_k1_MAT_s016
                                 + 0.945581 w_k1_MSFT_s016
                                 + 1.03831 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.800455 w_k1_ADM_s016
                                 + 1.08271 w_k1_AXP_s016
                                 + 1.09218 w_k1_DIS_s016
                                 + 0.959067 w_k1_FDX_s016
                                 + 1.63672 w_k1_IBM_s016
                                 + 1.19999 w_k1_INTC_s016
                                 + 1.48035 w_k1_KO_s016 + 1.7256 w_k1_MAT_s016
                                 + 1.10398 w_k1_MSFT_s016
                                 + 1.10198 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.07347 w_k1_ADM_s016
                                 + 0.799472 w_k1_AXP_s016
                                 + 1.19411 w_k1_DIS_s016
                                 + 1.07585 w_k1_FDX_s016
                                 + 0.848435 w_k1_IBM_s016
                                 + 0.985167 w_k1_INTC_s016
                                 + 0.943167 w_k1_KO_s016
                                 + 0.808275 w_k1_MAT_s016
                                 + 1.08052 w_k1_MSFT_s016 + 1.0403 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.917232 w_k1_ADM_s017
                                 + 0.914843 w_k1_AXP_s017
                                 + 0.883028 w_k1_DIS_s017
                                 + 0.835981 w_k1_FDX_s017
                                 + 1.35804 w_k1_IBM_s017
                                 + 0.99783 w_k1_INTC_s017
                                 + 0.964215 w_k1_KO_s017
                                 + 1.13101 w_k1_MAT_s017
                                 + 0.874933 w_k1_MSFT_s017
                                 + 0.953553 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.818911 w_k1_ADM_s017
                                 + 0.793157 w_k1_AXP_s017
                                 + 0.675336 w_k1_DIS_s017
                                 + 0.820352 w_k1_FDX_s017
                                 + 0.896566 w_k1_IBM_s017
                                 + 0.875346 w_k1_INTC_s017
                                 + 0.759124 w_k1_KO_s017
                                 + 0.917341 w_k1_MAT_s017
                                 + 0.772793 w_k1_MSFT_s017
                                 + 0.780853 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.923484 w_k1_ADM_s017
                                 + 0.997552 w_k1_AXP_s017
                                 + 1.28657 w_k1_DIS_s017
                                 + 1.03542 w_k1_FDX_s017
                                 + 1.07888 w_k1_IBM_s017
                                 + 1.0769 w_k1_INTC_s017 + 1.14604 w_k1_KO_s017
                                 + 1.10833 w_k1_MAT_s017
                                 + 1.09309 w_k1_MSFT_s017
                                 + 1.18317 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.82766 w_k1_ADM_s017
                                 + 1.6001 w_k1_AXP_s017 + 1.51833 w_k1_DIS_s017
                                 + 1.32845 w_k1_FDX_s017
                                 + 1.30967 w_k1_IBM_s017
                                 + 1.51813 w_k1_INTC_s017
                                 + 1.47998 w_k1_KO_s017 + 1.52252 w_k1_MAT_s017
                                 + 1.34462 w_k1_MSFT_s017
                                 + 1.47587 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.44455 w_k1_ADM_s017
                                 + 1.08474 w_k1_AXP_s017
                                 + 1.29831 w_k1_DIS_s017
                                 + 1.35462 w_k1_FDX_s017
                                 + 0.741907 w_k1_IBM_s017
                                 + 1.20377 w_k1_INTC_s017
                                 + 1.08802 w_k1_KO_s017
                                 + 0.672031 w_k1_MAT_s017
                                 + 1.21326 w_k1_MSFT_s017
                                 + 1.17259 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.909179 w_k1_ADM_s017
                                 + 0.89942 w_k1_AXP_s017
                                 + 0.94433 w_k1_DIS_s017 + 1.0341 w_k1_FDX_s017
                                 + 0.735194 w_k1_IBM_s017
                                 + 1.02047 w_k1_INTC_s017
                                 + 0.92202 w_k1_KO_s017 + 0.75427 w_k1_MAT_s017
                                 + 0.994059 w_k1_MSFT_s017
                                 + 0.93456 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.44509 w_k1_ADM_s017
                                 + 1.49666 w_k1_AXP_s017
                                 + 1.31956 w_k1_DIS_s017 + 1.0723 w_k1_FDX_s017
                                 + 1.83715 w_k1_IBM_s017
                                 + 1.06341 w_k1_INTC_s017
                                 + 1.45522 w_k1_KO_s017 + 1.50547 w_k1_MAT_s017
                                 + 1.16093 w_k1_MSFT_s017
                                 + 1.19581 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.840678 w_k1_ADM_s017
                                 + 0.590747 w_k1_AXP_s017
                                 + 0.705739 w_k1_DIS_s017
                                 + 0.726617 w_k1_FDX_s017
                                 + 0.754087 w_k1_IBM_s017
                                 + 0.816391 w_k1_INTC_s017
                                 + 0.819108 w_k1_KO_s017
                                 + 0.666565 w_k1_MAT_s017
                                 + 0.886119 w_k1_MSFT_s017
                                 + 0.778411 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.45741 w_k1_ADM_s100
                                 + 1.02654 w_k1_AXP_s100
                                 + 1.02979 w_k1_DIS_s100
                                 + 1.02979 w_k1_FDX_s100
                                 + 0.97915 w_k1_IBM_s100
                                 + 0.955997 w_k1_INTC_s100
                                 + 0.900335 w_k1_KO_s100
                                 + 0.88065 w_k1_MAT_s100
                                 + 1.11868 w_k1_MSFT_s100
                                 + 0.94016 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.37597 w_k1_ADM_s100
                                 + 1.00926 w_k1_AXP_s100
                                 + 0.873396 w_k1_DIS_s100
                                 + 0.912121 w_k1_FDX_s100
                                 + 0.672445 w_k1_IBM_s100
                                 + 0.940965 w_k1_INTC_s100
                                 + 0.768824 w_k1_KO_s100
                                 + 0.783902 w_k1_MAT_s100
                                 + 0.921513 w_k1_MSFT_s100
                                 + 0.902954 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.933158 w_k1_ADM_s100
                                 + 0.830794 w_k1_AXP_s100
                                 + 1.10047 w_k1_DIS_s100
                                 + 0.868815 w_k1_FDX_s100
                                 + 1.17354 w_k1_IBM_s100
                                 + 0.887948 w_k1_INTC_s100
                                 + 1.00756 w_k1_KO_s100 + 1.18462 w_k1_MAT_s100
                                 + 0.926565 w_k1_MSFT_s100
                                 + 0.999935 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.41816 w_k1_ADM_s100
                                 + 1.12025 w_k1_AXP_s100
                                 + 1.47579 w_k1_DIS_s100 + 1.3164 w_k1_FDX_s100
                                 + 0.807117 w_k1_IBM_s100
                                 + 1.11521 w_k1_INTC_s100
                                 + 0.995015 w_k1_KO_s100
                                 + 1.03254 w_k1_MAT_s100
                                 + 1.23524 w_k1_MSFT_s100
                                 + 1.11416 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.57348 w_k1_ADM_s100
                                 + 0.989322 w_k1_AXP_s100
                                 + 1.29567 w_k1_DIS_s100
                                 + 1.07988 w_k1_FDX_s100
                                 + 0.774458 w_k1_IBM_s100
                                 + 0.931633 w_k1_INTC_s100
                                 + 0.947548 w_k1_KO_s100
                                 + 0.859854 w_k1_MAT_s100
                                 + 1.08611 w_k1_MSFT_s100
                                 + 1.17008 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.33451 w_k1_ADM_s100
                                 + 1.21713 w_k1_AXP_s100
                                 + 1.49451 w_k1_DIS_s100
                                 + 1.19696 w_k1_FDX_s100
                                 + 1.37234 w_k1_IBM_s100
                                 + 1.25567 w_k1_INTC_s100
                                 + 1.23065 w_k1_KO_s100 + 1.61088 w_k1_MAT_s100
                                 + 1.2514 w_k1_MSFT_s100 + 1.09437 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.911749 w_k1_ADM_s100
                                 + 0.860963 w_k1_AXP_s100
                                 + 0.819198 w_k1_DIS_s100
                                 + 0.753172 w_k1_FDX_s100
                                 + 1.10367 w_k1_IBM_s100
                                 + 0.735343 w_k1_INTC_s100
                                 + 0.864458 w_k1_KO_s100
                                 + 1.22516 w_k1_MAT_s100
                                 + 0.758504 w_k1_MSFT_s100
                                 + 0.940903 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.03559 w_k1_ADM_s100
                                 + 0.742253 w_k1_AXP_s100
                                 + 0.800838 w_k1_DIS_s100
                                 + 0.841427 w_k1_FDX_s100
                                 + 0.755959 w_k1_IBM_s100
                                 + 0.856466 w_k1_INTC_s100
                                 + 0.776255 w_k1_KO_s100
                                 + 0.619186 w_k1_MAT_s100
                                 + 0.865071 w_k1_MSFT_s100
                                 + 0.88752 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.844633 w_k1_ADM_s101
                                 + 0.975806 w_k1_AXP_s101
                                 + 1.24984 w_k1_DIS_s101 + 1.0673 w_k1_FDX_s101
                                 + 1.07199 w_k1_IBM_s101
                                 + 1.08926 w_k1_INTC_s101
                                 + 1.17716 w_k1_KO_s101 + 1.50417 w_k1_MAT_s101
                                 + 1.0806 w_k1_MSFT_s101 + 1.13588 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.12088 w_k1_ADM_s101
                                 + 1.04367 w_k1_AXP_s101
                                 + 1.01698 w_k1_DIS_s101
                                 + 1.11161 w_k1_FDX_s101 + 1.0149 w_k1_IBM_s101
                                 + 1.10502 w_k1_INTC_s101
                                 + 0.894916 w_k1_KO_s101
                                 + 1.22397 w_k1_MAT_s101
                                 + 0.995167 w_k1_MSFT_s101
                                 + 1.00025 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.872139 w_k1_ADM_s101
                                 + 0.91232 w_k1_AXP_s101
                                 + 0.853597 w_k1_DIS_s101
                                 + 0.934496 w_k1_FDX_s101
                                 + 1.25863 w_k1_IBM_s101
                                 + 1.01771 w_k1_INTC_s101
                                 + 0.93707 w_k1_KO_s101 + 1.59788 w_k1_MAT_s101
                                 + 0.910106 w_k1_MSFT_s101
                                 + 1.06611 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.803222 w_k1_ADM_s101
                                 + 0.676667 w_k1_AXP_s101
                                 + 0.672085 w_k1_DIS_s101
                                 + 0.781217 w_k1_FDX_s101
                                 + 0.660607 w_k1_IBM_s101
                                 + 0.897286 w_k1_INTC_s101
                                 + 0.708368 w_k1_KO_s101
                                 + 0.645645 w_k1_MAT_s101
                                 + 0.781225 w_k1_MSFT_s101
                                 + 0.846805 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.10653 w_k1_ADM_s101
                                 + 0.945245 w_k1_AXP_s101
                                 + 1.23129 w_k1_DIS_s101
                                 + 1.01926 w_k1_FDX_s101
                                 + 1.53429 w_k1_IBM_s101
                                 + 1.10827 w_k1_INTC_s101
                                 + 1.37148 w_k1_KO_s101 + 1.30907 w_k1_MAT_s101
                                 + 1.0984 w_k1_MSFT_s101 + 1.25332 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.04381 w_k1_ADM_s101
                                 + 1.2059 w_k1_AXP_s101 + 1.4206 w_k1_DIS_s101
                                 + 1.26002 w_k1_FDX_s101
                                 + 0.948403 w_k1_IBM_s101
                                 + 1.18175 w_k1_INTC_s101
                                 + 1.05055 w_k1_KO_s101 + 1.03604 w_k1_MAT_s101
                                 + 1.24763 w_k1_MSFT_s101 + 1.1924 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.866262 w_k1_ADM_s101
                                 + 0.790342 w_k1_AXP_s101
                                 + 0.82432 w_k1_DIS_s101
                                 + 0.873518 w_k1_FDX_s101
                                 + 0.97368 w_k1_IBM_s101
                                 + 1.04737 w_k1_INTC_s101
                                 + 0.899703 w_k1_KO_s101
                                 + 0.993464 w_k1_MAT_s101
                                 + 1.0097 w_k1_MSFT_s101 + 0.95344 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.25667 w_k1_ADM_s101
                                 + 1.53449 w_k1_AXP_s101
                                 + 1.76348 w_k1_DIS_s101
                                 + 1.34991 w_k1_FDX_s101 + 1.4445 w_k1_IBM_s101
                                 + 1.29296 w_k1_INTC_s101
                                 + 1.48556 w_k1_KO_s101 + 1.8717 w_k1_MAT_s101
                                 + 1.28416 w_k1_MSFT_s101
                                 + 1.45302 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.28258 w_k1_ADM_s102
                                 + 1.04362 w_k1_AXP_s102
                                 + 1.00466 w_k1_DIS_s102
                                 + 1.05391 w_k1_FDX_s102
                                 + 0.982698 w_k1_IBM_s102
                                 + 0.959695 w_k1_INTC_s102
                                 + 0.937204 w_k1_KO_s102
                                 + 0.970187 w_k1_MAT_s102
                                 + 1.15562 w_k1_MSFT_s102
                                 + 0.874065 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.999271 w_k1_ADM_s102
                                 + 0.965883 w_k1_AXP_s102
                                 + 1.0101 w_k1_DIS_s102
                                 + 0.923967 w_k1_FDX_s102
                                 + 1.45879 w_k1_IBM_s102
                                 + 0.81097 w_k1_INTC_s102 + 1.0422 w_k1_KO_s102
                                 + 1.37876 w_k1_MAT_s102
                                 + 0.892355 w_k1_MSFT_s102
                                 + 1.00545 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.45321 w_k1_ADM_s102
                                 + 1.58714 w_k1_AXP_s102
                                 + 1.33987 w_k1_DIS_s102 + 1.3241 w_k1_FDX_s102
                                 + 1.63429 w_k1_IBM_s102
                                 + 1.20889 w_k1_INTC_s102
                                 + 1.41451 w_k1_KO_s102 + 2.17548 w_k1_MAT_s102
                                 + 1.25679 w_k1_MSFT_s102
                                 + 1.20216 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.827836 w_k1_ADM_s102
                                 + 1.24004 w_k1_AXP_s102
                                 + 0.875961 w_k1_DIS_s102
                                 + 0.950327 w_k1_FDX_s102
                                 + 1.19234 w_k1_IBM_s102
                                 + 0.899673 w_k1_INTC_s102
                                 + 0.951911 w_k1_KO_s102
                                 + 1.41906 w_k1_MAT_s102
                                 + 0.789744 w_k1_MSFT_s102
                                 + 0.92397 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.95135 w_k1_ADM_s102
                                 + 0.813229 w_k1_AXP_s102
                                 + 0.831145 w_k1_DIS_s102
                                 + 0.858908 w_k1_FDX_s102
                                 + 0.754632 w_k1_IBM_s102
                                 + 0.742097 w_k1_INTC_s102
                                 + 0.815137 w_k1_KO_s102
                                 + 0.480846 w_k1_MAT_s102
                                 + 0.888546 w_k1_MSFT_s102
                                 + 0.852615 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.24731 w_k1_ADM_s102
                                 + 1.14855 w_k1_AXP_s102
                                 + 1.33943 w_k1_DIS_s102
                                 + 1.29424 w_k1_FDX_s102 + 0.8195 w_k1_IBM_s102
                                 + 1.01116 w_k1_INTC_s102
                                 + 0.934356 w_k1_KO_s102
                                 + 0.678464 w_k1_MAT_s102
                                 + 1.23016 w_k1_MSFT_s102
                                 + 0.956119 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.848512 w_k1_ADM_s102
                                 + 0.824827 w_k1_AXP_s102
                                 + 0.69719 w_k1_DIS_s102
                                 + 0.710031 w_k1_FDX_s102
                                 + 1.16359 w_k1_IBM_s102
                                 + 0.821863 w_k1_INTC_s102
                                 + 0.937737 w_k1_KO_s102
                                 + 1.02552 w_k1_MAT_s102
                                 + 0.704104 w_k1_MSFT_s102
                                 + 0.798108 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.931282 w_k1_ADM_s102
                                 + 0.964791 w_k1_AXP_s102
                                 + 0.609368 w_k1_DIS_s102
                                 + 0.807872 w_k1_FDX_s102
                                 + 0.679505 w_k1_IBM_s102
                                 + 0.65068 w_k1_INTC_s102
                                 + 0.618776 w_k1_KO_s102
                                 + 0.848255 w_k1_MAT_s102
                                 + 0.712354 w_k1_MSFT_s102
                                 + 0.73212 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.61574 w_k1_ADM_s103
                                 + 1.64041 w_k1_AXP_s103
                                 + 1.88566 w_k1_DIS_s103 + 1.3505 w_k1_FDX_s103
                                 + 1.39708 w_k1_IBM_s103
                                 + 1.12952 w_k1_INTC_s103
                                 + 1.44561 w_k1_KO_s103 + 2.47537 w_k1_MAT_s103
                                 + 1.49599 w_k1_MSFT_s103
                                 + 1.29389 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.766768 w_k1_ADM_s103
                                 + 0.993261 w_k1_AXP_s103
                                 + 0.798941 w_k1_DIS_s103
                                 + 0.958098 w_k1_FDX_s103
                                 + 0.843414 w_k1_IBM_s103
                                 + 0.993139 w_k1_INTC_s103
                                 + 0.811353 w_k1_KO_s103
                                 + 1.18354 w_k1_MAT_s103
                                 + 0.94834 w_k1_MSFT_s103
                                 + 0.907081 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.803375 w_k1_ADM_s103
                                 + 0.979014 w_k1_AXP_s103
                                 + 1.05959 w_k1_DIS_s103
                                 + 0.970744 w_k1_FDX_s103
                                 + 0.851936 w_k1_IBM_s103
                                 + 0.958839 w_k1_INTC_s103
                                 + 0.982251 w_k1_KO_s103 + 1.0784 w_k1_MAT_s103
                                 + 0.972826 w_k1_MSFT_s103
                                 + 0.994653 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.949578 w_k1_ADM_s103
                                 + 1.19779 w_k1_AXP_s103
                                 + 1.13011 w_k1_DIS_s103 + 1.0491 w_k1_FDX_s103
                                 + 1.50565 w_k1_IBM_s103
                                 + 0.897793 w_k1_INTC_s103
                                 + 1.16806 w_k1_KO_s103 + 2.05831 w_k1_MAT_s103
                                 + 1.05044 w_k1_MSFT_s103
                                 + 0.961028 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.19982 w_k1_ADM_s103
                                 + 1.41286 w_k1_AXP_s103
                                 + 1.41796 w_k1_DIS_s103
                                 + 1.30712 w_k1_FDX_s103
                                 + 0.936692 w_k1_IBM_s103
                                 + 1.34547 w_k1_INTC_s103
                                 + 1.14002 w_k1_KO_s103
                                 + 0.884934 w_k1_MAT_s103
                                 + 1.23809 w_k1_MSFT_s103 + 1.079 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.896278 w_k1_ADM_s103
                                 + 1.47124 w_k1_AXP_s103
                                 + 1.03119 w_k1_DIS_s103
                                 + 1.11723 w_k1_FDX_s103
                                 + 1.24833 w_k1_IBM_s103
                                 + 1.01306 w_k1_INTC_s103
                                 + 1.02877 w_k1_KO_s103 + 1.62116 w_k1_MAT_s103
                                 + 0.95583 w_k1_MSFT_s103
                                 + 1.11065 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.866142 w_k1_ADM_s103
                                 + 1.04299 w_k1_AXP_s103
                                 + 1.51649 w_k1_DIS_s103
                                 + 1.14848 w_k1_FDX_s103
                                 + 1.17346 w_k1_IBM_s103
                                 + 1.09294 w_k1_INTC_s103
                                 + 1.31936 w_k1_KO_s103 + 1.46684 w_k1_MAT_s103
                                 + 1.35149 w_k1_MSFT_s103
                                 + 0.980824 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.726015 w_k1_ADM_s103
                                 + 0.713429 w_k1_AXP_s103
                                 + 0.586155 w_k1_DIS_s103
                                 + 0.712322 w_k1_FDX_s103
                                 + 0.788612 w_k1_IBM_s103
                                 + 0.799569 w_k1_INTC_s103
                                 + 0.572737 w_k1_KO_s103
                                 + 0.639694 w_k1_MAT_s103
                                 + 0.763111 w_k1_MSFT_s103
                                 + 0.732435 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.997319 w_k1_ADM_s104
                                 + 0.961365 w_k1_AXP_s104
                                 + 1.21623 w_k1_DIS_s104
                                 + 0.929176 w_k1_FDX_s104
                                 + 1.03335 w_k1_IBM_s104
                                 + 0.906812 w_k1_INTC_s104 + 1.025 w_k1_KO_s104
                                 + 1.44049 w_k1_MAT_s104
                                 + 0.982581 w_k1_MSFT_s104 + 1.092 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.992182 w_k1_ADM_s104
                                 + 0.983402 w_k1_AXP_s104
                                 + 1.18955 w_k1_DIS_s104
                                 + 1.04332 w_k1_FDX_s104
                                 + 1.01815 w_k1_IBM_s104
                                 + 1.11575 w_k1_INTC_s104
                                 + 1.04535 w_k1_KO_s104 + 1.69576 w_k1_MAT_s104
                                 + 1.24172 w_k1_MSFT_s104
                                 + 1.08797 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.32079 w_k1_ADM_s104
                                 + 1.45387 w_k1_AXP_s104 + 1.4216 w_k1_DIS_s104
                                 + 1.35224 w_k1_FDX_s104
                                 + 1.09961 w_k1_IBM_s104
                                 + 1.4738 w_k1_INTC_s104 + 1.22782 w_k1_KO_s104
                                 + 1.27318 w_k1_MAT_s104
                                 + 1.29375 w_k1_MSFT_s104
                                 + 1.25344 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.868145 w_k1_ADM_s104
                                 + 1.18451 w_k1_AXP_s104
                                 + 0.915318 w_k1_DIS_s104
                                 + 0.881652 w_k1_FDX_s104 + 1.371 w_k1_IBM_s104
                                 + 0.938242 w_k1_INTC_s104
                                 + 1.05072 w_k1_KO_s104 + 1.97271 w_k1_MAT_s104
                                 + 0.957603 w_k1_MSFT_s104
                                 + 0.97134 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.23601 w_k1_ADM_s104
                                 + 1.0606 w_k1_AXP_s104
                                 + 0.966668 w_k1_DIS_s104
                                 + 1.05661 w_k1_FDX_s104
                                 + 0.760836 w_k1_IBM_s104
                                 + 1.15326 w_k1_INTC_s104
                                 + 0.887254 w_k1_KO_s104
                                 + 0.793858 w_k1_MAT_s104
                                 + 1.07112 w_k1_MSFT_s104
                                 + 0.960357 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.12177 w_k1_ADM_s104
                                 + 1.39398 w_k1_AXP_s104
                                 + 1.14158 w_k1_DIS_s104 + 1.0131 w_k1_FDX_s104
                                 + 1.39897 w_k1_IBM_s104
                                 + 0.943773 w_k1_INTC_s104
                                 + 1.12509 w_k1_KO_s104 + 2.45261 w_k1_MAT_s104
                                 + 1.02565 w_k1_MSFT_s104
                                 + 1.17869 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.774322 w_k1_ADM_s104
                                 + 0.926207 w_k1_AXP_s104
                                 + 0.678886 w_k1_DIS_s104
                                 + 0.793413 w_k1_FDX_s104
                                 + 1.0234 w_k1_IBM_s104
                                 + 0.984646 w_k1_INTC_s104
                                 + 0.839083 w_k1_KO_s104 + 1.5761 w_k1_MAT_s104
                                 + 0.886824 w_k1_MSFT_s104
                                 + 0.871471 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.947022 w_k1_ADM_s104
                                 + 0.797777 w_k1_AXP_s104
                                 + 0.699396 w_k1_DIS_s104
                                 + 0.840183 w_k1_FDX_s104
                                 + 0.760017 w_k1_IBM_s104
                                 + 0.956943 w_k1_INTC_s104
                                 + 0.677161 w_k1_KO_s104
                                 + 0.744025 w_k1_MAT_s104
                                 + 0.827529 w_k1_MSFT_s104
                                 + 0.808083 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.41107 w_k1_ADM_s105
                                 + 0.901039 w_k1_AXP_s105
                                 + 1.11247 w_k1_DIS_s105
                                 + 1.01475 w_k1_FDX_s105
                                 + 1.29494 w_k1_IBM_s105
                                 + 1.08972 w_k1_INTC_s105
                                 + 1.23933 w_k1_KO_s105 + 1.00755 w_k1_MAT_s105
                                 + 1.08039 w_k1_MSFT_s105
                                 + 1.11823 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.06908 w_k1_ADM_s105
                                 + 0.806899 w_k1_AXP_s105
                                 + 0.706203 w_k1_DIS_s105
                                 + 0.861145 w_k1_FDX_s105
                                 + 0.729412 w_k1_IBM_s105
                                 + 0.694469 w_k1_INTC_s105
                                 + 0.887633 w_k1_KO_s105
                                 + 0.716368 w_k1_MAT_s105
                                 + 0.833672 w_k1_MSFT_s105
                                 + 0.878359 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.27228 w_k1_ADM_s105
                                 + 1.35707 w_k1_AXP_s105
                                 + 1.10467 w_k1_DIS_s105
                                 + 1.22255 w_k1_FDX_s105
                                 + 0.915452 w_k1_IBM_s105
                                 + 0.939467 w_k1_INTC_s105
                                 + 0.995146 w_k1_KO_s105
                                 + 1.23497 w_k1_MAT_s105
                                 + 1.07815 w_k1_MSFT_s105
                                 + 1.00583 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.29694 w_k1_ADM_s105
                                 + 1.05386 w_k1_AXP_s105 + 1.2543 w_k1_DIS_s105
                                 + 1.29317 w_k1_FDX_s105
                                 + 0.639044 w_k1_IBM_s105
                                 + 1.09424 w_k1_INTC_s105
                                 + 1.07898 w_k1_KO_s105
                                 + 0.667042 w_k1_MAT_s105
                                 + 1.22204 w_k1_MSFT_s105
                                 + 1.12606 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.917692 w_k1_ADM_s105
                                 + 1.19872 w_k1_AXP_s105
                                 + 0.727535 w_k1_DIS_s105
                                 + 0.94451 w_k1_FDX_s105
                                 + 1.37481 w_k1_IBM_s105
                                 + 0.937122 w_k1_INTC_s105
                                 + 1.10508 w_k1_KO_s105 + 1.52348 w_k1_MAT_s105
                                 + 0.8176 w_k1_MSFT_s105
                                 + 0.982837 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.20148 w_k1_ADM_s105
                                 + 0.923267 w_k1_AXP_s105
                                 + 1.13552 w_k1_DIS_s105
                                 + 1.12923 w_k1_FDX_s105
                                 + 0.895347 w_k1_IBM_s105
                                 + 0.926904 w_k1_INTC_s105
                                 + 1.12592 w_k1_KO_s105 + 1.01307 w_k1_MAT_s105
                                 + 1.10005 w_k1_MSFT_s105
                                 + 1.10789 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.50874 w_k1_ADM_s105
                                 + 1.45793 w_k1_AXP_s105 + 1.5785 w_k1_DIS_s105
                                 + 1.37386 w_k1_FDX_s105 + 1.4421 w_k1_IBM_s105
                                 + 1.10099 w_k1_INTC_s105 + 1.5217 w_k1_KO_s105
                                 + 1.98035 w_k1_MAT_s105
                                 + 1.33182 w_k1_MSFT_s105
                                 + 1.28854 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.0621 w_k1_ADM_s105
                                 + 0.737592 w_k1_AXP_s105
                                 + 0.847272 w_k1_DIS_s105
                                 + 0.866418 w_k1_FDX_s105
                                 + 1.08671 w_k1_IBM_s105
                                 + 0.838238 w_k1_INTC_s105
                                 + 0.983941 w_k1_KO_s105
                                 + 0.901373 w_k1_MAT_s105
                                 + 0.917752 w_k1_MSFT_s105
                                 + 0.923749 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.1659 w_k1_ADM_s106
                                 + 0.881161 w_k1_AXP_s106
                                 + 0.916637 w_k1_DIS_s106
                                 + 1.0025 w_k1_FDX_s106 + 1.2098 w_k1_IBM_s106
                                 + 0.706352 w_k1_INTC_s106
                                 + 0.912578 w_k1_KO_s106
                                 + 1.26714 w_k1_MAT_s106
                                 + 0.986373 w_k1_MSFT_s106
                                 + 0.970685 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.24785 w_k1_ADM_s106
                                 + 0.945547 w_k1_AXP_s106
                                 + 1.24697 w_k1_DIS_s106
                                 + 1.33045 w_k1_FDX_s106
                                 + 0.802431 w_k1_IBM_s106
                                 + 1.09947 w_k1_INTC_s106
                                 + 0.999828 w_k1_KO_s106
                                 + 0.690309 w_k1_MAT_s106
                                 + 1.2504 w_k1_MSFT_s106 + 1.05178 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.1425 w_k1_ADM_s106
                                 + 1.28105 w_k1_AXP_s106
                                 + 0.945292 w_k1_DIS_s106
                                 + 1.16396 w_k1_FDX_s106
                                 + 0.80691 w_k1_IBM_s106
                                 + 0.790941 w_k1_INTC_s106
                                 + 0.8498 w_k1_KO_s106 + 1.07419 w_k1_MAT_s106
                                 + 0.910346 w_k1_MSFT_s106
                                 + 0.911966 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.87419 w_k1_ADM_s106
                                 + 0.778151 w_k1_AXP_s106
                                 + 0.836957 w_k1_DIS_s106
                                 + 0.894612 w_k1_FDX_s106
                                 + 0.88119 w_k1_IBM_s106
                                 + 0.932548 w_k1_INTC_s106
                                 + 0.883156 w_k1_KO_s106
                                 + 0.836106 w_k1_MAT_s106
                                 + 0.957902 w_k1_MSFT_s106
                                 + 0.891902 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.815896 w_k1_ADM_s106
                                 + 0.722859 w_k1_AXP_s106
                                 + 0.599736 w_k1_DIS_s106
                                 + 0.766772 w_k1_FDX_s106
                                 + 0.656298 w_k1_IBM_s106
                                 + 0.714576 w_k1_INTC_s106
                                 + 0.623089 w_k1_KO_s106
                                 + 0.761271 w_k1_MAT_s106
                                 + 0.775727 w_k1_MSFT_s106
                                 + 0.81444 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.970533 w_k1_ADM_s106
                                 + 1.10239 w_k1_AXP_s106
                                 + 1.38636 w_k1_DIS_s106 + 1.2634 w_k1_FDX_s106
                                 + 1.43484 w_k1_IBM_s106
                                 + 1.11998 w_k1_INTC_s106
                                 + 1.34259 w_k1_KO_s106 + 1.50136 w_k1_MAT_s106
                                 + 1.33574 w_k1_MSFT_s106
                                 + 1.14829 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.17884 w_k1_ADM_s106
                                 + 0.800146 w_k1_AXP_s106
                                 + 1.24288 w_k1_DIS_s106
                                 + 1.01158 w_k1_FDX_s106
                                 + 0.982318 w_k1_IBM_s106
                                 + 0.768813 w_k1_INTC_s106
                                 + 1.17656 w_k1_KO_s106
                                 + 0.945692 w_k1_MAT_s106
                                 + 1.07814 w_k1_MSFT_s106 + 1.0304 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.12958 w_k1_ADM_s106
                                 + 0.736275 w_k1_AXP_s106
                                 + 0.984679 w_k1_DIS_s106
                                 + 1.01532 w_k1_FDX_s106
                                 + 0.699268 w_k1_IBM_s106
                                 + 0.776802 w_k1_INTC_s106
                                 + 0.878018 w_k1_KO_s106
                                 + 0.71429 w_k1_MAT_s106
                                 + 1.10152 w_k1_MSFT_s106
                                 + 0.991314 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.20266 w_k1_ADM_s107
                                 + 1.19082 w_k1_AXP_s107
                                 + 1.28428 w_k1_DIS_s107
                                 + 1.11616 w_k1_FDX_s107 + 1.1169 w_k1_IBM_s107
                                 + 0.820022 w_k1_INTC_s107
                                 + 1.06998 w_k1_KO_s107 + 1.34958 w_k1_MAT_s107
                                 + 0.988128 w_k1_MSFT_s107
                                 + 1.04416 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.2502 w_k1_ADM_s107
                                 + 1.57002 w_k1_AXP_s107
                                 + 1.04787 w_k1_DIS_s107
                                 + 1.23135 w_k1_FDX_s107
                                 + 0.991578 w_k1_IBM_s107
                                 + 0.982365 w_k1_INTC_s107
                                 + 0.9175 w_k1_KO_s107 + 1.51494 w_k1_MAT_s107
                                 + 0.997167 w_k1_MSFT_s107
                                 + 0.951269 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.959585 w_k1_ADM_s107
                                 + 1.09357 w_k1_AXP_s107
                                 + 0.811937 w_k1_DIS_s107
                                 + 0.874305 w_k1_FDX_s107
                                 + 1.13121 w_k1_IBM_s107
                                 + 0.835947 w_k1_INTC_s107
                                 + 0.8996 w_k1_KO_s107 + 1.69965 w_k1_MAT_s107
                                 + 0.835926 w_k1_MSFT_s107
                                 + 0.969271 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.646771 w_k1_ADM_s107
                                 + 0.77053 w_k1_AXP_s107
                                 + 0.612717 w_k1_DIS_s107
                                 + 0.699048 w_k1_FDX_s107
                                 + 0.934462 w_k1_IBM_s107
                                 + 0.703669 w_k1_INTC_s107
                                 + 0.716692 w_k1_KO_s107
                                 + 1.26394 w_k1_MAT_s107
                                 + 0.731703 w_k1_MSFT_s107
                                 + 0.857757 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.31918 w_k1_ADM_s107
                                 + 1.4668 w_k1_AXP_s107 + 1.72028 w_k1_DIS_s107
                                 + 1.37612 w_k1_FDX_s107
                                 + 1.53003 w_k1_IBM_s107
                                 + 1.25208 w_k1_INTC_s107
                                 + 1.36972 w_k1_KO_s107 + 1.88236 w_k1_MAT_s107
                                 + 1.33574 w_k1_MSFT_s107 + 1.199 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.802361 w_k1_ADM_s107
                                 + 0.88088 w_k1_AXP_s107
                                 + 0.822036 w_k1_DIS_s107
                                 + 0.859343 w_k1_FDX_s107
                                 + 0.721077 w_k1_IBM_s107
                                 + 0.883967 w_k1_INTC_s107
                                 + 0.846073 w_k1_KO_s107
                                 + 1.01834 w_k1_MAT_s107
                                 + 0.897932 w_k1_MSFT_s107
                                 + 0.901207 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.1648 w_k1_ADM_s107
                                 + 0.98302 w_k1_AXP_s107
                                 + 1.50685 w_k1_DIS_s107 + 1.1856 w_k1_FDX_s107
                                 + 1.20207 w_k1_IBM_s107
                                 + 1.18063 w_k1_INTC_s107
                                 + 1.26272 w_k1_KO_s107 + 1.16997 w_k1_MAT_s107
                                 + 1.32334 w_k1_MSFT_s107
                                 + 1.17939 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.948033 w_k1_ADM_s107
                                 + 1.05421 w_k1_AXP_s107
                                 + 1.16379 w_k1_DIS_s107
                                 + 1.07904 w_k1_FDX_s107
                                 + 0.848995 w_k1_IBM_s107
                                 + 1.17227 w_k1_INTC_s107
                                 + 0.941843 w_k1_KO_s107
                                 + 0.754478 w_k1_MAT_s107
                                 + 1.05688 w_k1_MSFT_s107
                                 + 0.996739 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.869993 w_k1_ADM_s110
                                 + 0.988633 w_k1_AXP_s110 + w_k1_DIS_s110
                                 + 0.984885 w_k1_FDX_s110 + 1.325 w_k1_IBM_s110
                                 + 1.13441 w_k1_INTC_s110 + 1.2399 w_k1_KO_s110
                                 + 1.76021 w_k1_MAT_s110
                                 + 1.0658 w_k1_MSFT_s110 + 1.09025 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.3352 w_k1_ADM_s110
                                 + 1.38346 w_k1_AXP_s110
                                 + 1.44234 w_k1_DIS_s110
                                 + 1.36166 w_k1_FDX_s110
                                 + 1.26123 w_k1_IBM_s110
                                 + 1.12472 w_k1_INTC_s110
                                 + 1.15675 w_k1_KO_s110 + 2.02662 w_k1_MAT_s110
                                 + 1.22412 w_k1_MSFT_s110
                                 + 1.35063 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.717655 w_k1_ADM_s110
                                 + 0.669742 w_k1_AXP_s110
                                 + 0.611819 w_k1_DIS_s110
                                 + 0.765463 w_k1_FDX_s110
                                 + 0.64026 w_k1_IBM_s110
                                 + 0.660266 w_k1_INTC_s110
                                 + 0.629936 w_k1_KO_s110
                                 + 0.942325 w_k1_MAT_s110
                                 + 0.797823 w_k1_MSFT_s110
                                 + 0.808371 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.05185 w_k1_ADM_s110
                                 + 1.0978 w_k1_AXP_s110 + 1.28704 w_k1_DIS_s110
                                 + 1.10183 w_k1_FDX_s110
                                 + 1.77378 w_k1_IBM_s110
                                 + 1.10292 w_k1_INTC_s110
                                 + 1.34443 w_k1_KO_s110 + 1.77782 w_k1_MAT_s110
                                 + 1.14381 w_k1_MSFT_s110
                                 + 1.11047 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.753806 w_k1_ADM_s110
                                 + 0.785594 w_k1_AXP_s110
                                 + 0.693489 w_k1_DIS_s110
                                 + 0.821689 w_k1_FDX_s110
                                 + 1.19866 w_k1_IBM_s110
                                 + 0.935528 w_k1_INTC_s110
                                 + 0.833774 w_k1_KO_s110
                                 + 1.21676 w_k1_MAT_s110
                                 + 0.868938 w_k1_MSFT_s110
                                 + 0.846456 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.20294 w_k1_ADM_s110
                                 + 1.10404 w_k1_AXP_s110
                                 + 0.964529 w_k1_DIS_s110
                                 + 1.15054 w_k1_FDX_s110
                                 + 0.920541 w_k1_IBM_s110
                                 + 0.892414 w_k1_INTC_s110
                                 + 0.908295 w_k1_KO_s110
                                 + 1.53254 w_k1_MAT_s110
                                 + 1.0975 w_k1_MSFT_s110 + 1.0227 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.861652 w_k1_ADM_s110
                                 + 0.678743 w_k1_AXP_s110
                                 + 1.26975 w_k1_DIS_s110
                                 + 1.06773 w_k1_FDX_s110
                                 + 1.09902 w_k1_IBM_s110
                                 + 0.985015 w_k1_INTC_s110
                                 + 1.06773 w_k1_KO_s110 + 1.14795 w_k1_MAT_s110
                                 + 1.22063 w_k1_MSFT_s110
                                 + 1.00788 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.00003 w_k1_ADM_s110
                                 + 0.708391 w_k1_AXP_s110
                                 + 0.82543 w_k1_DIS_s110
                                 + 0.922098 w_k1_FDX_s110
                                 + 0.837548 w_k1_IBM_s110
                                 + 0.797656 w_k1_INTC_s110
                                 + 0.807438 w_k1_KO_s110
                                 + 0.887408 w_k1_MAT_s110
                                 + 0.943763 w_k1_MSFT_s110
                                 + 0.910935 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.31751 w_k1_ADM_s111
                                 + 1.21287 w_k1_AXP_s111
                                 + 1.29179 w_k1_DIS_s111
                                 + 1.25323 w_k1_FDX_s111
                                 + 1.36258 w_k1_IBM_s111
                                 + 1.0368 w_k1_INTC_s111 + 1.36524 w_k1_KO_s111
                                 + 1.15701 w_k1_MAT_s111
                                 + 1.18565 w_k1_MSFT_s111
                                 + 1.25233 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.771266 w_k1_ADM_s111
                                 + 0.83113 w_k1_AXP_s111
                                 + 0.913818 w_k1_DIS_s111
                                 + 0.887754 w_k1_FDX_s111
                                 + 1.21383 w_k1_IBM_s111
                                 + 0.780135 w_k1_INTC_s111
                                 + 1.06383 w_k1_KO_s111
                                 + 0.906512 w_k1_MAT_s111
                                 + 0.886471 w_k1_MSFT_s111
                                 + 0.933177 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.854621 w_k1_ADM_s111
                                 + 0.819374 w_k1_AXP_s111
                                 + 0.817822 w_k1_DIS_s111
                                 + 0.923881 w_k1_FDX_s111
                                 + 0.780797 w_k1_IBM_s111
                                 + 0.774365 w_k1_INTC_s111
                                 + 0.866009 w_k1_KO_s111
                                 + 0.640749 w_k1_MAT_s111
                                 + 0.995916 w_k1_MSFT_s111
                                 + 0.95186 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.15055 w_k1_ADM_s111
                                 + 0.83903 w_k1_AXP_s111
                                 + 1.14371 w_k1_DIS_s111 + 1.0711 w_k1_FDX_s111
                                 + 0.842033 w_k1_IBM_s111
                                 + 1.0877 w_k1_INTC_s111 + 1.07245 w_k1_KO_s111
                                 + 0.58797 w_k1_MAT_s111
                                 + 1.14751 w_k1_MSFT_s111
                                 + 1.11372 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.13616 w_k1_ADM_s111
                                 + 1.06477 w_k1_AXP_s111
                                 + 0.818412 w_k1_DIS_s111
                                 + 0.975726 w_k1_FDX_s111
                                 + 0.836961 w_k1_IBM_s111
                                 + 0.781162 w_k1_INTC_s111
                                 + 0.911789 w_k1_KO_s111
                                 + 1.01723 w_k1_MAT_s111
                                 + 0.911288 w_k1_MSFT_s111
                                 + 1.02913 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.36017 w_k1_ADM_s111
                                 + 1.13487 w_k1_AXP_s111
                                 + 1.44837 w_k1_DIS_s111
                                 + 1.32843 w_k1_FDX_s111
                                 + 0.817975 w_k1_IBM_s111
                                 + 1.09095 w_k1_INTC_s111
                                 + 1.17831 w_k1_KO_s111
                                 + 0.660622 w_k1_MAT_s111
                                 + 1.41639 w_k1_MSFT_s111 + 1.1057 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.06981 w_k1_ADM_s111
                                 + 1.27906 w_k1_AXP_s111
                                 + 1.08982 w_k1_DIS_s111 + 1.0193 w_k1_FDX_s111
                                 + 1.53564 w_k1_IBM_s111
                                 + 0.865938 w_k1_INTC_s111
                                 + 1.2255 w_k1_KO_s111 + 1.38408 w_k1_MAT_s111
                                 + 1.1401 w_k1_MSFT_s111 + 1.07588 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.810928 w_k1_ADM_s111
                                 + 0.609061 w_k1_AXP_s111
                                 + 0.57794 w_k1_DIS_s111
                                 + 0.776051 w_k1_FDX_s111
                                 + 0.534004 w_k1_IBM_s111
                                 + 0.729383 w_k1_INTC_s111
                                 + 0.628806 w_k1_KO_s111
                                 + 0.453018 w_k1_MAT_s111
                                 + 0.842349 w_k1_MSFT_s111
                                 + 0.754807 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.48497 w_k1_ADM_s112
                                 + 1.30859 w_k1_AXP_s112
                                 + 1.94946 w_k1_DIS_s112
                                 + 1.45917 w_k1_FDX_s112
                                 + 1.34281 w_k1_IBM_s112
                                 + 1.28073 w_k1_INTC_s112
                                 + 1.40038 w_k1_KO_s112
                                 + 0.748269 w_k1_MAT_s112
                                 + 1.19886 w_k1_MSFT_s112
                                 + 1.32005 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.769345 w_k1_ADM_s112
                                 + 0.738345 w_k1_AXP_s112
                                 + 0.800273 w_k1_DIS_s112
                                 + 0.769949 w_k1_FDX_s112
                                 + 0.907135 w_k1_IBM_s112
                                 + 0.743907 w_k1_INTC_s112
                                 + 0.734881 w_k1_KO_s112
                                 + 0.643002 w_k1_MAT_s112
                                 + 0.71467 w_k1_MSFT_s112
                                 + 0.896691 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.757586 w_k1_ADM_s112
                                 + 0.47306 w_k1_AXP_s112
                                 + 0.650874 w_k1_DIS_s112
                                 + 0.787269 w_k1_FDX_s112
                                 + 0.491301 w_k1_IBM_s112
                                 + 0.774811 w_k1_INTC_s112
                                 + 0.494473 w_k1_KO_s112
                                 + 0.230144 w_k1_MAT_s112
                                 + 0.684201 w_k1_MSFT_s112
                                 + 0.715552 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.17459 w_k1_ADM_s112
                                 + 1.17659 w_k1_AXP_s112 + 1.5114 w_k1_DIS_s112
                                 + 1.2387 w_k1_FDX_s112 + 1.45577 w_k1_IBM_s112
                                 + 1.14228 w_k1_INTC_s112
                                 + 1.35256 w_k1_KO_s112 + 1.43733 w_k1_MAT_s112
                                 + 1.12607 w_k1_MSFT_s112
                                 + 1.19439 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.944913 w_k1_ADM_s112
                                 + 0.786969 w_k1_AXP_s112
                                 + 0.96141 w_k1_DIS_s112
                                 + 1.00692 w_k1_FDX_s112
                                 + 0.745974 w_k1_IBM_s112
                                 + 1.0406 w_k1_INTC_s112
                                 + 0.750835 w_k1_KO_s112
                                 + 0.483965 w_k1_MAT_s112
                                 + 0.935336 w_k1_MSFT_s112
                                 + 0.858375 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.19751 w_k1_ADM_s112
                                 + 1.14494 w_k1_AXP_s112 + 1.3715 w_k1_DIS_s112
                                 + 1.24088 w_k1_FDX_s112
                                 + 0.930813 w_k1_IBM_s112
                                 + 1.21334 w_k1_INTC_s112
                                 + 0.939086 w_k1_KO_s112
                                 + 1.04187 w_k1_MAT_s112
                                 + 1.09724 w_k1_MSFT_s112 + 1.0733 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.0949 w_k1_ADM_s112
                                 + 0.878714 w_k1_AXP_s112
                                 + 1.19496 w_k1_DIS_s112
                                 + 1.14496 w_k1_FDX_s112
                                 + 0.797774 w_k1_IBM_s112
                                 + 1.15062 w_k1_INTC_s112
                                 + 0.839189 w_k1_KO_s112
                                 + 0.482244 w_k1_MAT_s112
                                 + 1.20046 w_k1_MSFT_s112
                                 + 1.00892 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.83327 w_k1_ADM_s112
                                 + 0.902732 w_k1_AXP_s112
                                 + 1.0141 w_k1_DIS_s112
                                 + 0.923917 w_k1_FDX_s112
                                 + 1.31919 w_k1_IBM_s112
                                 + 0.911747 w_k1_INTC_s112
                                 + 1.07594 w_k1_KO_s112 + 1.21796 w_k1_MAT_s112
                                 + 0.823587 w_k1_MSFT_s112
                                 + 1.0322 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.16414 w_k1_ADM_s113
                                 + 1.16868 w_k1_AXP_s113
                                 + 1.11404 w_k1_DIS_s113
                                 + 1.00086 w_k1_FDX_s113
                                 + 1.04375 w_k1_IBM_s113
                                 + 0.862367 w_k1_INTC_s113
                                 + 1.05001 w_k1_KO_s113 + 1.26738 w_k1_MAT_s113
                                 + 0.998335 w_k1_MSFT_s113
                                 + 1.05853 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.790611 w_k1_ADM_s113
                                 + 0.95877 w_k1_AXP_s113
                                 + 1.02179 w_k1_DIS_s113
                                 + 0.826658 w_k1_FDX_s113
                                 + 1.46808 w_k1_IBM_s113
                                 + 0.729375 w_k1_INTC_s113
                                 + 1.24754 w_k1_KO_s113 + 1.35476 w_k1_MAT_s113
                                 + 0.962572 w_k1_MSFT_s113
                                 + 1.01586 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.10265 w_k1_ADM_s113
                                 + 1.02775 w_k1_AXP_s113
                                 + 1.05979 w_k1_DIS_s113
                                 + 1.05618 w_k1_FDX_s113
                                 + 0.796477 w_k1_IBM_s113
                                 + 0.898733 w_k1_INTC_s113
                                 + 0.882211 w_k1_KO_s113
                                 + 0.677672 w_k1_MAT_s113
                                 + 1.06786 w_k1_MSFT_s113
                                 + 0.974792 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.912664 w_k1_ADM_s113
                                 + 0.764977 w_k1_AXP_s113
                                 + 0.715533 w_k1_DIS_s113
                                 + 0.803181 w_k1_FDX_s113
                                 + 0.828856 w_k1_IBM_s113
                                 + 0.701459 w_k1_INTC_s113
                                 + 0.920209 w_k1_KO_s113
                                 + 0.787281 w_k1_MAT_s113
                                 + 0.813935 w_k1_MSFT_s113
                                 + 0.876434 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.21233 w_k1_ADM_s113
                                 + 1.66597 w_k1_AXP_s113
                                 + 1.70783 w_k1_DIS_s113
                                 + 1.40109 w_k1_FDX_s113
                                 + 1.19472 w_k1_IBM_s113
                                 + 1.13041 w_k1_INTC_s113
                                 + 1.42298 w_k1_KO_s113 + 1.56097 w_k1_MAT_s113
                                 + 1.23051 w_k1_MSFT_s113
                                 + 1.16927 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.06052 w_k1_ADM_s113
                                 + 1.2299 w_k1_AXP_s113 + 1.30222 w_k1_DIS_s113
                                 + 1.12175 w_k1_FDX_s113
                                 + 1.49822 w_k1_IBM_s113
                                 + 1.05962 w_k1_INTC_s113
                                 + 1.40774 w_k1_KO_s113 + 1.40297 w_k1_MAT_s113
                                 + 1.07218 w_k1_MSFT_s113
                                 + 1.26518 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.843416 w_k1_ADM_s113
                                 + 0.837906 w_k1_AXP_s113
                                 + 1.22602 w_k1_DIS_s113
                                 + 0.908548 w_k1_FDX_s113
                                 + 1.11206 w_k1_IBM_s113
                                 + 0.959609 w_k1_INTC_s113
                                 + 1.23261 w_k1_KO_s113 + 1.24835 w_k1_MAT_s113
                                 + 1.11064 w_k1_MSFT_s113
                                 + 1.01553 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.37261 w_k1_ADM_s113
                                 + 1.275 w_k1_AXP_s113 + 1.2405 w_k1_DIS_s113
                                 + 1.33128 w_k1_FDX_s113
                                 + 0.741128 w_k1_IBM_s113
                                 + 1.09229 w_k1_INTC_s113
                                 + 1.06055 w_k1_KO_s113 + 1.17616 w_k1_MAT_s113
                                 + 1.18139 w_k1_MSFT_s113
                                 + 1.16653 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.934599 w_k1_ADM_s114
                                 + 0.653687 w_k1_AXP_s114
                                 + 0.597031 w_k1_DIS_s114
                                 + 0.755099 w_k1_FDX_s114
                                 + 0.665013 w_k1_IBM_s114
                                 + 0.806583 w_k1_INTC_s114
                                 + 0.742011 w_k1_KO_s114
                                 + 0.478726 w_k1_MAT_s114
                                 + 0.783003 w_k1_MSFT_s114
                                 + 0.841847 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.04282 w_k1_ADM_s114
                                 + 0.888429 w_k1_AXP_s114
                                 + 0.902068 w_k1_DIS_s114
                                 + 1.04277 w_k1_FDX_s114
                                 + 0.699619 w_k1_IBM_s114
                                 + 1.00947 w_k1_INTC_s114
                                 + 0.819105 w_k1_KO_s114
                                 + 0.355876 w_k1_MAT_s114
                                 + 0.933149 w_k1_MSFT_s114
                                 + 0.958024 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.976408 w_k1_ADM_s114
                                 + 0.872115 w_k1_AXP_s114
                                 + 1.22112 w_k1_DIS_s114
                                 + 0.999137 w_k1_FDX_s114
                                 + 0.982308 w_k1_IBM_s114
                                 + 1.04014 w_k1_INTC_s114
                                 + 1.02448 w_k1_KO_s114
                                 + 0.685816 w_k1_MAT_s114
                                 + 1.03441 w_k1_MSFT_s114
                                 + 1.07817 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.02929 w_k1_ADM_s114
                                 + 0.865131 w_k1_AXP_s114
                                 + 0.832924 w_k1_DIS_s114
                                 + 0.819366 w_k1_FDX_s114
                                 + 1.06559 w_k1_IBM_s114
                                 + 0.862937 w_k1_INTC_s114
                                 + 0.959814 w_k1_KO_s114
                                 + 0.835508 w_k1_MAT_s114
                                 + 0.892345 w_k1_MSFT_s114
                                 + 1.00964 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.38019 w_k1_ADM_s114
                                 + 1.2158 w_k1_AXP_s114 + 1.58287 w_k1_DIS_s114
                                 + 1.4144 w_k1_FDX_s114
                                 + 0.819264 w_k1_IBM_s114
                                 + 1.35869 w_k1_INTC_s114 + 1.1295 w_k1_KO_s114
                                 + 0.829281 w_k1_MAT_s114
                                 + 1.42784 w_k1_MSFT_s114
                                 + 1.13035 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.19505 w_k1_ADM_s114
                                 + 1.08849 w_k1_AXP_s114
                                 + 1.07478 w_k1_DIS_s114
                                 + 0.966397 w_k1_FDX_s114
                                 + 1.36359 w_k1_IBM_s114
                                 + 1.02806 w_k1_INTC_s114
                                 + 1.29055 w_k1_KO_s114 + 1.22445 w_k1_MAT_s114
                                 + 1.01442 w_k1_MSFT_s114
                                 + 1.15264 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.07174 w_k1_ADM_s114
                                 + 1.38426 w_k1_AXP_s114
                                 + 1.45189 w_k1_DIS_s114
                                 + 1.20404 w_k1_FDX_s114
                                 + 1.34622 w_k1_IBM_s114
                                 + 1.11769 w_k1_INTC_s114
                                 + 1.37244 w_k1_KO_s114 + 1.48359 w_k1_MAT_s114
                                 + 1.16468 w_k1_MSFT_s114
                                 + 1.14074 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.91961 w_k1_ADM_s114
                                 + 0.762268 w_k1_AXP_s114
                                 + 0.720348 w_k1_DIS_s114
                                 + 0.860044 w_k1_FDX_s114
                                 + 0.568042 w_k1_IBM_s114
                                 + 0.850462 w_k1_INTC_s114
                                 + 0.650467 w_k1_KO_s114
                                 + 0.417303 w_k1_MAT_s114
                                 + 0.958279 w_k1_MSFT_s114
                                 + 0.88222 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.17887 w_k1_ADM_s115
                                 + 1.12119 w_k1_AXP_s115
                                 + 0.90806 w_k1_DIS_s115
                                 + 1.04396 w_k1_FDX_s115
                                 + 0.862704 w_k1_IBM_s115
                                 + 1.00915 w_k1_INTC_s115
                                 + 1.09639 w_k1_KO_s115
                                 + 0.865472 w_k1_MAT_s115
                                 + 1.08045 w_k1_MSFT_s115
                                 + 0.999406 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.908823 w_k1_ADM_s115
                                 + 0.987931 w_k1_AXP_s115
                                 + 1.16793 w_k1_DIS_s115
                                 + 1.03882 w_k1_FDX_s115
                                 + 0.99358 w_k1_IBM_s115
                                 + 0.981112 w_k1_INTC_s115
                                 + 1.16278 w_k1_KO_s115 + 0.95631 w_k1_MAT_s115
                                 + 1.11886 w_k1_MSFT_s115
                                 + 0.900125 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.65865 w_k1_ADM_s115
                                 + 1.41143 w_k1_AXP_s115
                                 + 1.62593 w_k1_DIS_s115
                                 + 1.40098 w_k1_FDX_s115
                                 + 1.10865 w_k1_IBM_s115
                                 + 1.26318 w_k1_INTC_s115
                                 + 1.36726 w_k1_KO_s115
                                 + 0.958372 w_k1_MAT_s115
                                 + 1.32524 w_k1_MSFT_s115
                                 + 1.19079 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.06389 w_k1_ADM_s115
                                 + 0.916712 w_k1_AXP_s115
                                 + 0.945777 w_k1_DIS_s115
                                 + 1.08947 w_k1_FDX_s115
                                 + 0.698216 w_k1_IBM_s115
                                 + 0.950023 w_k1_INTC_s115
                                 + 0.84376 w_k1_KO_s115
                                 + 0.524872 w_k1_MAT_s115
                                 + 1.11421 w_k1_MSFT_s115
                                 + 0.873794 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.892667 w_k1_ADM_s115
                                 + 1.17804 w_k1_AXP_s115 + 1.5354 w_k1_DIS_s115
                                 + 1.13949 w_k1_FDX_s115
                                 + 1.84303 w_k1_IBM_s115
                                 + 1.20983 w_k1_INTC_s115
                                 + 1.75053 w_k1_KO_s115 + 1.72518 w_k1_MAT_s115
                                 + 1.36693 w_k1_MSFT_s115 + 1.0616 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.782818 w_k1_ADM_s115
                                 + 0.954673 w_k1_AXP_s115
                                 + 1.11751 w_k1_DIS_s115
                                 + 1.07902 w_k1_FDX_s115
                                 + 1.33227 w_k1_IBM_s115
                                 + 1.00922 w_k1_INTC_s115
                                 + 1.27805 w_k1_KO_s115 + 1.17744 w_k1_MAT_s115
                                 + 1.10569 w_k1_MSFT_s115
                                 + 0.933279 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.855597 w_k1_ADM_s115
                                 + 0.638852 w_k1_AXP_s115
                                 + 0.753486 w_k1_DIS_s115
                                 + 0.80252 w_k1_FDX_s115 + 1.0456 w_k1_IBM_s115
                                 + 0.832327 w_k1_INTC_s115
                                 + 1.00293 w_k1_KO_s115
                                 + 0.697993 w_k1_MAT_s115
                                 + 0.837694 w_k1_MSFT_s115
                                 + 0.908168 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.86104 w_k1_ADM_s115
                                 + 0.59817 w_k1_AXP_s115
                                 + 0.622571 w_k1_DIS_s115
                                 + 0.717202 w_k1_FDX_s115
                                 + 0.677359 w_k1_IBM_s115
                                 + 0.661437 w_k1_INTC_s115
                                 + 0.807371 w_k1_KO_s115
                                 + 0.52535 w_k1_MAT_s115
                                 + 0.786034 w_k1_MSFT_s115
                                 + 0.78466 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.05802 w_k1_ADM_s116
                                 + 0.872439 w_k1_AXP_s116
                                 + 0.812871 w_k1_DIS_s116
                                 + 0.967434 w_k1_FDX_s116
                                 + 0.909953 w_k1_IBM_s116
                                 + 1.00433 w_k1_INTC_s116
                                 + 1.03636 w_k1_KO_s116
                                 + 0.860839 w_k1_MAT_s116
                                 + 0.942097 w_k1_MSFT_s116
                                 + 0.97661 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.14324 w_k1_ADM_s116
                                 + 1.40825 w_k1_AXP_s116
                                 + 1.84109 w_k1_DIS_s116
                                 + 1.26526 w_k1_FDX_s116
                                 + 1.57005 w_k1_IBM_s116
                                 + 1.27481 w_k1_INTC_s116
                                 + 1.68255 w_k1_KO_s116 + 1.74853 w_k1_MAT_s116
                                 + 1.36175 w_k1_MSFT_s116
                                 + 1.43096 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.09771 w_k1_ADM_s116
                                 + 1.07778 w_k1_AXP_s116
                                 + 0.943292 w_k1_DIS_s116
                                 + 1.01101 w_k1_FDX_s116
                                 + 1.09163 w_k1_IBM_s116
                                 + 1.04648 w_k1_INTC_s116
                                 + 1.01464 w_k1_KO_s116 + 1.38591 w_k1_MAT_s116
                                 + 1.02101 w_k1_MSFT_s116
                                 + 0.952558 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.863082 w_k1_ADM_s116
                                 + 0.657408 w_k1_AXP_s116
                                 + 0.538948 w_k1_DIS_s116
                                 + 0.692176 w_k1_FDX_s116
                                 + 0.825746 w_k1_IBM_s116
                                 + 0.778853 w_k1_INTC_s116
                                 + 0.743066 w_k1_KO_s116
                                 + 0.781013 w_k1_MAT_s116
                                 + 0.684789 w_k1_MSFT_s116
                                 + 0.849099 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.26202 w_k1_ADM_s116
                                 + 1.29398 w_k1_AXP_s116
                                 + 1.16888 w_k1_DIS_s116
                                 + 1.25023 w_k1_FDX_s116
                                 + 0.811189 w_k1_IBM_s116
                                 + 1.12071 w_k1_INTC_s116
                                 + 0.971391 w_k1_KO_s116
                                 + 1.13798 w_k1_MAT_s116
                                 + 1.18092 w_k1_MSFT_s116
                                 + 1.11687 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.24495 w_k1_ADM_s116
                                 + 0.853744 w_k1_AXP_s116
                                 + 1.24482 w_k1_DIS_s116
                                 + 1.13342 w_k1_FDX_s116
                                 + 0.895578 w_k1_IBM_s116
                                 + 0.92356 w_k1_INTC_s116
                                 + 1.14927 w_k1_KO_s116
                                 + 0.999648 w_k1_MAT_s116
                                 + 1.252 w_k1_MSFT_s116 + 0.982874 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.03762 w_k1_ADM_s116
                                 + 0.817762 w_k1_AXP_s116
                                 + 0.945627 w_k1_DIS_s116
                                 + 0.802501 w_k1_FDX_s116
                                 + 1.37463 w_k1_IBM_s116
                                 + 0.868067 w_k1_INTC_s116
                                 + 1.20219 w_k1_KO_s116 + 1.1363 w_k1_MAT_s116
                                 + 1.03833 w_k1_MSFT_s116
                                 + 1.06793 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.19936 w_k1_ADM_s116
                                 + 1.26467 w_k1_AXP_s116
                                 + 1.11999 w_k1_DIS_s116
                                 + 1.11476 w_k1_FDX_s116
                                 + 1.29667 w_k1_IBM_s116
                                 + 0.953801 w_k1_INTC_s116
                                 + 1.19703 w_k1_KO_s116 + 1.63029 w_k1_MAT_s116
                                 + 1.13511 w_k1_MSFT_s116
                                 + 1.13511 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.35871 w_k1_ADM_s117
                                 + 1.40336 w_k1_AXP_s117
                                 + 1.23511 w_k1_DIS_s117
                                 + 1.20333 w_k1_FDX_s117
                                 + 1.41707 w_k1_IBM_s117
                                 + 1.40061 w_k1_INTC_s117
                                 + 1.28672 w_k1_KO_s117 + 2.03809 w_k1_MAT_s117
                                 + 1.2648 w_k1_MSFT_s117 + 1.18761 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.03879 w_k1_ADM_s117
                                 + 0.957378 w_k1_AXP_s117
                                 + 1.34772 w_k1_DIS_s117
                                 + 1.10338 w_k1_FDX_s117
                                 + 0.97925 w_k1_IBM_s117
                                 + 1.19617 w_k1_INTC_s117
                                 + 1.12665 w_k1_KO_s117 + 1.0737 w_k1_MAT_s117
                                 + 1.15849 w_k1_MSFT_s117
                                 + 1.08096 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.788586 w_k1_ADM_s117
                                 + 0.747906 w_k1_AXP_s117
                                 + 0.688891 w_k1_DIS_s117
                                 + 0.823133 w_k1_FDX_s117
                                 + 0.978924 w_k1_IBM_s117
                                 + 0.905217 w_k1_INTC_s117
                                 + 0.786582 w_k1_KO_s117
                                 + 0.998897 w_k1_MAT_s117
                                 + 0.796311 w_k1_MSFT_s117
                                 + 0.817832 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.12839 w_k1_ADM_s117
                                 + 1.02783 w_k1_AXP_s117
                                 + 0.87149 w_k1_DIS_s117
                                 + 0.979547 w_k1_FDX_s117
                                 + 0.846284 w_k1_IBM_s117
                                 + 0.999482 w_k1_INTC_s117
                                 + 0.828533 w_k1_KO_s117
                                 + 0.737013 w_k1_MAT_s117
                                 + 0.909856 w_k1_MSFT_s117
                                 + 0.912703 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.40093 w_k1_ADM_s117
                                 + 1.42768 w_k1_AXP_s117
                                 + 1.32692 w_k1_DIS_s117
                                 + 1.30659 w_k1_FDX_s117
                                 + 0.842498 w_k1_IBM_s117
                                 + 1.22948 w_k1_INTC_s117
                                 + 1.14019 w_k1_KO_s117 + 1.18275 w_k1_MAT_s117
                                 + 1.30337 w_k1_MSFT_s117
                                 + 1.03566 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.03972 w_k1_ADM_s117
                                 + 1.15882 w_k1_AXP_s117
                                 + 1.28959 w_k1_DIS_s117
                                 + 1.04502 w_k1_FDX_s117
                                 + 1.74986 w_k1_IBM_s117
                                 + 1.25038 w_k1_INTC_s117
                                 + 1.27085 w_k1_KO_s117 + 1.89254 w_k1_MAT_s117
                                 + 1.23881 w_k1_MSFT_s117
                                 + 1.01095 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.840081 w_k1_ADM_s117
                                 + 0.611146 w_k1_AXP_s117
                                 + 0.727783 w_k1_DIS_s117
                                 + 0.859949 w_k1_FDX_s117
                                 + 0.583871 w_k1_IBM_s117
                                 + 1.03961 w_k1_INTC_s117
                                 + 0.685232 w_k1_KO_s117
                                 + 0.704866 w_k1_MAT_s117
                                 + 0.933731 w_k1_MSFT_s117
                                 + 0.795784 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.827225 w_k1_ADM_s117
                                 + 0.973385 w_k1_AXP_s117
                                 + 0.888118 w_k1_DIS_s117
                                 + 0.853933 w_k1_FDX_s117 + 1.364 w_k1_IBM_s117
                                 + 1.07964 w_k1_INTC_s117 + 1.1348 w_k1_KO_s117
                                 + 1.30599 w_k1_MAT_s117
                                 + 0.966321 w_k1_MSFT_s117
                                 + 0.992237 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00625287529015266 w_k1_ADM_s000
                                 + 0.01196221568263 w_k1_ADM_s001
                                 + 0.00888351385142007 w_k1_ADM_s002
                                 + 0.0118624237697605 w_k1_ADM_s003
                                 + 0.00607005213416567 w_k1_ADM_s004
                                 + 0.00499206515761694 w_k1_ADM_s005
                                 + 0.00424302671365192 w_k1_ADM_s006
                                 + 0.00762574005122927 w_k1_ADM_s007
                                 + 0.0271189552398909 w_k1_ADM_s010
                                 + 0.0555672594777159 w_k1_ADM_s011
                                 + 0.0243487359502865 w_k1_ADM_s012
                                 + 0.0280417471735667 w_k1_ADM_s013
                                 + 0.0780871300338746 w_k1_ADM_s014
                                 + 0.024561872335665 w_k1_ADM_s015
                                 + 0.0180033389187931 w_k1_ADM_s016
                                 + 0.0223854484062898 w_k1_ADM_s017
                                 + 0.0372440240771414 w_k1_ADM_s100
                                 + 0.0440580699025019 w_k1_ADM_s101
                                 + 0.030037792792507 w_k1_ADM_s102
                                 + 0.0132905725703195 w_k1_ADM_s103
                                 + 0.0245922454939142 w_k1_ADM_s104
                                 + 0.0588043287413605 w_k1_ADM_s105
                                 + 0.0593935187217435 w_k1_ADM_s106
                                 + 0.0351173260218629 w_k1_ADM_s107
                                 + 0.0259443209266092 w_k1_ADM_s110
                                 + 0.0217825959532453 w_k1_ADM_s111
                                 + 0.0396792311021389 w_k1_ADM_s112
                                 + 0.0524070824103312 w_k1_ADM_s113
                                 + 0.0955362646208619 w_k1_ADM_s114
                                 + 0.0580931342017564 w_k1_ADM_s115
                                 + 0.0779786487830943 w_k1_ADM_s116
                                 + 0.0254582806542682 w_k1_ADM_s117
                                 + 0.005794122784693 w_k1_AXP_s000
                                 + 0.0130375988991425 w_k1_AXP_s001
                                 + 0.00903726482537716 w_k1_AXP_s002
                                 + 0.0131633338815939 w_k1_AXP_s003
                                 + 0.00585564742880199 w_k1_AXP_s004
                                 + 0.00466195666483348 w_k1_AXP_s005
                                 + 0.00420913612684934 w_k1_AXP_s006
                                 + 0.00845146971532479 w_k1_AXP_s007
                                 + 0.0267613223064517 w_k1_AXP_s010
                                 + 0.058708466817454 w_k1_AXP_s011
                                 + 0.0278660106144134 w_k1_AXP_s012
                                 + 0.0273475563461317 w_k1_AXP_s013
                                 + 0.0688106049801079 w_k1_AXP_s014
                                 + 0.0226548788568733 w_k1_AXP_s015
                                 + 0.0210123012790437 w_k1_AXP_s016
                                 + 0.0205900832447231 w_k1_AXP_s017
                                 + 0.0294181273122709 w_k1_AXP_s100
                                 + 0.0440097299359993 w_k1_AXP_s101
                                 + 0.0291090329678746 w_k1_AXP_s102
                                 + 0.0160614249059864 w_k1_AXP_s103
                                 + 0.0249271336563503 w_k1_AXP_s104
                                 + 0.0509567876085985 w_k1_AXP_s105
                                 + 0.0524581006246724 w_k1_AXP_s106
                                 + 0.0384385033086446 w_k1_AXP_s107
                                 + 0.0233964434630056 w_k1_AXP_s110
                                 + 0.0198978378711011 w_k1_AXP_s111
                                 + 0.0368738125852537 w_k1_AXP_s112
                                 + 0.0527526716801001 w_k1_AXP_s113
                                 + 0.0875130607622114 w_k1_AXP_s114
                                 + 0.0558047226641925 w_k1_AXP_s115
                                 + 0.0691954954566077 w_k1_AXP_s116
                                 + 0.0241276419542653 w_k1_AXP_s117
                                 + 0.00613430638707678 w_k1_DIS_s000
                                 + 0.0126364946959431 w_k1_DIS_s001
                                 + 0.00906259634875134 w_k1_DIS_s002
                                 + 0.0114813190496037 w_k1_DIS_s003
                                 + 0.00557464778788665 w_k1_DIS_s004
                                 + 0.00459751253386174 w_k1_DIS_s005
                                 + 0.00410741902319803 w_k1_DIS_s006
                                 + 0.0078024426096275 w_k1_DIS_s007
                                 + 0.029155610300052 w_k1_DIS_s010
                                 + 0.0609649215640341 w_k1_DIS_s011
                                 + 0.0283296263375775 w_k1_DIS_s012
                                 + 0.0294892928452428 w_k1_DIS_s013
                                 + 0.0716324195021564 w_k1_DIS_s014
                                 + 0.0218286232534743 w_k1_DIS_s015
                                 + 0.0239757417680617 w_k1_DIS_s016
                                 + 0.0220697278747915 w_k1_DIS_s017
                                 + 0.0337412544639317 w_k1_DIS_s100
                                 + 0.0488588188457832 w_k1_DIS_s101
                                 + 0.0278465352073822 w_k1_DIS_s102
                                 + 0.0155043009235858 w_k1_DIS_s103
                                 + 0.0232267031018717 w_k1_DIS_s104
                                 + 0.0487507174539174 w_k1_DIS_s105
                                 + 0.0584132341647826 w_k1_DIS_s106
                                 + 0.0365665705706874 w_k1_DIS_s107
                                 + 0.0257387974730693 w_k1_DIS_s110
                                 + 0.0208721896826845 w_k1_DIS_s111
                                 + 0.0455578500233287 w_k1_DIS_s112
                                 + 0.0543317978319188 w_k1_DIS_s113
                                 + 0.0942075920344471 w_k1_DIS_s114
                                 + 0.0626492081621199 w_k1_DIS_s115
                                 + 0.0724673239819425 w_k1_DIS_s116
                                 + 0.0253876310722466 w_k1_DIS_s117
                                 + 0.00619771081399217 w_k1_FDX_s000
                                 + 0.0130528234408183 w_k1_FDX_s001
                                 + 0.00881849081474603 w_k1_FDX_s002
                                 + 0.0133121636077649 w_k1_FDX_s003
                                 + 0.00515765905016086 w_k1_FDX_s004
                                 + 0.00439995604572602 w_k1_FDX_s005
                                 + 0.00428243436495907 w_k1_FDX_s006
                                 + 0.00780714840668047 w_k1_FDX_s007
                                 + 0.027532793624106 w_k1_FDX_s010
                                 + 0.0618727766584931 w_k1_FDX_s011
                                 + 0.026069238255096 w_k1_FDX_s012
                                 + 0.0282991013357312 w_k1_FDX_s013
                                 + 0.0696323230916889 w_k1_FDX_s014
                                 + 0.0235189921550421 w_k1_FDX_s015
                                 + 0.0212909381035661 w_k1_FDX_s016
                                 + 0.0214002455937017 w_k1_FDX_s017
                                 + 0.0300459304079793 w_k1_FDX_s100
                                 + 0.0465820605937655 w_k1_FDX_s101
                                 + 0.0281126897640322 w_k1_FDX_s102
                                 + 0.0150643094004185 w_k1_FDX_s103
                                 + 0.0233253230809282 w_k1_FDX_s104
                                 + 0.051900055792349 w_k1_FDX_s105
                                 + 0.0603468188218613 w_k1_FDX_s106
                                 + 0.0351439797371037 w_k1_FDX_s107
                                 + 0.0264082926596849 w_k1_FDX_s110
                                 + 0.0213136427440759 w_k1_FDX_s111
                                 + 0.0413788431000405 w_k1_FDX_s112
                                 + 0.0506569006427725 w_k1_FDX_s113
                                 + 0.0894985924466838 w_k1_FDX_s114
                                 + 0.0612517538815139 w_k1_FDX_s115
                                 + 0.0716960713923296 w_k1_FDX_s116
                                 + 0.0248749818848459 w_k1_FDX_s117
                                 + 0.00677389185396414 w_k1_IBM_s000
                                 + 0.0129560016246232 w_k1_IBM_s001
                                 + 0.00847052838672421 w_k1_IBM_s002
                                 + 0.0126969615955775 w_k1_IBM_s003
                                 + 0.00576724966195185 w_k1_IBM_s004
                                 + 0.00460024648715174 w_k1_IBM_s005
                                 + 0.00432828118244679 w_k1_IBM_s006
                                 + 0.00855601114126784 w_k1_IBM_s007
                                 + 0.0279976521015737 w_k1_IBM_s010
                                 + 0.0634413545092207 w_k1_IBM_s011
                                 + 0.0273223597142496 w_k1_IBM_s012
                                 + 0.0295093848828573 w_k1_IBM_s013
                                 + 0.0700480787795925 w_k1_IBM_s014
                                 + 0.0236711171997752 w_k1_IBM_s015
                                 + 0.0223948213413724 w_k1_IBM_s016
                                 + 0.021095932063098 w_k1_IBM_s017
                                 + 0.0300506244562764 w_k1_IBM_s100
                                 + 0.0482377772419917 w_k1_IBM_s101
                                 + 0.0289297149658999 w_k1_IBM_s102
                                 + 0.0144710600650912 w_k1_IBM_s103
                                 + 0.0223733912428869 w_k1_IBM_s104
                                 + 0.0510245650690659 w_k1_IBM_s105
                                 + 0.0592147155558538 w_k1_IBM_s106
                                 + 0.0357945938636492 w_k1_IBM_s107
                                 + 0.0266952686096318 w_k1_IBM_s110
                                 + 0.0208531218254392 w_k1_IBM_s111
                                 + 0.0416806003524739 w_k1_IBM_s112
                                 + 0.0494627457942901 w_k1_IBM_s113
                                 + 0.0887802251845179 w_k1_IBM_s114
                                 + 0.0613953558827977 w_k1_IBM_s115
                                 + 0.0727405506254303 w_k1_IBM_s116
                                 + 0.024320088667519 w_k1_IBM_s117
                                 + 0.00505217171149412 w_k1_INTC_s000
                                 + 0.0108898416480016 w_k1_INTC_s001
                                 + 0.00846027216526195 w_k1_INTC_s002
                                 + 0.0124518761638935 w_k1_INTC_s003
                                 + 0.00504352948466155 w_k1_INTC_s004
                                 + 0.0044071741095451 w_k1_INTC_s005
                                 + 0.00435457714790304 w_k1_INTC_s006
                                 + 0.00691932714176327 w_k1_INTC_s007
                                 + 0.0278501425794393 w_k1_INTC_s010
                                 + 0.0673745205302681 w_k1_INTC_s011
                                 + 0.0267195577596121 w_k1_INTC_s012
                                 + 0.0279143062696588 w_k1_INTC_s013
                                 + 0.0682328414076418 w_k1_INTC_s014
                                 + 0.0238962864980209 w_k1_INTC_s015
                                 + 0.0242627653258696 w_k1_INTC_s016
                                 + 0.0221675775037116 w_k1_INTC_s017
                                 + 0.0295787893198244 w_k1_INTC_s100
                                 + 0.0495493087493729 w_k1_INTC_s101
                                 + 0.0246602969518832 w_k1_INTC_s102
                                 + 0.0149209372163603 w_k1_INTC_s103
                                 + 0.0255087342155698 w_k1_INTC_s104
                                 + 0.045099188178032 w_k1_INTC_s105
                                 + 0.0510594902358946 w_k1_INTC_s106
                                 + 0.032638888924234 w_k1_INTC_s107
                                 + 0.0233845841036038 w_k1_INTC_s110
                                 + 0.0185263204865389 w_k1_INTC_s111
                                 + 0.0400270079071593 w_k1_INTC_s112
                                 + 0.0442585452467231 w_k1_INTC_s113
                                 + 0.0899102418747432 w_k1_INTC_s114
                                 + 0.0576125517231017 w_k1_INTC_s115
                                 + 0.0689671024380262 w_k1_INTC_s116
                                 + 0.0273683641753686 w_k1_INTC_s117
                                 + 0.00589674863217759 w_k1_KO_s000
                                 + 0.0121515279281963 w_k1_KO_s001
                                 + 0.00923524833715355 w_k1_KO_s002
                                 + 0.0115629989864558 w_k1_KO_s003
                                 + 0.00569847753588002 w_k1_KO_s004
                                 + 0.00455994974514498 w_k1_KO_s005
                                 + 0.00426065851237728 w_k1_KO_s006
                                 + 0.0082625078998798 w_k1_KO_s007
                                 + 0.0313732992609088 w_k1_KO_s010
                                 + 0.0649384762333824 w_k1_KO_s011
                                 + 0.0280565800124639 w_k1_KO_s012
                                 + 0.0306534255028692 w_k1_KO_s013
                                 + 0.0688749272060434 w_k1_KO_s014
                                 + 0.022972052173863 w_k1_KO_s015
                                 + 0.0228070644149576 w_k1_KO_s016
                                 + 0.0216085449324049 w_k1_KO_s017
                                 + 0.0287522685610014 w_k1_KO_s100
                                 + 0.04665026627565 w_k1_KO_s101
                                 + 0.0263377806810313 w_k1_KO_s102
                                 + 0.0142465697563623 w_k1_KO_s103
                                 + 0.0219754805714841 w_k1_KO_s104
                                 + 0.0534425479696573 w_k1_KO_s105
                                 + 0.0565770301475503 w_k1_KO_s106
                                 + 0.0336789960812086 w_k1_KO_s107
                                 + 0.0243389482659223 w_k1_KO_s110
                                 + 0.021830166895338 w_k1_KO_s111
                                 + 0.0382429236045721 w_k1_KO_s112
                                 + 0.0531494812701995 w_k1_KO_s113
                                 + 0.0918305126428759 w_k1_KO_s114
                                 + 0.0661573690881864 w_k1_KO_s115
                                 + 0.0768371323427536 w_k1_KO_s116
                                 + 0.0241892207568432 w_k1_KO_s117
                                 + 0.0075085711910733 w_k1_MAT_s000
                                 + 0.013698271371801 w_k1_MAT_s001
                                 + 0.00946603648122058 w_k1_MAT_s002
                                 + 0.0117361769641649 w_k1_MAT_s003
                                 + 0.00546549331595942 w_k1_MAT_s004
                                 + 0.00448318853161157 w_k1_MAT_s005
                                 + 0.00457381872617841 w_k1_MAT_s006
                                 + 0.00882293578983498 w_k1_MAT_s007
                                 + 0.0284815964120727 w_k1_MAT_s010
                                 + 0.0677919750384578 w_k1_MAT_s011
                                 + 0.0303390442476853 w_k1_MAT_s012
                                 + 0.0285562377473305 w_k1_MAT_s013
                                 + 0.0690377684018372 w_k1_MAT_s014
                                 + 0.0184269182261207 w_k1_MAT_s015
                                 + 0.0212576758099197 w_k1_MAT_s016
                                 + 0.0203225230128007 w_k1_MAT_s017
                                 + 0.0318790606390072 w_k1_MAT_s100
                                 + 0.053729495329051 w_k1_MAT_s101
                                 + 0.028211144572203 w_k1_MAT_s102
                                 + 0.0180069190700928 w_k1_MAT_s103
                                 + 0.0295714971470027 w_k1_MAT_s104
                                 + 0.054313153235992 w_k1_MAT_s105
                                 + 0.0616721175598917 w_k1_MAT_s106
                                 + 0.0466146042979704 w_k1_MAT_s107
                                 + 0.0342053369222316 w_k1_MAT_s110
                                 + 0.0177395384080071 w_k1_MAT_s111
                                 + 0.0339202743048442 w_k1_MAT_s112
                                 + 0.0521374401569129 w_k1_MAT_s113
                                 + 0.0741994538436077 w_k1_MAT_s114
                                 + 0.0523189927379187 w_k1_MAT_s115
                                 + 0.078991640948426 w_k1_MAT_s116
                                 + 0.0272903964900411 w_k1_MAT_s117
                                 + 0.00620083527482865 w_k1_MSFT_s000
                                 + 0.0119221358677836 w_k1_MSFT_s001
                                 + 0.00807470059531078 w_k1_MSFT_s002
                                 + 0.0120819389423654 w_k1_MSFT_s003
                                 + 0.0054900147780772 w_k1_MSFT_s004
                                 + 0.00404590950186251 w_k1_MSFT_s005
                                 + 0.00432712140442929 w_k1_MSFT_s006
                                 + 0.00801907050731766 w_k1_MSFT_s007
                                 + 0.0260456372630861 w_k1_MSFT_s010
                                 + 0.0637034367238516 w_k1_MSFT_s011
                                 + 0.0273566043198446 w_k1_MSFT_s012
                                 + 0.0290170695796694 w_k1_MSFT_s013
                                 + 0.0685706195687698 w_k1_MSFT_s014
                                 + 0.0230158903645487 w_k1_MSFT_s015
                                 + 0.0212859813050515 w_k1_MSFT_s016
                                 + 0.021499108039942 w_k1_MSFT_s017
                                 + 0.0310869491417981 w_k1_MSFT_s100
                                 + 0.047790630298125 w_k1_MSFT_s101
                                 + 0.0274140308346421 w_k1_MSFT_s102
                                 + 0.0151597952862187 w_k1_MSFT_s103
                                 + 0.0240841714671251 w_k1_MSFT_s104
                                 + 0.0498802505868201 w_k1_MSFT_s105
                                 + 0.0611596715234775 w_k1_MSFT_s106
                                 + 0.0338296026570821 w_k1_MSFT_s107
                                 + 0.0266051504931791 w_k1_MSFT_s110
                                 + 0.0216214823564087 w_k1_MSFT_s111
                                 + 0.0377588069979274 w_k1_MSFT_s112
                                 + 0.0507032971112841 w_k1_MSFT_s113
                                 + 0.0908010972059226 w_k1_MSFT_s114
                                 + 0.0639081346033107 w_k1_MSFT_s115
                                 + 0.0742757545425701 w_k1_MSFT_s116
                                 + 0.0256385515647961 w_k1_MSFT_s117
                                 + 0.00573918339111623 w_k1_PG_s000
                                 + 0.0121801102501125 w_k1_PG_s001
                                 + 0.00919328665863934 w_k1_PG_s002
                                 + 0.0114496041741787 w_k1_PG_s003
                                 + 0.00542141624248431 w_k1_PG_s004
                                 + 0.00442773326412547 w_k1_PG_s005
                                 + 0.00472995704280826 w_k1_PG_s006
                                 + 0.00715536498163822 w_k1_PG_s007
                                 + 0.028810482059327 w_k1_PG_s010
                                 + 0.0619244866930476 w_k1_PG_s011
                                 + 0.0267637321783651 w_k1_PG_s012
                                 + 0.0292025144546643 w_k1_PG_s013
                                 + 0.071734571557479 w_k1_PG_s014
                                 + 0.022913805839926 w_k1_PG_s015
                                 + 0.0225328161868055 w_k1_PG_s016
                                 + 0.0217528794898235 w_k1_PG_s017
                                 + 0.0297558582550649 w_k1_PG_s100
                                 + 0.0491306852009766 w_k1_PG_s101
                                 + 0.0255743410344532 w_k1_PG_s102
                                 + 0.014266507228306 w_k1_PG_s103
                                 + 0.0232051095494859 w_k1_PG_s104
                                 + 0.0507053226921589 w_k1_PG_s105
                                 + 0.0569627135288034 w_k1_PG_s106
                                 + 0.0344587315075501 w_k1_PG_s107
                                 + 0.0257305786270575 w_k1_PG_s110
                                 + 0.0216407036222416 w_k1_PG_s111
                                 + 0.0404478003008564 w_k1_PG_s112
                                 + 0.0509044470426192 w_k1_PG_s113
                                 + 0.0929461461065587 w_k1_PG_s114
                                 + 0.0550850567908384 w_k1_PG_s115
                                 + 0.0729577979531304 w_k1_PG_s116
                                 + 0.0237782046342682 w_k1_PG_s117 >= 1.209
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
