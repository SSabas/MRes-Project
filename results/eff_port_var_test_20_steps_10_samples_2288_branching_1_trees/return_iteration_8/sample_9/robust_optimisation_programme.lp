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
 w_k1_ADM_s0_return_constraint:  1.08702 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.890511 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.02838 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.01783 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.918784 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.04509 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.785584 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.08502 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.957829 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.32286 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.21987 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.830396 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.39256 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.27867 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.17878 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.972673 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.03453 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.862348 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.725612 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.43661 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.10897 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.90868 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.04331 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.882313 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.05435 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.15501 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.22251 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.93473 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.723104 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.30314 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.09178 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.857824 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.71877 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.97155 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.29269 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.16331 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.06104 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.58411 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18092 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.865425 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.11489 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.944284 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.841786 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.25976 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.675039 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.938847 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.01091 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.32267 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.51104 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.851291 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.70945 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.28996 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.23803 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.798038 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.22485 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.672389 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.802092 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.5929 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.1946 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.07549 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.816205 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.01406 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.20986 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.03035 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.20778 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.710103 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.555761 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.18108 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.05729 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.878484 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.40076 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.734139 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.16569 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.24043 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.803937 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.08763 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25043 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.848033 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.17507 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.873371 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.798297 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.2161 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.611021 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.923883 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.988505 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.42324 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.22667 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.791646 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.48195 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.54837 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.48538 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.08331 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.18392 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.619436 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.935501 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.90343 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.23655 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.893533 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.838202 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.78506 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.14644 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.19868 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.24343 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.788608 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.497498 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.53939 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.878774 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.855799 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.64571 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.722902 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.1141 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.01109 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.00759 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.42463 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.14622 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.887662 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.0628 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.930738 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.909322 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.15661 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.735566 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.04004 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.98092 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.06471 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.33612 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.800385 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.3255 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.4133 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.46246 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.02532 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.10059 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.785397 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.856401 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.52167 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.18287 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.1216 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.899716 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.817518 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.01482 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.955243 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.20245 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.755495 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.650944 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.32493 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.970192 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.794613 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.28981 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.817982 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.19597 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.04143 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.991063 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.3402 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17673 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.891743 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.24325 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.751013 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.866174 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.18624 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.772016 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.711983 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.992428 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.78315 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.10526 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.23568 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.41514 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.790186 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.862518 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.891569 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.65847 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.817547 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.32855 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.23348 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.05434 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.724699 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.678393 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.956861 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.22771 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.74743 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.756149 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.06243 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.571311 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.15765 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.85958 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.3488 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.43661 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.678314 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.707642 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.09463 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.769101 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.802586 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.23236 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.869385 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.927292 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.907554 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.937722 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.21808 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.768511 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.14093 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.877543 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.15259 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.35899 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.941854 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.28353 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.22886 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.27777 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.873031 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.0128 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.802227 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.82608 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.25346 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.07756 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.993843 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.912311 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.993175 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.1689 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.0453 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.08495 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.885459 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.724415 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.07532 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.910625 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.950703 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.16767 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.813857 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.05225 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.0471 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.891785 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.143 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19005 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.836557 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.09996 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.849026 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.90459 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.19826 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.778348 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.844911 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.95288 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.43147 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.14008 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.01754 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.39753 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.20016 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.14832 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.962365 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.3227 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.746925 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.995928 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.45219 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.14949 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.802888 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.878568 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.823263 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.20218 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.30892 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.00903 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.00261 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.607802 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.25142 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.876474 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.02525 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.41672 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.889385 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.937984 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.01739 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.84788 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.08453 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.18485 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.907926 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.24418 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.639834 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.78319 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.27475 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.784499 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.696565 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.04176 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.47014 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.2787 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.30232 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.92071 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.11486 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.07065 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.972136 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 2.08836 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.948474 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.60712 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.75119 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.27287 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.1763 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.696814 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.01918 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.42472 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.65762 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.623909 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.833913 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.487463 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.08975 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.959151 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.12706 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.32152 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.583507 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.569922 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.29994 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.729099 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.830823 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13857 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.886218 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.996503 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.953764 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.928956 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.21074 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.775869 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.06981 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.99726 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.18535 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.24047 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.882853 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.2966 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.46455 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.33496 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.02068 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.11119 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.779089 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.849821 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.27373 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.17053 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.943319 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.945583 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.800702 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.04429 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.982702 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.13257 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.778896 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.676897 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.25035 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.85661 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.84651 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.16602 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.835693 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.11854 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.959083 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.935278 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.13375 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12927 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.896508 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.0876 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.878535 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.952015 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.16302 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.834159 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.83486 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.972669 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.05174 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.0876 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.871205 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.17153 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.06243 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.19953 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.04498 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.02212 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.81638 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.946287 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.27686 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.05114 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.987862 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.881096 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.883746 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.08696 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.05484 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.962566 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.918475 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.701351 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.19716 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.955085 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.989328 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.23241 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.830834 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.03044 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.994769 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.95033 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.08422 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.519000589576902 z_k1_s0000
                                 - 0.723001217261402 z_k1_s0001
                                 - 0.30800012627945 z_k1_s0002
                                 - 0.670999203560978 z_k1_s0003
                                 - 0.418000292686554 z_k1_s0004
                                 - 0.848001920997417 z_k1_s0005
                                 - 1.27500038489852 z_k1_s0006
                                 - 0.899001936393358 z_k1_s0007
                                 - 0.636001378077078 z_k1_s0010
                                 - 0.191000468652853 z_k1_s0011
                                 - 0.471999652814582 z_k1_s0012
                                 - 0.286000900005843 z_k1_s0013
                                 - 0.480001862003745 z_k1_s0014
                                 - 0.38700131274627 z_k1_s0015
                                 - 0.517000350972569 z_k1_s0016
                                 - 0.38200160750748 z_k1_s0017
                                 - 0.375998812803575 z_k1_s0020
                                 - 0.255999158138723 z_k1_s0021
                                 - 0.494998526205189 z_k1_s0022
                                 - 0.359998569695951 z_k1_s0023
                                 - 0.227999225573637 z_k1_s0024
                                 - 0.759998732907476 z_k1_s0025
                                 - 0.983995827636529 z_k1_s0026
                                 - 0.48399759511515 z_k1_s0027
                                 - 0.253999485109895 z_k1_s0030
                                 - 0.383999322203716 z_k1_s0031
                                 - 0.337998520806243 z_k1_s0032
                                 - 0.153999760006629 z_k1_s0033
                                 - 0.374998621888008 z_k1_s0034
                                 - 0.377998855326577 z_k1_s0035
                                 - 0.703999220372467 z_k1_s0036
                                 - 0.190999583289002 z_k1_s0037
                                 - 0.335000231939179 z_k1_s0040
                                 - 0.437999596087766 z_k1_s0041
                                 - 0.150000079102622 z_k1_s0042
                                 - 0.204000003343584 z_k1_s0043
                                 - 0.246999004191179 z_k1_s0044
                                 - 0.285999844431719 z_k1_s0045
                                 - 0.188999971743326 z_k1_s0046
                                 - 0.520000147875777 z_k1_s0047
                                 - 0.329999762436247 z_k1_s0050
                                 - 0.596999126133381 z_k1_s0051
                                 - 0.435999140998849 z_k1_s0052
                                 - 0.188999645419204 z_k1_s0053
                                 - 0.413000804691915 z_k1_s0054
                                 - 0.392999385776293 z_k1_s0055
                                 - 0.616000280253455 z_k1_s0056
                                 - 0.902998736669013 z_k1_s0057
                                 - 0.237000553851671 z_k1_s0060
                                 - 0.165999898728927 z_k1_s0061
                                 - 0.287999491390171 z_k1_s0062
                                 - 0.219999663339679 z_k1_s0063
                                 - 0.170999908050272 z_k1_s0064
                                 - 0.357998875089504 z_k1_s0065
                                 - 0.128999904430925 z_k1_s0066
                                 - 0.297999136633113 z_k1_s0067
                                 - 0.143999227896864 z_k1_s0070
                                 - 0.27599867458492 z_k1_s0071
                                 - 0.467996718418481 z_k1_s0072
                                 - 0.290998478320561 z_k1_s0073
                                 - 0.331997962457882 z_k1_s0074
                                 - 0.467996718418481 z_k1_s0075
                                 - 0.644995576313461 z_k1_s0076
                                 - 0.464997313688707 z_k1_s0077
                                 - 0.0770000744991616 z_k1_s0100
                                 - 0.0840000426535892 z_k1_s0101
                                 - 0.166000337149033 z_k1_s0102
                                 - 0.208999701554327 z_k1_s0103
                                 - 0.0780000547783492 z_k1_s0104
                                 - 0.113000107950716 z_k1_s0105
                                 - 0.0870000896912665 z_k1_s0106
                                 - 0.248000631644476 z_k1_s0107
                                 - 0.232000412142604 z_k1_s0110
                                 - 0.56400143454548 z_k1_s0111
                                 - 0.203000220187107 z_k1_s0112
                                 - 0.361000465357458 z_k1_s0113
                                 - 0.822002289976103 z_k1_s0114
                                 - 0.393000780480891 z_k1_s0115
                                 - 0.724999181380562 z_k1_s0116
                                 - 0.445000169055097 z_k1_s0117
                                 - 0.204000591367899 z_k1_s0120
                                 - 0.109000559731133 z_k1_s0121
                                 - 0.187000630253996 z_k1_s0122
                                 - 0.120000596804897 z_k1_s0123
                                 - 0.0900001302033534 z_k1_s0124
                                 - 0.142000670952426 z_k1_s0125
                                 - 0.055000079568716 z_k1_s0126
                                 - 0.151000028012101 z_k1_s0127
                                 - 0.448999180555051 z_k1_s0130
                                 - 0.324999407943409 z_k1_s0131
                                 - 1.23199974831905 z_k1_s0132
                                 - 0.728997443757398 z_k1_s0133
                                 - 0.355999472646134 z_k1_s0134
                                 - 0.491999615348627 z_k1_s0135
                                 - 0.370999205960811 z_k1_s0136
                                 - 0.907996562655692 z_k1_s0137
                                 - 0.754998408701605 z_k1_s0140
                                 - 0.362999890600712 z_k1_s0141
                                 - 0.397000304634814 z_k1_s0142
                                 - 0.241999927067141 z_k1_s0143
                                 - 0.214999813981174 z_k1_s0144
                                 - 0.408998725117944 z_k1_s0145
                                 - 0.435001414496268 z_k1_s0146
                                 - 0.851000440562346 z_k1_s0147
                                 - 0.147000110320268 z_k1_s0150
                                 - 0.212000674034783 z_k1_s0151
                                 - 0.348999360783297 z_k1_s0152
                                 - 0.328999989440543 z_k1_s0153
                                 - 0.468999450840658 z_k1_s0154
                                 - 0.116000022688769 z_k1_s0155
                                 - 0.113000020437335 z_k1_s0156
                                 - 0.196000404560413 z_k1_s0157
                                 - 0.297999247094763 z_k1_s0160
                                 - 0.308000610586278 z_k1_s0161
                                 - 0.21899978739878 z_k1_s0162
                                 - 0.519000410790576 z_k1_s0163
                                 - 0.274999697760152 z_k1_s0164
                                 - 0.479999231668906 z_k1_s0165
                                 - 0.291998428999854 z_k1_s0166
                                 - 0.368000514544891 z_k1_s0167
                                 - 0.0670001371200383 z_k1_s0170
                                 - 0.14100058729477 z_k1_s0171
                                 - 0.170999729526513 z_k1_s0172
                                 - 0.189999699473903 z_k1_s0173
                                 - 0.241000294070345 z_k1_s0174
                                 - 0.123000203854639 z_k1_s0175
                                 - 0.132999789631732 z_k1_s0176
                                 - 0.217000467084331 z_k1_s0177
                                 - 0.940001417987567 z_k1_s1000
                                 - 0.29000038627269 z_k1_s1001
                                 - 0.475000864991792 z_k1_s1002
                                 - 0.73000064047908 z_k1_s1003
                                 - 0.517001982894797 z_k1_s1004
                                 - 1.11300267719302 z_k1_s1005
                                 - 0.353000619525237 z_k1_s1006
                                 - 0.394000358866809 z_k1_s1007
                                 - 0.434999264901862 z_k1_s1010
                                 - 0.296000825401848 z_k1_s1011
                                 - 0.202999894254142 z_k1_s1012
                                 - 0.490999677599444 z_k1_s1013
                                 - 0.237999974215176 z_k1_s1014
                                 - 0.145999974211833 z_k1_s1015
                                 - 0.443001018858227 z_k1_s1016
                                 - 0.1209999509968 z_k1_s1017
                                 - 0.234999633773037 z_k1_s1020
                                 - 0.325999493770678 z_k1_s1021
                                 - 0.554998373542138 z_k1_s1022
                                 - 0.34899947243269 z_k1_s1023
                                 - 0.810998728874624 z_k1_s1024
                                 - 0.557998111394236 z_k1_s1025
                                 - 0.644999031087624 z_k1_s1026
                                 - 0.781998422310162 z_k1_s1027
                                 - 0.563998688028939 z_k1_s1030
                                 - 0.171999855427029 z_k1_s1031
                                 - 0.374000663195081 z_k1_s1032
                                 - 0.296999820184981 z_k1_s1033
                                 - 0.236999837101164 z_k1_s1034
                                 - 0.337000289307082 z_k1_s1035
                                 - 0.146999862475439 z_k1_s1036
                                 - 0.27400069138872 z_k1_s1037
                                 - 0.363000608350119 z_k1_s1040
                                 - 0.613000936245049 z_k1_s1041
                                 - 0.422002164491664 z_k1_s1042
                                 - 0.48000285370699 z_k1_s1043
                                 - 0.54500284879148 z_k1_s1044
                                 - 0.439002625607051 z_k1_s1045
                                 - 0.182000361937371 z_k1_s1046
                                 - 0.71300121768334 z_k1_s1047
                                 - 0.15699989615514 z_k1_s1050
                                 - 0.712000416525514 z_k1_s1051
                                 - 0.391000554889822 z_k1_s1052
                                 - 0.321000203135806 z_k1_s1053
                                 - 0.664999755911963 z_k1_s1054
                                 - 0.455001091150708 z_k1_s1055
                                 - 0.287000110341024 z_k1_s1056
                                 - 0.427001496849283 z_k1_s1057
                                 - 0.273999435385063 z_k1_s1060
                                 - 0.0689998655261583 z_k1_s1061
                                 - 0.26799965702607 z_k1_s1062
                                 - 0.205999432318832 z_k1_s1063
                                 - 0.228999588694029 z_k1_s1064
                                 - 0.0949999110808522 z_k1_s1065
                                 - 0.240999145412016 z_k1_s1066
                                 - 0.126999885894398 z_k1_s1067
                                 - 0.388999683008717 z_k1_s1070
                                 - 0.535001503100077 z_k1_s1071
                                 - 0.157000164054137 z_k1_s1072
                                 - 0.217000469775871 z_k1_s1073
                                 - 0.503001144381491 z_k1_s1074
                                 - 0.483002020809221 z_k1_s1075
                                 - 0.434002113553713 z_k1_s1076
                                 - 0.217000469775871 z_k1_s1077
                                 - 0.115000060082611 z_k1_s1100
                                 - 0.0920001109861296 z_k1_s1101
                                 - 0.166000371235954 z_k1_s1102
                                 - 0.178999651711776 z_k1_s1103
                                 - 0.192999360839899 z_k1_s1104
                                 - 0.426000137761616 z_k1_s1105
                                 - 0.109000004970441 z_k1_s1106
                                 - 0.292999755042392 z_k1_s1107
                                 - 0.706999728355664 z_k1_s1110
                                 - 1.2680026237073 z_k1_s1111
                                 - 0.758004084031591 z_k1_s1112
                                 - 1.31900054837129 z_k1_s1113
                                 - 0.646998386979242 z_k1_s1114
                                 - 0.694002653230074 z_k1_s1115
                                 - 0.450001056906454 z_k1_s1116
                                 - 0.588000926566256 z_k1_s1117
                                 - 0.504001296101447 z_k1_s1120
                                 - 0.757998823265073 z_k1_s1121
                                 - 0.607999309901159 z_k1_s1122
                                 - 0.209999776709275 z_k1_s1123
                                 - 0.733999633926521 z_k1_s1124
                                 - 0.61199764812493 z_k1_s1125
                                 - 0.566999168115145 z_k1_s1126
                                 - 0.587000019229927 z_k1_s1127
                                 - 0.850997681091714 z_k1_s1130
                                 - 0.77300074873495 z_k1_s1131
                                 - 0.49100084537476 z_k1_s1132
                                 - 0.3699997073351 z_k1_s1133
                                 - 0.496999276787616 z_k1_s1134
                                 - 0.299999961695908 z_k1_s1135
                                 - 0.457998645110528 z_k1_s1136
                                 - 0.590999244574345 z_k1_s1137
                                 - 0.102000067137303 z_k1_s1140
                                 - 0.27900095954995 z_k1_s1141
                                 - 0.0900000225093448 z_k1_s1142
                                 - 0.248001026377924 z_k1_s1143
                                 - 0.253999955990699 z_k1_s1144
                                 - 0.377999688678801 z_k1_s1145
                                 - 0.104000178642078 z_k1_s1146
                                 - 0.10600013404668 z_k1_s1147
                                 - 0.20500076801698 z_k1_s1150
                                 - 0.184000131946426 z_k1_s1151
                                 - 0.441000825069899 z_k1_s1152
                                 - 0.262001125407202 z_k1_s1153
                                 - 0.145000188116556 z_k1_s1154
                                 - 0.249999498165701 z_k1_s1155
                                 - 0.232999650242544 z_k1_s1156
                                 - 0.123000092209843 z_k1_s1157
                                 - 0.422998682207635 z_k1_s1160
                                 - 0.399999164908837 z_k1_s1161
                                 - 0.214999505542333 z_k1_s1162
                                 - 0.752000359253107 z_k1_s1163
                                 - 1.01999639117522 z_k1_s1164
                                 - 0.22899975799713 z_k1_s1165
                                 - 0.50399773999333 z_k1_s1166
                                 - 0.510000222083928 z_k1_s1167
                                 - 0.235999526986485 z_k1_s1170
                                 - 0.229999640879867 z_k1_s1171
                                 - 0.234999545968715 z_k1_s1172
                                 - 0.460998690378524 z_k1_s1173
                                 - 0.575999369416928 z_k1_s1174
                                 - 0.399998131097639 z_k1_s1175
                                 - 0.302999399975008 z_k1_s1176
                                 - 0.42099944966774 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.26506 w_k1_ADM_s000
                                 + 1.78179 w_k1_AXP_s000
                                 + 1.31798 w_k1_DIS_s000
                                 + 1.29415 w_k1_FDX_s000
                                 + 1.02854 w_k1_IBM_s000
                                 + 1.27508 w_k1_INTC_s000
                                 + 1.06906 w_k1_KO_s000 + 1.14166 w_k1_MAT_s000
                                 + 1.22623 w_k1_MSFT_s000 + 1.0397 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.07053 w_k1_ADM_s000
                                 + 0.964807 w_k1_AXP_s000 + 1.376 w_k1_DIS_s000
                                 + 1.25361 w_k1_FDX_s000
                                 + 0.891192 w_k1_IBM_s000
                                 + 1.09633 w_k1_INTC_s000 + 1.0788 w_k1_KO_s000
                                 + 0.791185 w_k1_MAT_s000
                                 + 1.32232 w_k1_MSFT_s000
                                 + 0.950939 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.801124 w_k1_ADM_s000
                                 + 0.742383 w_k1_AXP_s000
                                 + 0.626609 w_k1_DIS_s000
                                 + 0.640856 w_k1_FDX_s000
                                 + 1.31611 w_k1_IBM_s000
                                 + 0.959489 w_k1_INTC_s000
                                 + 0.948322 w_k1_KO_s000
                                 + 1.04497 w_k1_MAT_s000
                                 + 0.831736 w_k1_MSFT_s000
                                 + 0.749354 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.01433 w_k1_ADM_s000
                                 + 0.98968 w_k1_AXP_s000
                                 + 0.944918 w_k1_DIS_s000
                                 + 0.978901 w_k1_FDX_s000
                                 + 1.06642 w_k1_IBM_s000
                                 + 1.14315 w_k1_INTC_s000
                                 + 1.05542 w_k1_KO_s000
                                 + 0.927261 w_k1_MAT_s000
                                 + 1.13038 w_k1_MSFT_s000
                                 + 0.967151 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.27618 w_k1_ADM_s000
                                 + 1.50989 w_k1_AXP_s000
                                 + 1.26107 w_k1_DIS_s000
                                 + 1.14218 w_k1_FDX_s000
                                 + 1.26514 w_k1_IBM_s000
                                 + 1.02676 w_k1_INTC_s000
                                 + 1.18605 w_k1_KO_s000 + 1.34262 w_k1_MAT_s000
                                 + 1.20346 w_k1_MSFT_s000
                                 + 1.18057 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.71224 w_k1_ADM_s000
                                 + 0.952789 w_k1_AXP_s000
                                 + 0.70519 w_k1_DIS_s000
                                 + 0.752182 w_k1_FDX_s000
                                 + 1.04145 w_k1_IBM_s000
                                 + 1.07788 w_k1_INTC_s000
                                 + 0.846419 w_k1_KO_s000
                                 + 1.06184 w_k1_MAT_s000
                                 + 0.832741 w_k1_MSFT_s000
                                 + 0.905078 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.756176 w_k1_ADM_s000
                                 + 0.799768 w_k1_AXP_s000
                                 + 0.783641 w_k1_DIS_s000
                                 + 0.908547 w_k1_FDX_s000
                                 + 0.682516 w_k1_IBM_s000
                                 + 0.875463 w_k1_INTC_s000
                                 + 0.70764 w_k1_KO_s000
                                 + 0.728176 w_k1_MAT_s000
                                 + 0.895519 w_k1_MSFT_s000
                                 + 0.817723 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.03137 w_k1_ADM_s000
                                 + 1.23411 w_k1_AXP_s000
                                 + 1.19239 w_k1_DIS_s000
                                 + 1.04512 w_k1_FDX_s000
                                 + 1.56006 w_k1_IBM_s000
                                 + 1.24826 w_k1_INTC_s000
                                 + 1.30154 w_k1_KO_s000 + 1.19477 w_k1_MAT_s000
                                 + 1.09559 w_k1_MSFT_s000
                                 + 1.05529 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.881893 w_k1_ADM_s001
                                 + 0.84038 w_k1_AXP_s001
                                 + 0.699456 w_k1_DIS_s001
                                 + 0.832117 w_k1_FDX_s001
                                 + 1.00175 w_k1_IBM_s001
                                 + 0.714251 w_k1_INTC_s001
                                 + 0.830127 w_k1_KO_s001
                                 + 1.08543 w_k1_MAT_s001
                                 + 0.790662 w_k1_MSFT_s001
                                 + 0.969207 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.21994 w_k1_ADM_s001
                                 + 1.16134 w_k1_AXP_s001
                                 + 1.29804 w_k1_DIS_s001
                                 + 1.01379 w_k1_FDX_s001
                                 + 1.60354 w_k1_IBM_s001
                                 + 0.690524 w_k1_INTC_s001
                                 + 1.32234 w_k1_KO_s001 + 1.6469 w_k1_MAT_s001
                                 + 1.03683 w_k1_MSFT_s001
                                 + 1.22126 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.05032 w_k1_ADM_s001
                                 + 1.39192 w_k1_AXP_s001
                                 + 0.873422 w_k1_DIS_s001
                                 + 0.971957 w_k1_FDX_s001
                                 + 1.19829 w_k1_IBM_s001
                                 + 0.965137 w_k1_INTC_s001
                                 + 0.94288 w_k1_KO_s001 + 1.64154 w_k1_MAT_s001
                                 + 0.873881 w_k1_MSFT_s001
                                 + 1.11336 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.10182 w_k1_ADM_s001
                                 + 1.0028 w_k1_AXP_s001 + 1.13403 w_k1_DIS_s001
                                 + 1.16557 w_k1_FDX_s001
                                 + 0.777358 w_k1_IBM_s001
                                 + 0.953148 w_k1_INTC_s001
                                 + 0.950475 w_k1_KO_s001
                                 + 0.751829 w_k1_MAT_s001
                                 + 1.0969 w_k1_MSFT_s001 + 1.15959 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.09779 w_k1_ADM_s001
                                 + 1.21342 w_k1_AXP_s001
                                 + 1.29954 w_k1_DIS_s001 + 1.1968 w_k1_FDX_s001
                                 + 1.12702 w_k1_IBM_s001
                                 + 1.22152 w_k1_INTC_s001
                                 + 1.19809 w_k1_KO_s001 + 1.1787 w_k1_MAT_s001
                                 + 1.13935 w_k1_MSFT_s001 + 1.2 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.07726 w_k1_ADM_s001
                                 + 1.82854 w_k1_AXP_s001
                                 + 1.52673 w_k1_DIS_s001
                                 + 1.38919 w_k1_FDX_s001
                                 + 1.16983 w_k1_IBM_s001
                                 + 1.26167 w_k1_INTC_s001
                                 + 1.29117 w_k1_KO_s001 + 1.64137 w_k1_MAT_s001
                                 + 1.3087 w_k1_MSFT_s001 + 1.51275 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.14148 w_k1_ADM_s001
                                 + 1.08533 w_k1_AXP_s001
                                 + 0.94089 w_k1_DIS_s001
                                 + 1.02282 w_k1_FDX_s001
                                 + 0.780813 w_k1_IBM_s001
                                 + 0.875485 w_k1_INTC_s001
                                 + 0.962389 w_k1_KO_s001
                                 + 0.800111 w_k1_MAT_s001
                                 + 0.960592 w_k1_MSFT_s001
                                 + 1.01388 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.668217 w_k1_ADM_s001
                                 + 0.777749 w_k1_AXP_s001
                                 + 0.712478 w_k1_DIS_s001
                                 + 0.753626 w_k1_FDX_s001
                                 + 0.691389 w_k1_IBM_s001
                                 + 0.745835 w_k1_INTC_s001
                                 + 0.761699 w_k1_KO_s001
                                 + 0.876539 w_k1_MAT_s001
                                 + 0.805998 w_k1_MSFT_s001
                                 + 0.898457 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.16737 w_k1_ADM_s002
                                 + 0.771591 w_k1_AXP_s002
                                 + 1.04514 w_k1_DIS_s002 + 0.9998 w_k1_FDX_s002
                                 + 0.728388 w_k1_IBM_s002
                                 + 1.01861 w_k1_INTC_s002
                                 + 0.936904 w_k1_KO_s002
                                 + 0.653557 w_k1_MAT_s002
                                 + 1.09052 w_k1_MSFT_s002
                                 + 0.885812 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.854862 w_k1_ADM_s002
                                 + 0.850353 w_k1_AXP_s002
                                 + 0.639207 w_k1_DIS_s002
                                 + 0.828972 w_k1_FDX_s002
                                 + 0.60401 w_k1_IBM_s002
                                 + 0.921547 w_k1_INTC_s002
                                 + 0.643346 w_k1_KO_s002
                                 + 0.632085 w_k1_MAT_s002
                                 + 0.738262 w_k1_MSFT_s002
                                 + 0.728351 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.828137 w_k1_ADM_s002
                                 + 0.792147 w_k1_AXP_s002
                                 + 0.665708 w_k1_DIS_s002
                                 + 0.741709 w_k1_FDX_s002
                                 + 0.87632 w_k1_IBM_s002
                                 + 1.10782 w_k1_INTC_s002
                                 + 0.847508 w_k1_KO_s002
                                 + 0.793895 w_k1_MAT_s002
                                 + 0.792371 w_k1_MSFT_s002
                                 + 0.816975 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.4745 w_k1_ADM_s002
                                 + 1.25523 w_k1_AXP_s002
                                 + 1.67134 w_k1_DIS_s002
                                 + 1.37311 w_k1_FDX_s002
                                 + 1.03925 w_k1_IBM_s002
                                 + 1.60405 w_k1_INTC_s002
                                 + 1.51121 w_k1_KO_s002 + 1.06367 w_k1_MAT_s002
                                 + 1.46905 w_k1_MSFT_s002 + 1.1052 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.37652 w_k1_ADM_s002
                                 + 0.780202 w_k1_AXP_s002
                                 + 0.944054 w_k1_DIS_s002
                                 + 0.971948 w_k1_FDX_s002
                                 + 0.963285 w_k1_IBM_s002
                                 + 1.15228 w_k1_INTC_s002
                                 + 1.17274 w_k1_KO_s002
                                 + 0.735448 w_k1_MAT_s002
                                 + 1.05095 w_k1_MSFT_s002
                                 + 1.05765 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.33365 w_k1_ADM_s002
                                 + 1.25806 w_k1_AXP_s002
                                 + 1.26339 w_k1_DIS_s002
                                 + 1.18689 w_k1_FDX_s002
                                 + 1.03593 w_k1_IBM_s002
                                 + 1.26735 w_k1_INTC_s002
                                 + 1.15899 w_k1_KO_s002 + 1.14971 w_k1_MAT_s002
                                 + 1.25107 w_k1_MSFT_s002
                                 + 1.15797 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.956794 w_k1_ADM_s002
                                 + 1.14947 w_k1_AXP_s002
                                 + 0.994084 w_k1_DIS_s002
                                 + 1.00976 w_k1_FDX_s002
                                 + 1.30007 w_k1_IBM_s002
                                 + 1.22652 w_k1_INTC_s002 + 1.0701 w_k1_KO_s002
                                 + 1.45523 w_k1_MAT_s002
                                 + 0.960208 w_k1_MSFT_s002
                                 + 1.02054 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.08689 w_k1_ADM_s002
                                 + 1.09132 w_k1_AXP_s002
                                 + 0.743115 w_k1_DIS_s002
                                 + 0.989912 w_k1_FDX_s002
                                 + 0.850681 w_k1_IBM_s002
                                 + 1.10185 w_k1_INTC_s002
                                 + 0.886499 w_k1_KO_s002
                                 + 1.08546 w_k1_MAT_s002
                                 + 1.02446 w_k1_MSFT_s002
                                 + 0.995418 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.786052 w_k1_ADM_s003
                                 + 0.957028 w_k1_AXP_s003
                                 + 0.872981 w_k1_DIS_s003
                                 + 0.868551 w_k1_FDX_s003
                                 + 1.62295 w_k1_IBM_s003
                                 + 1.10276 w_k1_INTC_s003
                                 + 1.25128 w_k1_KO_s003 + 1.8222 w_k1_MAT_s003
                                 + 1.00569 w_k1_MSFT_s003
                                 + 1.11475 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.858273 w_k1_ADM_s003
                                 + 1.09095 w_k1_AXP_s003
                                 + 1.08515 w_k1_DIS_s003
                                 + 1.16104 w_k1_FDX_s003
                                 + 1.16251 w_k1_IBM_s003
                                 + 1.20383 w_k1_INTC_s003 + 1.0317 w_k1_KO_s003
                                 + 1.38388 w_k1_MAT_s003
                                 + 1.14352 w_k1_MSFT_s003
                                 + 1.09484 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.19823 w_k1_ADM_s003
                                 + 0.989564 w_k1_AXP_s003
                                 + 1.40432 w_k1_DIS_s003
                                 + 1.17522 w_k1_FDX_s003
                                 + 0.763253 w_k1_IBM_s003
                                 + 1.2003 w_k1_INTC_s003 + 1.16678 w_k1_KO_s003
                                 + 0.768555 w_k1_MAT_s003
                                 + 1.38741 w_k1_MSFT_s003
                                 + 1.15594 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.801668 w_k1_ADM_s003
                                 + 0.7725 w_k1_AXP_s003
                                 + 0.870081 w_k1_DIS_s003
                                 + 0.822394 w_k1_FDX_s003
                                 + 1.12436 w_k1_IBM_s003
                                 + 0.958313 w_k1_INTC_s003
                                 + 1.07366 w_k1_KO_s003 + 1.17124 w_k1_MAT_s003
                                 + 1.01884 w_k1_MSFT_s003
                                 + 1.01179 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.687721 w_k1_ADM_s003
                                 + 0.715351 w_k1_AXP_s003
                                 + 0.731145 w_k1_DIS_s003
                                 + 0.919624 w_k1_FDX_s003
                                 + 0.593459 w_k1_IBM_s003
                                 + 0.988326 w_k1_INTC_s003
                                 + 0.767368 w_k1_KO_s003
                                 + 0.491032 w_k1_MAT_s003
                                 + 0.858083 w_k1_MSFT_s003
                                 + 0.81476 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.36462 w_k1_ADM_s003
                                 + 1.57589 w_k1_AXP_s003
                                 + 1.68406 w_k1_DIS_s003
                                 + 1.56521 w_k1_FDX_s003
                                 + 1.25532 w_k1_IBM_s003
                                 + 1.2294 w_k1_INTC_s003 + 1.46548 w_k1_KO_s003
                                 + 1.94598 w_k1_MAT_s003
                                 + 1.43728 w_k1_MSFT_s003
                                 + 1.30789 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.837539 w_k1_ADM_s003
                                 + 0.851054 w_k1_AXP_s003
                                 + 0.687568 w_k1_DIS_s003
                                 + 0.851616 w_k1_FDX_s003
                                 + 0.855943 w_k1_IBM_s003
                                 + 0.722916 w_k1_INTC_s003
                                 + 0.796083 w_k1_KO_s003
                                 + 1.22118 w_k1_MAT_s003
                                 + 0.802314 w_k1_MSFT_s003
                                 + 0.945887 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.596029 w_k1_ADM_s003
                                 + 0.816638 w_k1_AXP_s003
                                 + 0.587324 w_k1_DIS_s003
                                 + 0.670717 w_k1_FDX_s003
                                 + 1.10673 w_k1_IBM_s003
                                 + 0.851099 w_k1_INTC_s003
                                 + 0.93177 w_k1_KO_s003 + 1.23286 w_k1_MAT_s003
                                 + 0.736487 w_k1_MSFT_s003
                                 + 0.889453 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.15728 w_k1_ADM_s004
                                 + 1.20771 w_k1_AXP_s004
                                 + 1.05039 w_k1_DIS_s004
                                 + 1.04664 w_k1_FDX_s004
                                 + 1.08478 w_k1_IBM_s004
                                 + 0.856004 w_k1_INTC_s004
                                 + 1.07974 w_k1_KO_s004 + 1.06465 w_k1_MAT_s004
                                 + 1.00523 w_k1_MSFT_s004
                                 + 1.06634 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.968438 w_k1_ADM_s004
                                 + 0.808641 w_k1_AXP_s004
                                 + 1.06896 w_k1_DIS_s004
                                 + 0.894021 w_k1_FDX_s004
                                 + 1.17786 w_k1_IBM_s004
                                 + 0.760473 w_k1_INTC_s004
                                 + 1.09545 w_k1_KO_s004 + 1.12749 w_k1_MAT_s004
                                 + 0.874678 w_k1_MSFT_s004
                                 + 1.0951 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.3229 w_k1_ADM_s004
                                 + 1.50733 w_k1_AXP_s004
                                 + 1.70419 w_k1_DIS_s004
                                 + 1.40566 w_k1_FDX_s004
                                 + 1.07356 w_k1_IBM_s004
                                 + 1.21851 w_k1_INTC_s004
                                 + 1.14299 w_k1_KO_s004 + 1.30504 w_k1_MAT_s004
                                 + 1.28191 w_k1_MSFT_s004
                                 + 1.20229 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.50526 w_k1_ADM_s004
                                 + 1.45659 w_k1_AXP_s004
                                 + 1.62119 w_k1_DIS_s004
                                 + 1.35654 w_k1_FDX_s004
                                 + 1.36803 w_k1_IBM_s004
                                 + 1.06553 w_k1_INTC_s004
                                 + 1.30046 w_k1_KO_s004 + 1.71628 w_k1_MAT_s004
                                 + 1.22341 w_k1_MSFT_s004
                                 + 1.28094 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.44368 w_k1_ADM_s004
                                 + 1.14903 w_k1_AXP_s004
                                 + 1.22174 w_k1_DIS_s004
                                 + 1.17886 w_k1_FDX_s004
                                 + 0.893574 w_k1_IBM_s004
                                 + 1.10169 w_k1_INTC_s004
                                 + 1.04916 w_k1_KO_s004 + 1.17989 w_k1_MAT_s004
                                 + 1.20011 w_k1_MSFT_s004
                                 + 1.11612 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.926597 w_k1_ADM_s004
                                 + 0.674026 w_k1_AXP_s004
                                 + 0.599876 w_k1_DIS_s004
                                 + 0.73725 w_k1_FDX_s004
                                 + 0.71745 w_k1_IBM_s004
                                 + 0.662787 w_k1_INTC_s004
                                 + 0.75535 w_k1_KO_s004
                                 + 0.868648 w_k1_MAT_s004
                                 + 0.784113 w_k1_MSFT_s004
                                 + 0.848619 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.13705 w_k1_ADM_s004
                                 + 0.984378 w_k1_AXP_s004
                                 + 0.880258 w_k1_DIS_s004
                                 + 0.966323 w_k1_FDX_s004
                                 + 0.645449 w_k1_IBM_s004
                                 + 0.805751 w_k1_INTC_s004
                                 + 0.748394 w_k1_KO_s004
                                 + 0.687988 w_k1_MAT_s004
                                 + 0.839204 w_k1_MSFT_s004
                                 + 0.925181 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.945193 w_k1_ADM_s004
                                 + 0.872971 w_k1_AXP_s004
                                 + 0.919176 w_k1_DIS_s004
                                 + 0.866903 w_k1_FDX_s004
                                 + 0.84649 w_k1_IBM_s004
                                 + 0.821519 w_k1_INTC_s004
                                 + 0.876435 w_k1_KO_s004
                                 + 0.726108 w_k1_MAT_s004
                                 + 0.893737 w_k1_MSFT_s004
                                 + 0.955255 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.25073 w_k1_ADM_s005
                                 + 1.26988 w_k1_AXP_s005
                                 + 0.831293 w_k1_DIS_s005
                                 + 0.952609 w_k1_FDX_s005
                                 + 1.09633 w_k1_IBM_s005
                                 + 0.831121 w_k1_INTC_s005
                                 + 0.896574 w_k1_KO_s005
                                 + 1.18793 w_k1_MAT_s005
                                 + 0.87529 w_k1_MSFT_s005
                                 + 1.02084 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.06598 w_k1_ADM_s005
                                 + 0.901847 w_k1_AXP_s005
                                 + 0.958385 w_k1_DIS_s005
                                 + 1.0332 w_k1_FDX_s005 + 1.01851 w_k1_IBM_s005
                                 + 0.932996 w_k1_INTC_s005
                                 + 0.95215 w_k1_KO_s005 + 1.13267 w_k1_MAT_s005
                                 + 1.07111 w_k1_MSFT_s005
                                 + 0.977266 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.81354 w_k1_ADM_s005
                                 + 0.696853 w_k1_AXP_s005
                                 + 0.691394 w_k1_DIS_s005
                                 + 0.685892 w_k1_FDX_s005
                                 + 0.860614 w_k1_IBM_s005
                                 + 0.77387 w_k1_INTC_s005
                                 + 0.759831 w_k1_KO_s005 + 0.5915 w_k1_MAT_s005
                                 + 0.771217 w_k1_MSFT_s005
                                 + 0.819316 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.849087 w_k1_ADM_s005
                                 + 0.821759 w_k1_AXP_s005
                                 + 1.27042 w_k1_DIS_s005
                                 + 0.935774 w_k1_FDX_s005
                                 + 1.10752 w_k1_IBM_s005
                                 + 0.801466 w_k1_INTC_s005
                                 + 1.16775 w_k1_KO_s005
                                 + 0.939845 w_k1_MAT_s005
                                 + 1.01586 w_k1_MSFT_s005
                                 + 1.06125 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.0075 w_k1_ADM_s005
                                 + 1.09318 w_k1_AXP_s005
                                 + 1.54632 w_k1_DIS_s005
                                 + 1.24217 w_k1_FDX_s005
                                 + 0.884345 w_k1_IBM_s005
                                 + 1.15293 w_k1_INTC_s005
                                 + 1.22957 w_k1_KO_s005
                                 + 0.802484 w_k1_MAT_s005
                                 + 1.39536 w_k1_MSFT_s005
                                 + 1.09677 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.36997 w_k1_ADM_s005
                                 + 1.51112 w_k1_AXP_s005
                                 + 1.49579 w_k1_DIS_s005
                                 + 1.30087 w_k1_FDX_s005
                                 + 1.27131 w_k1_IBM_s005
                                 + 1.08648 w_k1_INTC_s005
                                 + 1.28932 w_k1_KO_s005 + 1.31921 w_k1_MAT_s005
                                 + 1.14164 w_k1_MSFT_s005
                                 + 1.18653 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.08944 w_k1_ADM_s005
                                 + 1.12874 w_k1_AXP_s005
                                 + 1.11804 w_k1_DIS_s005
                                 + 1.07348 w_k1_FDX_s005
                                 + 1.54588 w_k1_IBM_s005
                                 + 0.876953 w_k1_INTC_s005
                                 + 1.22394 w_k1_KO_s005 + 1.33066 w_k1_MAT_s005
                                 + 1.19791 w_k1_MSFT_s005
                                 + 1.06198 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.15985 w_k1_ADM_s005
                                 + 0.85124 w_k1_AXP_s005
                                 + 1.00759 w_k1_DIS_s005
                                 + 0.976063 w_k1_FDX_s005
                                 + 0.707768 w_k1_IBM_s005
                                 + 0.852462 w_k1_INTC_s005
                                 + 0.85208 w_k1_KO_s005
                                 + 0.575119 w_k1_MAT_s005
                                 + 1.02786 w_k1_MSFT_s005
                                 + 0.928724 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00963 w_k1_ADM_s006
                                 + 1.39825 w_k1_AXP_s006
                                 + 1.25386 w_k1_DIS_s006
                                 + 1.09802 w_k1_FDX_s006
                                 + 1.55746 w_k1_IBM_s006
                                 + 0.933286 w_k1_INTC_s006
                                 + 1.15725 w_k1_KO_s006 + 1.48598 w_k1_MAT_s006
                                 + 1.08254 w_k1_MSFT_s006
                                 + 1.06043 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.957137 w_k1_ADM_s006
                                 + 1.64157 w_k1_AXP_s006
                                 + 1.99875 w_k1_DIS_s006
                                 + 1.57675 w_k1_FDX_s006
                                 + 1.19697 w_k1_IBM_s006
                                 + 1.59193 w_k1_INTC_s006
                                 + 1.43597 w_k1_KO_s006 + 1.23135 w_k1_MAT_s006
                                 + 1.60085 w_k1_MSFT_s006
                                 + 1.29772 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.922582 w_k1_ADM_s006
                                 + 0.697925 w_k1_AXP_s006
                                 + 0.593987 w_k1_DIS_s006
                                 + 0.773943 w_k1_FDX_s006
                                 + 0.719066 w_k1_IBM_s006
                                 + 0.81809 w_k1_INTC_s006
                                 + 0.860721 w_k1_KO_s006
                                 + 0.695205 w_k1_MAT_s006
                                 + 0.821774 w_k1_MSFT_s006
                                 + 0.81557 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.05373 w_k1_ADM_s006
                                 + 1.13139 w_k1_AXP_s006
                                 + 0.923281 w_k1_DIS_s006
                                 + 0.997324 w_k1_FDX_s006
                                 + 1.01529 w_k1_IBM_s006
                                 + 0.974455 w_k1_INTC_s006
                                 + 0.921051 w_k1_KO_s006
                                 + 0.950576 w_k1_MAT_s006
                                 + 0.895266 w_k1_MSFT_s006
                                 + 0.977643 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.844439 w_k1_ADM_s006
                                 + 0.765044 w_k1_AXP_s006
                                 + 1.16848 w_k1_DIS_s006
                                 + 0.987294 w_k1_FDX_s006
                                 + 0.747712 w_k1_IBM_s006
                                 + 0.898323 w_k1_INTC_s006
                                 + 0.899448 w_k1_KO_s006
                                 + 0.707026 w_k1_MAT_s006
                                 + 1.12048 w_k1_MSFT_s006
                                 + 1.02797 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.01036 w_k1_ADM_s006
                                 + 0.864191 w_k1_AXP_s006
                                 + 0.768968 w_k1_DIS_s006
                                 + 0.805612 w_k1_FDX_s006
                                 + 1.07299 w_k1_IBM_s006
                                 + 0.985979 w_k1_INTC_s006
                                 + 0.987469 w_k1_KO_s006
                                 + 0.970632 w_k1_MAT_s006
                                 + 0.918319 w_k1_MSFT_s006
                                 + 0.981053 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.613222 w_k1_ADM_s006
                                 + 1.07461 w_k1_AXP_s006
                                 + 0.763008 w_k1_DIS_s006
                                 + 0.861432 w_k1_FDX_s006
                                 + 1.46334 w_k1_IBM_s006
                                 + 1.04844 w_k1_INTC_s006
                                 + 1.02831 w_k1_KO_s006 + 1.4554 w_k1_MAT_s006
                                 + 0.8452 w_k1_MSFT_s006 + 1.06817 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.35276 w_k1_ADM_s006
                                 + 1.18005 w_k1_AXP_s006
                                 + 1.30005 w_k1_DIS_s006
                                 + 1.19475 w_k1_FDX_s006
                                 + 0.960442 w_k1_IBM_s006
                                 + 1.12696 w_k1_INTC_s006
                                 + 1.14161 w_k1_KO_s006
                                 + 0.941914 w_k1_MAT_s006
                                 + 1.12216 w_k1_MSFT_s006
                                 + 1.19708 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.843945 w_k1_ADM_s007
                                 + 0.881758 w_k1_AXP_s007
                                 + 0.719824 w_k1_DIS_s007
                                 + 0.873259 w_k1_FDX_s007
                                 + 0.75784 w_k1_IBM_s007
                                 + 0.89751 w_k1_INTC_s007
                                 + 0.716507 w_k1_KO_s007
                                 + 0.60539 w_k1_MAT_s007
                                 + 0.771707 w_k1_MSFT_s007
                                 + 0.788905 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.845065 w_k1_ADM_s007
                                 + 0.489603 w_k1_AXP_s007
                                 + 0.635084 w_k1_DIS_s007
                                 + 0.645791 w_k1_FDX_s007
                                 + 0.766345 w_k1_IBM_s007
                                 + 0.843006 w_k1_INTC_s007
                                 + 0.697537 w_k1_KO_s007
                                 + 0.458758 w_k1_MAT_s007
                                 + 0.768315 w_k1_MSFT_s007
                                 + 0.786528 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.01719 w_k1_ADM_s007
                                 + 0.80756 w_k1_AXP_s007
                                 + 1.03748 w_k1_DIS_s007
                                 + 0.91021 w_k1_FDX_s007
                                 + 1.00131 w_k1_IBM_s007
                                 + 1.05067 w_k1_INTC_s007
                                 + 1.09569 w_k1_KO_s007
                                 + 0.792752 w_k1_MAT_s007
                                 + 1.00465 w_k1_MSFT_s007 + 1.0955 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.50098 w_k1_ADM_s007
                                 + 1.73019 w_k1_AXP_s007 + 1.4821 w_k1_DIS_s007
                                 + 1.4251 w_k1_FDX_s007 + 1.2674 w_k1_IBM_s007
                                 + 1.4173 w_k1_INTC_s007 + 1.24117 w_k1_KO_s007
                                 + 1.0404 w_k1_MAT_s007
                                 + 1.11777 w_k1_MSFT_s007 + 1.0943 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.1407 w_k1_ADM_s007
                                 + 0.818 w_k1_AXP_s007 + 0.71417 w_k1_DIS_s007
                                 + 0.890258 w_k1_FDX_s007
                                 + 0.827968 w_k1_IBM_s007
                                 + 1.07663 w_k1_INTC_s007
                                 + 0.806534 w_k1_KO_s007
                                 + 0.525695 w_k1_MAT_s007
                                 + 0.755909 w_k1_MSFT_s007
                                 + 0.965733 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.09457 w_k1_ADM_s007
                                 + 1.22387 w_k1_AXP_s007
                                 + 1.09102 w_k1_DIS_s007
                                 + 1.15609 w_k1_FDX_s007
                                 + 1.01185 w_k1_IBM_s007
                                 + 1.01858 w_k1_INTC_s007
                                 + 0.938486 w_k1_KO_s007
                                 + 0.815397 w_k1_MAT_s007
                                 + 0.946131 w_k1_MSFT_s007
                                 + 1.09065 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.07444 w_k1_ADM_s007
                                 + 1.02667 w_k1_AXP_s007
                                 + 0.915605 w_k1_DIS_s007
                                 + 0.899419 w_k1_FDX_s007
                                 + 1.2342 w_k1_IBM_s007
                                 + 0.820199 w_k1_INTC_s007
                                 + 0.871162 w_k1_KO_s007
                                 + 1.04967 w_k1_MAT_s007
                                 + 0.878627 w_k1_MSFT_s007
                                 + 1.05192 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.2284 w_k1_ADM_s007
                                 + 1.3399 w_k1_AXP_s007 + 1.7063 w_k1_DIS_s007
                                 + 1.22707 w_k1_FDX_s007
                                 + 1.48211 w_k1_IBM_s007
                                 + 1.11705 w_k1_INTC_s007
                                 + 1.49861 w_k1_KO_s007 + 1.22693 w_k1_MAT_s007
                                 + 1.29978 w_k1_MSFT_s007
                                 + 1.25454 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.29416 w_k1_ADM_s010
                                 + 1.18796 w_k1_AXP_s010
                                 + 1.30486 w_k1_DIS_s010
                                 + 1.12993 w_k1_FDX_s010
                                 + 0.816942 w_k1_IBM_s010
                                 + 0.961805 w_k1_INTC_s010
                                 + 1.10512 w_k1_KO_s010
                                 + 0.657839 w_k1_MAT_s010
                                 + 1.19659 w_k1_MSFT_s010
                                 + 1.18424 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.13776 w_k1_ADM_s010
                                 + 1.25639 w_k1_AXP_s010
                                 + 1.42419 w_k1_DIS_s010 + 1.1529 w_k1_FDX_s010
                                 + 1.37163 w_k1_IBM_s010
                                 + 0.712701 w_k1_INTC_s010
                                 + 1.1713 w_k1_KO_s010 + 1.78427 w_k1_MAT_s010
                                 + 1.09147 w_k1_MSFT_s010
                                 + 1.10012 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.11425 w_k1_ADM_s010
                                 + 0.895098 w_k1_AXP_s010
                                 + 0.852136 w_k1_DIS_s010
                                 + 1.01321 w_k1_FDX_s010
                                 + 0.706471 w_k1_IBM_s010
                                 + 0.757014 w_k1_INTC_s010
                                 + 0.773235 w_k1_KO_s010
                                 + 0.821649 w_k1_MAT_s010
                                 + 0.949178 w_k1_MSFT_s010
                                 + 0.836027 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.15813 w_k1_ADM_s010
                                 + 0.891006 w_k1_AXP_s010
                                 + 1.02219 w_k1_DIS_s010
                                 + 0.952212 w_k1_FDX_s010
                                 + 1.28476 w_k1_IBM_s010
                                 + 0.919112 w_k1_INTC_s010
                                 + 1.04922 w_k1_KO_s010 + 1.14798 w_k1_MAT_s010
                                 + 0.950815 w_k1_MSFT_s010
                                 + 0.997542 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.15159 w_k1_ADM_s010
                                 + 0.951032 w_k1_AXP_s010
                                 + 0.902642 w_k1_DIS_s010
                                 + 0.993651 w_k1_FDX_s010
                                 + 1.04116 w_k1_IBM_s010
                                 + 1.01975 w_k1_INTC_s010
                                 + 0.940293 w_k1_KO_s010
                                 + 0.791024 w_k1_MAT_s010
                                 + 1.07516 w_k1_MSFT_s010
                                 + 0.931436 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.64936 w_k1_ADM_s010
                                 + 1.26383 w_k1_AXP_s010 + 1.5997 w_k1_DIS_s010
                                 + 1.40547 w_k1_FDX_s010
                                 + 1.12005 w_k1_IBM_s010
                                 + 1.14522 w_k1_INTC_s010
                                 + 1.20841 w_k1_KO_s010 + 0.91774 w_k1_MAT_s010
                                 + 1.22107 w_k1_MSFT_s010
                                 + 1.08545 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.12347 w_k1_ADM_s010
                                 + 1.11092 w_k1_AXP_s010
                                 + 1.06467 w_k1_DIS_s010
                                 + 1.14342 w_k1_FDX_s010
                                 + 0.87845 w_k1_IBM_s010
                                 + 1.05605 w_k1_INTC_s010
                                 + 0.830962 w_k1_KO_s010
                                 + 1.04527 w_k1_MAT_s010
                                 + 0.953425 w_k1_MSFT_s010
                                 + 0.802755 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.04678 w_k1_ADM_s010
                                 + 0.76111 w_k1_AXP_s010
                                 + 0.688771 w_k1_DIS_s010
                                 + 0.826878 w_k1_FDX_s010
                                 + 0.971965 w_k1_IBM_s010
                                 + 0.780915 w_k1_INTC_s010
                                 + 0.734609 w_k1_KO_s010
                                 + 0.89803 w_k1_MAT_s010
                                 + 0.816912 w_k1_MSFT_s010
                                 + 0.940205 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.01915 w_k1_ADM_s011
                                 + 0.861672 w_k1_AXP_s011
                                 + 0.921735 w_k1_DIS_s011
                                 + 1.04955 w_k1_FDX_s011
                                 + 0.579995 w_k1_IBM_s011
                                 + 0.990537 w_k1_INTC_s011
                                 + 0.739233 w_k1_KO_s011
                                 + 0.483782 w_k1_MAT_s011
                                 + 1.01462 w_k1_MSFT_s011
                                 + 0.810193 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.949628 w_k1_ADM_s011
                                 + 1.44888 w_k1_AXP_s011
                                 + 1.08824 w_k1_DIS_s011
                                 + 1.05297 w_k1_FDX_s011
                                 + 1.47051 w_k1_IBM_s011
                                 + 1.17912 w_k1_INTC_s011
                                 + 1.19505 w_k1_KO_s011 + 1.79983 w_k1_MAT_s011
                                 + 1.06778 w_k1_MSFT_s011
                                 + 1.14404 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.825514 w_k1_ADM_s011
                                 + 0.920274 w_k1_AXP_s011
                                 + 0.907717 w_k1_DIS_s011
                                 + 1.00592 w_k1_FDX_s011
                                 + 0.744995 w_k1_IBM_s011
                                 + 1.34954 w_k1_INTC_s011
                                 + 0.857684 w_k1_KO_s011
                                 + 0.910851 w_k1_MAT_s011
                                 + 1.04621 w_k1_MSFT_s011
                                 + 0.915255 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.894176 w_k1_ADM_s011
                                 + 0.603095 w_k1_AXP_s011
                                 + 0.711073 w_k1_DIS_s011
                                 + 0.746767 w_k1_FDX_s011
                                 + 0.817773 w_k1_IBM_s011
                                 + 0.84445 w_k1_INTC_s011
                                 + 0.799503 w_k1_KO_s011
                                 + 0.885287 w_k1_MAT_s011
                                 + 0.871346 w_k1_MSFT_s011
                                 + 0.842721 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.28386 w_k1_ADM_s011
                                 + 1.30661 w_k1_AXP_s011
                                 + 1.25176 w_k1_DIS_s011
                                 + 1.24934 w_k1_FDX_s011
                                 + 0.936225 w_k1_IBM_s011
                                 + 1.16957 w_k1_INTC_s011
                                 + 0.97605 w_k1_KO_s011 + 0.93133 w_k1_MAT_s011
                                 + 1.16098 w_k1_MSFT_s011
                                 + 1.06846 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.12918 w_k1_ADM_s011
                                 + 1.33407 w_k1_AXP_s011
                                 + 1.14075 w_k1_DIS_s011
                                 + 1.09421 w_k1_FDX_s011 + 1.0473 w_k1_IBM_s011
                                 + 1.03686 w_k1_INTC_s011
                                 + 1.05707 w_k1_KO_s011 + 1.24797 w_k1_MAT_s011
                                 + 1.09693 w_k1_MSFT_s011
                                 + 0.963562 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.969704 w_k1_ADM_s011
                                 + 0.900164 w_k1_AXP_s011
                                 + 0.897608 w_k1_DIS_s011
                                 + 0.923306 w_k1_FDX_s011
                                 + 1.14854 w_k1_IBM_s011
                                 + 0.854473 w_k1_INTC_s011
                                 + 1.0415 w_k1_KO_s011 + 0.981717 w_k1_MAT_s011
                                 + 0.9526 w_k1_MSFT_s011
                                 + 0.999319 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.722831 w_k1_ADM_s011
                                 + 0.880403 w_k1_AXP_s011
                                 + 0.569068 w_k1_DIS_s011
                                 + 0.784043 w_k1_FDX_s011
                                 + 0.754139 w_k1_IBM_s011
                                 + 0.787205 w_k1_INTC_s011
                                 + 0.729714 w_k1_KO_s011
                                 + 0.935653 w_k1_MAT_s011
                                 + 0.693688 w_k1_MSFT_s011
                                 + 0.787782 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.69431 w_k1_ADM_s012
                                 + 0.710997 w_k1_AXP_s012
                                 + 0.690731 w_k1_DIS_s012
                                 + 0.810686 w_k1_FDX_s012
                                 + 0.889275 w_k1_IBM_s012
                                 + 0.848151 w_k1_INTC_s012
                                 + 0.811039 w_k1_KO_s012
                                 + 0.907789 w_k1_MAT_s012
                                 + 0.843299 w_k1_MSFT_s012
                                 + 1.10288 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.18888 w_k1_ADM_s012
                                 + 1.46443 w_k1_AXP_s012
                                 + 1.24791 w_k1_DIS_s012
                                 + 1.28262 w_k1_FDX_s012
                                 + 1.09779 w_k1_IBM_s012
                                 + 1.33155 w_k1_INTC_s012
                                 + 1.23111 w_k1_KO_s012 + 1.43632 w_k1_MAT_s012
                                 + 1.18419 w_k1_MSFT_s012
                                 + 1.31968 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.09014 w_k1_ADM_s012
                                 + 1.09425 w_k1_AXP_s012
                                 + 1.33712 w_k1_DIS_s012
                                 + 1.19389 w_k1_FDX_s012
                                 + 0.864615 w_k1_IBM_s012
                                 + 1.1247 w_k1_INTC_s012 + 1.10113 w_k1_KO_s012
                                 + 1.11316 w_k1_MAT_s012
                                 + 1.1059 w_k1_MSFT_s012 + 1.51234 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.07788 w_k1_ADM_s012
                                 + 1.16175 w_k1_AXP_s012
                                 + 1.45827 w_k1_DIS_s012
                                 + 1.19712 w_k1_FDX_s012
                                 + 1.41403 w_k1_IBM_s012
                                 + 0.989528 w_k1_INTC_s012
                                 + 1.37936 w_k1_KO_s012 + 1.69095 w_k1_MAT_s012
                                 + 1.16701 w_k1_MSFT_s012
                                 + 1.58234 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.12701 w_k1_ADM_s012
                                 + 1.03003 w_k1_AXP_s012
                                 + 0.926083 w_k1_DIS_s012
                                 + 0.797653 w_k1_FDX_s012
                                 + 1.34238 w_k1_IBM_s012
                                 + 0.784971 w_k1_INTC_s012
                                 + 1.09284 w_k1_KO_s012 + 1.24199 w_k1_MAT_s012
                                 + 0.890147 w_k1_MSFT_s012
                                 + 1.0968 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.941418 w_k1_ADM_s012
                                 + 0.921134 w_k1_AXP_s012
                                 + 0.667122 w_k1_DIS_s012
                                 + 0.880675 w_k1_FDX_s012
                                 + 0.632291 w_k1_IBM_s012
                                 + 0.864252 w_k1_INTC_s012
                                 + 0.662649 w_k1_KO_s012
                                 + 0.958518 w_k1_MAT_s012
                                 + 0.778862 w_k1_MSFT_s012
                                 + 1.06968 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.776096 w_k1_ADM_s012
                                 + 1.36051 w_k1_AXP_s012
                                 + 1.53292 w_k1_DIS_s012
                                 + 1.11191 w_k1_FDX_s012
                                 + 1.47666 w_k1_IBM_s012
                                 + 1.18253 w_k1_INTC_s012
                                 + 1.29878 w_k1_KO_s012 + 2.07745 w_k1_MAT_s012
                                 + 1.08519 w_k1_MSFT_s012 + 1.3943 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.990344 w_k1_ADM_s012
                                 + 1.13157 w_k1_AXP_s012
                                 + 1.01482 w_k1_DIS_s012
                                 + 0.909986 w_k1_FDX_s012
                                 + 0.955616 w_k1_IBM_s012
                                 + 0.856365 w_k1_INTC_s012
                                 + 0.914786 w_k1_KO_s012
                                 + 1.23548 w_k1_MAT_s012
                                 + 0.920241 w_k1_MSFT_s012
                                 + 1.20586 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.977088 w_k1_ADM_s013
                                 + 0.971104 w_k1_AXP_s013
                                 + 0.92529 w_k1_DIS_s013
                                 + 0.983529 w_k1_FDX_s013
                                 + 0.766118 w_k1_IBM_s013
                                 + 0.945008 w_k1_INTC_s013
                                 + 0.901721 w_k1_KO_s013
                                 + 0.750073 w_k1_MAT_s013
                                 + 0.947546 w_k1_MSFT_s013
                                 + 1.02496 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.840363 w_k1_ADM_s013
                                 + 1.36164 w_k1_AXP_s013
                                 + 1.24042 w_k1_DIS_s013
                                 + 1.08152 w_k1_FDX_s013
                                 + 1.14717 w_k1_IBM_s013
                                 + 1.36776 w_k1_INTC_s013
                                 + 1.26834 w_k1_KO_s013 + 1.07312 w_k1_MAT_s013
                                 + 1.13016 w_k1_MSFT_s013
                                 + 1.09391 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.688046 w_k1_ADM_s013
                                 + 0.878805 w_k1_AXP_s013
                                 + 0.897039 w_k1_DIS_s013
                                 + 0.741495 w_k1_FDX_s013
                                 + 1.23505 w_k1_IBM_s013
                                 + 0.661917 w_k1_INTC_s013
                                 + 0.977417 w_k1_KO_s013
                                 + 1.16475 w_k1_MAT_s013
                                 + 0.722316 w_k1_MSFT_s013
                                 + 1.01943 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.17408 w_k1_ADM_s013
                                 + 1.09117 w_k1_AXP_s013
                                 + 1.53522 w_k1_DIS_s013
                                 + 1.21484 w_k1_FDX_s013
                                 + 1.29138 w_k1_IBM_s013
                                 + 1.06817 w_k1_INTC_s013
                                 + 1.22934 w_k1_KO_s013 + 1.11912 w_k1_MAT_s013
                                 + 1.22839 w_k1_MSFT_s013
                                 + 1.27618 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.24641 w_k1_ADM_s013
                                 + 1.62255 w_k1_AXP_s013
                                 + 1.36085 w_k1_DIS_s013
                                 + 1.29223 w_k1_FDX_s013
                                 + 1.15729 w_k1_IBM_s013
                                 + 1.28469 w_k1_INTC_s013
                                 + 1.26226 w_k1_KO_s013 + 1.43908 w_k1_MAT_s013
                                 + 1.27439 w_k1_MSFT_s013
                                 + 1.37204 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.993074 w_k1_ADM_s013
                                 + 1.16707 w_k1_AXP_s013
                                 + 1.04431 w_k1_DIS_s013
                                 + 1.16448 w_k1_FDX_s013
                                 + 0.768371 w_k1_IBM_s013
                                 + 1.05387 w_k1_INTC_s013
                                 + 0.863488 w_k1_KO_s013
                                 + 1.04682 w_k1_MAT_s013
                                 + 1.00498 w_k1_MSFT_s013
                                 + 1.05554 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.801465 w_k1_ADM_s013
                                 + 0.744524 w_k1_AXP_s013
                                 + 0.801732 w_k1_DIS_s013
                                 + 0.870094 w_k1_FDX_s013
                                 + 0.581618 w_k1_IBM_s013
                                 + 0.850449 w_k1_INTC_s013
                                 + 0.70796 w_k1_KO_s013
                                 + 0.695569 w_k1_MAT_s013
                                 + 0.859413 w_k1_MSFT_s013
                                 + 0.796782 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.662988 w_k1_ADM_s013
                                 + 0.792983 w_k1_AXP_s013
                                 + 0.825822 w_k1_DIS_s013
                                 + 0.854123 w_k1_FDX_s013
                                 + 0.87017 w_k1_IBM_s013
                                 + 0.878982 w_k1_INTC_s013
                                 + 0.910867 w_k1_KO_s013
                                 + 0.93094 w_k1_MAT_s013
                                 + 0.833277 w_k1_MSFT_s013
                                 + 0.910929 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.07565 w_k1_ADM_s014
                                 + 1.09458 w_k1_AXP_s014
                                 + 1.04515 w_k1_DIS_s014
                                 + 1.21353 w_k1_FDX_s014
                                 + 0.829677 w_k1_IBM_s014
                                 + 1.10693 w_k1_INTC_s014
                                 + 0.932292 w_k1_KO_s014
                                 + 0.696282 w_k1_MAT_s014
                                 + 1.12805 w_k1_MSFT_s014
                                 + 1.17317 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.32352 w_k1_ADM_s014
                                 + 1.4889 w_k1_AXP_s014 + 1.46688 w_k1_DIS_s014
                                 + 1.33132 w_k1_FDX_s014
                                 + 0.966194 w_k1_IBM_s014
                                 + 1.32654 w_k1_INTC_s014
                                 + 1.19347 w_k1_KO_s014
                                 + 0.931197 w_k1_MAT_s014
                                 + 1.30062 w_k1_MSFT_s014
                                 + 1.29151 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.66946 w_k1_ADM_s014
                                 + 1.52036 w_k1_AXP_s014 + 1.4176 w_k1_DIS_s014
                                 + 1.24757 w_k1_FDX_s014
                                 + 1.56258 w_k1_IBM_s014 + 1.054 w_k1_INTC_s014
                                 + 1.43234 w_k1_KO_s014 + 1.46086 w_k1_MAT_s014
                                 + 1.21393 w_k1_MSFT_s014
                                 + 1.25261 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.06998 w_k1_ADM_s014
                                 + 0.962554 w_k1_AXP_s014
                                 + 0.895019 w_k1_DIS_s014
                                 + 0.947244 w_k1_FDX_s014
                                 + 1.18103 w_k1_IBM_s014
                                 + 0.976346 w_k1_INTC_s014
                                 + 1.15361 w_k1_KO_s014 + 1.00434 w_k1_MAT_s014
                                 + 1.04634 w_k1_MSFT_s014
                                 + 1.07303 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.01385 w_k1_ADM_s014
                                 + 1.02199 w_k1_AXP_s014
                                 + 0.979752 w_k1_DIS_s014
                                 + 0.908357 w_k1_FDX_s014
                                 + 1.6489 w_k1_IBM_s014
                                 + 1.10062 w_k1_INTC_s014
                                 + 1.20144 w_k1_KO_s014 + 1.10913 w_k1_MAT_s014
                                 + 1.02018 w_k1_MSFT_s014
                                 + 1.26123 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.905051 w_k1_ADM_s014
                                 + 1.05264 w_k1_AXP_s014
                                 + 0.738551 w_k1_DIS_s014
                                 + 0.81376 w_k1_FDX_s014 + 1.124 w_k1_IBM_s014
                                 + 0.897287 w_k1_INTC_s014
                                 + 1.01188 w_k1_KO_s014 + 1.1835 w_k1_MAT_s014
                                 + 0.844186 w_k1_MSFT_s014
                                 + 0.893216 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.40371 w_k1_ADM_s014
                                 + 1.04023 w_k1_AXP_s014
                                 + 1.30896 w_k1_DIS_s014
                                 + 1.11975 w_k1_FDX_s014
                                 + 1.09711 w_k1_IBM_s014
                                 + 1.04434 w_k1_INTC_s014
                                 + 1.30686 w_k1_KO_s014
                                 + 0.931419 w_k1_MAT_s014
                                 + 1.11905 w_k1_MSFT_s014
                                 + 1.15461 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.19268 w_k1_ADM_s014
                                 + 0.873605 w_k1_AXP_s014
                                 + 0.671997 w_k1_DIS_s014
                                 + 0.802898 w_k1_FDX_s014
                                 + 0.753044 w_k1_IBM_s014
                                 + 0.813292 w_k1_INTC_s014
                                 + 0.797196 w_k1_KO_s014
                                 + 0.657491 w_k1_MAT_s014
                                 + 0.927386 w_k1_MSFT_s014
                                 + 0.896901 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.986785 w_k1_ADM_s015
                                 + 1.65948 w_k1_AXP_s015
                                 + 1.77344 w_k1_DIS_s015 + 1.3129 w_k1_FDX_s015
                                 + 1.63628 w_k1_IBM_s015
                                 + 1.44271 w_k1_INTC_s015
                                 + 1.40332 w_k1_KO_s015 + 2.28166 w_k1_MAT_s015
                                 + 1.43354 w_k1_MSFT_s015
                                 + 1.30304 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.825272 w_k1_ADM_s015
                                 + 0.980219 w_k1_AXP_s015
                                 + 1.11665 w_k1_DIS_s015
                                 + 0.980562 w_k1_FDX_s015
                                 + 1.54565 w_k1_IBM_s015
                                 + 1.16517 w_k1_INTC_s015
                                 + 1.23343 w_k1_KO_s015 + 1.28228 w_k1_MAT_s015
                                 + 1.25792 w_k1_MSFT_s015
                                 + 1.21997 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.01212 w_k1_ADM_s015
                                 + 1.04894 w_k1_AXP_s015
                                 + 0.994163 w_k1_DIS_s015
                                 + 1.05844 w_k1_FDX_s015 + 1.125 w_k1_IBM_s015
                                 + 1.20897 w_k1_INTC_s015
                                 + 0.967873 w_k1_KO_s015
                                 + 0.783542 w_k1_MAT_s015
                                 + 1.15226 w_k1_MSFT_s015
                                 + 1.11753 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.910379 w_k1_ADM_s015
                                 + 0.874904 w_k1_AXP_s015
                                 + 1.14422 w_k1_DIS_s015
                                 + 1.09792 w_k1_FDX_s015
                                 + 0.887913 w_k1_IBM_s015
                                 + 1.22915 w_k1_INTC_s015
                                 + 0.908534 w_k1_KO_s015
                                 + 0.786657 w_k1_MAT_s015
                                 + 1.24363 w_k1_MSFT_s015
                                 + 1.10427 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.714205 w_k1_ADM_s015
                                 + 0.700834 w_k1_AXP_s015
                                 + 0.650119 w_k1_DIS_s015
                                 + 0.788975 w_k1_FDX_s015
                                 + 0.982552 w_k1_IBM_s015
                                 + 0.935064 w_k1_INTC_s015
                                 + 0.84151 w_k1_KO_s015
                                 + 0.889036 w_k1_MAT_s015
                                 + 0.965812 w_k1_MSFT_s015
                                 + 0.916065 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.682557 w_k1_ADM_s015
                                 + 0.699366 w_k1_AXP_s015
                                 + 0.921696 w_k1_DIS_s015
                                 + 0.906215 w_k1_FDX_s015
                                 + 0.690206 w_k1_IBM_s015
                                 + 0.983889 w_k1_INTC_s015
                                 + 0.722392 w_k1_KO_s015
                                 + 0.628491 w_k1_MAT_s015
                                 + 1.01582 w_k1_MSFT_s015
                                 + 0.915651 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.903223 w_k1_ADM_s015
                                 + 0.632017 w_k1_AXP_s015
                                 + 0.878831 w_k1_DIS_s015
                                 + 0.779094 w_k1_FDX_s015
                                 + 1.21733 w_k1_IBM_s015
                                 + 0.883688 w_k1_INTC_s015
                                 + 1.00938 w_k1_KO_s015
                                 + 0.789514 w_k1_MAT_s015
                                 + 1.02384 w_k1_MSFT_s015
                                 + 1.00361 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.10632 w_k1_ADM_s015
                                 + 1.2009 w_k1_AXP_s015 + 1.10623 w_k1_DIS_s015
                                 + 1.34341 w_k1_FDX_s015
                                 + 0.674341 w_k1_IBM_s015
                                 + 1.37995 w_k1_INTC_s015
                                 + 0.846615 w_k1_KO_s015
                                 + 0.845993 w_k1_MAT_s015
                                 + 1.41952 w_k1_MSFT_s015
                                 + 1.16538 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.41445 w_k1_ADM_s016
                                 + 1.59061 w_k1_AXP_s016
                                 + 1.37654 w_k1_DIS_s016
                                 + 1.28634 w_k1_FDX_s016
                                 + 1.15756 w_k1_IBM_s016
                                 + 1.04498 w_k1_INTC_s016 + 1.1411 w_k1_KO_s016
                                 + 1.35667 w_k1_MAT_s016
                                 + 1.06118 w_k1_MSFT_s016
                                 + 1.35664 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.907584 w_k1_ADM_s016
                                 + 0.771583 w_k1_AXP_s016
                                 + 0.66881 w_k1_DIS_s016
                                 + 0.777764 w_k1_FDX_s016
                                 + 0.69655 w_k1_IBM_s016
                                 + 0.845904 w_k1_INTC_s016
                                 + 0.675617 w_k1_KO_s016
                                 + 0.576518 w_k1_MAT_s016
                                 + 0.828966 w_k1_MSFT_s016
                                 + 0.88221 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.693727 w_k1_ADM_s016
                                 + 0.679058 w_k1_AXP_s016
                                 + 0.791838 w_k1_DIS_s016
                                 + 0.773382 w_k1_FDX_s016
                                 + 1.12345 w_k1_IBM_s016
                                 + 0.76777 w_k1_INTC_s016
                                 + 0.921443 w_k1_KO_s016
                                 + 1.02242 w_k1_MAT_s016
                                 + 0.763502 w_k1_MSFT_s016
                                 + 0.948597 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.20047 w_k1_ADM_s016
                                 + 1.03715 w_k1_AXP_s016
                                 + 1.45562 w_k1_DIS_s016
                                 + 1.15684 w_k1_FDX_s016
                                 + 1.19219 w_k1_IBM_s016
                                 + 1.08529 w_k1_INTC_s016
                                 + 1.19622 w_k1_KO_s016
                                 + 0.999351 w_k1_MAT_s016
                                 + 1.10825 w_k1_MSFT_s016
                                 + 1.09951 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.20884 w_k1_ADM_s016
                                 + 1.07706 w_k1_AXP_s016
                                 + 1.00026 w_k1_DIS_s016
                                 + 1.10903 w_k1_FDX_s016
                                 + 0.769434 w_k1_IBM_s016
                                 + 1.0192 w_k1_INTC_s016
                                 + 0.891035 w_k1_KO_s016
                                 + 0.926139 w_k1_MAT_s016
                                 + 0.936962 w_k1_MSFT_s016
                                 + 1.05199 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.845539 w_k1_ADM_s016
                                 + 1.02284 w_k1_AXP_s016
                                 + 0.79126 w_k1_DIS_s016
                                 + 0.853438 w_k1_FDX_s016
                                 + 1.24604 w_k1_IBM_s016
                                 + 0.943433 w_k1_INTC_s016
                                 + 0.870965 w_k1_KO_s016 + 1.4161 w_k1_MAT_s016
                                 + 0.89252 w_k1_MSFT_s016
                                 + 0.96674 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.12217 w_k1_ADM_s016
                                 + 0.837943 w_k1_AXP_s016
                                 + 0.922721 w_k1_DIS_s016
                                 + 0.944162 w_k1_FDX_s016
                                 + 0.849349 w_k1_IBM_s016
                                 + 0.809261 w_k1_INTC_s016
                                 + 0.989645 w_k1_KO_s016
                                 + 0.997443 w_k1_MAT_s016
                                 + 0.951564 w_k1_MSFT_s016
                                 + 1.09311 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.939756 w_k1_ADM_s016
                                 + 1.10808 w_k1_AXP_s016
                                 + 1.02745 w_k1_DIS_s016
                                 + 1.01781 w_k1_FDX_s016
                                 + 1.13215 w_k1_IBM_s016
                                 + 0.950866 w_k1_INTC_s016
                                 + 1.01212 w_k1_KO_s016 + 1.32848 w_k1_MAT_s016
                                 + 0.886069 w_k1_MSFT_s016
                                 + 1.0737 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.56945 w_k1_ADM_s017
                                 + 1.45865 w_k1_AXP_s017
                                 + 1.34843 w_k1_DIS_s017
                                 + 1.21711 w_k1_FDX_s017
                                 + 1.06019 w_k1_IBM_s017
                                 + 0.999611 w_k1_INTC_s017
                                 + 1.12496 w_k1_KO_s017 + 1.03797 w_k1_MAT_s017
                                 + 1.11303 w_k1_MSFT_s017
                                 + 1.00824 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.22082 w_k1_ADM_s017
                                 + 0.671191 w_k1_AXP_s017
                                 + 0.842937 w_k1_DIS_s017
                                 + 0.976304 w_k1_FDX_s017
                                 + 0.653989 w_k1_IBM_s017
                                 + 0.797657 w_k1_INTC_s017
                                 + 0.712968 w_k1_KO_s017
                                 + 0.454855 w_k1_MAT_s017
                                 + 0.932209 w_k1_MSFT_s017
                                 + 0.923154 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.935338 w_k1_ADM_s017
                                 + 0.896735 w_k1_AXP_s017
                                 + 0.655386 w_k1_DIS_s017
                                 + 0.790839 w_k1_FDX_s017
                                 + 1.19738 w_k1_IBM_s017
                                 + 0.748353 w_k1_INTC_s017
                                 + 0.876226 w_k1_KO_s017
                                 + 1.21729 w_k1_MAT_s017
                                 + 0.775563 w_k1_MSFT_s017
                                 + 0.880901 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.960042 w_k1_ADM_s017
                                 + 0.690744 w_k1_AXP_s017
                                 + 0.806922 w_k1_DIS_s017
                                 + 0.809057 w_k1_FDX_s017
                                 + 0.790752 w_k1_IBM_s017
                                 + 0.775269 w_k1_INTC_s017
                                 + 0.84087 w_k1_KO_s017
                                 + 0.799603 w_k1_MAT_s017
                                 + 0.852471 w_k1_MSFT_s017
                                 + 0.917535 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.22197 w_k1_ADM_s017
                                 + 0.954422 w_k1_AXP_s017
                                 + 0.99997 w_k1_DIS_s017
                                 + 0.977993 w_k1_FDX_s017
                                 + 0.993994 w_k1_IBM_s017
                                 + 0.798387 w_k1_INTC_s017
                                 + 0.9046 w_k1_KO_s017 + 1.1855 w_k1_MAT_s017
                                 + 1.02325 w_k1_MSFT_s017
                                 + 1.00966 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.34023 w_k1_ADM_s017
                                 + 1.23873 w_k1_AXP_s017
                                 + 0.941335 w_k1_DIS_s017
                                 + 1.01939 w_k1_FDX_s017
                                 + 0.981554 w_k1_IBM_s017
                                 + 0.851124 w_k1_INTC_s017
                                 + 0.996961 w_k1_KO_s017
                                 + 0.923511 w_k1_MAT_s017
                                 + 0.869906 w_k1_MSFT_s017
                                 + 1.02853 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.43806 w_k1_ADM_s017
                                 + 1.47276 w_k1_AXP_s017
                                 + 1.55464 w_k1_DIS_s017
                                 + 1.41268 w_k1_FDX_s017
                                 + 1.20244 w_k1_IBM_s017
                                 + 1.13067 w_k1_INTC_s017
                                 + 1.31872 w_k1_KO_s017
                                 + 0.887018 w_k1_MAT_s017
                                 + 1.28838 w_k1_MSFT_s017 + 1.2309 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.40632 w_k1_ADM_s017
                                 + 1.43622 w_k1_AXP_s017
                                 + 1.13877 w_k1_DIS_s017
                                 + 1.19476 w_k1_FDX_s017
                                 + 1.31041 w_k1_IBM_s017
                                 + 1.11217 w_k1_INTC_s017
                                 + 1.23878 w_k1_KO_s017 + 1.42174 w_k1_MAT_s017
                                 + 1.23626 w_k1_MSFT_s017
                                 + 1.22091 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.821601 w_k1_ADM_s100
                                 + 0.773643 w_k1_AXP_s100
                                 + 0.822377 w_k1_DIS_s100
                                 + 0.842081 w_k1_FDX_s100
                                 + 1.04497 w_k1_IBM_s100
                                 + 0.827035 w_k1_INTC_s100
                                 + 0.794576 w_k1_KO_s100
                                 + 0.735012 w_k1_MAT_s100
                                 + 0.759244 w_k1_MSFT_s100
                                 + 0.859885 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.08141 w_k1_ADM_s100
                                 + 1.69476 w_k1_AXP_s100
                                 + 1.35809 w_k1_DIS_s100
                                 + 1.33433 w_k1_FDX_s100
                                 + 1.52326 w_k1_IBM_s100
                                 + 1.32998 w_k1_INTC_s100
                                 + 1.23577 w_k1_KO_s100 + 1.78213 w_k1_MAT_s100
                                 + 1.14469 w_k1_MSFT_s100 + 1.2582 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.05543 w_k1_ADM_s100
                                 + 1.35135 w_k1_AXP_s100
                                 + 0.988206 w_k1_DIS_s100
                                 + 1.05899 w_k1_FDX_s100
                                 + 1.47832 w_k1_IBM_s100
                                 + 0.996179 w_k1_INTC_s100
                                 + 1.18343 w_k1_KO_s100 + 1.551 w_k1_MAT_s100
                                 + 0.922538 w_k1_MSFT_s100 + 1.143 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.789793 w_k1_ADM_s100
                                 + 0.863314 w_k1_AXP_s100
                                 + 1.07578 w_k1_DIS_s100
                                 + 0.970737 w_k1_FDX_s100
                                 + 1.24007 w_k1_IBM_s100
                                 + 1.09699 w_k1_INTC_s100
                                 + 1.16899 w_k1_KO_s100 + 1.15866 w_k1_MAT_s100
                                 + 1.11276 w_k1_MSFT_s100
                                 + 1.02443 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.816921 w_k1_ADM_s100
                                 + 0.766635 w_k1_AXP_s100
                                 + 1.05896 w_k1_DIS_s100
                                 + 0.944984 w_k1_FDX_s100
                                 + 0.836773 w_k1_IBM_s100
                                 + 1.0326 w_k1_INTC_s100
                                 + 0.952131 w_k1_KO_s100
                                 + 0.832059 w_k1_MAT_s100
                                 + 0.92076 w_k1_MSFT_s100
                                 + 0.945875 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.04616 w_k1_ADM_s100
                                 + 1.09078 w_k1_AXP_s100
                                 + 1.16571 w_k1_DIS_s100
                                 + 1.15656 w_k1_FDX_s100
                                 + 0.834633 w_k1_IBM_s100
                                 + 1.26108 w_k1_INTC_s100
                                 + 0.822819 w_k1_KO_s100
                                 + 0.928465 w_k1_MAT_s100
                                 + 1.19154 w_k1_MSFT_s100
                                 + 1.10233 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.31123 w_k1_ADM_s100
                                 + 1.21266 w_k1_AXP_s100
                                 + 1.74672 w_k1_DIS_s100
                                 + 1.45024 w_k1_FDX_s100
                                 + 1.27476 w_k1_IBM_s100
                                 + 1.47787 w_k1_INTC_s100 + 1.3008 w_k1_KO_s100
                                 + 1.29966 w_k1_MAT_s100
                                 + 1.29892 w_k1_MSFT_s100 + 1.3568 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.684704 w_k1_ADM_s100
                                 + 0.568788 w_k1_AXP_s100
                                 + 0.638344 w_k1_DIS_s100
                                 + 0.727429 w_k1_FDX_s100
                                 + 0.75874 w_k1_IBM_s100
                                 + 0.757073 w_k1_INTC_s100
                                 + 0.695571 w_k1_KO_s100
                                 + 0.707368 w_k1_MAT_s100
                                 + 0.69307 w_k1_MSFT_s100 + 0.7728 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.18103 w_k1_ADM_s101
                                 + 1.30901 w_k1_AXP_s101
                                 + 1.21557 w_k1_DIS_s101
                                 + 1.27287 w_k1_FDX_s101
                                 + 1.04491 w_k1_IBM_s101
                                 + 1.02062 w_k1_INTC_s101
                                 + 1.11065 w_k1_KO_s101 + 1.4145 w_k1_MAT_s101
                                 + 1.21564 w_k1_MSFT_s101
                                 + 1.24011 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.878006 w_k1_ADM_s101
                                 + 0.808671 w_k1_AXP_s101
                                 + 0.612325 w_k1_DIS_s101
                                 + 0.829348 w_k1_FDX_s101
                                 + 0.884438 w_k1_IBM_s101
                                 + 0.832737 w_k1_INTC_s101
                                 + 0.885094 w_k1_KO_s101
                                 + 1.04652 w_k1_MAT_s101
                                 + 0.811838 w_k1_MSFT_s101
                                 + 0.884581 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.35417 w_k1_ADM_s101
                                 + 1.40904 w_k1_AXP_s101
                                 + 1.50139 w_k1_DIS_s101
                                 + 1.39285 w_k1_FDX_s101
                                 + 1.26513 w_k1_IBM_s101
                                 + 1.23373 w_k1_INTC_s101
                                 + 1.34027 w_k1_KO_s101 + 1.42096 w_k1_MAT_s101
                                 + 1.412 w_k1_MSFT_s101 + 1.23642 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.00659 w_k1_ADM_s101
                                 + 0.993649 w_k1_AXP_s101
                                 + 0.883644 w_k1_DIS_s101
                                 + 1.03807 w_k1_FDX_s101
                                 + 0.933904 w_k1_IBM_s101
                                 + 0.91178 w_k1_INTC_s101
                                 + 1.03166 w_k1_KO_s101
                                 + 0.709675 w_k1_MAT_s101
                                 + 0.955827 w_k1_MSFT_s101
                                 + 0.994444 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.731052 w_k1_ADM_s101
                                 + 0.645702 w_k1_AXP_s101
                                 + 0.510144 w_k1_DIS_s101
                                 + 0.678918 w_k1_FDX_s101
                                 + 0.942756 w_k1_IBM_s101
                                 + 0.763945 w_k1_INTC_s101
                                 + 0.75299 w_k1_KO_s101
                                 + 0.899393 w_k1_MAT_s101
                                 + 0.767409 w_k1_MSFT_s101
                                 + 0.820187 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.934932 w_k1_ADM_s101
                                 + 0.919226 w_k1_AXP_s101
                                 + 0.952176 w_k1_DIS_s101
                                 + 1.04223 w_k1_FDX_s101
                                 + 0.721764 w_k1_IBM_s101
                                 + 0.820294 w_k1_INTC_s101
                                 + 0.896245 w_k1_KO_s101
                                 + 0.808161 w_k1_MAT_s101
                                 + 0.980412 w_k1_MSFT_s101
                                 + 1.02658 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.02694 w_k1_ADM_s101
                                 + 1.28102 w_k1_AXP_s101
                                 + 0.929233 w_k1_DIS_s101
                                 + 0.982952 w_k1_FDX_s101
                                 + 1.44109 w_k1_IBM_s101
                                 + 0.801859 w_k1_INTC_s101
                                 + 1.18377 w_k1_KO_s101 + 1.73275 w_k1_MAT_s101
                                 + 1.03906 w_k1_MSFT_s101
                                 + 1.02438 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.944264 w_k1_ADM_s101
                                 + 0.812179 w_k1_AXP_s101
                                 + 0.685613 w_k1_DIS_s101
                                 + 0.946135 w_k1_FDX_s101
                                 + 0.514133 w_k1_IBM_s101
                                 + 0.742213 w_k1_INTC_s101
                                 + 0.718826 w_k1_KO_s101
                                 + 0.529962 w_k1_MAT_s101
                                 + 0.895543 w_k1_MSFT_s101
                                 + 0.941055 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.04295 w_k1_ADM_s102
                                 + 1.12203 w_k1_AXP_s102
                                 + 1.39468 w_k1_DIS_s102
                                 + 1.30777 w_k1_FDX_s102
                                 + 0.803411 w_k1_IBM_s102
                                 + 1.28014 w_k1_INTC_s102 + 1.1567 w_k1_KO_s102
                                 + 1.04636 w_k1_MAT_s102
                                 + 1.41843 w_k1_MSFT_s102
                                 + 1.08926 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.814367 w_k1_ADM_s102
                                 + 1.09874 w_k1_AXP_s102
                                 + 1.14968 w_k1_DIS_s102
                                 + 0.964107 w_k1_FDX_s102
                                 + 1.48495 w_k1_IBM_s102
                                 + 0.854887 w_k1_INTC_s102
                                 + 1.12044 w_k1_KO_s102 + 1.60109 w_k1_MAT_s102
                                 + 1.01069 w_k1_MSFT_s102
                                 + 0.975518 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.797395 w_k1_ADM_s102
                                 + 0.569117 w_k1_AXP_s102
                                 + 0.704746 w_k1_DIS_s102
                                 + 0.76546 w_k1_FDX_s102
                                 + 0.771123 w_k1_IBM_s102
                                 + 0.858818 w_k1_INTC_s102
                                 + 0.82864 w_k1_KO_s102
                                 + 0.558238 w_k1_MAT_s102
                                 + 0.867858 w_k1_MSFT_s102
                                 + 0.775352 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.57017 w_k1_ADM_s102
                                 + 1.34441 w_k1_AXP_s102
                                 + 1.78715 w_k1_DIS_s102
                                 + 1.34548 w_k1_FDX_s102
                                 + 1.47013 w_k1_IBM_s102
                                 + 1.1932 w_k1_INTC_s102 + 1.43912 w_k1_KO_s102
                                 + 1.18577 w_k1_MAT_s102
                                 + 1.21036 w_k1_MSFT_s102
                                 + 1.16655 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.770418 w_k1_ADM_s102
                                 + 0.778965 w_k1_AXP_s102
                                 + 0.870668 w_k1_DIS_s102
                                 + 0.831984 w_k1_FDX_s102
                                 + 1.14675 w_k1_IBM_s102
                                 + 0.764345 w_k1_INTC_s102
                                 + 0.855342 w_k1_KO_s102
                                 + 1.10902 w_k1_MAT_s102
                                 + 0.855752 w_k1_MSFT_s102
                                 + 0.887903 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05175 w_k1_ADM_s102
                                 + 1.05039 w_k1_AXP_s102 + 1.1634 w_k1_DIS_s102
                                 + 1.07504 w_k1_FDX_s102 + 0.9157 w_k1_IBM_s102
                                 + 0.852211 w_k1_INTC_s102
                                 + 0.986018 w_k1_KO_s102
                                 + 0.897121 w_k1_MAT_s102
                                 + 1.00773 w_k1_MSFT_s102
                                 + 1.00257 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.942797 w_k1_ADM_s102
                                 + 1.12367 w_k1_AXP_s102
                                 + 1.14377 w_k1_DIS_s102
                                 + 1.06173 w_k1_FDX_s102 + 1.1918 w_k1_IBM_s102
                                 + 1.02231 w_k1_INTC_s102
                                 + 1.09064 w_k1_KO_s102 + 1.31757 w_k1_MAT_s102
                                 + 1.0538 w_k1_MSFT_s102 + 1.14427 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.06783 w_k1_ADM_s102
                                 + 0.895558 w_k1_AXP_s102
                                 + 0.801452 w_k1_DIS_s102
                                 + 0.960464 w_k1_FDX_s102
                                 + 0.725485 w_k1_IBM_s102
                                 + 0.789263 w_k1_INTC_s102
                                 + 0.778427 w_k1_KO_s102
                                 + 0.682987 w_k1_MAT_s102
                                 + 0.872665 w_k1_MSFT_s102
                                 + 0.879814 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.930567 w_k1_ADM_s103
                                 + 1.27065 w_k1_AXP_s103
                                 + 1.45243 w_k1_DIS_s103
                                 + 1.17312 w_k1_FDX_s103
                                 + 1.16957 w_k1_IBM_s103
                                 + 1.24046 w_k1_INTC_s103
                                 + 1.30402 w_k1_KO_s103 + 0.84366 w_k1_MAT_s103
                                 + 1.31481 w_k1_MSFT_s103
                                 + 1.13977 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.874961 w_k1_ADM_s103
                                 + 1.2096 w_k1_AXP_s103
                                 + 0.836686 w_k1_DIS_s103
                                 + 1.0867 w_k1_FDX_s103
                                 + 0.811858 w_k1_IBM_s103
                                 + 1.03856 w_k1_INTC_s103
                                 + 0.917511 w_k1_KO_s103
                                 + 0.728039 w_k1_MAT_s103
                                 + 0.946943 w_k1_MSFT_s103
                                 + 1.01463 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.825828 w_k1_ADM_s103
                                 + 1.31514 w_k1_AXP_s103 + 1.0973 w_k1_DIS_s103
                                 + 1.13303 w_k1_FDX_s103
                                 + 0.954601 w_k1_IBM_s103
                                 + 1.04679 w_k1_INTC_s103
                                 + 0.920125 w_k1_KO_s103
                                 + 1.07178 w_k1_MAT_s103
                                 + 1.15529 w_k1_MSFT_s103
                                 + 1.05883 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.58812 w_k1_ADM_s103
                                 + 0.744651 w_k1_AXP_s103
                                 + 0.552435 w_k1_DIS_s103
                                 + 0.629214 w_k1_FDX_s103
                                 + 0.705557 w_k1_IBM_s103
                                 + 0.753668 w_k1_INTC_s103
                                 + 0.667134 w_k1_KO_s103
                                 + 0.69103 w_k1_MAT_s103
                                 + 0.709486 w_k1_MSFT_s103
                                 + 0.769506 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.61499 w_k1_ADM_s103
                                 + 0.786459 w_k1_AXP_s103
                                 + 0.650231 w_k1_DIS_s103
                                 + 0.734138 w_k1_FDX_s103
                                 + 0.998907 w_k1_IBM_s103
                                 + 0.856668 w_k1_INTC_s103
                                 + 0.933502 w_k1_KO_s103
                                 + 0.930306 w_k1_MAT_s103
                                 + 0.762119 w_k1_MSFT_s103
                                 + 0.923324 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.853487 w_k1_ADM_s103
                                 + 1.02891 w_k1_AXP_s103
                                 + 0.728041 w_k1_DIS_s103
                                 + 0.780638 w_k1_FDX_s103
                                 + 1.23708 w_k1_IBM_s103
                                 + 0.799278 w_k1_INTC_s103
                                 + 1.00893 w_k1_KO_s103 + 1.52978 w_k1_MAT_s103
                                 + 0.899385 w_k1_MSFT_s103
                                 + 0.989238 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.0904 w_k1_ADM_s103
                                 + 1.92055 w_k1_AXP_s103
                                 + 1.80347 w_k1_DIS_s103
                                 + 1.44228 w_k1_FDX_s103
                                 + 1.54499 w_k1_IBM_s103
                                 + 1.18221 w_k1_INTC_s103
                                 + 1.76956 w_k1_KO_s103 + 2.14381 w_k1_MAT_s103
                                 + 1.46859 w_k1_MSFT_s103
                                 + 1.34217 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.81771 w_k1_ADM_s103
                                 + 0.868576 w_k1_AXP_s103
                                 + 0.90815 w_k1_DIS_s103
                                 + 1.00837 w_k1_FDX_s103
                                 + 0.762472 w_k1_IBM_s103
                                 + 0.99561 w_k1_INTC_s103
                                 + 0.885813 w_k1_KO_s103
                                 + 0.773131 w_k1_MAT_s103
                                 + 1.02774 w_k1_MSFT_s103
                                 + 0.930348 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.21401 w_k1_ADM_s104
                                 + 1.38393 w_k1_AXP_s104
                                 + 1.31641 w_k1_DIS_s104
                                 + 1.13297 w_k1_FDX_s104
                                 + 1.56221 w_k1_IBM_s104
                                 + 1.22725 w_k1_INTC_s104
                                 + 1.53992 w_k1_KO_s104 + 2.13553 w_k1_MAT_s104
                                 + 1.21957 w_k1_MSFT_s104
                                 + 1.18307 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.966244 w_k1_ADM_s104
                                 + 0.811858 w_k1_AXP_s104
                                 + 1.07439 w_k1_DIS_s104
                                 + 0.915287 w_k1_FDX_s104
                                 + 1.2021 w_k1_IBM_s104
                                 + 1.14636 w_k1_INTC_s104
                                 + 1.02398 w_k1_KO_s104 + 1.1135 w_k1_MAT_s104
                                 + 1.09774 w_k1_MSFT_s104
                                 + 1.04955 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.730826 w_k1_ADM_s104
                                 + 1.03148 w_k1_AXP_s104
                                 + 0.786211 w_k1_DIS_s104
                                 + 0.891981 w_k1_FDX_s104
                                 + 0.969834 w_k1_IBM_s104
                                 + 0.823463 w_k1_INTC_s104
                                 + 0.873606 w_k1_KO_s104
                                 + 1.91355 w_k1_MAT_s104
                                 + 0.804985 w_k1_MSFT_s104
                                 + 1.01234 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.868898 w_k1_ADM_s104
                                 + 0.670673 w_k1_AXP_s104
                                 + 0.756674 w_k1_DIS_s104
                                 + 0.964498 w_k1_FDX_s104
                                 + 0.586119 w_k1_IBM_s104
                                 + 0.997005 w_k1_INTC_s104
                                 + 0.736476 w_k1_KO_s104
                                 + 0.641115 w_k1_MAT_s104
                                 + 0.986226 w_k1_MSFT_s104
                                 + 0.934613 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.667947 w_k1_ADM_s104
                                 + 0.626179 w_k1_AXP_s104
                                 + 0.580885 w_k1_DIS_s104
                                 + 0.746346 w_k1_FDX_s104
                                 + 0.92342 w_k1_IBM_s104
                                 + 0.861447 w_k1_INTC_s104
                                 + 0.676203 w_k1_KO_s104
                                 + 1.06999 w_k1_MAT_s104
                                 + 0.786213 w_k1_MSFT_s104
                                 + 0.855869 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.24468 w_k1_ADM_s104
                                 + 1.25866 w_k1_AXP_s104
                                 + 1.40751 w_k1_DIS_s104
                                 + 1.41655 w_k1_FDX_s104
                                 + 1.02651 w_k1_IBM_s104
                                 + 1.23084 w_k1_INTC_s104
                                 + 1.05117 w_k1_KO_s104 + 1.00842 w_k1_MAT_s104
                                 + 1.21223 w_k1_MSFT_s104
                                 + 1.11454 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.12345 w_k1_ADM_s104
                                 + 1.03196 w_k1_AXP_s104
                                 + 1.57061 w_k1_DIS_s104
                                 + 1.13481 w_k1_FDX_s104
                                 + 1.03598 w_k1_IBM_s104
                                 + 1.37185 w_k1_INTC_s104
                                 + 1.35926 w_k1_KO_s104
                                 + 0.965791 w_k1_MAT_s104
                                 + 1.26046 w_k1_MSFT_s104
                                 + 1.28558 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.04268 w_k1_ADM_s104
                                 + 1.05276 w_k1_AXP_s104
                                 + 1.04337 w_k1_DIS_s104
                                 + 1.09743 w_k1_FDX_s104
                                 + 0.835228 w_k1_IBM_s104
                                 + 1.11552 w_k1_INTC_s104
                                 + 0.893924 w_k1_KO_s104 + 1.1866 w_k1_MAT_s104
                                 + 1.15934 w_k1_MSFT_s104
                                 + 1.09889 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.02781 w_k1_ADM_s105
                                 + 0.91784 w_k1_AXP_s105
                                 + 1.31181 w_k1_DIS_s105
                                 + 1.15828 w_k1_FDX_s105
                                 + 0.671919 w_k1_IBM_s105
                                 + 1.00104 w_k1_INTC_s105
                                 + 0.943806 w_k1_KO_s105
                                 + 0.745129 w_k1_MAT_s105
                                 + 1.35524 w_k1_MSFT_s105
                                 + 1.04222 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.09607 w_k1_ADM_s105
                                 + 1.28314 w_k1_AXP_s105
                                 + 1.47886 w_k1_DIS_s105 + 1.322 w_k1_FDX_s105
                                 + 1.02588 w_k1_IBM_s105
                                 + 1.26238 w_k1_INTC_s105
                                 + 1.05903 w_k1_KO_s105
                                 + 0.832602 w_k1_MAT_s105
                                 + 1.25092 w_k1_MSFT_s105
                                 + 1.12045 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.857216 w_k1_ADM_s105
                                 + 0.694438 w_k1_AXP_s105
                                 + 0.787325 w_k1_DIS_s105
                                 + 0.912548 w_k1_FDX_s105
                                 + 0.630641 w_k1_IBM_s105
                                 + 0.891447 w_k1_INTC_s105
                                 + 0.743639 w_k1_KO_s105
                                 + 0.530468 w_k1_MAT_s105
                                 + 1.06308 w_k1_MSFT_s105
                                 + 0.830434 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.625941 w_k1_ADM_s105
                                 + 0.693341 w_k1_AXP_s105
                                 + 0.641763 w_k1_DIS_s105
                                 + 0.734378 w_k1_FDX_s105
                                 + 0.81317 w_k1_IBM_s105
                                 + 0.883446 w_k1_INTC_s105
                                 + 0.704821 w_k1_KO_s105
                                 + 0.930262 w_k1_MAT_s105
                                 + 0.863677 w_k1_MSFT_s105
                                 + 0.832204 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.877351 w_k1_ADM_s105
                                 + 0.978438 w_k1_AXP_s105
                                 + 0.975502 w_k1_DIS_s105
                                 + 1.03999 w_k1_FDX_s105
                                 + 1.00796 w_k1_IBM_s105
                                 + 0.91824 w_k1_INTC_s105
                                 + 0.819239 w_k1_KO_s105
                                 + 1.14549 w_k1_MAT_s105
                                 + 0.992528 w_k1_MSFT_s105
                                 + 0.954588 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.903191 w_k1_ADM_s105
                                 + 1.08267 w_k1_AXP_s105
                                 + 1.28023 w_k1_DIS_s105
                                 + 1.07294 w_k1_FDX_s105
                                 + 1.56672 w_k1_IBM_s105
                                 + 0.983865 w_k1_INTC_s105
                                 + 1.2954 w_k1_KO_s105 + 1.60681 w_k1_MAT_s105
                                 + 1.11814 w_k1_MSFT_s105
                                 + 1.13737 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.636272 w_k1_ADM_s105
                                 + 0.708981 w_k1_AXP_s105
                                 + 0.660471 w_k1_DIS_s105
                                 + 0.669383 w_k1_FDX_s105
                                 + 1.14811 w_k1_IBM_s105
                                 + 0.848768 w_k1_INTC_s105
                                 + 0.926844 w_k1_KO_s105
                                 + 1.02138 w_k1_MAT_s105
                                 + 0.799086 w_k1_MSFT_s105
                                 + 0.796173 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.982729 w_k1_ADM_s105
                                 + 1.03585 w_k1_AXP_s105
                                 + 0.975618 w_k1_DIS_s105
                                 + 0.976707 w_k1_FDX_s105
                                 + 0.898369 w_k1_IBM_s105
                                 + 0.927821 w_k1_INTC_s105
                                 + 0.871198 w_k1_KO_s105
                                 + 1.02239 w_k1_MAT_s105
                                 + 1.01047 w_k1_MSFT_s105
                                 + 0.842954 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.846956 w_k1_ADM_s106
                                 + 0.756375 w_k1_AXP_s106
                                 + 0.584804 w_k1_DIS_s106
                                 + 0.777421 w_k1_FDX_s106
                                 + 0.813793 w_k1_IBM_s106
                                 + 0.774068 w_k1_INTC_s106
                                 + 0.741706 w_k1_KO_s106 + 1.1829 w_k1_MAT_s106
                                 + 0.744214 w_k1_MSFT_s106
                                 + 0.956499 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.00599 w_k1_ADM_s106
                                 + 0.906255 w_k1_AXP_s106
                                 + 0.932933 w_k1_DIS_s106
                                 + 0.879839 w_k1_FDX_s106
                                 + 1.24797 w_k1_IBM_s106
                                 + 0.991682 w_k1_INTC_s106
                                 + 0.993561 w_k1_KO_s106
                                 + 1.48501 w_k1_MAT_s106
                                 + 0.920982 w_k1_MSFT_s106
                                 + 0.920095 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.21247 w_k1_ADM_s106
                                 + 1.16257 w_k1_AXP_s106
                                 + 1.59845 w_k1_DIS_s106
                                 + 1.31015 w_k1_FDX_s106
                                 + 0.894824 w_k1_IBM_s106
                                 + 1.06792 w_k1_INTC_s106
                                 + 1.11501 w_k1_KO_s106 + 0.86284 w_k1_MAT_s106
                                 + 1.21629 w_k1_MSFT_s106 + 1.1058 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.06663 w_k1_ADM_s106
                                 + 0.963079 w_k1_AXP_s106
                                 + 1.12383 w_k1_DIS_s106
                                 + 0.988346 w_k1_FDX_s106
                                 + 1.00396 w_k1_IBM_s106
                                 + 0.920391 w_k1_INTC_s106
                                 + 0.94734 w_k1_KO_s106
                                 + 0.845411 w_k1_MAT_s106
                                 + 0.916152 w_k1_MSFT_s106
                                 + 0.951845 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.04471 w_k1_ADM_s106
                                 + 1.15805 w_k1_AXP_s106
                                 + 0.944362 w_k1_DIS_s106
                                 + 1.16832 w_k1_FDX_s106
                                 + 0.649089 w_k1_IBM_s106
                                 + 1.24607 w_k1_INTC_s106
                                 + 0.85088 w_k1_KO_s106
                                 + 0.584566 w_k1_MAT_s106
                                 + 1.13196 w_k1_MSFT_s106
                                 + 1.09532 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.669987 w_k1_ADM_s106
                                 + 0.83861 w_k1_AXP_s106
                                 + 0.679088 w_k1_DIS_s106
                                 + 0.74633 w_k1_FDX_s106
                                 + 1.47059 w_k1_IBM_s106
                                 + 0.830928 w_k1_INTC_s106
                                 + 0.898377 w_k1_KO_s106
                                 + 1.51956 w_k1_MAT_s106
                                 + 0.802633 w_k1_MSFT_s106
                                 + 0.932784 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.05591 w_k1_ADM_s106
                                 + 1.18456 w_k1_AXP_s106
                                 + 1.13606 w_k1_DIS_s106
                                 + 1.05172 w_k1_FDX_s106
                                 + 1.26217 w_k1_IBM_s106
                                 + 0.953021 w_k1_INTC_s106
                                 + 1.12961 w_k1_KO_s106 + 1.33218 w_k1_MAT_s106
                                 + 1.07472 w_k1_MSFT_s106
                                 + 1.12978 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.04943 w_k1_ADM_s106
                                 + 1.39813 w_k1_AXP_s106
                                 + 1.31096 w_k1_DIS_s106
                                 + 1.09439 w_k1_FDX_s106
                                 + 1.78103 w_k1_IBM_s106
                                 + 1.01545 w_k1_INTC_s106
                                 + 1.29488 w_k1_KO_s106 + 2.29171 w_k1_MAT_s106
                                 + 1.01261 w_k1_MSFT_s106
                                 + 1.31927 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.13586 w_k1_ADM_s107
                                 + 1.22808 w_k1_AXP_s107
                                 + 1.15703 w_k1_DIS_s107
                                 + 1.06927 w_k1_FDX_s107
                                 + 1.17515 w_k1_IBM_s107
                                 + 1.47832 w_k1_INTC_s107
                                 + 1.33281 w_k1_KO_s107 + 1.33949 w_k1_MAT_s107
                                 + 1.22592 w_k1_MSFT_s107
                                 + 1.04089 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.924906 w_k1_ADM_s107
                                 + 0.892228 w_k1_AXP_s107
                                 + 0.724731 w_k1_DIS_s107
                                 + 0.883047 w_k1_FDX_s107
                                 + 0.693582 w_k1_IBM_s107
                                 + 0.880845 w_k1_INTC_s107
                                 + 0.815255 w_k1_KO_s107
                                 + 0.824325 w_k1_MAT_s107
                                 + 0.89475 w_k1_MSFT_s107
                                 + 0.846986 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.762826 w_k1_ADM_s107
                                 + 1.00683 w_k1_AXP_s107
                                 + 0.773724 w_k1_DIS_s107
                                 + 0.743835 w_k1_FDX_s107
                                 + 1.28867 w_k1_IBM_s107
                                 + 1.04741 w_k1_INTC_s107
                                 + 1.12641 w_k1_KO_s107 + 1.26145 w_k1_MAT_s107
                                 + 0.945789 w_k1_MSFT_s107
                                 + 0.916055 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.4399 w_k1_ADM_s107
                                 + 1.58766 w_k1_AXP_s107
                                 + 1.91861 w_k1_DIS_s107
                                 + 1.42115 w_k1_FDX_s107
                                 + 1.38566 w_k1_IBM_s107
                                 + 1.52716 w_k1_INTC_s107
                                 + 1.56497 w_k1_KO_s107 + 1.37165 w_k1_MAT_s107
                                 + 1.39588 w_k1_MSFT_s107
                                 + 1.17412 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.1569 w_k1_ADM_s107
                                 + 1.37027 w_k1_AXP_s107
                                 + 1.25662 w_k1_DIS_s107
                                 + 1.29508 w_k1_FDX_s107
                                 + 0.842777 w_k1_IBM_s107
                                 + 1.14522 w_k1_INTC_s107
                                 + 1.10633 w_k1_KO_s107
                                 + 0.990833 w_k1_MAT_s107
                                 + 1.1999 w_k1_MSFT_s107 + 1.11817 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.877775 w_k1_ADM_s107
                                 + 1.13654 w_k1_AXP_s107
                                 + 1.02816 w_k1_DIS_s107
                                 + 0.936289 w_k1_FDX_s107
                                 + 1.1485 w_k1_IBM_s107
                                 + 1.14964 w_k1_INTC_s107
                                 + 1.07752 w_k1_KO_s107
                                 + 0.953368 w_k1_MAT_s107
                                 + 0.973272 w_k1_MSFT_s107
                                 + 0.967397 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.730854 w_k1_ADM_s107
                                 + 0.902439 w_k1_AXP_s107
                                 + 0.766645 w_k1_DIS_s107
                                 + 0.801885 w_k1_FDX_s107
                                 + 0.913984 w_k1_IBM_s107
                                 + 0.984003 w_k1_INTC_s107
                                 + 0.985631 w_k1_KO_s107
                                 + 1.04341 w_k1_MAT_s107
                                 + 0.866568 w_k1_MSFT_s107
                                 + 0.866468 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.02366 w_k1_ADM_s107
                                 + 1.34819 w_k1_AXP_s107
                                 + 1.30002 w_k1_DIS_s107
                                 + 1.08595 w_k1_FDX_s107
                                 + 1.59028 w_k1_IBM_s107
                                 + 1.38956 w_k1_INTC_s107
                                 + 1.47514 w_k1_KO_s107 + 2.09262 w_k1_MAT_s107
                                 + 1.16004 w_k1_MSFT_s107
                                 + 1.21359 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.18592 w_k1_ADM_s110
                                 + 1.15374 w_k1_AXP_s110
                                 + 1.08858 w_k1_DIS_s110
                                 + 1.18333 w_k1_FDX_s110
                                 + 0.753316 w_k1_IBM_s110
                                 + 1.10066 w_k1_INTC_s110
                                 + 0.850265 w_k1_KO_s110
                                 + 0.584136 w_k1_MAT_s110
                                 + 1.00009 w_k1_MSFT_s110
                                 + 0.923869 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.07692 w_k1_ADM_s110
                                 + 1.50898 w_k1_AXP_s110
                                 + 1.02846 w_k1_DIS_s110
                                 + 1.13841 w_k1_FDX_s110
                                 + 1.19185 w_k1_IBM_s110
                                 + 1.18795 w_k1_INTC_s110
                                 + 1.07328 w_k1_KO_s110 + 1.61811 w_k1_MAT_s110
                                 + 0.944016 w_k1_MSFT_s110
                                 + 1.05825 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.27533 w_k1_ADM_s110
                                 + 1.23167 w_k1_AXP_s110
                                 + 1.34913 w_k1_DIS_s110
                                 + 1.18033 w_k1_FDX_s110
                                 + 1.58769 w_k1_IBM_s110
                                 + 1.28518 w_k1_INTC_s110
                                 + 1.43635 w_k1_KO_s110 + 1.2566 w_k1_MAT_s110
                                 + 1.16176 w_k1_MSFT_s110
                                 + 1.11642 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.27179 w_k1_ADM_s110
                                 + 1.24349 w_k1_AXP_s110
                                 + 1.17239 w_k1_DIS_s110
                                 + 1.24915 w_k1_FDX_s110
                                 + 0.969255 w_k1_IBM_s110
                                 + 1.19537 w_k1_INTC_s110
                                 + 1.12853 w_k1_KO_s110 + 1.07695 w_k1_MAT_s110
                                 + 1.09467 w_k1_MSFT_s110
                                 + 1.09014 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.812687 w_k1_ADM_s110
                                 + 0.732115 w_k1_AXP_s110
                                 + 0.686322 w_k1_DIS_s110
                                 + 0.849864 w_k1_FDX_s110
                                 + 0.690184 w_k1_IBM_s110
                                 + 0.803392 w_k1_INTC_s110
                                 + 0.801374 w_k1_KO_s110
                                 + 0.777055 w_k1_MAT_s110
                                 + 0.855025 w_k1_MSFT_s110
                                 + 0.840366 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.956466 w_k1_ADM_s110
                                 + 0.867129 w_k1_AXP_s110
                                 + 0.887553 w_k1_DIS_s110
                                 + 0.95687 w_k1_FDX_s110
                                 + 0.864712 w_k1_IBM_s110
                                 + 1.01334 w_k1_INTC_s110
                                 + 0.955184 w_k1_KO_s110
                                 + 0.81626 w_k1_MAT_s110
                                 + 0.984808 w_k1_MSFT_s110
                                 + 1.04492 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.32055 w_k1_ADM_s110
                                 + 1.20613 w_k1_AXP_s110
                                 + 1.58848 w_k1_DIS_s110
                                 + 1.46807 w_k1_FDX_s110
                                 + 1.07997 w_k1_IBM_s110
                                 + 1.24033 w_k1_INTC_s110
                                 + 1.27669 w_k1_KO_s110 + 1.12002 w_k1_MAT_s110
                                 + 1.34849 w_k1_MSFT_s110
                                 + 1.17719 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.05021 w_k1_ADM_s110
                                 + 0.897405 w_k1_AXP_s110
                                 + 0.959301 w_k1_DIS_s110
                                 + 0.962937 w_k1_FDX_s110
                                 + 1.18398 w_k1_IBM_s110
                                 + 1.02348 w_k1_INTC_s110
                                 + 1.19619 w_k1_KO_s110 + 1.08664 w_k1_MAT_s110
                                 + 1.04249 w_k1_MSFT_s110
                                 + 1.05977 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.983378 w_k1_ADM_s111
                                 + 0.723937 w_k1_AXP_s111
                                 + 0.878759 w_k1_DIS_s111
                                 + 0.831464 w_k1_FDX_s111
                                 + 1.09697 w_k1_IBM_s111
                                 + 0.894462 w_k1_INTC_s111
                                 + 1.22391 w_k1_KO_s111 + 1.08139 w_k1_MAT_s111
                                 + 0.901103 w_k1_MSFT_s111
                                 + 0.901053 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.45502 w_k1_ADM_s111
                                 + 1.02531 w_k1_AXP_s111
                                 + 1.16008 w_k1_DIS_s111
                                 + 1.22553 w_k1_FDX_s111
                                 + 0.765541 w_k1_IBM_s111
                                 + 0.963193 w_k1_INTC_s111
                                 + 0.99536 w_k1_KO_s111
                                 + 0.913675 w_k1_MAT_s111
                                 + 1.1471 w_k1_MSFT_s111 + 1.00377 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.09957 w_k1_ADM_s111
                                 + 1.24806 w_k1_AXP_s111
                                 + 1.02421 w_k1_DIS_s111
                                 + 1.08097 w_k1_FDX_s111
                                 + 1.12501 w_k1_IBM_s111
                                 + 0.924082 w_k1_INTC_s111
                                 + 1.04348 w_k1_KO_s111 + 1.14894 w_k1_MAT_s111
                                 + 0.953118 w_k1_MSFT_s111
                                 + 0.977265 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.842828 w_k1_ADM_s111
                                 + 0.682446 w_k1_AXP_s111
                                 + 0.647077 w_k1_DIS_s111
                                 + 0.783966 w_k1_FDX_s111
                                 + 0.797764 w_k1_IBM_s111
                                 + 0.795855 w_k1_INTC_s111
                                 + 0.838992 w_k1_KO_s111
                                 + 0.744518 w_k1_MAT_s111
                                 + 0.739942 w_k1_MSFT_s111
                                 + 0.864997 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.61273 w_k1_ADM_s111
                                 + 1.49532 w_k1_AXP_s111
                                 + 1.64265 w_k1_DIS_s111
                                 + 1.44578 w_k1_FDX_s111 + 1.3673 w_k1_IBM_s111
                                 + 1.00969 w_k1_INTC_s111 + 1.3983 w_k1_KO_s111
                                 + 1.39191 w_k1_MAT_s111
                                 + 1.2315 w_k1_MSFT_s111 + 1.052 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.0611 w_k1_ADM_s111
                                 + 1.09527 w_k1_AXP_s111
                                 + 1.31828 w_k1_DIS_s111
                                 + 1.10719 w_k1_FDX_s111
                                 + 1.00492 w_k1_IBM_s111
                                 + 0.830111 w_k1_INTC_s111
                                 + 1.27377 w_k1_KO_s111 + 1.16586 w_k1_MAT_s111
                                 + 1.15209 w_k1_MSFT_s111
                                 + 1.00051 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.0182 w_k1_ADM_s111
                                 + 0.776725 w_k1_AXP_s111
                                 + 0.707934 w_k1_DIS_s111
                                 + 0.903719 w_k1_FDX_s111
                                 + 0.547075 w_k1_IBM_s111
                                 + 0.711575 w_k1_INTC_s111
                                 + 0.65793 w_k1_KO_s111
                                 + 0.468848 w_k1_MAT_s111
                                 + 0.754465 w_k1_MSFT_s111
                                 + 0.748288 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.881044 w_k1_ADM_s111
                                 + 0.970587 w_k1_AXP_s111
                                 + 1.28416 w_k1_DIS_s111
                                 + 0.997508 w_k1_FDX_s111
                                 + 1.70716 w_k1_IBM_s111
                                 + 1.00083 w_k1_INTC_s111
                                 + 1.50624 w_k1_KO_s111 + 1.72294 w_k1_MAT_s111
                                 + 1.05488 w_k1_MSFT_s111 + 1.1182 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.775649 w_k1_ADM_s112
                                 + 1.15278 w_k1_AXP_s112
                                 + 0.874591 w_k1_DIS_s112
                                 + 1.07299 w_k1_FDX_s112
                                 + 0.901725 w_k1_IBM_s112
                                 + 0.967656 w_k1_INTC_s112
                                 + 0.9431 w_k1_KO_s112 + 1.28263 w_k1_MAT_s112
                                 + 1.04227 w_k1_MSFT_s112
                                 + 0.984305 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.662101 w_k1_ADM_s112
                                 + 0.709985 w_k1_AXP_s112
                                 + 0.795845 w_k1_DIS_s112
                                 + 0.893667 w_k1_FDX_s112
                                 + 0.735454 w_k1_IBM_s112
                                 + 0.785925 w_k1_INTC_s112
                                 + 0.813058 w_k1_KO_s112
                                 + 0.906703 w_k1_MAT_s112
                                 + 0.940159 w_k1_MSFT_s112
                                 + 0.804514 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.748464 w_k1_ADM_s112
                                 + 1.19984 w_k1_AXP_s112
                                 + 1.09153 w_k1_DIS_s112
                                 + 0.94876 w_k1_FDX_s112
                                 + 1.84971 w_k1_IBM_s112
                                 + 1.0113 w_k1_INTC_s112 + 1.2824 w_k1_KO_s112
                                 + 1.94159 w_k1_MAT_s112
                                 + 0.977528 w_k1_MSFT_s112
                                 + 1.03998 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.783518 w_k1_ADM_s112
                                 + 1.11558 w_k1_AXP_s112
                                 + 0.964795 w_k1_DIS_s112
                                 + 1.16033 w_k1_FDX_s112
                                 + 0.609648 w_k1_IBM_s112
                                 + 0.863421 w_k1_INTC_s112
                                 + 0.756547 w_k1_KO_s112
                                 + 0.958459 w_k1_MAT_s112
                                 + 0.953334 w_k1_MSFT_s112
                                 + 0.990045 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.04223 w_k1_ADM_s112
                                 + 1.20867 w_k1_AXP_s112
                                 + 1.24243 w_k1_DIS_s112
                                 + 1.15773 w_k1_FDX_s112
                                 + 1.05065 w_k1_IBM_s112
                                 + 1.19167 w_k1_INTC_s112
                                 + 1.15001 w_k1_KO_s112 + 1.09832 w_k1_MAT_s112
                                 + 1.19564 w_k1_MSFT_s112
                                 + 1.14581 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.654067 w_k1_ADM_s112
                                 + 0.940146 w_k1_AXP_s112
                                 + 0.963423 w_k1_DIS_s112
                                 + 0.891202 w_k1_FDX_s112
                                 + 1.22698 w_k1_IBM_s112
                                 + 1.06927 w_k1_INTC_s112
                                 + 1.09799 w_k1_KO_s112 + 1.0349 w_k1_MAT_s112
                                 + 0.952422 w_k1_MSFT_s112
                                 + 0.916843 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.12759 w_k1_ADM_s112
                                 + 1.32824 w_k1_AXP_s112
                                 + 1.83365 w_k1_DIS_s112
                                 + 1.57609 w_k1_FDX_s112
                                 + 1.00469 w_k1_IBM_s112
                                 + 1.48669 w_k1_INTC_s112
                                 + 1.31691 w_k1_KO_s112 + 1.16378 w_k1_MAT_s112
                                 + 1.54493 w_k1_MSFT_s112
                                 + 1.23494 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.596269 w_k1_ADM_s112
                                 + 0.782155 w_k1_AXP_s112
                                 + 0.593761 w_k1_DIS_s112
                                 + 0.682899 w_k1_FDX_s112
                                 + 0.956635 w_k1_IBM_s112
                                 + 0.796505 w_k1_INTC_s112
                                 + 0.804457 w_k1_KO_s112
                                 + 0.88591 w_k1_MAT_s112
                                 + 0.864564 w_k1_MSFT_s112
                                 + 0.802998 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.1064 w_k1_ADM_s113
                                 + 1.15405 w_k1_AXP_s113
                                 + 0.996834 w_k1_DIS_s113
                                 + 1.07071 w_k1_FDX_s113
                                 + 1.19533 w_k1_IBM_s113
                                 + 1.15277 w_k1_INTC_s113
                                 + 1.00326 w_k1_KO_s113 + 1.28118 w_k1_MAT_s113
                                 + 0.913645 w_k1_MSFT_s113
                                 + 1.04173 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.895123 w_k1_ADM_s113
                                 + 1.24745 w_k1_AXP_s113
                                 + 1.27301 w_k1_DIS_s113
                                 + 1.04652 w_k1_FDX_s113
                                 + 1.57289 w_k1_IBM_s113
                                 + 1.08718 w_k1_INTC_s113
                                 + 1.22211 w_k1_KO_s113 + 1.76444 w_k1_MAT_s113
                                 + 0.989507 w_k1_MSFT_s113
                                 + 1.11718 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.80888 w_k1_ADM_s113
                                 + 0.804325 w_k1_AXP_s113
                                 + 0.824127 w_k1_DIS_s113
                                 + 0.86457 w_k1_FDX_s113
                                 + 1.09723 w_k1_IBM_s113
                                 + 0.944014 w_k1_INTC_s113
                                 + 0.876053 w_k1_KO_s113 + 1.3299 w_k1_MAT_s113
                                 + 0.882863 w_k1_MSFT_s113
                                 + 0.882568 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.962072 w_k1_ADM_s113
                                 + 0.869759 w_k1_AXP_s113
                                 + 0.530697 w_k1_DIS_s113
                                 + 0.845904 w_k1_FDX_s113
                                 + 0.696812 w_k1_IBM_s113
                                 + 0.745112 w_k1_INTC_s113
                                 + 0.587489 w_k1_KO_s113
                                 + 0.83193 w_k1_MAT_s113
                                 + 0.632544 w_k1_MSFT_s113
                                 + 0.822341 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.835699 w_k1_ADM_s113
                                 + 0.700902 w_k1_AXP_s113
                                 + 0.616401 w_k1_DIS_s113
                                 + 0.743006 w_k1_FDX_s113
                                 + 0.995533 w_k1_IBM_s113
                                 + 0.924198 w_k1_INTC_s113
                                 + 0.727339 w_k1_KO_s113
                                 + 0.991708 w_k1_MAT_s113
                                 + 0.739353 w_k1_MSFT_s113
                                 + 0.805076 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.35431 w_k1_ADM_s113
                                 + 1.43681 w_k1_AXP_s113
                                 + 1.59766 w_k1_DIS_s113
                                 + 1.49407 w_k1_FDX_s113
                                 + 0.973467 w_k1_IBM_s113
                                 + 1.30395 w_k1_INTC_s113
                                 + 1.07397 w_k1_KO_s113 + 1.39519 w_k1_MAT_s113
                                 + 1.26258 w_k1_MSFT_s113
                                 + 1.16329 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.856999 w_k1_ADM_s113
                                 + 0.880579 w_k1_AXP_s113
                                 + 0.86186 w_k1_DIS_s113
                                 + 0.952111 w_k1_FDX_s113
                                 + 0.824148 w_k1_IBM_s113
                                 + 0.856434 w_k1_INTC_s113
                                 + 0.866143 w_k1_KO_s113
                                 + 0.997274 w_k1_MAT_s113
                                 + 0.885913 w_k1_MSFT_s113
                                 + 0.985811 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.04816 w_k1_ADM_s113
                                 + 1.15316 w_k1_AXP_s113
                                 + 1.09174 w_k1_DIS_s113
                                 + 1.17655 w_k1_FDX_s113
                                 + 0.738469 w_k1_IBM_s113
                                 + 1.13442 w_k1_INTC_s113
                                 + 0.782621 w_k1_KO_s113
                                 + 0.810637 w_k1_MAT_s113
                                 + 1.03631 w_k1_MSFT_s113
                                 + 1.03911 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.05854 w_k1_ADM_s114
                                 + 1.52858 w_k1_AXP_s114
                                 + 1.28657 w_k1_DIS_s114
                                 + 1.16153 w_k1_FDX_s114
                                 + 1.57184 w_k1_IBM_s114
                                 + 0.906541 w_k1_INTC_s114
                                 + 1.27266 w_k1_KO_s114 + 2.32134 w_k1_MAT_s114
                                 + 1.02042 w_k1_MSFT_s114
                                 + 1.11997 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.916083 w_k1_ADM_s114
                                 + 1.02369 w_k1_AXP_s114
                                 + 1.12997 w_k1_DIS_s114
                                 + 1.12505 w_k1_FDX_s114
                                 + 0.94347 w_k1_IBM_s114
                                 + 1.06959 w_k1_INTC_s114
                                 + 1.00888 w_k1_KO_s114 + 1.31026 w_k1_MAT_s114
                                 + 0.964262 w_k1_MSFT_s114
                                 + 1.07367 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.867966 w_k1_ADM_s114
                                 + 0.969965 w_k1_AXP_s114
                                 + 1.39687 w_k1_DIS_s114
                                 + 1.11991 w_k1_FDX_s114
                                 + 0.916207 w_k1_IBM_s114
                                 + 1.09314 w_k1_INTC_s114
                                 + 0.929527 w_k1_KO_s114 + 1.2643 w_k1_MAT_s114
                                 + 1.04891 w_k1_MSFT_s114
                                 + 1.00602 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.921953 w_k1_ADM_s114
                                 + 1.0551 w_k1_AXP_s114 + 1.04009 w_k1_DIS_s114
                                 + 0.90939 w_k1_FDX_s114
                                 + 1.32175 w_k1_IBM_s114
                                 + 0.818041 w_k1_INTC_s114
                                 + 1.15206 w_k1_KO_s114 + 2.00946 w_k1_MAT_s114
                                 + 0.766258 w_k1_MSFT_s114
                                 + 1.03643 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.16942 w_k1_ADM_s114
                                 + 1.08604 w_k1_AXP_s114
                                 + 0.886068 w_k1_DIS_s114
                                 + 1.04437 w_k1_FDX_s114
                                 + 0.824676 w_k1_IBM_s114
                                 + 1.09303 w_k1_INTC_s114
                                 + 0.961796 w_k1_KO_s114
                                 + 1.02333 w_k1_MAT_s114
                                 + 1.04574 w_k1_MSFT_s114
                                 + 0.959304 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.816741 w_k1_ADM_s114
                                 + 0.802744 w_k1_AXP_s114
                                 + 0.718105 w_k1_DIS_s114
                                 + 0.815986 w_k1_FDX_s114
                                 + 0.851091 w_k1_IBM_s114
                                 + 0.711528 w_k1_INTC_s114
                                 + 0.829828 w_k1_KO_s114
                                 + 1.47954 w_k1_MAT_s114
                                 + 0.720119 w_k1_MSFT_s114
                                 + 0.934076 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.867681 w_k1_ADM_s114
                                 + 0.675204 w_k1_AXP_s114
                                 + 0.660219 w_k1_DIS_s114
                                 + 0.878881 w_k1_FDX_s114
                                 + 0.570426 w_k1_IBM_s114
                                 + 0.851773 w_k1_INTC_s114
                                 + 0.645603 w_k1_KO_s114
                                 + 0.527273 w_k1_MAT_s114
                                 + 0.757084 w_k1_MSFT_s114
                                 + 0.801059 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.26144 w_k1_ADM_s114
                                 + 1.6165 w_k1_AXP_s114 + 1.72543 w_k1_DIS_s114
                                 + 1.40114 w_k1_FDX_s114
                                 + 1.08044 w_k1_IBM_s114
                                 + 1.28022 w_k1_INTC_s114
                                 + 1.30171 w_k1_KO_s114 + 1.39973 w_k1_MAT_s114
                                 + 1.2035 w_k1_MSFT_s114 + 1.10153 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.915312 w_k1_ADM_s115
                                 + 1.25125 w_k1_AXP_s115
                                 + 1.06197 w_k1_DIS_s115 + 1.0149 w_k1_FDX_s115
                                 + 1.07341 w_k1_IBM_s115
                                 + 0.812407 w_k1_INTC_s115
                                 + 1.07691 w_k1_KO_s115 + 1.33156 w_k1_MAT_s115
                                 + 0.921851 w_k1_MSFT_s115
                                 + 1.07342 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.39054 w_k1_ADM_s115
                                 + 1.4423 w_k1_AXP_s115 + 1.36337 w_k1_DIS_s115
                                 + 1.31696 w_k1_FDX_s115
                                 + 1.13011 w_k1_IBM_s115
                                 + 1.12105 w_k1_INTC_s115
                                 + 1.16738 w_k1_KO_s115 + 1.02663 w_k1_MAT_s115
                                 + 1.16938 w_k1_MSFT_s115 + 1.1423 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.995384 w_k1_ADM_s115
                                 + 0.828945 w_k1_AXP_s115
                                 + 0.812894 w_k1_DIS_s115
                                 + 0.885452 w_k1_FDX_s115
                                 + 0.862574 w_k1_IBM_s115
                                 + 0.991824 w_k1_INTC_s115
                                 + 0.868846 w_k1_KO_s115
                                 + 0.632369 w_k1_MAT_s115
                                 + 0.839753 w_k1_MSFT_s115
                                 + 0.956035 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.845203 w_k1_ADM_s115
                                 + 0.667151 w_k1_AXP_s115
                                 + 0.867967 w_k1_DIS_s115
                                 + 0.863694 w_k1_FDX_s115
                                 + 0.923395 w_k1_IBM_s115
                                 + 0.76278 w_k1_INTC_s115
                                 + 0.957755 w_k1_KO_s115
                                 + 0.727782 w_k1_MAT_s115
                                 + 0.889985 w_k1_MSFT_s115
                                 + 0.972502 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.19586 w_k1_ADM_s115
                                 + 1.20709 w_k1_AXP_s115
                                 + 1.07179 w_k1_DIS_s115
                                 + 1.12035 w_k1_FDX_s115
                                 + 0.809023 w_k1_IBM_s115
                                 + 0.980511 w_k1_INTC_s115
                                 + 0.956875 w_k1_KO_s115
                                 + 0.718824 w_k1_MAT_s115
                                 + 1.04538 w_k1_MSFT_s115
                                 + 1.01741 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.09252 w_k1_ADM_s115
                                 + 0.966291 w_k1_AXP_s115
                                 + 1.35624 w_k1_DIS_s115
                                 + 1.13159 w_k1_FDX_s115
                                 + 1.11464 w_k1_IBM_s115
                                 + 0.892202 w_k1_INTC_s115
                                 + 1.24493 w_k1_KO_s115 + 1.01457 w_k1_MAT_s115
                                 + 1.0459 w_k1_MSFT_s115 + 1.10203 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.897564 w_k1_ADM_s115
                                 + 1.11902 w_k1_AXP_s115
                                 + 0.976015 w_k1_DIS_s115
                                 + 0.886574 w_k1_FDX_s115
                                 + 1.47981 w_k1_IBM_s115
                                 + 0.976496 w_k1_INTC_s115
                                 + 1.22184 w_k1_KO_s115 + 1.1661 w_k1_MAT_s115
                                 + 0.984932 w_k1_MSFT_s115
                                 + 0.988607 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.2804 w_k1_ADM_s115
                                 + 1.36026 w_k1_AXP_s115
                                 + 1.71218 w_k1_DIS_s115
                                 + 1.26952 w_k1_FDX_s115
                                 + 1.63473 w_k1_IBM_s115
                                 + 1.2122 w_k1_INTC_s115 + 1.69343 w_k1_KO_s115
                                 + 1.35576 w_k1_MAT_s115
                                 + 1.35854 w_k1_MSFT_s115
                                 + 1.24851 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.964396 w_k1_ADM_s116
                                 + 0.936619 w_k1_AXP_s116
                                 + 0.831344 w_k1_DIS_s116
                                 + 0.885738 w_k1_FDX_s116
                                 + 1.1662 w_k1_IBM_s116
                                 + 0.958924 w_k1_INTC_s116
                                 + 0.987241 w_k1_KO_s116
                                 + 1.00405 w_k1_MAT_s116
                                 + 0.992386 w_k1_MSFT_s116
                                 + 0.927466 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.902942 w_k1_ADM_s116
                                 + 0.982866 w_k1_AXP_s116
                                 + 1.06328 w_k1_DIS_s116
                                 + 0.903963 w_k1_FDX_s116
                                 + 1.46646 w_k1_IBM_s116
                                 + 1.06317 w_k1_INTC_s116
                                 + 1.34896 w_k1_KO_s116 + 1.50723 w_k1_MAT_s116
                                 + 1.10742 w_k1_MSFT_s116
                                 + 1.02756 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.06132 w_k1_ADM_s116
                                 + 0.95048 w_k1_AXP_s116
                                 + 0.737858 w_k1_DIS_s116
                                 + 1.04001 w_k1_FDX_s116
                                 + 0.634074 w_k1_IBM_s116
                                 + 1.10121 w_k1_INTC_s116
                                 + 0.783442 w_k1_KO_s116
                                 + 0.399526 w_k1_MAT_s116
                                 + 0.887235 w_k1_MSFT_s116
                                 + 0.824557 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.15961 w_k1_ADM_s116
                                 + 0.854993 w_k1_AXP_s116
                                 + 1.09805 w_k1_DIS_s116
                                 + 1.11245 w_k1_FDX_s116
                                 + 0.764817 w_k1_IBM_s116
                                 + 1.10985 w_k1_INTC_s116
                                 + 0.979855 w_k1_KO_s116
                                 + 0.522537 w_k1_MAT_s116
                                 + 1.16995 w_k1_MSFT_s116
                                 + 0.908162 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.1568 w_k1_ADM_s116
                                 + 1.36212 w_k1_AXP_s116
                                 + 1.31015 w_k1_DIS_s116
                                 + 1.22863 w_k1_FDX_s116
                                 + 1.21072 w_k1_IBM_s116
                                 + 1.08935 w_k1_INTC_s116
                                 + 1.28018 w_k1_KO_s116 + 1.2279 w_k1_MAT_s116
                                 + 1.09806 w_k1_MSFT_s116
                                 + 1.17947 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.11122 w_k1_ADM_s116
                                 + 1.37226 w_k1_AXP_s116
                                 + 1.59836 w_k1_DIS_s116
                                 + 1.17641 w_k1_FDX_s116
                                 + 1.82763 w_k1_IBM_s116
                                 + 1.26491 w_k1_INTC_s116
                                 + 1.78309 w_k1_KO_s116 + 1.8579 w_k1_MAT_s116
                                 + 1.26354 w_k1_MSFT_s116
                                 + 1.14029 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.850748 w_k1_ADM_s116
                                 + 0.656524 w_k1_AXP_s116
                                 + 0.725687 w_k1_DIS_s116
                                 + 0.787972 w_k1_FDX_s116
                                 + 0.753645 w_k1_IBM_s116
                                 + 0.911055 w_k1_INTC_s116
                                 + 0.777237 w_k1_KO_s116
                                 + 0.512851 w_k1_MAT_s116
                                 + 0.858667 w_k1_MSFT_s116
                                 + 0.779427 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.937881 w_k1_ADM_s116
                                 + 0.975765 w_k1_AXP_s116
                                 + 0.768082 w_k1_DIS_s116
                                 + 0.907795 w_k1_FDX_s116
                                 + 0.991234 w_k1_IBM_s116
                                 + 1.01295 w_k1_INTC_s116
                                 + 1.09381 w_k1_KO_s116 + 0.73888 w_k1_MAT_s116
                                 + 0.892519 w_k1_MSFT_s116
                                 + 0.936244 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.13686 w_k1_ADM_s117
                                 + 1.18911 w_k1_AXP_s117
                                 + 1.17448 w_k1_DIS_s117
                                 + 1.04761 w_k1_FDX_s117
                                 + 1.81044 w_k1_IBM_s117
                                 + 0.959049 w_k1_INTC_s117
                                 + 1.58554 w_k1_KO_s117 + 1.57154 w_k1_MAT_s117
                                 + 1.3008 w_k1_MSFT_s117 + 1.26514 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.997889 w_k1_ADM_s117
                                 + 1.67085 w_k1_AXP_s117
                                 + 1.69578 w_k1_DIS_s117
                                 + 1.42922 w_k1_FDX_s117
                                 + 1.29892 w_k1_IBM_s117
                                 + 1.55472 w_k1_INTC_s117
                                 + 1.35572 w_k1_KO_s117 + 1.21757 w_k1_MAT_s117
                                 + 1.6244 w_k1_MSFT_s117 + 1.3702 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.890656 w_k1_ADM_s117
                                 + 1.10256 w_k1_AXP_s117
                                 + 1.06512 w_k1_DIS_s117
                                 + 1.17147 w_k1_FDX_s117
                                 + 0.786233 w_k1_IBM_s117
                                 + 1.13313 w_k1_INTC_s117
                                 + 0.981126 w_k1_KO_s117
                                 + 0.881412 w_k1_MAT_s117
                                 + 1.19696 w_k1_MSFT_s117
                                 + 0.970998 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.726927 w_k1_ADM_s117
                                 + 0.722188 w_k1_AXP_s117
                                 + 0.617419 w_k1_DIS_s117
                                 + 0.811753 w_k1_FDX_s117
                                 + 0.704671 w_k1_IBM_s117
                                 + 0.778643 w_k1_INTC_s117
                                 + 0.819761 w_k1_KO_s117
                                 + 0.828321 w_k1_MAT_s117
                                 + 0.844978 w_k1_MSFT_s117
                                 + 0.799207 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.804408 w_k1_ADM_s117
                                 + 0.864351 w_k1_AXP_s117
                                 + 0.921307 w_k1_DIS_s117
                                 + 0.881016 w_k1_FDX_s117
                                 + 1.02127 w_k1_IBM_s117
                                 + 0.790959 w_k1_INTC_s117
                                 + 1.01428 w_k1_KO_s117
                                 + 0.938692 w_k1_MAT_s117
                                 + 0.950944 w_k1_MSFT_s117
                                 + 0.917264 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.14949 w_k1_ADM_s117
                                 + 1.37594 w_k1_AXP_s117
                                 + 1.36281 w_k1_DIS_s117
                                 + 1.19864 w_k1_FDX_s117
                                 + 0.989173 w_k1_IBM_s117
                                 + 0.897528 w_k1_INTC_s117
                                 + 1.14267 w_k1_KO_s117 + 1.15254 w_k1_MAT_s117
                                 + 1.11757 w_k1_MSFT_s117
                                 + 1.10572 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.848675 w_k1_ADM_s117
                                 + 1.33203 w_k1_AXP_s117
                                 + 0.801376 w_k1_DIS_s117
                                 + 0.948819 w_k1_FDX_s117
                                 + 1.3226 w_k1_IBM_s117
                                 + 1.00162 w_k1_INTC_s117
                                 + 0.939822 w_k1_KO_s117
                                 + 1.29266 w_k1_MAT_s117
                                 + 0.84559 w_k1_MSFT_s117
                                 + 0.895982 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.958748 w_k1_ADM_s117
                                 + 0.933694 w_k1_AXP_s117
                                 + 1.09262 w_k1_DIS_s117
                                 + 1.00216 w_k1_FDX_s117
                                 + 1.18237 w_k1_IBM_s117
                                 + 0.93909 w_k1_INTC_s117
                                 + 1.07636 w_k1_KO_s117 + 1.01959 w_k1_MAT_s117
                                 + 1.12267 w_k1_MSFT_s117
                                 + 1.03417 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0538667511419948 w_k1_ADM_s000
                                 + 0.0339401426908367 w_k1_ADM_s001
                                 + 0.0439346942613101 w_k1_ADM_s002
                                 + 0.0253487701958491 w_k1_ADM_s003
                                 + 0.0264536970879596 w_k1_ADM_s004
                                 + 0.0423724372374974 w_k1_ADM_s005
                                 + 0.0188402453252786 w_k1_ADM_s006
                                 + 0.0342276571732768 w_k1_ADM_s007
                                 + 0.0125578303806285 w_k1_ADM_s010
                                 + 0.0378621125489056 w_k1_ADM_s011
                                 + 0.010317716942767 w_k1_ADM_s012
                                 + 0.0424705003878728 w_k1_ADM_s013
                                 + 0.0442799217163067 w_k1_ADM_s014
                                 + 0.0170580140385408 w_k1_ADM_s015
                                 + 0.0288780322346997 w_k1_ADM_s016
                                 + 0.0157541740558753 w_k1_ADM_s017
                                 + 0.044831622632216 w_k1_ADM_s100
                                 + 0.0242245118388586 w_k1_ADM_s101
                                 + 0.0415594458665621 w_k1_ADM_s102
                                 + 0.0197658313769694 w_k1_ADM_s103
                                 + 0.0361682974078629 w_k1_ADM_s104
                                 + 0.0307449545919052 w_k1_ADM_s105
                                 + 0.0153678312278131 w_k1_ADM_s106
                                 + 0.0291411406808432 w_k1_ADM_s107
                                 + 0.016907667999092 w_k1_ADM_s110
                                 + 0.072414719897421 w_k1_ADM_s111
                                 + 0.0366704049136051 w_k1_ADM_s112
                                 + 0.0422020260160242 w_k1_ADM_s113
                                 + 0.0150003369726651 w_k1_ADM_s114
                                 + 0.019170593017437 w_k1_ADM_s115
                                 + 0.0421082021888833 w_k1_ADM_s116
                                 + 0.0262614143311612 w_k1_ADM_s117
                                 + 0.0608330839723414 w_k1_AXP_s000
                                 + 0.038483953806053 w_k1_AXP_s001
                                 + 0.0414507895147498 w_k1_AXP_s002
                                 + 0.0273450378812177 w_k1_AXP_s003
                                 + 0.02398584605804 w_k1_AXP_s004
                                 + 0.0392592769069386 w_k1_AXP_s005
                                 + 0.0198427427031522 w_k1_AXP_s006
                                 + 0.0327310932340278 w_k1_AXP_s007
                                 + 0.0103421575487417 w_k1_AXP_s010
                                 + 0.0406433666816991 w_k1_AXP_s011
                                 + 0.0111790337124353 w_k1_AXP_s012
                                 + 0.0490477336008216 w_k1_AXP_s013
                                 + 0.0404959391544247 w_k1_AXP_s014
                                 + 0.0182228568078906 w_k1_AXP_s015
                                 + 0.0283824908956038 w_k1_AXP_s016
                                 + 0.0136686787626942 w_k1_AXP_s017
                                 + 0.0475338404627464 w_k1_AXP_s100
                                 + 0.0253635279689852 w_k1_AXP_s101
                                 + 0.0404986897837106 w_k1_AXP_s102
                                 + 0.02691165036322 w_k1_AXP_s103
                                 + 0.0358950953157928 w_k1_AXP_s104
                                 + 0.0334084943259869 w_k1_AXP_s105
                                 + 0.015876410318922 w_k1_AXP_s106
                                 + 0.0338007828437888 w_k1_AXP_s107
                                 + 0.0160365087263004 w_k1_AXP_s110
                                 + 0.0630592329807015 w_k1_AXP_s111
                                 + 0.047577103050789 w_k1_AXP_s112
                                 + 0.0452731989813086 w_k1_AXP_s113
                                 + 0.0161135015727027 w_k1_AXP_s114
                                 + 0.019068942592411 w_k1_AXP_s115
                                 + 0.0416877362634453 w_k1_AXP_s116
                                 + 0.0310188112947354 w_k1_AXP_s117
                                 + 0.0570215054788527 w_k1_DIS_s000
                                 + 0.0340260604512643 w_k1_DIS_s001
                                 + 0.0400107110273044 w_k1_DIS_s002
                                 + 0.0275406103183531 w_k1_DIS_s003
                                 + 0.0252410974119661 w_k1_DIS_s004
                                 + 0.0421307902561591 w_k1_DIS_s005
                                 + 0.0196408866329551 w_k1_DIS_s006
                                 + 0.0332744537819044 w_k1_DIS_s007
                                 + 0.0108981303287043 w_k1_DIS_s010
                                 + 0.036498419040253 w_k1_DIS_s011
                                 + 0.0111759590658407 w_k1_DIS_s012
                                 + 0.050884609686286 w_k1_DIS_s013
                                 + 0.0375492814071075 w_k1_DIS_s014
                                 + 0.0194878870713043 w_k1_DIS_s015
                                 + 0.0284749210437752 w_k1_DIS_s016
                                 + 0.0128524324899534 w_k1_DIS_s017
                                 + 0.0513462474237265 w_k1_DIS_s100
                                 + 0.0220371392764846 w_k1_DIS_s101
                                 + 0.0443714206355955 w_k1_DIS_s102
                                 + 0.0245094074744834 w_k1_DIS_s103
                                 + 0.0379570836414868 w_k1_DIS_s104
                                 + 0.0361011377004753 w_k1_DIS_s105
                                 + 0.0160555367989528 w_k1_DIS_s106
                                 + 0.031191457006554 w_k1_DIS_s107
                                 + 0.0161839537557392 w_k1_DIS_s110
                                 + 0.0677345717489585 w_k1_DIS_s111
                                 + 0.048000723107278 w_k1_DIS_s112
                                 + 0.0425894291152104 w_k1_DIS_s113
                                 + 0.0157821806428206 w_k1_DIS_s114
                                 + 0.0198693870353849 w_k1_DIS_s115
                                 + 0.0422118055177554 w_k1_DIS_s116
                                 + 0.0298074015221428 w_k1_DIS_s117
                                 + 0.0564549772296623 w_k1_FDX_s000
                                 + 0.0344374858394872 w_k1_FDX_s001
                                 + 0.0404595518675821 w_k1_FDX_s002
                                 + 0.0285447016865831 w_k1_FDX_s003
                                 + 0.0236524409465488 w_k1_FDX_s004
                                 + 0.0397399547512094 w_k1_FDX_s005
                                 + 0.0188867231020996 w_k1_FDX_s006
                                 + 0.0313197900455771 w_k1_FDX_s007
                                 + 0.0109192057177315 w_k1_FDX_s010
                                 + 0.037864389257808 w_k1_FDX_s011
                                 + 0.0106750838294904 w_k1_FDX_s012
                                 + 0.0472353582450028 w_k1_FDX_s013
                                 + 0.0382248365642354 w_k1_FDX_s014
                                 + 0.0195798314530743 w_k1_FDX_s015
                                 + 0.027575319907112 w_k1_FDX_s016
                                 + 0.0131639121550143 w_k1_FDX_s017
                                 + 0.0496452923087865 w_k1_FDX_s100
                                 + 0.0245530469064065 w_k1_FDX_s101
                                 + 0.0422652914026009 w_k1_FDX_s102
                                 + 0.0238455565715146 w_k1_FDX_s103
                                 + 0.0382935491479565 w_k1_FDX_s104
                                 + 0.0350460696464107 w_k1_FDX_s105
                                 + 0.0155933726276513 w_k1_FDX_s106
                                 + 0.0300087724411709 w_k1_FDX_s107
                                 + 0.0167415940360098 w_k1_FDX_s110
                                 + 0.0669227159514996 w_k1_FDX_s111
                                 + 0.047283917204623 w_k1_FDX_s112
                                 + 0.044065214006257 w_k1_FDX_s113
                                 + 0.0157232494293748 w_k1_FDX_s114
                                 + 0.0187522035765792 w_k1_FDX_s115
                                 + 0.0417912241747964 w_k1_FDX_s116
                                 + 0.0292178635570045 w_k1_FDX_s117
                                 + 0.0598376039857765 w_k1_IBM_s000
                                 + 0.033927985722526 w_k1_IBM_s001
                                 + 0.0393432943204151 w_k1_IBM_s002
                                 + 0.0280078507528843 w_k1_IBM_s003
                                 + 0.0230748320984224 w_k1_IBM_s004
                                 + 0.040106214403856 w_k1_IBM_s005
                                 + 0.01985239385018 w_k1_IBM_s006
                                 + 0.0339171672207654 w_k1_IBM_s007
                                 + 0.0108915992479023 w_k1_IBM_s010
                                 + 0.0375983076475357 w_k1_IBM_s011
                                 + 0.0106855492546733 w_k1_IBM_s012
                                 + 0.049757339843421 w_k1_IBM_s013
                                 + 0.0381559997884272 w_k1_IBM_s014
                                 + 0.0206356951461501 w_k1_IBM_s015
                                 + 0.0289860496095417 w_k1_IBM_s016
                                 + 0.0132280804447171 w_k1_IBM_s017
                                 + 0.0514197140886083 w_k1_IBM_s100
                                 + 0.0246206511418241 w_k1_IBM_s101
                                 + 0.04390950477853 w_k1_IBM_s102
                                 + 0.0245551540026791 w_k1_IBM_s103
                                 + 0.0373255687748633 w_k1_IBM_s104
                                 + 0.0343979198455336 w_k1_IBM_s105
                                 + 0.015744358312719 w_k1_IBM_s106
                                 + 0.0305161877355937 w_k1_IBM_s107
                                 + 0.0159953024169659 w_k1_IBM_s110
                                 + 0.0648333284742172 w_k1_IBM_s111
                                 + 0.0491788092997205 w_k1_IBM_s112
                                 + 0.0463034183169777 w_k1_IBM_s113
                                 + 0.0153884181954933 w_k1_IBM_s114
                                 + 0.0199215171937996 w_k1_IBM_s115
                                 + 0.0433017388912927 w_k1_IBM_s116
                                 + 0.031180732763897 w_k1_IBM_s117
                                 + 0.0609862477920272 w_k1_INTC_s000
                                 + 0.0312643658460021 w_k1_INTC_s001
                                 + 0.047108230344012 w_k1_INTC_s002
                                 + 0.0280247450823031 w_k1_INTC_s003
                                 + 0.0206114335486021 w_k1_INTC_s004
                                 + 0.0353327336754117 w_k1_INTC_s005
                                 + 0.0191311308361953 w_k1_INTC_s006
                                 + 0.0314863258088175 w_k1_INTC_s007
                                 + 0.0094617908144111 w_k1_INTC_s010
                                 + 0.0381230871499576 w_k1_INTC_s011
                                 + 0.010349487994399 w_k1_INTC_s012
                                 + 0.045524849202207 w_k1_INTC_s013
                                 + 0.0372200180714866 w_k1_INTC_s014
                                 + 0.0220841861790976 w_k1_INTC_s015
                                 + 0.0262269854932686 w_k1_INTC_s016
                                 + 0.0114353342603588 w_k1_INTC_s017
                                 + 0.0519451579870398 w_k1_INTC_s100
                                 + 0.0213520441842434 w_k1_INTC_s101
                                 + 0.0384460025627479 w_k1_INTC_s102
                                 + 0.0241255902656701 w_k1_INTC_s103
                                 + 0.0402915803906946 w_k1_INTC_s104
                                 + 0.0338618736885073 w_k1_INTC_s105
                                 + 0.0147924922112673 w_k1_INTC_s106
                                 + 0.0340207735793514 w_k1_INTC_s107
                                 + 0.0168499115630725 w_k1_INTC_s110
                                 + 0.0574197226079315 w_k1_INTC_s111
                                 + 0.0471919453519769 w_k1_INTC_s112
                                 + 0.044737961071378 w_k1_INTC_s113
                                 + 0.0146301648156734 w_k1_INTC_s114
                                 + 0.0175191009757595 w_k1_INTC_s115
                                 + 0.0427882944329315 w_k1_INTC_s116
                                 + 0.0272260663085544 w_k1_INTC_s117
                                 + 0.0562094819642319 w_k1_KO_s000
                                 + 0.0336070082281192 w_k1_KO_s001
                                 + 0.0411077153803766 w_k1_KO_s002
                                 + 0.0285383107093883 w_k1_KO_s003
                                 + 0.0235062647491515 w_k1_KO_s004
                                 + 0.0395418183824546 w_k1_KO_s005
                                 + 0.0194332688824899 w_k1_KO_s006
                                 + 0.0313537492145183 w_k1_KO_s007
                                 + 0.00995498140349101 w_k1_KO_s010
                                 + 0.0360579744185247 w_k1_KO_s011
                                 + 0.0107309820177813 w_k1_KO_s012
                                 + 0.0488136131576069 w_k1_KO_s013
                                 + 0.0390398851033005 w_k1_KO_s014
                                 + 0.0186293298031821 w_k1_KO_s015
                                 + 0.0269530423443126 w_k1_KO_s016
                                 + 0.012703414307267 w_k1_KO_s017
                                 + 0.0466206186073063 w_k1_KO_s100
                                 + 0.0244519245812652 w_k1_KO_s101
                                 + 0.0415530080790782 w_k1_KO_s102
                                 + 0.0249963134168523 w_k1_KO_s103
                                 + 0.036236204639104 w_k1_KO_s104
                                 + 0.0319142751974692 w_k1_KO_s105
                                 + 0.0148263796244817 w_k1_KO_s106
                                 + 0.0329587270159324 w_k1_KO_s107
                                 + 0.0168817929408766 w_k1_KO_s110
                                 + 0.0699545345386907 w_k1_KO_s111
                                 + 0.0476517101131555 w_k1_KO_s112
                                 + 0.039888760563262 w_k1_KO_s113
                                 + 0.0154375453603646 w_k1_KO_s114
                                 + 0.0201261941089856 w_k1_KO_s115
                                 + 0.0452615066305244 w_k1_KO_s116
                                 + 0.0307367768971452 w_k1_KO_s117
                                 + 0.05572775352204 w_k1_MAT_s000
                                 + 0.039442112426685 w_k1_MAT_s001
                                 + 0.0418220641306767 w_k1_MAT_s002
                                 + 0.0344928802582639 w_k1_MAT_s003
                                 + 0.0244384621835224 w_k1_MAT_s004
                                 + 0.0369263740245696 w_k1_MAT_s005
                                 + 0.0190274970950949 w_k1_MAT_s006
                                 + 0.0269123689365172 w_k1_MAT_s007
                                 + 0.0105590912191143 w_k1_MAT_s010
                                 + 0.040159546391222 w_k1_MAT_s011
                                 + 0.0130153269827074 w_k1_MAT_s012
                                 + 0.0506704547828559 w_k1_MAT_s013
                                 + 0.0337393473092722 w_k1_MAT_s014
                                 + 0.0188440673228296 w_k1_MAT_s015
                                 + 0.0303897584635599 w_k1_MAT_s016
                                 + 0.0131954888626864 w_k1_MAT_s017
                                 + 0.0499132380891434 w_k1_MAT_s100
                                 + 0.0272576455736806 w_k1_MAT_s101
                                 + 0.042754347965891 w_k1_MAT_s102
                                 + 0.0247012482916734 w_k1_MAT_s103
                                 + 0.0462070268542814 w_k1_MAT_s104
                                 + 0.034383748264783 w_k1_MAT_s105
                                 + 0.0172229902105103 w_k1_MAT_s106
                                 + 0.0332358332258174 w_k1_MAT_s107
                                 + 0.0155557714626328 w_k1_MAT_s110
                                 + 0.067097514661035 w_k1_MAT_s111
                                 + 0.053349028684492 w_k1_MAT_s112
                                 + 0.0526226163171714 w_k1_MAT_s113
                                 + 0.0223687529220647 w_k1_MAT_s114
                                 + 0.017277569351595 w_k1_MAT_s115
                                 + 0.0381966540052409 w_k1_MAT_s116
                                 + 0.0306253155476126 w_k1_MAT_s117
                                 + 0.0594304913642542 w_k1_MSFT_s000
                                 + 0.0328496072557845 w_k1_MSFT_s001
                                 + 0.0415121129469399 w_k1_MSFT_s002
                                 + 0.028909665681845 w_k1_MSFT_s003
                                 + 0.023057582147378 w_k1_MSFT_s004
                                 + 0.041475621541784 w_k1_MSFT_s005
                                 + 0.0191972480641864 w_k1_MSFT_s006
                                 + 0.0298347259371127 w_k1_MSFT_s007
                                 + 0.0104749129817061 w_k1_MSFT_s010
                                 + 0.0374930694400039 w_k1_MSFT_s011
                                 + 0.0103731098660768 w_k1_MSFT_s012
                                 + 0.0460172114227453 w_k1_MSFT_s013
                                 + 0.0389954939159485 w_k1_MSFT_s014
                                 + 0.0225342356222364 w_k1_MSFT_s015
                                 + 0.0260394407820146 w_k1_MSFT_s016
                                 + 0.0129383090878644 w_k1_MSFT_s017
                                 + 0.0482998418863466 w_k1_MSFT_s100
                                 + 0.024195025707054 w_k1_MSFT_s101
                                 + 0.0418533759200361 w_k1_MSFT_s102
                                 + 0.0252842069736218 w_k1_MSFT_s103
                                 + 0.0394539249197566 w_k1_MSFT_s104
                                 + 0.0362592923312969 w_k1_MSFT_s105
                                 + 0.0150523033878105 w_k1_MSFT_s106
                                 + 0.0310843571464138 w_k1_MSFT_s107
                                 + 0.0162764048039472 w_k1_MSFT_s110
                                 + 0.0634616912134202 w_k1_MSFT_s111
                                 + 0.0487643357886013 w_k1_MSFT_s112
                                 + 0.039743642412056 w_k1_MSFT_s113
                                 + 0.0140167833020802 w_k1_MSFT_s114
                                 + 0.0181729920331143 w_k1_MSFT_s115
                                 + 0.0423062165918721 w_k1_MSFT_s116
                                 + 0.030750441797956 w_k1_MSFT_s117
                                 + 0.0535918502626205 w_k1_PG_s000
                                 + 0.0373565434146618 w_k1_PG_s001
                                 + 0.0392897947711445 w_k1_PG_s002
                                 + 0.028857888342438 w_k1_PG_s003
                                 + 0.0246851021904516 w_k1_PG_s004
                                 + 0.0389018988149788 w_k1_PG_s005
                                 + 0.0193823004223728 w_k1_PG_s006
                                 + 0.0325469592203368 w_k1_PG_s007
                                 + 0.0102918265637559 w_k1_PG_s010
                                 + 0.0365525071569252 w_k1_PG_s011
                                 + 0.0135090323361316 w_k1_PG_s012
                                 + 0.0513249244423004 w_k1_PG_s013
                                 + 0.0401352775515043 w_k1_PG_s014
                                 + 0.0208117598073667 w_k1_PG_s015
                                 + 0.0292202754470061 w_k1_PG_s016
                                 + 0.0132116791931715 w_k1_PG_s017
                                 + 0.0496328235975105 w_k1_PG_s100
                                 + 0.0245330258813794 w_k1_PG_s101
                                 + 0.0411702508087032 w_k1_PG_s102
                                 + 0.0244630719424471 w_k1_PG_s103
                                 + 0.039218798036058 w_k1_PG_s104
                                 + 0.0329397684298083 w_k1_PG_s105
                                 + 0.0159726554812609 w_k1_PG_s106
                                 + 0.0292574407898731 w_k1_PG_s107
                                 + 0.0163021715380573 w_k1_PG_s110
                                 + 0.0616076152299559 w_k1_PG_s111
                                 + 0.0451982768666542 w_k1_PG_s112
                                 + 0.0430242082679075 w_k1_PG_s113
                                 + 0.0155818571958881 w_k1_PG_s114
                                 + 0.019135876360911 w_k1_PG_s115
                                 + 0.0399805290719812 w_k1_PG_s116
                                 + 0.0288783184734058 w_k1_PG_s117 >= 1.11
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
