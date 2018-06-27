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
 w_k1_ADM_s0_return_constraint:  1.19378 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.970116 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.10612 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.914908 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.01592 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.903808 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.846108 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.11442 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.02338 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.898716 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.873431 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.01812 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.09879 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.25733 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.758903 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.92511 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.679136 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.949319 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.784746 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.999169 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.04285 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.925058 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.44577 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.92927 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.09519 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.08666 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.26699 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.43169 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.770508 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.974831 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.07402 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.891388 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.866671 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.42834 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.715247 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.796066 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.33321 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.11809 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.31606 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.933856 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.11339 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.764405 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.08702 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.01106 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.793328 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.22157 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.02809 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.81427 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.733969 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.24364 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.4515 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.06167 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.679089 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.16061 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.651192 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.39535 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.919062 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.24065 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.857516 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.956397 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.31415 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.929464 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.00319 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.912363 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.30996 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.7934 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.635267 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.928737 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.16784 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.796544 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.922243 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.60503 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.518127 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.671971 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.1347 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.992083 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.41515 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.911299 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.14815 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.706261 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.06945 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.98424 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.641031 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.46447 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.09484 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.810014 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.675517 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.15943 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.58372 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.871521 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.597806 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.02118 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.730445 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.47364 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.903408 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.36012 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.00042 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.38475 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.46891 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.17541 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.20597 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.883776 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.38374 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.89624 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.613199 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.840568 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.38386 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.894231 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.788895 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.82197 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.459143 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.728456 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.45003 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.915061 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.23327 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.937792 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.12104 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.78292 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.12973 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.921238 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.697836 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.31365 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.0532 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.826438 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.847303 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.00442 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.37804 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.02991 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.768395 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.933363 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.745894 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.19214 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.940888 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.37617 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.06876 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.18423 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.34932 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.979851 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.10771 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.973794 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.21259 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.35878 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.691746 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.889021 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.06481 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.909485 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.92262 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.41367 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.659536 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.804994 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.26147 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.10371 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.24471 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.957871 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.09991 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.867805 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.849237 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.23685 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.935594 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.700758 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.978082 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.848469 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.518178 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.60414 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.13755 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.649545 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.660941 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.58425 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.17563 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.52185 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.933043 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.683646 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.650234 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.00933 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.846695 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.11923 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.923991 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.693078 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.2871 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.78827 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.73958 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.17779 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.6191 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.62745 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.947334 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.33774 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.550602 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.866579 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.931813 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.684176 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15675 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.930233 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.03232 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.805612 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.07476 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.864452 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.768693 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.34223 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.07329 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.00411 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.864342 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.00073 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.3431 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.04249 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.846606 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.10587 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.948356 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.2723 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.03097 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.2964 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.14892 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.20367 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.19633 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.12078 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.01892 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.938111 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.18778 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.33199 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.794585 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.926172 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.15109 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.15314 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.0324 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.44574 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.84293 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.907536 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.44293 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.26924 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.2549 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.958682 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.08305 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.820043 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.944902 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.05122 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.81876 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.06293 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.9949 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.848568 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.620127 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.33602 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.19994 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.809034 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.670799 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.15552 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.909823 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.40381 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.989567 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.00768 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.817073 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.30039 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.11911 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.12356 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.941511 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.721592 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.32717 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.64642 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.791703 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.984403 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.42247 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.76355 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.888513 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.35566 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.550854 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.920062 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.27861 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.889466 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.3482 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.998295 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.10636 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.744745 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.850193 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.12022 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.13204 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.749677 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.12076 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.872337 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.506368 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.84522 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.2814 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.832799 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.738624 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.53214 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.05066 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.57431 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.01988 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.935759 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.656005 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.00722 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.11043 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.841654 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.805785 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.714981 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.42896 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 2.00702 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.739422 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.02278 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.54643 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.683534 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.17681 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.4633 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.491647 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.693334 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.11699 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.653887 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16568 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.971032 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.10079 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.845257 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.05111 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.895839 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.715118 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.33439 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.00069 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.903146 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.903824 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.00132 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.42298 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.910426 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.852012 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.983095 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.829302 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.09548 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.889103 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.22024 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.07188 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.2236 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.37537 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.02751 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.05828 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.965318 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.18185 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.32911 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.792412 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.926205 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.15603 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.00753 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.883878 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.3717 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.731694 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.882655 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.34208 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.05964 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.19156 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.947216 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.08795 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.856685 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.992747 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.968696 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.82635 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.0453 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.925071 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.851027 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.809227 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.09114 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.10671 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.86872 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.804728 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.0271 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.842185 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.13709 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.04124 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.03724 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.917733 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.10585 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.2114 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.03647 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.03493 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.90047 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.18464 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.38491 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.758095 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.937726 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.21219 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.936887 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.974862 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.31093 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.727399 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.916458 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.16623 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.977309 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.119999984576008 z_k1_s0000
                                 - 0.205000545867374 z_k1_s0001
                                 - 0.423999921408559 z_k1_s0002
                                 - 0.297000341295637 z_k1_s0003
                                 - 0.120999978424141 z_k1_s0004
                                 - 0.183000319808428 z_k1_s0005
                                 - 0.252000618129647 z_k1_s0006
                                 - 0.205000545867374 z_k1_s0007
                                 - 0.217000401832142 z_k1_s0010
                                 - 0.0919999928137337 z_k1_s0011
                                 - 0.134999623020372 z_k1_s0012
                                 - 0.214999355032472 z_k1_s0013
                                 - 0.157000301823248 z_k1_s0014
                                 - 0.0679999528101761 z_k1_s0015
                                 - 0.0949999978141784 z_k1_s0016
                                 - 0.224999773033713 z_k1_s0017
                                 - 0.359000635796494 z_k1_s0020
                                 - 0.38500046007168 z_k1_s0021
                                 - 0.221000516842038 z_k1_s0022
                                 - 0.440999803210131 z_k1_s0023
                                 - 0.422000550240714 z_k1_s0024
                                 - 0.603999320167013 z_k1_s0025
                                 - 0.633002835864031 z_k1_s0026
                                 - 0.956002199361882 z_k1_s0027
                                 - 0.316999084470688 z_k1_s0030
                                 - 0.276000627865716 z_k1_s0031
                                 - 0.239999809470967 z_k1_s0032
                                 - 0.209999741224694 z_k1_s0033
                                 - 0.215999754873948 z_k1_s0034
                                 - 0.349999159541589 z_k1_s0035
                                 - 0.59800054204325 z_k1_s0036
                                 - 0.24799892750426 z_k1_s0037
                                 - 0.406000694834411 z_k1_s0040
                                 - 0.130000206057781 z_k1_s0041
                                 - 0.28800137989921 z_k1_s0042
                                 - 0.365999947430557 z_k1_s0043
                                 - 0.279000433681682 z_k1_s0044
                                 - 0.128000168687589 z_k1_s0045
                                 - 0.129000298522922 z_k1_s0046
                                 - 0.49700161712652 z_k1_s0047
                                 - 0.413999189576252 z_k1_s0050
                                 - 0.184000138967254 z_k1_s0051
                                 - 0.694998826935989 z_k1_s0052
                                 - 0.76200093507528 z_k1_s0053
                                 - 0.787999680681443 z_k1_s0054
                                 - 0.482001877777356 z_k1_s0055
                                 - 0.296999933548212 z_k1_s0056
                                 - 0.461998444541574 z_k1_s0057
                                 - 0.182000029642544 z_k1_s0060
                                 - 0.484999606852393 z_k1_s0061
                                 - 0.167000021894533 z_k1_s0062
                                 - 0.490000896768375 z_k1_s0063
                                 - 0.16500008522813 z_k1_s0064
                                 - 0.117000060434492 z_k1_s0065
                                 - 0.150000077480118 z_k1_s0066
                                 - 0.17499996166014 z_k1_s0067
                                 - 0.12100003713087 z_k1_s0070
                                 - 0.408999657905105 z_k1_s0071
                                 - 0.160000062652392 z_k1_s0072
                                 - 0.281999679924779 z_k1_s0073
                                 - 0.170000784068268 z_k1_s0074
                                 - 0.132000010688217 z_k1_s0075
                                 - 0.137000043396109 z_k1_s0076
                                 - 0.228999792421194 z_k1_s0077
                                 - 0.224999830009724 z_k1_s0100
                                 - 0.0709998389588556 z_k1_s0101
                                 - 0.118000125649081 z_k1_s0102
                                 - 0.148999278829974 z_k1_s0103
                                 - 0.0679998949252768 z_k1_s0104
                                 - 0.0739998724922414 z_k1_s0105
                                 - 0.0979996037604855 z_k1_s0106
                                 - 0.0919995366937138 z_k1_s0107
                                 - 0.121000484888038 z_k1_s0110
                                 - 0.0620000279919905 z_k1_s0111
                                 - 0.131000043208867 z_k1_s0112
                                 - 0.0900000756916353 z_k1_s0113
                                 - 0.187999798039933 z_k1_s0114
                                 - 0.186000083975971 z_k1_s0115
                                 - 0.138999887465733 z_k1_s0116
                                 - 0.0710000256811438 z_k1_s0117
                                 - 0.0230000365403606 z_k1_s0120
                                 - 0.0610000511417666 z_k1_s0121
                                 - 0.053999950059801 z_k1_s0122
                                 - 0.0379999783013589 z_k1_s0123
                                 - 0.0750002533056978 z_k1_s0124
                                 - 0.0610000511417666 z_k1_s0125
                                 - 0.0240000302377574 z_k1_s0126
                                 - 0.0270000476299947 z_k1_s0127
                                 - 0.21199991529704 z_k1_s0130
                                 - 0.066000047490585 z_k1_s0131
                                 - 0.165000383226452 z_k1_s0132
                                 - 0.0740000083621729 z_k1_s0133
                                 - 0.12800000874538 z_k1_s0134
                                 - 0.118999603114862 z_k1_s0135
                                 - 0.124999520868556 z_k1_s0136
                                 - 0.168999623062276 z_k1_s0137
                                 - 0.115999741712547 z_k1_s0140
                                 - 0.215999028596598 z_k1_s0141
                                 - 0.152999788605126 z_k1_s0142
                                 - 0.276999448532439 z_k1_s0143
                                 - 0.269999172245101 z_k1_s0144
                                 - 0.0849998653806543 z_k1_s0145
                                 - 0.247999629054108 z_k1_s0146
                                 - 0.180999193157322 z_k1_s0147
                                 - 0.0620001227557403 z_k1_s0150
                                 - 0.0710001129621997 z_k1_s0151
                                 - 0.0340000438466801 z_k1_s0152
                                 - 0.0310000185778407 z_k1_s0153
                                 - 0.0890000933751185 z_k1_s0154
                                 - 0.0850000026166265 z_k1_s0155
                                 - 0.0350000237362736 z_k1_s0156
                                 - 0.0210000056817583 z_k1_s0157
                                 - 0.096000104968637 z_k1_s0160
                                 - 0.102000111529177 z_k1_s0161
                                 - 0.090999786168889 z_k1_s0162
                                 - 0.13299936209372 z_k1_s0163
                                 - 0.130999986572137 z_k1_s0164
                                 - 0.153999855054557 z_k1_s0165
                                 - 0.120999348972641 z_k1_s0166
                                 - 0.111999809130778 z_k1_s0167
                                 - 0.065999819444855 z_k1_s0170
                                 - 0.0839998102933593 z_k1_s0171
                                 - 0.0629998797699686 z_k1_s0172
                                 - 0.105000093615936 z_k1_s0173
                                 - 0.0679998380279769 z_k1_s0174
                                 - 0.184999954942847 z_k1_s0175
                                 - 0.226999815989629 z_k1_s0176
                                 - 0.0839998102933593 z_k1_s0177
                                 - 0.372000110435662 z_k1_s1000
                                 - 0.321000128730796 z_k1_s1001
                                 - 0.279001119327472 z_k1_s1002
                                 - 0.137000105313179 z_k1_s1003
                                 - 0.180000167117335 z_k1_s1004
                                 - 0.313999602829759 z_k1_s1005
                                 - 0.29200091142857 z_k1_s1006
                                 - 0.178000135317122 z_k1_s1007
                                 - 0.643999185011651 z_k1_s1010
                                 - 0.231999978958293 z_k1_s1011
                                 - 0.460999814078301 z_k1_s1012
                                 - 0.366001353867257 z_k1_s1013
                                 - 0.200999972763002 z_k1_s1014
                                 - 0.300001323825666 z_k1_s1015
                                 - 0.196999955111993 z_k1_s1016
                                 - 0.211000016890524 z_k1_s1017
                                 - 0.411999113635642 z_k1_s1020
                                 - 0.196000301106949 z_k1_s1021
                                 - 0.638001100078854 z_k1_s1022
                                 - 0.438999098776852 z_k1_s1023
                                 - 0.295000127891279 z_k1_s1024
                                 - 0.631001763561377 z_k1_s1025
                                 - 0.574999947414923 z_k1_s1026
                                 - 0.376001508116238 z_k1_s1027
                                 - 0.495000668440284 z_k1_s1030
                                 - 0.58800336687903 z_k1_s1031
                                 - 0.326000329564975 z_k1_s1032
                                 - 1.00200063166486 z_k1_s1033
                                 - 0.362000320720558 z_k1_s1034
                                 - 0.790003609863512 z_k1_s1035
                                 - 0.59100029372139 z_k1_s1036
                                 - 1.11300279898073 z_k1_s1037
                                 - 0.701996246203927 z_k1_s1040
                                 - 0.522998638913617 z_k1_s1041
                                 - 0.386999075981381 z_k1_s1042
                                 - 1.09299476628466 z_k1_s1043
                                 - 1.01599578889611 z_k1_s1044
                                 - 0.423998847588216 z_k1_s1045
                                 - 0.756999770042362 z_k1_s1046
                                 - 1.05399497482889 z_k1_s1047
                                 - 0.209000098134451 z_k1_s1050
                                 - 0.439999482598734 z_k1_s1051
                                 - 0.275000020524261 z_k1_s1052
                                 - 0.193999865664001 z_k1_s1053
                                 - 0.663001471352524 z_k1_s1054
                                 - 0.342000004265441 z_k1_s1055
                                 - 0.946000263187486 z_k1_s1056
                                 - 0.369998627003339 z_k1_s1057
                                 - 0.546999444975199 z_k1_s1060
                                 - 1.19899583957396 z_k1_s1061
                                 - 0.615999492439955 z_k1_s1062
                                 - 0.99499890402262 z_k1_s1063
                                 - 1.1190016736666 z_k1_s1064
                                 - 0.669999570673198 z_k1_s1065
                                 - 1.74899471139606 z_k1_s1066
                                 - 2.55499381410183 z_k1_s1067
                                 - 0.438001148952142 z_k1_s1070
                                 - 0.408001191310527 z_k1_s1071
                                 - 0.906003185748301 z_k1_s1072
                                 - 1.10400141785912 z_k1_s1073
                                 - 0.351000713670018 z_k1_s1074
                                 - 0.527001519396881 z_k1_s1075
                                 - 0.384001039156755 z_k1_s1076
                                 - 0.533001324884724 z_k1_s1077
                                 - 0.751999152840488 z_k1_s1100
                                 - 0.515000647092623 z_k1_s1101
                                 - 0.974001345569455 z_k1_s1102
                                 - 0.60100202175385 z_k1_s1103
                                 - 0.817999210137773 z_k1_s1104
                                 - 0.480000266707396 z_k1_s1105
                                 - 0.992998362061524 z_k1_s1106
                                 - 0.367000468607759 z_k1_s1107
                                 - 0.219999740580948 z_k1_s1110
                                 - 0.165999727323892 z_k1_s1111
                                 - 0.313000047973339 z_k1_s1112
                                 - 0.193999767190477 z_k1_s1113
                                 - 0.352999150498093 z_k1_s1114
                                 - 0.48200020906103 z_k1_s1115
                                 - 0.319999501190604 z_k1_s1116
                                 - 0.178999825369003 z_k1_s1117
                                 - 1.67599880570741 z_k1_s1120
                                 - 0.686000619643025 z_k1_s1121
                                 - 0.828002901393999 z_k1_s1122
                                 - 0.933004138237829 z_k1_s1123
                                 - 0.684000519950727 z_k1_s1124
                                 - 1.62499706295418 z_k1_s1125
                                 - 0.335999960898471 z_k1_s1126
                                 - 1.22600035695119 z_k1_s1127
                                 - 1.00200542507343 z_k1_s1130
                                 - 0.353001129673617 z_k1_s1131
                                 - 0.65100321684004 z_k1_s1132
                                 - 0.43500317608567 z_k1_s1133
                                 - 0.429999446462482 z_k1_s1134
                                 - 0.7310029145998 z_k1_s1135
                                 - 0.389001422067018 z_k1_s1136
                                 - 0.293000927952288 z_k1_s1137
                                 - 0.242999988952972 z_k1_s1140
                                 - 0.501999953867458 z_k1_s1141
                                 - 0.363000123364303 z_k1_s1142
                                 - 0.209999920519858 z_k1_s1143
                                 - 0.638000693640248 z_k1_s1144
                                 - 0.667000296033025 z_k1_s1145
                                 - 0.303000056158638 z_k1_s1146
                                 - 0.220999943330896 z_k1_s1147
                                 - 0.52999825141572 z_k1_s1150
                                 - 1.26299865684526 z_k1_s1151
                                 - 0.373998998637853 z_k1_s1152
                                 - 0.618995466237357 z_k1_s1153
                                 - 0.618995466237357 z_k1_s1154
                                 - 0.542998042422705 z_k1_s1155
                                 - 0.376998634386688 z_k1_s1156
                                 - 1.54399368792086 z_k1_s1157
                                 - 0.498998355061573 z_k1_s1160
                                 - 0.454998495054159 z_k1_s1161
                                 - 0.437998446010763 z_k1_s1162
                                 - 0.417998417971365 z_k1_s1163
                                 - 0.368998424904563 z_k1_s1164
                                 - 1.26199788696024 z_k1_s1165
                                 - 1.15599526778049 z_k1_s1166
                                 - 0.444998228935384 z_k1_s1167
                                 - 0.347998670725319 z_k1_s1170
                                 - 0.304999853733089 z_k1_s1171
                                 - 0.177999837952449 z_k1_s1172
                                 - 0.310000174220419 z_k1_s1173
                                 - 0.223000014139238 z_k1_s1174
                                 - 0.260999003043989 z_k1_s1175
                                 - 0.23699992670406 z_k1_s1176
                                 - 0.599999066491518 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.26055 w_k1_ADM_s000
                                 + 1.3906 w_k1_AXP_s000 + 1.24065 w_k1_DIS_s000
                                 + 1.19429 w_k1_FDX_s000
                                 + 0.771294 w_k1_IBM_s000
                                 + 1.03899 w_k1_INTC_s000
                                 + 1.05409 w_k1_KO_s000 + 1.01266 w_k1_MAT_s000
                                 + 1.20368 w_k1_MSFT_s000
                                 + 1.04364 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.872351 w_k1_ADM_s000
                                 + 0.983203 w_k1_AXP_s000
                                 + 0.986644 w_k1_DIS_s000
                                 + 0.935013 w_k1_FDX_s000
                                 + 1.64844 w_k1_IBM_s000
                                 + 1.01076 w_k1_INTC_s000
                                 + 1.24052 w_k1_KO_s000 + 1.13638 w_k1_MAT_s000
                                 + 1.01289 w_k1_MSFT_s000
                                 + 0.953539 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.633409 w_k1_ADM_s000
                                 + 0.651716 w_k1_AXP_s000
                                 + 0.653437 w_k1_DIS_s000
                                 + 0.763006 w_k1_FDX_s000
                                 + 0.804255 w_k1_IBM_s000
                                 + 0.92424 w_k1_INTC_s000
                                 + 0.799831 w_k1_KO_s000 + 0.4888 w_k1_MAT_s000
                                 + 0.868653 w_k1_MSFT_s000
                                 + 0.741695 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.03843 w_k1_ADM_s000
                                 + 0.84497 w_k1_AXP_s000
                                 + 1.14855 w_k1_DIS_s000
                                 + 1.01293 w_k1_FDX_s000
                                 + 1.09663 w_k1_IBM_s000
                                 + 1.09633 w_k1_INTC_s000
                                 + 1.08756 w_k1_KO_s000
                                 + 0.789719 w_k1_MAT_s000
                                 + 1.23487 w_k1_MSFT_s000
                                 + 0.925654 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.30857 w_k1_ADM_s000
                                 + 1.25173 w_k1_AXP_s000 + 1.2763 w_k1_DIS_s000
                                 + 1.06946 w_k1_FDX_s000
                                 + 1.43281 w_k1_IBM_s000
                                 + 0.971878 w_k1_INTC_s000
                                 + 1.38389 w_k1_KO_s000 + 1.56372 w_k1_MAT_s000
                                 + 0.975686 w_k1_MSFT_s000
                                 + 1.07681 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.07626 w_k1_ADM_s000
                                 + 1.37636 w_k1_AXP_s000
                                 + 1.49352 w_k1_DIS_s000
                                 + 1.29866 w_k1_FDX_s000
                                 + 1.24358 w_k1_IBM_s000
                                 + 1.21607 w_k1_INTC_s000
                                 + 1.23238 w_k1_KO_s000 + 1.30205 w_k1_MAT_s000
                                 + 1.29214 w_k1_MSFT_s000
                                 + 1.20843 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.992793 w_k1_ADM_s000
                                 + 0.832843 w_k1_AXP_s000
                                 + 0.940816 w_k1_DIS_s000
                                 + 1.02485 w_k1_FDX_s000
                                 + 0.752277 w_k1_IBM_s000
                                 + 1.05661 w_k1_INTC_s000
                                 + 0.955429 w_k1_KO_s000
                                 + 0.794709 w_k1_MAT_s000
                                 + 1.07678 w_k1_MSFT_s000
                                 + 0.983721 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.04434 w_k1_ADM_s000
                                 + 1.25993 w_k1_AXP_s000
                                 + 0.973333 w_k1_DIS_s000
                                 + 1.07282 w_k1_FDX_s000
                                 + 0.958531 w_k1_IBM_s000
                                 + 1.06388 w_k1_INTC_s000
                                 + 0.893075 w_k1_KO_s000
                                 + 1.17583 w_k1_MAT_s000
                                 + 1.01133 w_k1_MSFT_s000
                                 + 1.01312 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.894443 w_k1_ADM_s001
                                 + 0.933897 w_k1_AXP_s001
                                 + 1.00849 w_k1_DIS_s001
                                 + 0.956051 w_k1_FDX_s001
                                 + 0.994959 w_k1_IBM_s001
                                 + 0.971885 w_k1_INTC_s001 + 1.057 w_k1_KO_s001
                                 + 1.12855 w_k1_MAT_s001
                                 + 0.920076 w_k1_MSFT_s001
                                 + 0.986445 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.47046 w_k1_ADM_s001
                                 + 1.22196 w_k1_AXP_s001
                                 + 1.39876 w_k1_DIS_s001
                                 + 1.38472 w_k1_FDX_s001
                                 + 0.873735 w_k1_IBM_s001
                                 + 1.13035 w_k1_INTC_s001
                                 + 0.995879 w_k1_KO_s001
                                 + 0.781948 w_k1_MAT_s001
                                 + 1.18244 w_k1_MSFT_s001
                                 + 0.982668 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.23529 w_k1_ADM_s001
                                 + 1.47417 w_k1_AXP_s001 + 1.1065 w_k1_DIS_s001
                                 + 1.14537 w_k1_FDX_s001
                                 + 1.28315 w_k1_IBM_s001
                                 + 1.29643 w_k1_INTC_s001
                                 + 1.10991 w_k1_KO_s001
                                 + 0.956294 w_k1_MAT_s001
                                 + 1.12387 w_k1_MSFT_s001
                                 + 0.952856 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.10278 w_k1_ADM_s001
                                 + 0.767458 w_k1_AXP_s001
                                 + 1.34818 w_k1_DIS_s001
                                 + 1.08264 w_k1_FDX_s001
                                 + 1.09537 w_k1_IBM_s001
                                 + 1.00406 w_k1_INTC_s001 + 1.2061 w_k1_KO_s001
                                 + 1.1936 w_k1_MAT_s001
                                 + 1.23529 w_k1_MSFT_s001
                                 + 1.05135 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.13278 w_k1_ADM_s001
                                 + 1.43613 w_k1_AXP_s001
                                 + 1.47011 w_k1_DIS_s001
                                 + 1.38692 w_k1_FDX_s001
                                 + 1.03479 w_k1_IBM_s001
                                 + 0.901781 w_k1_INTC_s001
                                 + 1.09908 w_k1_KO_s001 + 1.18344 w_k1_MAT_s001
                                 + 1.06243 w_k1_MSFT_s001
                                 + 1.00379 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.16194 w_k1_ADM_s001
                                 + 0.895677 w_k1_AXP_s001
                                 + 0.830733 w_k1_DIS_s001
                                 + 1.08741 w_k1_FDX_s001
                                 + 0.656368 w_k1_IBM_s001
                                 + 0.814622 w_k1_INTC_s001
                                 + 0.771112 w_k1_KO_s001
                                 + 0.743749 w_k1_MAT_s001
                                 + 0.893843 w_k1_MSFT_s001
                                 + 1.0097 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.854213 w_k1_ADM_s001
                                 + 0.469389 w_k1_AXP_s001
                                 + 0.788368 w_k1_DIS_s001
                                 + 0.823678 w_k1_FDX_s001
                                 + 0.754748 w_k1_IBM_s001
                                 + 0.727201 w_k1_INTC_s001
                                 + 0.768233 w_k1_KO_s001
                                 + 0.387512 w_k1_MAT_s001
                                 + 0.851066 w_k1_MSFT_s001
                                 + 0.853116 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.937529 w_k1_ADM_s001
                                 + 0.680782 w_k1_AXP_s001
                                 + 0.66792 w_k1_DIS_s001
                                 + 0.798763 w_k1_FDX_s001
                                 + 0.797465 w_k1_IBM_s001
                                 + 0.788143 w_k1_INTC_s001
                                 + 0.741392 w_k1_KO_s001
                                 + 0.619486 w_k1_MAT_s001
                                 + 0.810294 w_k1_MSFT_s001
                                 + 0.766987 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.02813 w_k1_ADM_s002
                                 + 1.25211 w_k1_AXP_s002
                                 + 1.43444 w_k1_DIS_s002
                                 + 1.31422 w_k1_FDX_s002
                                 + 0.836213 w_k1_IBM_s002
                                 + 1.43486 w_k1_INTC_s002
                                 + 1.34038 w_k1_KO_s002 + 1.08198 w_k1_MAT_s002
                                 + 1.58922 w_k1_MSFT_s002
                                 + 1.15337 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.904917 w_k1_ADM_s002
                                 + 1.39744 w_k1_AXP_s002 + 1.4532 w_k1_DIS_s002
                                 + 1.18669 w_k1_FDX_s002
                                 + 1.22575 w_k1_IBM_s002
                                 + 1.39918 w_k1_INTC_s002 + 1.4355 w_k1_KO_s002
                                 + 1.22085 w_k1_MAT_s002
                                 + 1.45233 w_k1_MSFT_s002
                                 + 1.29968 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.12455 w_k1_ADM_s002
                                 + 1.21342 w_k1_AXP_s002
                                 + 1.35138 w_k1_DIS_s002
                                 + 1.07049 w_k1_FDX_s002
                                 + 1.66725 w_k1_IBM_s002
                                 + 1.13067 w_k1_INTC_s002
                                 + 1.34408 w_k1_KO_s002 + 1.1961 w_k1_MAT_s002
                                 + 1.10052 w_k1_MSFT_s002
                                 + 1.19095 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.918721 w_k1_ADM_s002
                                 + 1.37811 w_k1_AXP_s002
                                 + 0.921125 w_k1_DIS_s002
                                 + 1.03892 w_k1_FDX_s002
                                 + 0.903666 w_k1_IBM_s002
                                 + 0.988026 w_k1_INTC_s002
                                 + 0.949235 w_k1_KO_s002
                                 + 1.13449 w_k1_MAT_s002
                                 + 0.981966 w_k1_MSFT_s002
                                 + 1.02208 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.882287 w_k1_ADM_s002
                                 + 1.01364 w_k1_AXP_s002
                                 + 0.751092 w_k1_DIS_s002
                                 + 0.7943 w_k1_FDX_s002
                                 + 0.978721 w_k1_IBM_s002
                                 + 0.938932 w_k1_INTC_s002
                                 + 0.901008 w_k1_KO_s002
                                 + 0.887944 w_k1_MAT_s002
                                 + 0.971879 w_k1_MSFT_s002
                                 + 0.935213 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.735154 w_k1_ADM_s002
                                 + 0.683528 w_k1_AXP_s002
                                 + 0.716685 w_k1_DIS_s002
                                 + 0.788183 w_k1_FDX_s002
                                 + 0.743575 w_k1_IBM_s002
                                 + 0.963621 w_k1_INTC_s002
                                 + 0.756075 w_k1_KO_s002
                                 + 0.402302 w_k1_MAT_s002
                                 + 0.929457 w_k1_MSFT_s002
                                 + 0.923564 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.904937 w_k1_ADM_s002
                                 + 0.772132 w_k1_AXP_s002
                                 + 0.926464 w_k1_DIS_s002
                                 + 0.822219 w_k1_FDX_s002
                                 + 1.22548 w_k1_IBM_s002
                                 + 0.899736 w_k1_INTC_s002
                                 + 1.08288 w_k1_KO_s002 + 0.95686 w_k1_MAT_s002
                                 + 0.960907 w_k1_MSFT_s002
                                 + 1.0272 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.11767 w_k1_ADM_s002
                                 + 1.1116 w_k1_AXP_s002 + 1.10496 w_k1_DIS_s002
                                 + 1.01884 w_k1_FDX_s002
                                 + 0.910394 w_k1_IBM_s002
                                 + 1.10688 w_k1_INTC_s002
                                 + 1.00774 w_k1_KO_s002
                                 + 0.743811 w_k1_MAT_s002
                                 + 1.12122 w_k1_MSFT_s002
                                 + 0.980475 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.965887 w_k1_ADM_s003
                                 + 1.0783 w_k1_AXP_s003
                                 + 0.964424 w_k1_DIS_s003
                                 + 0.991652 w_k1_FDX_s003
                                 + 1.33813 w_k1_IBM_s003
                                 + 0.994923 w_k1_INTC_s003
                                 + 0.985223 w_k1_KO_s003
                                 + 1.07217 w_k1_MAT_s003
                                 + 0.907946 w_k1_MSFT_s003
                                 + 1.07642 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.877981 w_k1_ADM_s003
                                 + 0.831295 w_k1_AXP_s003
                                 + 0.80193 w_k1_DIS_s003
                                 + 0.930139 w_k1_FDX_s003
                                 + 0.710363 w_k1_IBM_s003
                                 + 0.779747 w_k1_INTC_s003
                                 + 0.772847 w_k1_KO_s003
                                 + 0.859487 w_k1_MAT_s003
                                 + 0.960711 w_k1_MSFT_s003
                                 + 0.91773 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.826995 w_k1_ADM_s003
                                 + 0.621759 w_k1_AXP_s003
                                 + 0.614393 w_k1_DIS_s003
                                 + 0.725053 w_k1_FDX_s003
                                 + 0.76464 w_k1_IBM_s003
                                 + 0.725675 w_k1_INTC_s003
                                 + 0.702374 w_k1_KO_s003
                                 + 0.666512 w_k1_MAT_s003
                                 + 0.733037 w_k1_MSFT_s003
                                 + 0.764433 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.28886 w_k1_ADM_s003
                                 + 1.37851 w_k1_AXP_s003
                                 + 1.68466 w_k1_DIS_s003 + 1.3215 w_k1_FDX_s003
                                 + 1.13162 w_k1_IBM_s003
                                 + 1.1178 w_k1_INTC_s003 + 1.46162 w_k1_KO_s003
                                 + 1.01348 w_k1_MAT_s003
                                 + 1.1737 w_k1_MSFT_s003 + 1.16772 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.800667 w_k1_ADM_s003
                                 + 0.831056 w_k1_AXP_s003
                                 + 0.896727 w_k1_DIS_s003
                                 + 0.967581 w_k1_FDX_s003
                                 + 0.904897 w_k1_IBM_s003
                                 + 0.78007 w_k1_INTC_s003
                                 + 0.787559 w_k1_KO_s003
                                 + 0.681314 w_k1_MAT_s003
                                 + 0.853734 w_k1_MSFT_s003
                                 + 0.842293 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.04831 w_k1_ADM_s003
                                 + 0.811665 w_k1_AXP_s003
                                 + 0.939347 w_k1_DIS_s003
                                 + 0.955122 w_k1_FDX_s003
                                 + 1.10391 w_k1_IBM_s003
                                 + 0.915858 w_k1_INTC_s003
                                 + 1.04438 w_k1_KO_s003
                                 + 0.945892 w_k1_MAT_s003
                                 + 1.04741 w_k1_MSFT_s003
                                 + 1.03767 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.930836 w_k1_ADM_s003
                                 + 1.26003 w_k1_AXP_s003
                                 + 1.10488 w_k1_DIS_s003
                                 + 1.12356 w_k1_FDX_s003
                                 + 0.91628 w_k1_IBM_s003
                                 + 1.08163 w_k1_INTC_s003
                                 + 0.976621 w_k1_KO_s003
                                 + 1.12827 w_k1_MAT_s003
                                 + 1.07696 w_k1_MSFT_s003
                                 + 0.990858 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.97094 w_k1_ADM_s003
                                 + 1.19169 w_k1_AXP_s003
                                 + 1.15069 w_k1_DIS_s003
                                 + 1.06467 w_k1_FDX_s003
                                 + 1.67864 w_k1_IBM_s003
                                 + 1.03073 w_k1_INTC_s003
                                 + 1.28115 w_k1_KO_s003 + 1.57696 w_k1_MAT_s003
                                 + 0.991323 w_k1_MSFT_s003
                                 + 1.11345 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.13661 w_k1_ADM_s004
                                 + 1.06242 w_k1_AXP_s004
                                 + 1.43589 w_k1_DIS_s004
                                 + 1.08923 w_k1_FDX_s004
                                 + 1.44053 w_k1_IBM_s004
                                 + 1.19115 w_k1_INTC_s004
                                 + 1.39108 w_k1_KO_s004 + 1.26373 w_k1_MAT_s004
                                 + 1.06946 w_k1_MSFT_s004
                                 + 1.09764 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.8325 w_k1_ADM_s004
                                 + 1.06509 w_k1_AXP_s004
                                 + 0.698164 w_k1_DIS_s004
                                 + 0.844973 w_k1_FDX_s004
                                 + 1.21134 w_k1_IBM_s004
                                 + 0.899562 w_k1_INTC_s004
                                 + 0.866414 w_k1_KO_s004 + 1.738 w_k1_MAT_s004
                                 + 0.7108 w_k1_MSFT_s004
                                 + 0.860942 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.05969 w_k1_ADM_s004
                                 + 1.03542 w_k1_AXP_s004
                                 + 1.12904 w_k1_DIS_s004
                                 + 1.08705 w_k1_FDX_s004
                                 + 1.04164 w_k1_IBM_s004
                                 + 1.05694 w_k1_INTC_s004
                                 + 1.08202 w_k1_KO_s004
                                 + 0.701754 w_k1_MAT_s004
                                 + 0.971377 w_k1_MSFT_s004
                                 + 0.99932 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.26922 w_k1_ADM_s004
                                 + 1.23634 w_k1_AXP_s004 + 1.4121 w_k1_DIS_s004
                                 + 1.36118 w_k1_FDX_s004
                                 + 0.81673 w_k1_IBM_s004
                                 + 1.3789 w_k1_INTC_s004 + 1.19594 w_k1_KO_s004
                                 + 1.18932 w_k1_MAT_s004
                                 + 1.21188 w_k1_MSFT_s004
                                 + 1.16676 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.845478 w_k1_ADM_s004
                                 + 0.643062 w_k1_AXP_s004
                                 + 0.538024 w_k1_DIS_s004
                                 + 0.758855 w_k1_FDX_s004
                                 + 0.698089 w_k1_IBM_s004
                                 + 0.904162 w_k1_INTC_s004
                                 + 0.835581 w_k1_KO_s004
                                 + 0.698889 w_k1_MAT_s004
                                 + 0.795705 w_k1_MSFT_s004
                                 + 0.759309 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.972541 w_k1_ADM_s004
                                 + 0.73271 w_k1_AXP_s004
                                 + 0.716908 w_k1_DIS_s004
                                 + 0.759201 w_k1_FDX_s004
                                 + 1.22464 w_k1_IBM_s004
                                 + 0.784628 w_k1_INTC_s004
                                 + 0.923422 w_k1_KO_s004
                                 + 1.27334 w_k1_MAT_s004
                                 + 0.754748 w_k1_MSFT_s004
                                 + 0.90029 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.701315 w_k1_ADM_s004
                                 + 0.696884 w_k1_AXP_s004
                                 + 0.818581 w_k1_DIS_s004
                                 + 0.909191 w_k1_FDX_s004
                                 + 0.691679 w_k1_IBM_s004
                                 + 0.833779 w_k1_INTC_s004
                                 + 0.812388 w_k1_KO_s004
                                 + 0.62298 w_k1_MAT_s004
                                 + 0.839557 w_k1_MSFT_s004
                                 + 0.910393 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.88115 w_k1_ADM_s004
                                 + 0.849972 w_k1_AXP_s004
                                 + 0.998977 w_k1_DIS_s004
                                 + 0.920106 w_k1_FDX_s004
                                 + 0.962688 w_k1_IBM_s004
                                 + 0.929515 w_k1_INTC_s004
                                 + 1.15303 w_k1_KO_s004
                                 + 0.997989 w_k1_MAT_s004
                                 + 0.876569 w_k1_MSFT_s004
                                 + 0.881305 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.760739 w_k1_ADM_s005
                                 + 0.787384 w_k1_AXP_s005
                                 + 0.723128 w_k1_DIS_s005
                                 + 0.797112 w_k1_FDX_s005
                                 + 0.624384 w_k1_IBM_s005
                                 + 0.774298 w_k1_INTC_s005
                                 + 0.701174 w_k1_KO_s005
                                 + 0.455884 w_k1_MAT_s005
                                 + 0.909451 w_k1_MSFT_s005
                                 + 0.794014 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.675559 w_k1_ADM_s005
                                 + 0.586318 w_k1_AXP_s005
                                 + 0.492426 w_k1_DIS_s005
                                 + 0.608218 w_k1_FDX_s005
                                 + 0.760856 w_k1_IBM_s005
                                 + 0.760906 w_k1_INTC_s005
                                 + 0.680785 w_k1_KO_s005
                                 + 0.578825 w_k1_MAT_s005
                                 + 0.685927 w_k1_MSFT_s005
                                 + 0.749023 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.17198 w_k1_ADM_s005
                                 + 1.01578 w_k1_AXP_s005
                                 + 1.30676 w_k1_DIS_s005
                                 + 1.13851 w_k1_FDX_s005
                                 + 0.905844 w_k1_IBM_s005
                                 + 1.03941 w_k1_INTC_s005
                                 + 0.997625 w_k1_KO_s005
                                 + 0.909579 w_k1_MAT_s005
                                 + 1.20755 w_k1_MSFT_s005
                                 + 1.18417 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.13393 w_k1_ADM_s005
                                 + 1.08362 w_k1_AXP_s005
                                 + 0.988657 w_k1_DIS_s005
                                 + 1.11585 w_k1_FDX_s005
                                 + 0.805045 w_k1_IBM_s005
                                 + 1.19002 w_k1_INTC_s005
                                 + 0.886415 w_k1_KO_s005
                                 + 0.88013 w_k1_MAT_s005
                                 + 1.12743 w_k1_MSFT_s005
                                 + 0.911726 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.736859 w_k1_ADM_s005
                                 + 0.747956 w_k1_AXP_s005
                                 + 0.765774 w_k1_DIS_s005
                                 + 0.84693 w_k1_FDX_s005
                                 + 1.03492 w_k1_IBM_s005
                                 + 0.91926 w_k1_INTC_s005
                                 + 0.894899 w_k1_KO_s005
                                 + 0.860944 w_k1_MAT_s005
                                 + 0.933574 w_k1_MSFT_s005
                                 + 0.841225 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.01387 w_k1_ADM_s005
                                 + 1.44176 w_k1_AXP_s005
                                 + 1.40173 w_k1_DIS_s005
                                 + 1.22123 w_k1_FDX_s005
                                 + 1.23342 w_k1_IBM_s005
                                 + 1.38789 w_k1_INTC_s005 + 1.1632 w_k1_KO_s005
                                 + 1.40462 w_k1_MAT_s005
                                 + 1.15245 w_k1_MSFT_s005
                                 + 1.09049 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.45324 w_k1_ADM_s005
                                 + 1.58482 w_k1_AXP_s005
                                 + 2.19139 w_k1_DIS_s005
                                 + 1.49037 w_k1_FDX_s005
                                 + 1.50134 w_k1_IBM_s005
                                 + 1.33571 w_k1_INTC_s005
                                 + 1.61683 w_k1_KO_s005 + 1.45057 w_k1_MAT_s005
                                 + 1.60566 w_k1_MSFT_s005
                                 + 1.47201 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.834305 w_k1_ADM_s005
                                 + 1.20202 w_k1_AXP_s005
                                 + 0.96928 w_k1_DIS_s005
                                 + 0.915417 w_k1_FDX_s005
                                 + 1.55962 w_k1_IBM_s005
                                 + 1.19273 w_k1_INTC_s005
                                 + 1.18694 w_k1_KO_s005 + 1.29003 w_k1_MAT_s005
                                 + 0.979668 w_k1_MSFT_s005
                                 + 1.06527 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.22737 w_k1_ADM_s006
                                 + 1.05857 w_k1_AXP_s006
                                 + 1.41364 w_k1_DIS_s006
                                 + 1.01907 w_k1_FDX_s006
                                 + 1.16939 w_k1_IBM_s006
                                 + 0.845982 w_k1_INTC_s006
                                 + 1.21942 w_k1_KO_s006 + 1.22187 w_k1_MAT_s006
                                 + 0.959599 w_k1_MSFT_s006
                                 + 1.13962 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.15572 w_k1_ADM_s006
                                 + 1.0309 w_k1_AXP_s006
                                 + 0.818624 w_k1_DIS_s006
                                 + 0.960388 w_k1_FDX_s006
                                 + 0.797656 w_k1_IBM_s006
                                 + 0.929697 w_k1_INTC_s006
                                 + 0.756405 w_k1_KO_s006
                                 + 0.647461 w_k1_MAT_s006
                                 + 0.837423 w_k1_MSFT_s006
                                 + 1.00416 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.525 w_k1_ADM_s006
                                 + 1.35501 w_k1_AXP_s006
                                 + 1.05719 w_k1_DIS_s006 + 1.0832 w_k1_FDX_s006
                                 + 1.1396 w_k1_IBM_s006
                                 + 0.93099 w_k1_INTC_s006
                                 + 1.22786 w_k1_KO_s006 + 1.49502 w_k1_MAT_s006
                                 + 0.928123 w_k1_MSFT_s006
                                 + 1.21958 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.987756 w_k1_ADM_s006
                                 + 0.743024 w_k1_AXP_s006
                                 + 0.721591 w_k1_DIS_s006
                                 + 0.786775 w_k1_FDX_s006
                                 + 0.864088 w_k1_IBM_s006
                                 + 0.893087 w_k1_INTC_s006
                                 + 0.758949 w_k1_KO_s006
                                 + 0.759582 w_k1_MAT_s006
                                 + 0.754485 w_k1_MSFT_s006
                                 + 0.886404 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.82121 w_k1_ADM_s006
                                 + 1.40051 w_k1_AXP_s006
                                 + 1.52228 w_k1_DIS_s006
                                 + 1.35878 w_k1_FDX_s006
                                 + 0.943799 w_k1_IBM_s006
                                 + 1.35207 w_k1_INTC_s006
                                 + 1.43497 w_k1_KO_s006 + 0.98955 w_k1_MAT_s006
                                 + 1.10295 w_k1_MSFT_s006
                                 + 1.12649 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.95492 w_k1_ADM_s006
                                 + 1.11684 w_k1_AXP_s006
                                 + 1.11628 w_k1_DIS_s006
                                 + 0.895902 w_k1_FDX_s006
                                 + 1.4569 w_k1_IBM_s006
                                 + 0.904049 w_k1_INTC_s006
                                 + 1.17897 w_k1_KO_s006 + 1.24781 w_k1_MAT_s006
                                 + 0.892399 w_k1_MSFT_s006
                                 + 1.03661 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.49829 w_k1_ADM_s006
                                 + 1.09113 w_k1_AXP_s006
                                 + 1.23316 w_k1_DIS_s006
                                 + 1.09944 w_k1_FDX_s006
                                 + 0.866157 w_k1_IBM_s006
                                 + 1.26909 w_k1_INTC_s006
                                 + 1.05397 w_k1_KO_s006
                                 + 0.823632 w_k1_MAT_s006
                                 + 1.13339 w_k1_MSFT_s006
                                 + 1.00533 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.41237 w_k1_ADM_s006
                                 + 1.67087 w_k1_AXP_s006
                                 + 1.65615 w_k1_DIS_s006
                                 + 1.31585 w_k1_FDX_s006 + 1.6061 w_k1_IBM_s006
                                 + 1.33667 w_k1_INTC_s006
                                 + 1.47431 w_k1_KO_s006 + 1.68638 w_k1_MAT_s006
                                 + 1.12633 w_k1_MSFT_s006
                                 + 1.12553 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.779689 w_k1_ADM_s007
                                 + 0.899076 w_k1_AXP_s007
                                 + 0.950394 w_k1_DIS_s007
                                 + 0.954054 w_k1_FDX_s007
                                 + 1.26857 w_k1_IBM_s007
                                 + 1.00691 w_k1_INTC_s007
                                 + 1.01641 w_k1_KO_s007
                                 + 0.914514 w_k1_MAT_s007
                                 + 1.14264 w_k1_MSFT_s007
                                 + 1.02991 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.663974 w_k1_ADM_s007
                                 + 0.773036 w_k1_AXP_s007
                                 + 0.723972 w_k1_DIS_s007
                                 + 0.798262 w_k1_FDX_s007
                                 + 0.865522 w_k1_IBM_s007
                                 + 0.873868 w_k1_INTC_s007
                                 + 0.816524 w_k1_KO_s007
                                 + 0.644565 w_k1_MAT_s007
                                 + 0.937159 w_k1_MSFT_s007
                                 + 0.901622 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.867157 w_k1_ADM_s007
                                 + 1.35914 w_k1_AXP_s007
                                 + 1.18742 w_k1_DIS_s007
                                 + 1.03001 w_k1_FDX_s007
                                 + 1.64994 w_k1_IBM_s007
                                 + 1.05054 w_k1_INTC_s007 + 1.3731 w_k1_KO_s007
                                 + 1.21648 w_k1_MAT_s007
                                 + 1.10403 w_k1_MSFT_s007
                                 + 1.22413 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.864801 w_k1_ADM_s007
                                 + 1.13188 w_k1_AXP_s007
                                 + 1.08455 w_k1_DIS_s007
                                 + 1.18669 w_k1_FDX_s007
                                 + 0.917808 w_k1_IBM_s007
                                 + 1.17856 w_k1_INTC_s007
                                 + 1.00554 w_k1_KO_s007 + 1.11684 w_k1_MAT_s007
                                 + 1.19383 w_k1_MSFT_s007
                                 + 0.975068 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.861162 w_k1_ADM_s007
                                 + 1.41407 w_k1_AXP_s007
                                 + 1.60061 w_k1_DIS_s007
                                 + 1.39588 w_k1_FDX_s007
                                 + 0.983803 w_k1_IBM_s007
                                 + 1.40855 w_k1_INTC_s007
                                 + 1.15356 w_k1_KO_s007 + 1.07876 w_k1_MAT_s007
                                 + 1.40224 w_k1_MSFT_s007
                                 + 1.23469 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.911639 w_k1_ADM_s007
                                 + 1.10996 w_k1_AXP_s007
                                 + 0.887114 w_k1_DIS_s007
                                 + 1.13093 w_k1_FDX_s007
                                 + 0.656814 w_k1_IBM_s007
                                 + 1.00899 w_k1_INTC_s007
                                 + 0.80521 w_k1_KO_s007
                                 + 0.552256 w_k1_MAT_s007
                                 + 1.01475 w_k1_MSFT_s007
                                 + 1.06152 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.669884 w_k1_ADM_s007
                                 + 0.901861 w_k1_AXP_s007
                                 + 0.891949 w_k1_DIS_s007
                                 + 0.819915 w_k1_FDX_s007
                                 + 1.32347 w_k1_IBM_s007
                                 + 0.93318 w_k1_INTC_s007
                                 + 0.951836 w_k1_KO_s007
                                 + 1.13883 w_k1_MAT_s007
                                 + 0.895988 w_k1_MSFT_s007
                                 + 0.936749 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.869638 w_k1_ADM_s007
                                 + 0.754004 w_k1_AXP_s007
                                 + 1.13405 w_k1_DIS_s007
                                 + 0.976225 w_k1_FDX_s007
                                 + 0.834199 w_k1_IBM_s007
                                 + 1.01521 w_k1_INTC_s007
                                 + 0.948296 w_k1_KO_s007
                                 + 0.452515 w_k1_MAT_s007
                                 + 1.10262 w_k1_MSFT_s007
                                 + 1.01663 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.760971 w_k1_ADM_s010
                                 + 0.963895 w_k1_AXP_s010
                                 + 1.0389 w_k1_DIS_s010 + 1.05136 w_k1_FDX_s010
                                 + 0.815706 w_k1_IBM_s010
                                 + 1.08039 w_k1_INTC_s010
                                 + 0.962346 w_k1_KO_s010
                                 + 0.676319 w_k1_MAT_s010
                                 + 0.963027 w_k1_MSFT_s010
                                 + 0.956362 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.25776 w_k1_ADM_s010
                                 + 1.13607 w_k1_AXP_s010
                                 + 1.46021 w_k1_DIS_s010
                                 + 1.33796 w_k1_FDX_s010
                                 + 0.862074 w_k1_IBM_s010
                                 + 1.28536 w_k1_INTC_s010
                                 + 1.02396 w_k1_KO_s010
                                 + 0.772208 w_k1_MAT_s010
                                 + 1.15635 w_k1_MSFT_s010
                                 + 0.937035 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.17583 w_k1_ADM_s010
                                 + 1.00609 w_k1_AXP_s010
                                 + 0.901614 w_k1_DIS_s010
                                 + 1.09556 w_k1_FDX_s010
                                 + 0.810964 w_k1_IBM_s010
                                 + 0.88661 w_k1_INTC_s010
                                 + 0.92969 w_k1_KO_s010
                                 + 0.924793 w_k1_MAT_s010
                                 + 0.954538 w_k1_MSFT_s010
                                 + 0.985009 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.09313 w_k1_ADM_s010
                                 + 0.857296 w_k1_AXP_s010
                                 + 1.07666 w_k1_DIS_s010
                                 + 0.952137 w_k1_FDX_s010
                                 + 1.26413 w_k1_IBM_s010
                                 + 0.934922 w_k1_INTC_s010
                                 + 1.21709 w_k1_KO_s010 + 1.3516 w_k1_MAT_s010
                                 + 0.944986 w_k1_MSFT_s010
                                 + 0.948114 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.30831 w_k1_ADM_s010
                                 + 1.66502 w_k1_AXP_s010
                                 + 1.88238 w_k1_DIS_s010
                                 + 1.38315 w_k1_FDX_s010 + 1.6331 w_k1_IBM_s010
                                 + 1.47715 w_k1_INTC_s010
                                 + 1.43804 w_k1_KO_s010 + 1.16839 w_k1_MAT_s010
                                 + 1.22998 w_k1_MSFT_s010
                                 + 1.10855 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.768856 w_k1_ADM_s010
                                 + 1.2021 w_k1_AXP_s010 + 0.66544 w_k1_DIS_s010
                                 + 0.918516 w_k1_FDX_s010
                                 + 0.833945 w_k1_IBM_s010
                                 + 1.08684 w_k1_INTC_s010
                                 + 0.894039 w_k1_KO_s010
                                 + 1.07589 w_k1_MAT_s010
                                 + 0.728511 w_k1_MSFT_s010
                                 + 0.887971 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.794685 w_k1_ADM_s010
                                 + 0.65762 w_k1_AXP_s010
                                 + 0.727892 w_k1_DIS_s010
                                 + 0.764052 w_k1_FDX_s010
                                 + 0.855113 w_k1_IBM_s010
                                 + 0.660993 w_k1_INTC_s010
                                 + 1.00478 w_k1_KO_s010
                                 + 0.786094 w_k1_MAT_s010
                                 + 0.708249 w_k1_MSFT_s010
                                 + 0.797951 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.03937 w_k1_ADM_s010
                                 + 1.22291 w_k1_AXP_s010
                                 + 1.19647 w_k1_DIS_s010
                                 + 1.15233 w_k1_FDX_s010
                                 + 1.41051 w_k1_IBM_s010
                                 + 1.19541 w_k1_INTC_s010
                                 + 1.14812 w_k1_KO_s010 + 1.08396 w_k1_MAT_s010
                                 + 1.05817 w_k1_MSFT_s010
                                 + 0.925224 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.87326 w_k1_ADM_s011
                                 + 0.924177 w_k1_AXP_s011
                                 + 0.807449 w_k1_DIS_s011
                                 + 0.919833 w_k1_FDX_s011
                                 + 0.689871 w_k1_IBM_s011
                                 + 0.81145 w_k1_INTC_s011
                                 + 0.763454 w_k1_KO_s011
                                 + 0.577936 w_k1_MAT_s011
                                 + 0.839717 w_k1_MSFT_s011
                                 + 0.972696 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.758771 w_k1_ADM_s011
                                 + 0.634515 w_k1_AXP_s011
                                 + 0.968661 w_k1_DIS_s011
                                 + 0.780031 w_k1_FDX_s011
                                 + 1.44822 w_k1_IBM_s011
                                 + 0.830227 w_k1_INTC_s011
                                 + 1.38233 w_k1_KO_s011 + 1.35094 w_k1_MAT_s011
                                 + 0.967765 w_k1_MSFT_s011
                                 + 0.956906 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.01693 w_k1_ADM_s011
                                 + 1.327 w_k1_AXP_s011 + 1.22399 w_k1_DIS_s011
                                 + 1.08792 w_k1_FDX_s011
                                 + 1.17831 w_k1_IBM_s011
                                 + 0.948363 w_k1_INTC_s011
                                 + 1.16579 w_k1_KO_s011 + 1.26324 w_k1_MAT_s011
                                 + 0.932591 w_k1_MSFT_s011
                                 + 1.17002 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.10557 w_k1_ADM_s011
                                 + 1.44788 w_k1_AXP_s011
                                 + 1.14645 w_k1_DIS_s011
                                 + 1.14262 w_k1_FDX_s011
                                 + 1.81492 w_k1_IBM_s011
                                 + 1.03418 w_k1_INTC_s011
                                 + 1.45212 w_k1_KO_s011 + 2.47007 w_k1_MAT_s011
                                 + 1.10445 w_k1_MSFT_s011
                                 + 1.20507 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.09288 w_k1_ADM_s011
                                 + 0.958282 w_k1_AXP_s011
                                 + 1.20308 w_k1_DIS_s011
                                 + 1.02876 w_k1_FDX_s011
                                 + 1.02425 w_k1_IBM_s011
                                 + 0.919298 w_k1_INTC_s011
                                 + 1.08697 w_k1_KO_s011 + 1.12476 w_k1_MAT_s011
                                 + 1.00681 w_k1_MSFT_s011
                                 + 0.881983 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.954786 w_k1_ADM_s011
                                 + 0.639892 w_k1_AXP_s011
                                 + 0.821275 w_k1_DIS_s011
                                 + 0.804676 w_k1_FDX_s011
                                 + 0.940247 w_k1_IBM_s011
                                 + 0.750716 w_k1_INTC_s011
                                 + 1.03056 w_k1_KO_s011
                                 + 0.753819 w_k1_MAT_s011
                                 + 0.832452 w_k1_MSFT_s011
                                 + 0.947692 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.898808 w_k1_ADM_s011
                                 + 1.13154 w_k1_AXP_s011
                                 + 1.15525 w_k1_DIS_s011
                                 + 1.17593 w_k1_FDX_s011
                                 + 0.778914 w_k1_IBM_s011
                                 + 1.15031 w_k1_INTC_s011
                                 + 0.991936 w_k1_KO_s011
                                 + 0.931355 w_k1_MAT_s011
                                 + 1.14641 w_k1_MSFT_s011
                                 + 1.02319 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.744142 w_k1_ADM_s011
                                 + 0.509718 w_k1_AXP_s011
                                 + 0.629251 w_k1_DIS_s011
                                 + 0.752192 w_k1_FDX_s011
                                 + 0.625878 w_k1_IBM_s011
                                 + 0.740618 w_k1_INTC_s011
                                 + 0.706414 w_k1_KO_s011
                                 + 0.427161 w_k1_MAT_s011
                                 + 0.681859 w_k1_MSFT_s011
                                 + 0.844578 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.24415 w_k1_ADM_s012
                                 + 1.17012 w_k1_AXP_s012 + 1.0881 w_k1_DIS_s012
                                 + 0.907792 w_k1_FDX_s012
                                 + 1.3821 w_k1_IBM_s012
                                 + 0.897194 w_k1_INTC_s012
                                 + 1.28205 w_k1_KO_s012 + 1.69643 w_k1_MAT_s012
                                 + 1.05395 w_k1_MSFT_s012
                                 + 1.12362 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.06212 w_k1_ADM_s012
                                 + 1.32635 w_k1_AXP_s012
                                 + 0.997847 w_k1_DIS_s012
                                 + 1.02991 w_k1_FDX_s012
                                 + 1.08223 w_k1_IBM_s012
                                 + 0.90738 w_k1_INTC_s012
                                 + 1.03229 w_k1_KO_s012 + 1.08724 w_k1_MAT_s012
                                 + 0.936955 w_k1_MSFT_s012
                                 + 1.03098 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.917449 w_k1_ADM_s012
                                 + 0.908075 w_k1_AXP_s012
                                 + 0.830521 w_k1_DIS_s012
                                 + 0.832646 w_k1_FDX_s012
                                 + 1.10317 w_k1_IBM_s012
                                 + 0.876615 w_k1_INTC_s012
                                 + 1.01216 w_k1_KO_s012 + 1.17264 w_k1_MAT_s012
                                 + 0.857435 w_k1_MSFT_s012
                                 + 1.01155 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.47772 w_k1_ADM_s012
                                 + 1.73383 w_k1_AXP_s012
                                 + 1.77094 w_k1_DIS_s012
                                 + 1.40266 w_k1_FDX_s012
                                 + 1.00042 w_k1_IBM_s012
                                 + 1.47838 w_k1_INTC_s012
                                 + 1.25893 w_k1_KO_s012
                                 + 0.977817 w_k1_MAT_s012
                                 + 1.29434 w_k1_MSFT_s012 + 1.2231 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.05636 w_k1_ADM_s012
                                 + 1.15395 w_k1_AXP_s012
                                 + 0.994587 w_k1_DIS_s012
                                 + 1.06826 w_k1_FDX_s012
                                 + 0.787324 w_k1_IBM_s012
                                 + 1.00633 w_k1_INTC_s012
                                 + 0.857262 w_k1_KO_s012
                                 + 0.749996 w_k1_MAT_s012
                                 + 0.962726 w_k1_MSFT_s012
                                 + 0.954097 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.891989 w_k1_ADM_s012
                                 + 0.784922 w_k1_AXP_s012
                                 + 0.511427 w_k1_DIS_s012
                                 + 0.749447 w_k1_FDX_s012
                                 + 0.735541 w_k1_IBM_s012
                                 + 0.816943 w_k1_INTC_s012
                                 + 0.740072 w_k1_KO_s012
                                 + 0.921696 w_k1_MAT_s012
                                 + 0.764732 w_k1_MSFT_s012
                                 + 0.878361 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.88288 w_k1_ADM_s012
                                 + 1.04964 w_k1_AXP_s012
                                 + 0.981256 w_k1_DIS_s012
                                 + 0.912018 w_k1_FDX_s012
                                 + 1.6532 w_k1_IBM_s012
                                 + 1.02243 w_k1_INTC_s012
                                 + 1.28378 w_k1_KO_s012 + 1.73642 w_k1_MAT_s012
                                 + 0.983628 w_k1_MSFT_s012
                                 + 1.20398 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.29718 w_k1_ADM_s012
                                 + 1.51247 w_k1_AXP_s012
                                 + 1.51031 w_k1_DIS_s012
                                 + 1.33026 w_k1_FDX_s012
                                 + 1.30457 w_k1_IBM_s012
                                 + 1.07215 w_k1_INTC_s012
                                 + 1.56462 w_k1_KO_s012 + 1.71666 w_k1_MAT_s012
                                 + 1.21042 w_k1_MSFT_s012
                                 + 1.33664 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.05809 w_k1_ADM_s013
                                 + 0.914364 w_k1_AXP_s013
                                 + 0.777795 w_k1_DIS_s013
                                 + 0.89126 w_k1_FDX_s013
                                 + 0.872871 w_k1_IBM_s013
                                 + 0.881501 w_k1_INTC_s013
                                 + 0.876147 w_k1_KO_s013
                                 + 0.897862 w_k1_MAT_s013
                                 + 0.95932 w_k1_MSFT_s013
                                 + 0.926857 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.28025 w_k1_ADM_s013
                                 + 1.21423 w_k1_AXP_s013
                                 + 1.87008 w_k1_DIS_s013
                                 + 1.54841 w_k1_FDX_s013
                                 + 1.01157 w_k1_IBM_s013
                                 + 1.37135 w_k1_INTC_s013 + 1.4462 w_k1_KO_s013
                                 + 0.892134 w_k1_MAT_s013
                                 + 1.58606 w_k1_MSFT_s013 + 1.3429 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.21345 w_k1_ADM_s013
                                 + 0.984026 w_k1_AXP_s013
                                 + 1.16756 w_k1_DIS_s013
                                 + 1.14543 w_k1_FDX_s013
                                 + 1.04037 w_k1_IBM_s013
                                 + 1.04173 w_k1_INTC_s013
                                 + 0.997766 w_k1_KO_s013
                                 + 0.925143 w_k1_MAT_s013
                                 + 1.07001 w_k1_MSFT_s013
                                 + 0.99741 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.09422 w_k1_ADM_s013
                                 + 1.22952 w_k1_AXP_s013
                                 + 1.38846 w_k1_DIS_s013 + 1.3046 w_k1_FDX_s013
                                 + 1.12789 w_k1_IBM_s013
                                 + 1.30144 w_k1_INTC_s013
                                 + 1.17539 w_k1_KO_s013 + 1.79863 w_k1_MAT_s013
                                 + 1.40632 w_k1_MSFT_s013
                                 + 1.08093 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.61583 w_k1_ADM_s013
                                 + 1.23551 w_k1_AXP_s013
                                 + 1.37218 w_k1_DIS_s013
                                 + 1.12877 w_k1_FDX_s013
                                 + 1.64588 w_k1_IBM_s013
                                 + 1.21814 w_k1_INTC_s013
                                 + 1.70416 w_k1_KO_s013 + 1.47086 w_k1_MAT_s013
                                 + 1.36088 w_k1_MSFT_s013
                                 + 1.26253 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.23075 w_k1_ADM_s013
                                 + 0.864327 w_k1_AXP_s013
                                 + 1.01024 w_k1_DIS_s013
                                 + 1.16591 w_k1_FDX_s013
                                 + 0.725471 w_k1_IBM_s013
                                 + 1.1989 w_k1_INTC_s013
                                 + 0.830708 w_k1_KO_s013
                                 + 0.629884 w_k1_MAT_s013
                                 + 1.20938 w_k1_MSFT_s013
                                 + 0.974974 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.14375 w_k1_ADM_s013
                                 + 1.00736 w_k1_AXP_s013
                                 + 0.836166 w_k1_DIS_s013
                                 + 0.930284 w_k1_FDX_s013
                                 + 1.16589 w_k1_IBM_s013
                                 + 0.824883 w_k1_INTC_s013
                                 + 0.960589 w_k1_KO_s013
                                 + 1.21623 w_k1_MAT_s013
                                 + 0.959095 w_k1_MSFT_s013
                                 + 1.01036 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.911595 w_k1_ADM_s013
                                 + 0.673734 w_k1_AXP_s013
                                 + 0.608459 w_k1_DIS_s013
                                 + 0.776402 w_k1_FDX_s013
                                 + 0.686709 w_k1_IBM_s013
                                 + 0.858753 w_k1_INTC_s013
                                 + 0.722492 w_k1_KO_s013
                                 + 0.716425 w_k1_MAT_s013
                                 + 0.825777 w_k1_MSFT_s013
                                 + 0.914136 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.29059 w_k1_ADM_s014
                                 + 1.00784 w_k1_AXP_s014
                                 + 1.32287 w_k1_DIS_s014
                                 + 1.21338 w_k1_FDX_s014
                                 + 0.873116 w_k1_IBM_s014
                                 + 1.0468 w_k1_INTC_s014
                                 + 0.999493 w_k1_KO_s014
                                 + 0.904421 w_k1_MAT_s014
                                 + 1.32115 w_k1_MSFT_s014
                                 + 1.01945 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.13027 w_k1_ADM_s014
                                 + 1.07937 w_k1_AXP_s014
                                 + 0.900567 w_k1_DIS_s014
                                 + 0.973143 w_k1_FDX_s014
                                 + 0.935101 w_k1_IBM_s014
                                 + 0.976199 w_k1_INTC_s014
                                 + 0.954029 w_k1_KO_s014
                                 + 1.01146 w_k1_MAT_s014
                                 + 0.966154 w_k1_MSFT_s014
                                 + 0.929716 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.34282 w_k1_ADM_s014
                                 + 1.64632 w_k1_AXP_s014
                                 + 1.18771 w_k1_DIS_s014
                                 + 1.31188 w_k1_FDX_s014
                                 + 0.958011 w_k1_IBM_s014
                                 + 1.15512 w_k1_INTC_s014
                                 + 0.926271 w_k1_KO_s014
                                 + 1.17468 w_k1_MAT_s014
                                 + 1.20552 w_k1_MSFT_s014
                                 + 1.03289 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.01188 w_k1_ADM_s014
                                 + 0.83546 w_k1_AXP_s014
                                 + 0.87806 w_k1_DIS_s014
                                 + 1.02435 w_k1_FDX_s014
                                 + 0.692965 w_k1_IBM_s014
                                 + 0.917291 w_k1_INTC_s014
                                 + 0.805302 w_k1_KO_s014
                                 + 0.883591 w_k1_MAT_s014
                                 + 1.09534 w_k1_MSFT_s014
                                 + 0.809718 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.919539 w_k1_ADM_s014
                                 + 0.642183 w_k1_AXP_s014
                                 + 0.989217 w_k1_DIS_s014
                                 + 0.895112 w_k1_FDX_s014
                                 + 0.893951 w_k1_IBM_s014
                                 + 0.796522 w_k1_INTC_s014
                                 + 1.02085 w_k1_KO_s014
                                 + 0.803703 w_k1_MAT_s014
                                 + 1.00307 w_k1_MSFT_s014
                                 + 0.840498 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.87543 w_k1_ADM_s014
                                 + 1.15728 w_k1_AXP_s014 + 1.2359 w_k1_DIS_s014
                                 + 1.04257 w_k1_FDX_s014 + 1.5576 w_k1_IBM_s014
                                 + 1.31937 w_k1_INTC_s014
                                 + 1.24951 w_k1_KO_s014 + 1.58282 w_k1_MAT_s014
                                 + 1.23518 w_k1_MSFT_s014
                                 + 1.05936 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.31401 w_k1_ADM_s014
                                 + 1.26814 w_k1_AXP_s014 + 1.396 w_k1_DIS_s014
                                 + 1.1008 w_k1_FDX_s014 + 1.30699 w_k1_IBM_s014
                                 + 1.1205 w_k1_INTC_s014 + 1.26927 w_k1_KO_s014
                                 + 1.11498 w_k1_MAT_s014
                                 + 1.30499 w_k1_MSFT_s014
                                 + 1.10117 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.760216 w_k1_ADM_s014
                                 + 0.946858 w_k1_AXP_s014
                                 + 0.685692 w_k1_DIS_s014
                                 + 0.78861 w_k1_FDX_s014
                                 + 1.21636 w_k1_IBM_s014
                                 + 0.887643 w_k1_INTC_s014
                                 + 0.849632 w_k1_KO_s014
                                 + 1.09849 w_k1_MAT_s014
                                 + 0.927734 w_k1_MSFT_s014
                                 + 0.845383 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.779708 w_k1_ADM_s015
                                 + 0.76129 w_k1_AXP_s015
                                 + 0.748004 w_k1_DIS_s015
                                 + 0.789424 w_k1_FDX_s015
                                 + 0.92627 w_k1_IBM_s015
                                 + 0.837725 w_k1_INTC_s015
                                 + 0.865253 w_k1_KO_s015
                                 + 0.983143 w_k1_MAT_s015
                                 + 0.837168 w_k1_MSFT_s015
                                 + 0.852431 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.31766 w_k1_ADM_s015
                                 + 1.48877 w_k1_AXP_s015 + 1.4571 w_k1_DIS_s015
                                 + 1.39147 w_k1_FDX_s015
                                 + 1.36979 w_k1_IBM_s015
                                 + 1.43438 w_k1_INTC_s015
                                 + 1.32709 w_k1_KO_s015 + 1.62154 w_k1_MAT_s015
                                 + 1.46714 w_k1_MSFT_s015
                                 + 1.42741 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.0621 w_k1_ADM_s015
                                 + 1.14859 w_k1_AXP_s015
                                 + 1.32377 w_k1_DIS_s015
                                 + 1.28982 w_k1_FDX_s015
                                 + 0.883564 w_k1_IBM_s015
                                 + 1.35632 w_k1_INTC_s015
                                 + 0.935399 w_k1_KO_s015
                                 + 0.676304 w_k1_MAT_s015
                                 + 1.29822 w_k1_MSFT_s015
                                 + 1.04048 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.4639 w_k1_ADM_s015
                                 + 1.34368 w_k1_AXP_s015
                                 + 1.01386 w_k1_DIS_s015
                                 + 1.14338 w_k1_FDX_s015
                                 + 1.16491 w_k1_IBM_s015
                                 + 0.952284 w_k1_INTC_s015
                                 + 1.04473 w_k1_KO_s015 + 1.08966 w_k1_MAT_s015
                                 + 1.05951 w_k1_MSFT_s015
                                 + 1.06827 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.09272 w_k1_ADM_s015
                                 + 1.04361 w_k1_AXP_s015
                                 + 1.11068 w_k1_DIS_s015
                                 + 1.08811 w_k1_FDX_s015
                                 + 1.16805 w_k1_IBM_s015
                                 + 1.14608 w_k1_INTC_s015
                                 + 1.05549 w_k1_KO_s015
                                 + 0.910217 w_k1_MAT_s015
                                 + 1.13587 w_k1_MSFT_s015
                                 + 1.06735 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.02859 w_k1_ADM_s015
                                 + 0.973846 w_k1_AXP_s015
                                 + 0.798073 w_k1_DIS_s015
                                 + 0.995691 w_k1_FDX_s015
                                 + 0.791718 w_k1_IBM_s015
                                 + 0.880091 w_k1_INTC_s015
                                 + 0.769344 w_k1_KO_s015
                                 + 0.512095 w_k1_MAT_s015
                                 + 0.974639 w_k1_MSFT_s015
                                 + 0.92004 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.14521 w_k1_ADM_s015
                                 + 0.890452 w_k1_AXP_s015
                                 + 0.799057 w_k1_DIS_s015
                                 + 0.867609 w_k1_FDX_s015
                                 + 1.24055 w_k1_IBM_s015
                                 + 0.833752 w_k1_INTC_s015
                                 + 1.01464 w_k1_KO_s015
                                 + 0.883668 w_k1_MAT_s015
                                 + 0.907559 w_k1_MSFT_s015
                                 + 0.914441 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.909682 w_k1_ADM_s015
                                 + 0.785903 w_k1_AXP_s015
                                 + 0.504253 w_k1_DIS_s015
                                 + 0.718211 w_k1_FDX_s015
                                 + 0.783149 w_k1_IBM_s015
                                 + 0.959275 w_k1_INTC_s015
                                 + 0.650602 w_k1_KO_s015
                                 + 0.648496 w_k1_MAT_s015
                                 + 0.834531 w_k1_MSFT_s015
                                 + 0.886254 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.80402 w_k1_ADM_s016
                                 + 1.05358 w_k1_AXP_s016
                                 + 1.04086 w_k1_DIS_s016
                                 + 0.978294 w_k1_FDX_s016
                                 + 1.08149 w_k1_IBM_s016
                                 + 0.969216 w_k1_INTC_s016
                                 + 1.11742 w_k1_KO_s016 + 1.1372 w_k1_MAT_s016
                                 + 0.972879 w_k1_MSFT_s016
                                 + 1.06982 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.05011 w_k1_ADM_s016
                                 + 1.08312 w_k1_AXP_s016
                                 + 0.898265 w_k1_DIS_s016
                                 + 1.12928 w_k1_FDX_s016
                                 + 0.645267 w_k1_IBM_s016
                                 + 0.921507 w_k1_INTC_s016
                                 + 0.772033 w_k1_KO_s016
                                 + 0.765967 w_k1_MAT_s016
                                 + 0.895715 w_k1_MSFT_s016
                                 + 0.984663 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.846819 w_k1_ADM_s016
                                 + 1.03847 w_k1_AXP_s016
                                 + 0.949778 w_k1_DIS_s016
                                 + 0.933855 w_k1_FDX_s016
                                 + 0.834073 w_k1_IBM_s016
                                 + 0.888502 w_k1_INTC_s016
                                 + 1.01854 w_k1_KO_s016 + 1.17468 w_k1_MAT_s016
                                 + 0.852516 w_k1_MSFT_s016
                                 + 0.970484 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.06418 w_k1_ADM_s016
                                 + 1.23305 w_k1_AXP_s016
                                 + 1.27538 w_k1_DIS_s016
                                 + 1.25415 w_k1_FDX_s016
                                 + 1.35539 w_k1_IBM_s016
                                 + 1.25373 w_k1_INTC_s016
                                 + 1.38826 w_k1_KO_s016 + 1.64493 w_k1_MAT_s016
                                 + 1.35944 w_k1_MSFT_s016 + 1.1642 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.27274 w_k1_ADM_s016
                                 + 1.03798 w_k1_AXP_s016
                                 + 0.912181 w_k1_DIS_s016
                                 + 0.981186 w_k1_FDX_s016
                                 + 0.883673 w_k1_IBM_s016
                                 + 0.820112 w_k1_INTC_s016
                                 + 1.07274 w_k1_KO_s016 + 1.05491 w_k1_MAT_s016
                                 + 0.833131 w_k1_MSFT_s016
                                 + 1.07199 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.710974 w_k1_ADM_s016
                                 + 0.828483 w_k1_AXP_s016
                                 + 0.718781 w_k1_DIS_s016
                                 + 0.79694 w_k1_FDX_s016
                                 + 0.943052 w_k1_IBM_s016
                                 + 0.77111 w_k1_INTC_s016
                                 + 0.880623 w_k1_KO_s016
                                 + 0.897167 w_k1_MAT_s016
                                 + 0.863331 w_k1_MSFT_s016
                                 + 0.922196 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.13636 w_k1_ADM_s016
                                 + 1.30702 w_k1_AXP_s016
                                 + 1.41126 w_k1_DIS_s016
                                 + 1.23146 w_k1_FDX_s016
                                 + 0.829145 w_k1_IBM_s016
                                 + 0.928154 w_k1_INTC_s016
                                 + 1.16132 w_k1_KO_s016 + 1.14254 w_k1_MAT_s016
                                 + 1.12643 w_k1_MSFT_s016
                                 + 1.18373 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.739401 w_k1_ADM_s016
                                 + 0.828643 w_k1_AXP_s016
                                 + 0.640008 w_k1_DIS_s016
                                 + 0.834047 w_k1_FDX_s016
                                 + 0.671829 w_k1_IBM_s016
                                 + 0.812753 w_k1_INTC_s016
                                 + 0.751816 w_k1_KO_s016
                                 + 0.80915 w_k1_MAT_s016
                                 + 0.801059 w_k1_MSFT_s016
                                 + 0.815743 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.23926 w_k1_ADM_s017
                                 + 1.15547 w_k1_AXP_s017
                                 + 1.00072 w_k1_DIS_s017
                                 + 1.12179 w_k1_FDX_s017
                                 + 0.936634 w_k1_IBM_s017
                                 + 0.937942 w_k1_INTC_s017
                                 + 0.772702 w_k1_KO_s017
                                 + 0.65672 w_k1_MAT_s017
                                 + 1.02702 w_k1_MSFT_s017
                                 + 0.920977 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.11658 w_k1_ADM_s017
                                 + 1.38057 w_k1_AXP_s017
                                 + 1.13167 w_k1_DIS_s017
                                 + 1.08049 w_k1_FDX_s017
                                 + 1.18223 w_k1_IBM_s017
                                 + 1.06995 w_k1_INTC_s017
                                 + 0.929594 w_k1_KO_s017 + 1.3697 w_k1_MAT_s017
                                 + 1.00599 w_k1_MSFT_s017
                                 + 1.12706 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.837335 w_k1_ADM_s017
                                 + 1.24019 w_k1_AXP_s017 + 1.2766 w_k1_DIS_s017
                                 + 1.0307 w_k1_FDX_s017 + 1.52814 w_k1_IBM_s017
                                 + 1.0548 w_k1_INTC_s017 + 1.00446 w_k1_KO_s017
                                 + 1.37144 w_k1_MAT_s017
                                 + 0.956694 w_k1_MSFT_s017
                                 + 1.07542 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.1718 w_k1_ADM_s017
                                 + 0.983208 w_k1_AXP_s017
                                 + 1.06373 w_k1_DIS_s017
                                 + 1.06632 w_k1_FDX_s017
                                 + 0.736693 w_k1_IBM_s017
                                 + 0.981158 w_k1_INTC_s017
                                 + 0.876834 w_k1_KO_s017
                                 + 0.995576 w_k1_MAT_s017
                                 + 1.07201 w_k1_MSFT_s017
                                 + 0.984485 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.50704 w_k1_ADM_s017
                                 + 1.68007 w_k1_AXP_s017
                                 + 1.53363 w_k1_DIS_s017
                                 + 1.33962 w_k1_FDX_s017
                                 + 1.47952 w_k1_IBM_s017
                                 + 1.3747 w_k1_INTC_s017 + 1.29827 w_k1_KO_s017
                                 + 1.36071 w_k1_MAT_s017
                                 + 1.23659 w_k1_MSFT_s017
                                 + 1.16882 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.18374 w_k1_ADM_s017
                                 + 1.17357 w_k1_AXP_s017
                                 + 0.944728 w_k1_DIS_s017
                                 + 0.897833 w_k1_FDX_s017
                                 + 1.07341 w_k1_IBM_s017
                                 + 0.97072 w_k1_INTC_s017
                                 + 0.795897 w_k1_KO_s017 + 1.0493 w_k1_MAT_s017
                                 + 0.997749 w_k1_MSFT_s017
                                 + 1.02419 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.836665 w_k1_ADM_s017
                                 + 0.925574 w_k1_AXP_s017
                                 + 0.621735 w_k1_DIS_s017
                                 + 0.795771 w_k1_FDX_s017
                                 + 0.827164 w_k1_IBM_s017
                                 + 0.877857 w_k1_INTC_s017
                                 + 0.68293 w_k1_KO_s017
                                 + 0.911132 w_k1_MAT_s017
                                 + 0.790042 w_k1_MSFT_s017
                                 + 0.838319 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.39747 w_k1_ADM_s017
                                 + 1.45113 w_k1_AXP_s017
                                 + 1.28758 w_k1_DIS_s017
                                 + 1.32572 w_k1_FDX_s017
                                 + 0.876481 w_k1_IBM_s017 + 1.02 w_k1_INTC_s017
                                 + 0.884478 w_k1_KO_s017
                                 + 0.708731 w_k1_MAT_s017
                                 + 1.03007 w_k1_MSFT_s017
                                 + 1.03954 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.92624 w_k1_ADM_s100
                                 + 1.09046 w_k1_AXP_s100
                                 + 0.813102 w_k1_DIS_s100
                                 + 0.9801 w_k1_FDX_s100
                                 + 0.914139 w_k1_IBM_s100
                                 + 0.990271 w_k1_INTC_s100
                                 + 0.908206 w_k1_KO_s100
                                 + 0.987082 w_k1_MAT_s100
                                 + 0.783806 w_k1_MSFT_s100
                                 + 0.975781 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.00016 w_k1_ADM_s100
                                 + 1.18708 w_k1_AXP_s100
                                 + 1.28979 w_k1_DIS_s100
                                 + 1.14236 w_k1_FDX_s100
                                 + 1.34465 w_k1_IBM_s100
                                 + 1.15244 w_k1_INTC_s100
                                 + 1.31548 w_k1_KO_s100
                                 + 0.937282 w_k1_MAT_s100
                                 + 1.00071 w_k1_MSFT_s100
                                 + 1.07738 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.00693 w_k1_ADM_s100
                                 + 1.61694 w_k1_AXP_s100
                                 + 1.58386 w_k1_DIS_s100
                                 + 1.42981 w_k1_FDX_s100
                                 + 1.15433 w_k1_IBM_s100
                                 + 1.3515 w_k1_INTC_s100 + 1.14308 w_k1_KO_s100
                                 + 1.52258 w_k1_MAT_s100
                                 + 1.2968 w_k1_MSFT_s100 + 1.28502 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.946068 w_k1_ADM_s100
                                 + 1.05633 w_k1_AXP_s100
                                 + 1.14872 w_k1_DIS_s100
                                 + 1.19234 w_k1_FDX_s100
                                 + 0.768231 w_k1_IBM_s100
                                 + 1.20185 w_k1_INTC_s100
                                 + 1.07131 w_k1_KO_s100
                                 + 0.814929 w_k1_MAT_s100
                                 + 1.07117 w_k1_MSFT_s100
                                 + 1.03874 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.757431 w_k1_ADM_s100
                                 + 0.774147 w_k1_AXP_s100
                                 + 0.653945 w_k1_DIS_s100
                                 + 0.873714 w_k1_FDX_s100
                                 + 0.711085 w_k1_IBM_s100
                                 + 0.941902 w_k1_INTC_s100
                                 + 0.690968 w_k1_KO_s100
                                 + 0.429078 w_k1_MAT_s100
                                 + 0.752887 w_k1_MSFT_s100
                                 + 0.818157 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.934641 w_k1_ADM_s100
                                 + 1.08562 w_k1_AXP_s100
                                 + 0.847406 w_k1_DIS_s100
                                 + 0.94537 w_k1_FDX_s100
                                 + 1.23126 w_k1_IBM_s100
                                 + 0.962578 w_k1_INTC_s100
                                 + 1.11006 w_k1_KO_s100 + 1.2761 w_k1_MAT_s100
                                 + 0.889322 w_k1_MSFT_s100
                                 + 1.02448 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.831794 w_k1_ADM_s100
                                 + 0.743521 w_k1_AXP_s100
                                 + 0.606502 w_k1_DIS_s100
                                 + 0.729188 w_k1_FDX_s100
                                 + 0.945176 w_k1_IBM_s100
                                 + 0.812416 w_k1_INTC_s100
                                 + 0.694824 w_k1_KO_s100
                                 + 0.675047 w_k1_MAT_s100
                                 + 0.689897 w_k1_MSFT_s100
                                 + 0.866561 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.882658 w_k1_ADM_s100
                                 + 0.764444 w_k1_AXP_s100
                                 + 0.917401 w_k1_DIS_s100
                                 + 1.01036 w_k1_FDX_s100
                                 + 0.666524 w_k1_IBM_s100
                                 + 1.10806 w_k1_INTC_s100
                                 + 0.945001 w_k1_KO_s100
                                 + 0.580472 w_k1_MAT_s100
                                 + 0.983831 w_k1_MSFT_s100
                                 + 0.948622 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.982407 w_k1_ADM_s101
                                 + 0.832523 w_k1_AXP_s101
                                 + 0.801514 w_k1_DIS_s101
                                 + 0.824038 w_k1_FDX_s101
                                 + 0.962555 w_k1_IBM_s101
                                 + 0.741522 w_k1_INTC_s101
                                 + 0.876681 w_k1_KO_s101
                                 + 1.31831 w_k1_MAT_s101
                                 + 0.902548 w_k1_MSFT_s101
                                 + 0.900819 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.49072 w_k1_ADM_s101
                                 + 1.41756 w_k1_AXP_s101
                                 + 1.59328 w_k1_DIS_s101
                                 + 1.18464 w_k1_FDX_s101
                                 + 1.61735 w_k1_IBM_s101
                                 + 1.2302 w_k1_INTC_s101 + 1.38152 w_k1_KO_s101
                                 + 1.80838 w_k1_MAT_s101
                                 + 1.18241 w_k1_MSFT_s101
                                 + 1.30036 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.833197 w_k1_ADM_s101
                                 + 0.800567 w_k1_AXP_s101
                                 + 0.8007 w_k1_DIS_s101
                                 + 0.897485 w_k1_FDX_s101
                                 + 0.719566 w_k1_IBM_s101
                                 + 0.946341 w_k1_INTC_s101
                                 + 0.752797 w_k1_KO_s101
                                 + 0.628473 w_k1_MAT_s101
                                 + 0.953328 w_k1_MSFT_s101
                                 + 0.906081 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09695 w_k1_ADM_s101
                                 + 0.865321 w_k1_AXP_s101
                                 + 1.18891 w_k1_DIS_s101
                                 + 1.00842 w_k1_FDX_s101
                                 + 0.909312 w_k1_IBM_s101
                                 + 0.776861 w_k1_INTC_s101
                                 + 0.927563 w_k1_KO_s101
                                 + 0.982065 w_k1_MAT_s101
                                 + 1.0235 w_k1_MSFT_s101
                                 + 0.999289 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.13318 w_k1_ADM_s101
                                 + 1.09881 w_k1_AXP_s101
                                 + 1.11083 w_k1_DIS_s101 + 1.0622 w_k1_FDX_s101
                                 + 1.32165 w_k1_IBM_s101
                                 + 0.902451 w_k1_INTC_s101
                                 + 1.14841 w_k1_KO_s101 + 1.6411 w_k1_MAT_s101
                                 + 1.1685 w_k1_MSFT_s101 + 1.21765 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.42878 w_k1_ADM_s101
                                 + 0.999507 w_k1_AXP_s101
                                 + 1.13819 w_k1_DIS_s101
                                 + 1.14214 w_k1_FDX_s101
                                 + 0.93388 w_k1_IBM_s101
                                 + 1.08626 w_k1_INTC_s101
                                 + 1.12362 w_k1_KO_s101 + 1.01346 w_k1_MAT_s101
                                 + 1.26858 w_k1_MSFT_s101
                                 + 1.12587 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.26591 w_k1_ADM_s101
                                 + 1.06679 w_k1_AXP_s101
                                 + 1.67609 w_k1_DIS_s101 + 1.3219 w_k1_FDX_s101
                                 + 1.06449 w_k1_IBM_s101
                                 + 1.1719 w_k1_INTC_s101 + 1.23448 w_k1_KO_s101
                                 + 1.05059 w_k1_MAT_s101
                                 + 1.35472 w_k1_MSFT_s101
                                 + 1.25524 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.01166 w_k1_ADM_s101
                                 + 0.978752 w_k1_AXP_s101
                                 + 1.03561 w_k1_DIS_s101
                                 + 0.94135 w_k1_FDX_s101
                                 + 1.55542 w_k1_IBM_s101
                                 + 0.96365 w_k1_INTC_s101
                                 + 1.13626 w_k1_KO_s101 + 1.58527 w_k1_MAT_s101
                                 + 1.11453 w_k1_MSFT_s101
                                 + 1.17151 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.00393 w_k1_ADM_s102
                                 + 1.10407 w_k1_AXP_s102 + 1.2163 w_k1_DIS_s102
                                 + 1.13769 w_k1_FDX_s102
                                 + 0.954504 w_k1_IBM_s102
                                 + 1.14587 w_k1_INTC_s102
                                 + 1.10886 w_k1_KO_s102 + 0.98997 w_k1_MAT_s102
                                 + 1.07462 w_k1_MSFT_s102
                                 + 1.02873 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.09573 w_k1_ADM_s102
                                 + 1.16078 w_k1_AXP_s102
                                 + 0.869188 w_k1_DIS_s102
                                 + 0.910396 w_k1_FDX_s102
                                 + 1.50532 w_k1_IBM_s102
                                 + 1.09295 w_k1_INTC_s102 + 1.2743 w_k1_KO_s102
                                 + 1.80172 w_k1_MAT_s102
                                 + 0.86822 w_k1_MSFT_s102
                                 + 1.01795 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.831617 w_k1_ADM_s102
                                 + 0.721127 w_k1_AXP_s102
                                 + 0.816795 w_k1_DIS_s102
                                 + 0.837761 w_k1_FDX_s102
                                 + 0.66924 w_k1_IBM_s102
                                 + 1.01133 w_k1_INTC_s102
                                 + 1.03513 w_k1_KO_s102
                                 + 0.539689 w_k1_MAT_s102
                                 + 0.89519 w_k1_MSFT_s102
                                 + 0.910276 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.15903 w_k1_ADM_s102
                                 + 0.879732 w_k1_AXP_s102
                                 + 0.916792 w_k1_DIS_s102
                                 + 1.00898 w_k1_FDX_s102
                                 + 0.924122 w_k1_IBM_s102
                                 + 1.10566 w_k1_INTC_s102
                                 + 0.997851 w_k1_KO_s102
                                 + 0.984535 w_k1_MAT_s102
                                 + 1.07316 w_k1_MSFT_s102
                                 + 0.921614 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.987437 w_k1_ADM_s102
                                 + 0.809714 w_k1_AXP_s102
                                 + 1.07026 w_k1_DIS_s102
                                 + 0.993944 w_k1_FDX_s102
                                 + 1.22897 w_k1_IBM_s102
                                 + 1.20212 w_k1_INTC_s102
                                 + 1.32165 w_k1_KO_s102 + 1.21422 w_k1_MAT_s102
                                 + 1.16307 w_k1_MSFT_s102
                                 + 1.03136 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.00569 w_k1_ADM_s102
                                 + 0.740239 w_k1_AXP_s102
                                 + 0.714229 w_k1_DIS_s102
                                 + 0.786234 w_k1_FDX_s102
                                 + 1.03985 w_k1_IBM_s102
                                 + 0.798715 w_k1_INTC_s102
                                 + 0.99118 w_k1_KO_s102 + 1.19321 w_k1_MAT_s102
                                 + 0.855438 w_k1_MSFT_s102
                                 + 0.940536 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.19562 w_k1_ADM_s102
                                 + 1.31696 w_k1_AXP_s102
                                 + 1.25275 w_k1_DIS_s102
                                 + 1.17039 w_k1_FDX_s102
                                 + 1.19496 w_k1_IBM_s102
                                 + 1.15798 w_k1_INTC_s102
                                 + 1.39835 w_k1_KO_s102 + 1.33274 w_k1_MAT_s102
                                 + 1.14882 w_k1_MSFT_s102
                                 + 1.10474 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.32237 w_k1_ADM_s102
                                 + 1.25163 w_k1_AXP_s102
                                 + 1.73714 w_k1_DIS_s102
                                 + 1.32796 w_k1_FDX_s102
                                 + 1.33274 w_k1_IBM_s102
                                 + 1.41789 w_k1_INTC_s102
                                 + 1.46646 w_k1_KO_s102 + 1.14322 w_k1_MAT_s102
                                 + 1.25323 w_k1_MSFT_s102
                                 + 1.23463 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.05297 w_k1_ADM_s103
                                 + 1.17081 w_k1_AXP_s103
                                 + 0.923269 w_k1_DIS_s103
                                 + 1.01795 w_k1_FDX_s103
                                 + 0.962657 w_k1_IBM_s103
                                 + 0.976259 w_k1_INTC_s103
                                 + 1.11258 w_k1_KO_s103 + 1.22108 w_k1_MAT_s103
                                 + 0.886977 w_k1_MSFT_s103
                                 + 1.03125 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.945556 w_k1_ADM_s103
                                 + 0.736981 w_k1_AXP_s103
                                 + 1.05635 w_k1_DIS_s103
                                 + 1.06441 w_k1_FDX_s103
                                 + 0.684893 w_k1_IBM_s103
                                 + 1.06934 w_k1_INTC_s103
                                 + 0.975092 w_k1_KO_s103
                                 + 0.870724 w_k1_MAT_s103
                                 + 1.11566 w_k1_MSFT_s103
                                 + 0.962191 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.736653 w_k1_ADM_s103
                                 + 0.870313 w_k1_AXP_s103
                                 + 0.495008 w_k1_DIS_s103
                                 + 0.745524 w_k1_FDX_s103
                                 + 0.79906 w_k1_IBM_s103
                                 + 0.789385 w_k1_INTC_s103
                                 + 0.69963 w_k1_KO_s103
                                 + 0.892965 w_k1_MAT_s103
                                 + 0.723744 w_k1_MSFT_s103
                                 + 0.79474 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.14028 w_k1_ADM_s103
                                 + 1.22619 w_k1_AXP_s103
                                 + 1.49134 w_k1_DIS_s103
                                 + 1.32566 w_k1_FDX_s103
                                 + 1.24522 w_k1_IBM_s103
                                 + 1.39565 w_k1_INTC_s103
                                 + 1.37838 w_k1_KO_s103 + 1.21684 w_k1_MAT_s103
                                 + 1.27525 w_k1_MSFT_s103
                                 + 1.24543 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.954799 w_k1_ADM_s103
                                 + 1.21464 w_k1_AXP_s103
                                 + 1.24277 w_k1_DIS_s103
                                 + 1.01372 w_k1_FDX_s103
                                 + 1.59798 w_k1_IBM_s103
                                 + 1.15445 w_k1_INTC_s103
                                 + 1.56487 w_k1_KO_s103 + 1.56479 w_k1_MAT_s103
                                 + 1.07618 w_k1_MSFT_s103
                                 + 1.15474 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.774234 w_k1_ADM_s103
                                 + 0.737861 w_k1_AXP_s103
                                 + 0.649859 w_k1_DIS_s103
                                 + 0.817615 w_k1_FDX_s103
                                 + 0.760679 w_k1_IBM_s103
                                 + 0.767118 w_k1_INTC_s103
                                 + 0.832057 w_k1_KO_s103
                                 + 0.692504 w_k1_MAT_s103
                                 + 0.738014 w_k1_MSFT_s103
                                 + 0.903034 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.921268 w_k1_ADM_s103
                                 + 1.45947 w_k1_AXP_s103
                                 + 1.09021 w_k1_DIS_s103
                                 + 1.08989 w_k1_FDX_s103 + 1.1902 w_k1_IBM_s103
                                 + 1.12804 w_k1_INTC_s103
                                 + 1.21189 w_k1_KO_s103 + 1.68241 w_k1_MAT_s103
                                 + 1.09123 w_k1_MSFT_s103
                                 + 1.27036 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.823446 w_k1_ADM_s103
                                 + 0.820169 w_k1_AXP_s103
                                 + 0.954371 w_k1_DIS_s103
                                 + 0.912913 w_k1_FDX_s103
                                 + 1.04829 w_k1_IBM_s103
                                 + 0.881652 w_k1_INTC_s103
                                 + 1.10644 w_k1_KO_s103
                                 + 0.931025 w_k1_MAT_s103
                                 + 0.909349 w_k1_MSFT_s103
                                 + 1.04148 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.01288 w_k1_ADM_s104
                                 + 0.916208 w_k1_AXP_s104
                                 + 0.866218 w_k1_DIS_s104
                                 + 0.872425 w_k1_FDX_s104
                                 + 1.15201 w_k1_IBM_s104
                                 + 0.957256 w_k1_INTC_s104
                                 + 0.89663 w_k1_KO_s104 + 1.20798 w_k1_MAT_s104
                                 + 0.918391 w_k1_MSFT_s104
                                 + 1.05568 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.01827 w_k1_ADM_s104
                                 + 1.3575 w_k1_AXP_s104
                                 + 0.982686 w_k1_DIS_s104
                                 + 1.02188 w_k1_FDX_s104
                                 + 0.980516 w_k1_IBM_s104
                                 + 1.08938 w_k1_INTC_s104
                                 + 0.928617 w_k1_KO_s104
                                 + 1.39239 w_k1_MAT_s104
                                 + 0.888227 w_k1_MSFT_s104
                                 + 0.975696 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.811829 w_k1_ADM_s104
                                 + 0.725027 w_k1_AXP_s104
                                 + 0.623328 w_k1_DIS_s104
                                 + 0.741366 w_k1_FDX_s104
                                 + 0.721075 w_k1_IBM_s104
                                 + 0.83121 w_k1_INTC_s104
                                 + 0.685305 w_k1_KO_s104
                                 + 0.783054 w_k1_MAT_s104
                                 + 0.791313 w_k1_MSFT_s104
                                 + 0.799436 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.08252 w_k1_ADM_s104
                                 + 1.38512 w_k1_AXP_s104
                                 + 1.49059 w_k1_DIS_s104
                                 + 1.23006 w_k1_FDX_s104
                                 + 1.39846 w_k1_IBM_s104
                                 + 1.18371 w_k1_INTC_s104
                                 + 1.28677 w_k1_KO_s104 + 1.42108 w_k1_MAT_s104
                                 + 1.18301 w_k1_MSFT_s104 + 1.2838 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.13249 w_k1_ADM_s104
                                 + 1.17513 w_k1_AXP_s104
                                 + 1.33771 w_k1_DIS_s104
                                 + 1.27644 w_k1_FDX_s104
                                 + 0.816933 w_k1_IBM_s104
                                 + 0.998654 w_k1_INTC_s104
                                 + 0.933725 w_k1_KO_s104
                                 + 1.09289 w_k1_MAT_s104
                                 + 1.21556 w_k1_MSFT_s104
                                 + 1.08673 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.966169 w_k1_ADM_s104
                                 + 1.17865 w_k1_AXP_s104
                                 + 1.03094 w_k1_DIS_s104
                                 + 0.973081 w_k1_FDX_s104
                                 + 1.29215 w_k1_IBM_s104
                                 + 0.804037 w_k1_INTC_s104
                                 + 1.06822 w_k1_KO_s104 + 1.66299 w_k1_MAT_s104
                                 + 0.94286 w_k1_MSFT_s104
                                 + 1.10955 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.702191 w_k1_ADM_s104
                                 + 0.833708 w_k1_AXP_s104
                                 + 0.705278 w_k1_DIS_s104
                                 + 0.766588 w_k1_FDX_s104
                                 + 0.943858 w_k1_IBM_s104
                                 + 0.793774 w_k1_INTC_s104
                                 + 0.796923 w_k1_KO_s104
                                 + 0.833738 w_k1_MAT_s104
                                 + 0.804903 w_k1_MSFT_s104
                                 + 0.954671 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.02596 w_k1_ADM_s104
                                 + 0.962958 w_k1_AXP_s104
                                 + 0.914454 w_k1_DIS_s104
                                 + 0.959617 w_k1_FDX_s104
                                 + 0.813598 w_k1_IBM_s104
                                 + 1.06474 w_k1_INTC_s104
                                 + 0.868011 w_k1_KO_s104
                                 + 0.872301 w_k1_MAT_s104
                                 + 1.04089 w_k1_MSFT_s104
                                 + 0.927639 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.931825 w_k1_ADM_s105
                                 + 0.800038 w_k1_AXP_s105
                                 + 0.795461 w_k1_DIS_s105
                                 + 0.845377 w_k1_FDX_s105
                                 + 1.58596 w_k1_IBM_s105
                                 + 0.838945 w_k1_INTC_s105
                                 + 0.963006 w_k1_KO_s105 + 1.4432 w_k1_MAT_s105
                                 + 0.730198 w_k1_MSFT_s105
                                 + 0.867632 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.36843 w_k1_ADM_s105
                                 + 1.12794 w_k1_AXP_s105
                                 + 1.85314 w_k1_DIS_s105
                                 + 1.28319 w_k1_FDX_s105 + 1.413 w_k1_IBM_s105
                                 + 1.2848 w_k1_INTC_s105 + 1.49353 w_k1_KO_s105
                                 + 1.48154 w_k1_MAT_s105
                                 + 1.39211 w_k1_MSFT_s105
                                 + 1.28593 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.858275 w_k1_ADM_s105
                                 + 0.890866 w_k1_AXP_s105
                                 + 0.796166 w_k1_DIS_s105
                                 + 0.921844 w_k1_FDX_s105
                                 + 1.18531 w_k1_IBM_s105
                                 + 0.931431 w_k1_INTC_s105
                                 + 0.922761 w_k1_KO_s105
                                 + 0.919838 w_k1_MAT_s105
                                 + 0.94097 w_k1_MSFT_s105
                                 + 0.936825 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.84251 w_k1_ADM_s105
                                 + 0.592074 w_k1_AXP_s105
                                 + 0.924357 w_k1_DIS_s105
                                 + 0.801553 w_k1_FDX_s105
                                 + 1.07406 w_k1_IBM_s105
                                 + 0.933596 w_k1_INTC_s105
                                 + 0.973111 w_k1_KO_s105
                                 + 0.650191 w_k1_MAT_s105
                                 + 0.912341 w_k1_MSFT_s105
                                 + 1.00356 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.876267 w_k1_ADM_s105
                                 + 0.805252 w_k1_AXP_s105
                                 + 0.895566 w_k1_DIS_s105
                                 + 0.929171 w_k1_FDX_s105
                                 + 0.852325 w_k1_IBM_s105
                                 + 0.790706 w_k1_INTC_s105
                                 + 0.832713 w_k1_KO_s105
                                 + 0.850815 w_k1_MAT_s105
                                 + 0.854521 w_k1_MSFT_s105
                                 + 0.914632 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.671785 w_k1_ADM_s105
                                 + 0.634068 w_k1_AXP_s105
                                 + 0.52778 w_k1_DIS_s105
                                 + 0.643182 w_k1_FDX_s105
                                 + 0.949093 w_k1_IBM_s105
                                 + 0.680626 w_k1_INTC_s105
                                 + 0.836644 w_k1_KO_s105
                                 + 1.19532 w_k1_MAT_s105
                                 + 0.70283 w_k1_MSFT_s105
                                 + 0.846403 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.06588 w_k1_ADM_s105
                                 + 1.2815 w_k1_AXP_s105 + 1.25658 w_k1_DIS_s105
                                 + 1.26104 w_k1_FDX_s105
                                 + 1.06194 w_k1_IBM_s105
                                 + 1.22238 w_k1_INTC_s105
                                 + 1.18366 w_k1_KO_s105 + 1.37554 w_k1_MAT_s105
                                 + 1.14697 w_k1_MSFT_s105 + 1.15 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.737551 w_k1_ADM_s105
                                 + 0.689603 w_k1_AXP_s105
                                 + 0.664794 w_k1_DIS_s105
                                 + 0.858002 w_k1_FDX_s105
                                 + 0.614522 w_k1_IBM_s105
                                 + 1.01541 w_k1_INTC_s105
                                 + 0.753206 w_k1_KO_s105
                                 + 0.50218 w_k1_MAT_s105
                                 + 0.958239 w_k1_MSFT_s105
                                 + 0.814535 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.984736 w_k1_ADM_s106
                                 + 0.708888 w_k1_AXP_s106
                                 + 0.719376 w_k1_DIS_s106
                                 + 0.866254 w_k1_FDX_s106
                                 + 0.756254 w_k1_IBM_s106
                                 + 0.789726 w_k1_INTC_s106
                                 + 0.722262 w_k1_KO_s106
                                 + 0.578927 w_k1_MAT_s106
                                 + 0.740033 w_k1_MSFT_s106
                                 + 1.03811 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.711119 w_k1_ADM_s106
                                 + 0.616033 w_k1_AXP_s106
                                 + 0.563558 w_k1_DIS_s106
                                 + 0.72308 w_k1_FDX_s106
                                 + 0.640075 w_k1_IBM_s106
                                 + 0.793003 w_k1_INTC_s106
                                 + 0.617895 w_k1_KO_s106
                                 + 0.643859 w_k1_MAT_s106
                                 + 0.701398 w_k1_MSFT_s106
                                 + 0.880179 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.941104 w_k1_ADM_s106
                                 + 1.06721 w_k1_AXP_s106
                                 + 1.70301 w_k1_DIS_s106 + 1.2451 w_k1_FDX_s106
                                 + 1.21696 w_k1_IBM_s106
                                 + 1.23234 w_k1_INTC_s106
                                 + 1.20498 w_k1_KO_s106
                                 + 0.938239 w_k1_MAT_s106
                                 + 1.19948 w_k1_MSFT_s106
                                 + 1.29801 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.03074 w_k1_ADM_s106
                                 + 1.17637 w_k1_AXP_s106 + 1.33 w_k1_DIS_s106
                                 + 1.04794 w_k1_FDX_s106
                                 + 1.77863 w_k1_IBM_s106
                                 + 1.07659 w_k1_INTC_s106
                                 + 1.14943 w_k1_KO_s106 + 1.50122 w_k1_MAT_s106
                                 + 1.07541 w_k1_MSFT_s106
                                 + 1.20401 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.284 w_k1_ADM_s106
                                 + 1.13502 w_k1_AXP_s106
                                 + 1.17865 w_k1_DIS_s106 + 1.2304 w_k1_FDX_s106
                                 + 0.876617 w_k1_IBM_s106
                                 + 1.22821 w_k1_INTC_s106
                                 + 0.964297 w_k1_KO_s106
                                 + 0.679685 w_k1_MAT_s106
                                 + 1.16061 w_k1_MSFT_s106
                                 + 1.19957 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.5777 w_k1_ADM_s106
                                 + 1.5737 w_k1_AXP_s106 + 1.66007 w_k1_DIS_s106
                                 + 1.44555 w_k1_FDX_s106
                                 + 1.27882 w_k1_IBM_s106
                                 + 1.09639 w_k1_INTC_s106
                                 + 1.25358 w_k1_KO_s106 + 1.26353 w_k1_MAT_s106
                                 + 1.10729 w_k1_MSFT_s106
                                 + 1.43226 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.903001 w_k1_ADM_s106
                                 + 1.03932 w_k1_AXP_s106
                                 + 1.07492 w_k1_DIS_s106
                                 + 1.05038 w_k1_FDX_s106
                                 + 1.27144 w_k1_IBM_s106
                                 + 0.940832 w_k1_INTC_s106
                                 + 1.07786 w_k1_KO_s106 + 1.24975 w_k1_MAT_s106
                                 + 1.01124 w_k1_MSFT_s106
                                 + 1.11087 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.897302 w_k1_ADM_s106
                                 + 0.89653 w_k1_AXP_s106
                                 + 0.861531 w_k1_DIS_s106
                                 + 0.996383 w_k1_FDX_s106
                                 + 0.832339 w_k1_IBM_s106
                                 + 1.04487 w_k1_INTC_s106
                                 + 0.804002 w_k1_KO_s106
                                 + 0.738409 w_k1_MAT_s106
                                 + 0.911783 w_k1_MSFT_s106
                                 + 0.952814 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.16453 w_k1_ADM_s107
                                 + 0.849043 w_k1_AXP_s107
                                 + 0.835659 w_k1_DIS_s107
                                 + 0.859956 w_k1_FDX_s107
                                 + 0.871917 w_k1_IBM_s107
                                 + 0.952115 w_k1_INTC_s107
                                 + 0.7691 w_k1_KO_s107 + 0.804069 w_k1_MAT_s107
                                 + 0.839778 w_k1_MSFT_s107
                                 + 0.888541 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.20566 w_k1_ADM_s107
                                 + 1.48884 w_k1_AXP_s107
                                 + 1.63662 w_k1_DIS_s107
                                 + 1.21058 w_k1_FDX_s107
                                 + 1.48971 w_k1_IBM_s107
                                 + 1.25339 w_k1_INTC_s107
                                 + 1.42756 w_k1_KO_s107 + 1.51882 w_k1_MAT_s107
                                 + 1.23069 w_k1_MSFT_s107
                                 + 1.23287 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.11864 w_k1_ADM_s107
                                 + 1.27443 w_k1_AXP_s107
                                 + 1.28624 w_k1_DIS_s107 + 1.1539 w_k1_FDX_s107
                                 + 0.961373 w_k1_IBM_s107
                                 + 1.18586 w_k1_INTC_s107
                                 + 1.04604 w_k1_KO_s107
                                 + 0.928571 w_k1_MAT_s107
                                 + 1.07554 w_k1_MSFT_s107
                                 + 1.15272 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.05242 w_k1_ADM_s107
                                 + 0.952222 w_k1_AXP_s107
                                 + 1.11567 w_k1_DIS_s107
                                 + 0.923265 w_k1_FDX_s107
                                 + 1.34866 w_k1_IBM_s107
                                 + 0.919506 w_k1_INTC_s107
                                 + 1.19333 w_k1_KO_s107 + 1.74162 w_k1_MAT_s107
                                 + 1.00092 w_k1_MSFT_s107
                                 + 1.11432 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.11179 w_k1_ADM_s107
                                 + 0.759068 w_k1_AXP_s107
                                 + 1.12593 w_k1_DIS_s107
                                 + 1.02727 w_k1_FDX_s107
                                 + 0.682526 w_k1_IBM_s107
                                 + 1.00336 w_k1_INTC_s107
                                 + 0.947945 w_k1_KO_s107
                                 + 0.673724 w_k1_MAT_s107
                                 + 1.06428 w_k1_MSFT_s107
                                 + 1.06793 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.691353 w_k1_ADM_s107
                                 + 0.635993 w_k1_AXP_s107
                                 + 0.685274 w_k1_DIS_s107
                                 + 0.68807 w_k1_FDX_s107
                                 + 0.885524 w_k1_IBM_s107
                                 + 0.863767 w_k1_INTC_s107
                                 + 0.847695 w_k1_KO_s107
                                 + 0.886061 w_k1_MAT_s107
                                 + 0.813108 w_k1_MSFT_s107
                                 + 0.928184 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.03998 w_k1_ADM_s107
                                 + 1.11716 w_k1_AXP_s107
                                 + 0.75274 w_k1_DIS_s107
                                 + 0.956211 w_k1_FDX_s107
                                 + 0.74919 w_k1_IBM_s107
                                 + 0.929299 w_k1_INTC_s107
                                 + 0.694067 w_k1_KO_s107
                                 + 1.20045 w_k1_MAT_s107
                                 + 0.779677 w_k1_MSFT_s107
                                 + 0.939483 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.853507 w_k1_ADM_s107
                                 + 0.762235 w_k1_AXP_s107
                                 + 0.85505 w_k1_DIS_s107
                                 + 0.905664 w_k1_FDX_s107
                                 + 0.780953 w_k1_IBM_s107
                                 + 0.930859 w_k1_INTC_s107
                                 + 0.895149 w_k1_KO_s107
                                 + 0.950166 w_k1_MAT_s107
                                 + 0.912722 w_k1_MSFT_s107
                                 + 1.04868 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.07226 w_k1_ADM_s110
                                 + 1.12991 w_k1_AXP_s110
                                 + 0.931678 w_k1_DIS_s110
                                 + 1.06798 w_k1_FDX_s110
                                 + 0.770262 w_k1_IBM_s110
                                 + 1.10294 w_k1_INTC_s110
                                 + 0.910221 w_k1_KO_s110 + 1.0788 w_k1_MAT_s110
                                 + 0.919478 w_k1_MSFT_s110
                                 + 0.901572 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.20388 w_k1_ADM_s110
                                 + 1.67693 w_k1_AXP_s110
                                 + 1.41147 w_k1_DIS_s110
                                 + 1.33962 w_k1_FDX_s110
                                 + 1.77396 w_k1_IBM_s110
                                 + 1.30027 w_k1_INTC_s110
                                 + 1.41657 w_k1_KO_s110 + 2.32873 w_k1_MAT_s110
                                 + 1.22412 w_k1_MSFT_s110
                                 + 1.39281 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.838518 w_k1_ADM_s110
                                 + 1.01037 w_k1_AXP_s110
                                 + 0.829799 w_k1_DIS_s110
                                 + 0.928376 w_k1_FDX_s110
                                 + 1.23999 w_k1_IBM_s110
                                 + 1.11011 w_k1_INTC_s110
                                 + 1.00723 w_k1_KO_s110 + 1.42834 w_k1_MAT_s110
                                 + 0.983458 w_k1_MSFT_s110
                                 + 1.11492 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.961886 w_k1_ADM_s110
                                 + 1.04587 w_k1_AXP_s110 + 1.2293 w_k1_DIS_s110
                                 + 1.18352 w_k1_FDX_s110
                                 + 1.01656 w_k1_IBM_s110
                                 + 1.2572 w_k1_INTC_s110 + 1.06117 w_k1_KO_s110
                                 + 1.04412 w_k1_MAT_s110
                                 + 1.05608 w_k1_MSFT_s110 + 1.1465 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.18394 w_k1_ADM_s110
                                 + 0.833357 w_k1_AXP_s110
                                 + 0.8114 w_k1_DIS_s110
                                 + 0.945836 w_k1_FDX_s110
                                 + 0.816735 w_k1_IBM_s110
                                 + 0.966261 w_k1_INTC_s110
                                 + 0.896044 w_k1_KO_s110
                                 + 0.797072 w_k1_MAT_s110
                                 + 0.986921 w_k1_MSFT_s110
                                 + 0.937605 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.29868 w_k1_ADM_s110
                                 + 1.41805 w_k1_AXP_s110
                                 + 1.58767 w_k1_DIS_s110
                                 + 1.41958 w_k1_FDX_s110
                                 + 1.00688 w_k1_IBM_s110
                                 + 1.68685 w_k1_INTC_s110
                                 + 1.18595 w_k1_KO_s110 + 1.154 w_k1_MAT_s110
                                 + 1.35648 w_k1_MSFT_s110
                                 + 1.25496 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.962969 w_k1_ADM_s110
                                 + 0.662956 w_k1_AXP_s110
                                 + 0.558171 w_k1_DIS_s110
                                 + 0.743261 w_k1_FDX_s110
                                 + 0.825093 w_k1_IBM_s110
                                 + 0.828228 w_k1_INTC_s110
                                 + 0.703923 w_k1_KO_s110
                                 + 0.702547 w_k1_MAT_s110
                                 + 0.818575 w_k1_MSFT_s110
                                 + 0.754132 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.10301 w_k1_ADM_s110
                                 + 1.38882 w_k1_AXP_s110
                                 + 1.54279 w_k1_DIS_s110
                                 + 1.35241 w_k1_FDX_s110
                                 + 1.32547 w_k1_IBM_s110
                                 + 1.21836 w_k1_INTC_s110
                                 + 1.29707 w_k1_KO_s110 + 1.57683 w_k1_MAT_s110
                                 + 1.35884 w_k1_MSFT_s110
                                 + 1.35405 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.05927 w_k1_ADM_s111
                                 + 0.670353 w_k1_AXP_s111
                                 + 0.716716 w_k1_DIS_s111
                                 + 0.943294 w_k1_FDX_s111
                                 + 0.65502 w_k1_IBM_s111
                                 + 0.924222 w_k1_INTC_s111
                                 + 0.727467 w_k1_KO_s111
                                 + 0.433216 w_k1_MAT_s111
                                 + 0.951288 w_k1_MSFT_s111
                                 + 0.818729 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.868757 w_k1_ADM_s111
                                 + 0.887644 w_k1_AXP_s111
                                 + 0.73825 w_k1_DIS_s111
                                 + 0.858596 w_k1_FDX_s111
                                 + 1.18897 w_k1_IBM_s111
                                 + 0.778088 w_k1_INTC_s111
                                 + 0.949622 w_k1_KO_s111
                                 + 1.07008 w_k1_MAT_s111
                                 + 0.87757 w_k1_MSFT_s111
                                 + 0.88153 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.881845 w_k1_ADM_s111
                                 + 0.602887 w_k1_AXP_s111
                                 + 0.638628 w_k1_DIS_s111
                                 + 0.779558 w_k1_FDX_s111
                                 + 0.863011 w_k1_IBM_s111
                                 + 0.77258 w_k1_INTC_s111
                                 + 0.812985 w_k1_KO_s111
                                 + 0.64885 w_k1_MAT_s111
                                 + 0.851229 w_k1_MSFT_s111
                                 + 0.835099 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.20709 w_k1_ADM_s111
                                 + 1.11543 w_k1_AXP_s111 + 1.6224 w_k1_DIS_s111
                                 + 1.38283 w_k1_FDX_s111
                                 + 1.11992 w_k1_IBM_s111
                                 + 1.07303 w_k1_INTC_s111
                                 + 1.46054 w_k1_KO_s111
                                 + 0.928377 w_k1_MAT_s111
                                 + 1.40264 w_k1_MSFT_s111
                                 + 1.12994 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.00966 w_k1_ADM_s111
                                 + 0.889233 w_k1_AXP_s111
                                 + 1.16252 w_k1_DIS_s111
                                 + 1.03791 w_k1_FDX_s111
                                 + 1.32788 w_k1_IBM_s111
                                 + 0.849782 w_k1_INTC_s111
                                 + 1.13373 w_k1_KO_s111
                                 + 0.941816 w_k1_MAT_s111
                                 + 1.00368 w_k1_MSFT_s111
                                 + 1.09982 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.04746 w_k1_ADM_s111
                                 + 1.02809 w_k1_AXP_s111
                                 + 1.19862 w_k1_DIS_s111
                                 + 1.16892 w_k1_FDX_s111
                                 + 0.974866 w_k1_IBM_s111
                                 + 0.985803 w_k1_INTC_s111
                                 + 1.05155 w_k1_KO_s111
                                 + 0.769076 w_k1_MAT_s111
                                 + 1.09811 w_k1_MSFT_s111
                                 + 0.952901 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.9504 w_k1_ADM_s111
                                 + 0.865614 w_k1_AXP_s111
                                 + 0.790713 w_k1_DIS_s111
                                 + 0.994502 w_k1_FDX_s111
                                 + 0.886933 w_k1_IBM_s111
                                 + 0.917159 w_k1_INTC_s111
                                 + 1.00149 w_k1_KO_s111 + 1.03813 w_k1_MAT_s111
                                 + 0.993268 w_k1_MSFT_s111
                                 + 1.01504 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.15187 w_k1_ADM_s111
                                 + 1.34056 w_k1_AXP_s111
                                 + 1.66062 w_k1_DIS_s111
                                 + 1.24889 w_k1_FDX_s111
                                 + 1.60843 w_k1_IBM_s111
                                 + 0.845507 w_k1_INTC_s111
                                 + 1.40998 w_k1_KO_s111 + 1.50551 w_k1_MAT_s111
                                 + 1.21124 w_k1_MSFT_s111
                                 + 1.23808 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.08959 w_k1_ADM_s112
                                 + 1.17815 w_k1_AXP_s112
                                 + 1.31705 w_k1_DIS_s112
                                 + 1.07154 w_k1_FDX_s112
                                 + 1.25637 w_k1_IBM_s112
                                 + 1.08796 w_k1_INTC_s112
                                 + 1.21125 w_k1_KO_s112 + 1.01167 w_k1_MAT_s112
                                 + 1.11932 w_k1_MSFT_s112
                                 + 1.11061 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.34885 w_k1_ADM_s112
                                 + 1.5095 w_k1_AXP_s112 + 1.30618 w_k1_DIS_s112
                                 + 1.27225 w_k1_FDX_s112
                                 + 0.921473 w_k1_IBM_s112
                                 + 1.16335 w_k1_INTC_s112
                                 + 0.992176 w_k1_KO_s112
                                 + 0.969556 w_k1_MAT_s112
                                 + 1.16577 w_k1_MSFT_s112
                                 + 0.95773 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.36116 w_k1_ADM_s112
                                 + 1.62516 w_k1_AXP_s112
                                 + 1.62501 w_k1_DIS_s112
                                 + 1.36199 w_k1_FDX_s112
                                 + 1.55921 w_k1_IBM_s112
                                 + 1.51572 w_k1_INTC_s112
                                 + 1.52111 w_k1_KO_s112 + 1.35477 w_k1_MAT_s112
                                 + 1.45102 w_k1_MSFT_s112
                                 + 1.29415 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.0295 w_k1_ADM_s112
                                 + 1.08873 w_k1_AXP_s112
                                 + 0.842909 w_k1_DIS_s112
                                 + 0.82414 w_k1_FDX_s112
                                 + 0.959186 w_k1_IBM_s112
                                 + 0.796083 w_k1_INTC_s112
                                 + 0.885374 w_k1_KO_s112
                                 + 0.859298 w_k1_MAT_s112
                                 + 0.864023 w_k1_MSFT_s112
                                 + 0.930093 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.31337 w_k1_ADM_s112
                                 + 1.06725 w_k1_AXP_s112
                                 + 0.86186 w_k1_DIS_s112
                                 + 1.04677 w_k1_FDX_s112
                                 + 0.725892 w_k1_IBM_s112
                                 + 1.20097 w_k1_INTC_s112
                                 + 0.955097 w_k1_KO_s112
                                 + 0.585936 w_k1_MAT_s112
                                 + 1.0171 w_k1_MSFT_s112
                                 + 0.980359 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.920964 w_k1_ADM_s112
                                 + 0.71481 w_k1_AXP_s112
                                 + 0.678671 w_k1_DIS_s112
                                 + 0.774617 w_k1_FDX_s112
                                 + 0.811327 w_k1_IBM_s112
                                 + 0.839069 w_k1_INTC_s112
                                 + 0.685022 w_k1_KO_s112
                                 + 0.599451 w_k1_MAT_s112
                                 + 0.804202 w_k1_MSFT_s112
                                 + 0.826776 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.815582 w_k1_ADM_s112
                                 + 0.780374 w_k1_AXP_s112
                                 + 0.78925 w_k1_DIS_s112
                                 + 0.805947 w_k1_FDX_s112
                                 + 1.21062 w_k1_IBM_s112
                                 + 0.886599 w_k1_INTC_s112
                                 + 1.01776 w_k1_KO_s112
                                 + 0.845992 w_k1_MAT_s112
                                 + 0.799274 w_k1_MSFT_s112
                                 + 0.852843 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.052 w_k1_ADM_s112
                                 + 0.9182 w_k1_AXP_s112
                                 + 0.949481 w_k1_DIS_s112
                                 + 0.89467 w_k1_FDX_s112
                                 + 1.13652 w_k1_IBM_s112
                                 + 0.886815 w_k1_INTC_s112
                                 + 0.966308 w_k1_KO_s112
                                 + 1.00115 w_k1_MAT_s112
                                 + 0.972491 w_k1_MSFT_s112
                                 + 1.05824 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.876004 w_k1_ADM_s113
                                 + 1.18293 w_k1_AXP_s113
                                 + 1.10161 w_k1_DIS_s113
                                 + 0.973757 w_k1_FDX_s113
                                 + 1.40264 w_k1_IBM_s113
                                 + 1.22524 w_k1_INTC_s113
                                 + 1.27973 w_k1_KO_s113 + 1.41014 w_k1_MAT_s113
                                 + 1.06355 w_k1_MSFT_s113
                                 + 1.15215 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.33304 w_k1_ADM_s113
                                 + 1.54225 w_k1_AXP_s113
                                 + 1.78461 w_k1_DIS_s113
                                 + 1.55974 w_k1_FDX_s113 + 1.2006 w_k1_IBM_s113
                                 + 1.50715 w_k1_INTC_s113
                                 + 1.40345 w_k1_KO_s113 + 1.4686 w_k1_MAT_s113
                                 + 1.42167 w_k1_MSFT_s113
                                 + 1.21655 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.785573 w_k1_ADM_s113
                                 + 0.644648 w_k1_AXP_s113
                                 + 0.661243 w_k1_DIS_s113
                                 + 0.779165 w_k1_FDX_s113
                                 + 0.755976 w_k1_IBM_s113
                                 + 0.932628 w_k1_INTC_s113
                                 + 0.815879 w_k1_KO_s113
                                 + 0.658857 w_k1_MAT_s113
                                 + 0.950656 w_k1_MSFT_s113
                                 + 0.793038 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.861524 w_k1_ADM_s113
                                 + 0.93034 w_k1_AXP_s113
                                 + 1.03923 w_k1_DIS_s113
                                 + 0.971062 w_k1_FDX_s113
                                 + 1.03765 w_k1_IBM_s113
                                 + 1.21685 w_k1_INTC_s113
                                 + 1.12426 w_k1_KO_s113
                                 + 0.926587 w_k1_MAT_s113
                                 + 1.12646 w_k1_MSFT_s113 + 1.0686 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.21748 w_k1_ADM_s113
                                 + 1.3524 w_k1_AXP_s113 + 1.21 w_k1_DIS_s113
                                 + 1.21822 w_k1_FDX_s113
                                 + 0.986882 w_k1_IBM_s113
                                 + 1.19267 w_k1_INTC_s113
                                 + 0.953866 w_k1_KO_s113
                                 + 1.07708 w_k1_MAT_s113
                                 + 1.18344 w_k1_MSFT_s113
                                 + 1.11293 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.04859 w_k1_ADM_s113
                                 + 0.979543 w_k1_AXP_s113
                                 + 0.827674 w_k1_DIS_s113
                                 + 0.991605 w_k1_FDX_s113
                                 + 0.913933 w_k1_IBM_s113
                                 + 1.04971 w_k1_INTC_s113
                                 + 1.04762 w_k1_KO_s113
                                 + 0.923002 w_k1_MAT_s113
                                 + 0.991287 w_k1_MSFT_s113
                                 + 1.11731 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.649541 w_k1_ADM_s113
                                 + 0.635583 w_k1_AXP_s113
                                 + 0.457035 w_k1_DIS_s113
                                 + 0.683139 w_k1_FDX_s113
                                 + 0.949101 w_k1_IBM_s113
                                 + 0.873833 w_k1_INTC_s113
                                 + 0.76477 w_k1_KO_s113
                                 + 0.987968 w_k1_MAT_s113
                                 + 0.800444 w_k1_MSFT_s113
                                 + 0.826384 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.892755 w_k1_ADM_s113
                                 + 0.930699 w_k1_AXP_s113
                                 + 0.719225 w_k1_DIS_s113
                                 + 0.924678 w_k1_FDX_s113
                                 + 0.715172 w_k1_IBM_s113
                                 + 0.90715 w_k1_INTC_s113
                                 + 0.779878 w_k1_KO_s113
                                 + 0.913053 w_k1_MAT_s113
                                 + 0.92985 w_k1_MSFT_s113
                                 + 0.871542 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.10964 w_k1_ADM_s114
                                 + 1.09354 w_k1_AXP_s114
                                 + 1.10871 w_k1_DIS_s114
                                 + 0.930747 w_k1_FDX_s114
                                 + 1.5426 w_k1_IBM_s114
                                 + 0.931071 w_k1_INTC_s114
                                 + 1.40697 w_k1_KO_s114 + 1.62227 w_k1_MAT_s114
                                 + 0.975371 w_k1_MSFT_s114
                                 + 1.23511 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.12109 w_k1_ADM_s114
                                 + 1.0506 w_k1_AXP_s114 + 1.52646 w_k1_DIS_s114
                                 + 1.23721 w_k1_FDX_s114
                                 + 0.932195 w_k1_IBM_s114
                                 + 1.27389 w_k1_INTC_s114
                                 + 1.14701 w_k1_KO_s114
                                 + 0.736102 w_k1_MAT_s114
                                 + 1.17423 w_k1_MSFT_s114
                                 + 1.07312 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.922693 w_k1_ADM_s114
                                 + 0.838592 w_k1_AXP_s114
                                 + 0.769742 w_k1_DIS_s114
                                 + 0.769773 w_k1_FDX_s114
                                 + 1.29024 w_k1_IBM_s114
                                 + 0.790796 w_k1_INTC_s114
                                 + 1.07872 w_k1_KO_s114 + 1.18606 w_k1_MAT_s114
                                 + 0.768341 w_k1_MSFT_s114
                                 + 1.00184 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.901474 w_k1_ADM_s114
                                 + 0.749367 w_k1_AXP_s114
                                 + 0.658732 w_k1_DIS_s114
                                 + 0.839226 w_k1_FDX_s114
                                 + 0.535313 w_k1_IBM_s114
                                 + 0.937214 w_k1_INTC_s114
                                 + 0.763792 w_k1_KO_s114
                                 + 0.436152 w_k1_MAT_s114
                                 + 0.870993 w_k1_MSFT_s114
                                 + 0.775344 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.14275 w_k1_ADM_s114
                                 + 1.26487 w_k1_AXP_s114
                                 + 1.33625 w_k1_DIS_s114
                                 + 1.18522 w_k1_FDX_s114
                                 + 1.25158 w_k1_IBM_s114
                                 + 1.03429 w_k1_INTC_s114
                                 + 1.31067 w_k1_KO_s114 + 1.58416 w_k1_MAT_s114
                                 + 1.1216 w_k1_MSFT_s114 + 1.24286 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.873172 w_k1_ADM_s114
                                 + 0.975212 w_k1_AXP_s114
                                 + 0.866019 w_k1_DIS_s114
                                 + 0.97659 w_k1_FDX_s114
                                 + 0.797163 w_k1_IBM_s114
                                 + 0.948994 w_k1_INTC_s114
                                 + 0.931052 w_k1_KO_s114
                                 + 0.708443 w_k1_MAT_s114
                                 + 0.962719 w_k1_MSFT_s114
                                 + 1.03936 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.736101 w_k1_ADM_s114
                                 + 0.583154 w_k1_AXP_s114
                                 + 0.693123 w_k1_DIS_s114
                                 + 0.722608 w_k1_FDX_s114
                                 + 0.96866 w_k1_IBM_s114
                                 + 0.822231 w_k1_INTC_s114
                                 + 0.965776 w_k1_KO_s114
                                 + 0.696108 w_k1_MAT_s114
                                 + 0.752227 w_k1_MSFT_s114
                                 + 0.854855 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.726998 w_k1_ADM_s114
                                 + 0.578756 w_k1_AXP_s114
                                 + 0.605315 w_k1_DIS_s114
                                 + 0.742879 w_k1_FDX_s114
                                 + 0.679625 w_k1_IBM_s114
                                 + 0.734478 w_k1_INTC_s114
                                 + 0.712059 w_k1_KO_s114
                                 + 0.658726 w_k1_MAT_s114
                                 + 0.738115 w_k1_MSFT_s114
                                 + 0.789037 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.714303 w_k1_ADM_s115
                                 + 1.00076 w_k1_AXP_s115
                                 + 0.683566 w_k1_DIS_s115
                                 + 0.766537 w_k1_FDX_s115
                                 + 1.14809 w_k1_IBM_s115
                                 + 0.940967 w_k1_INTC_s115
                                 + 0.746574 w_k1_KO_s115
                                 + 1.18541 w_k1_MAT_s115
                                 + 0.806635 w_k1_MSFT_s115
                                 + 0.941812 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.1861 w_k1_ADM_s115
                                 + 1.06195 w_k1_AXP_s115
                                 + 1.21745 w_k1_DIS_s115
                                 + 1.04822 w_k1_FDX_s115
                                 + 1.39675 w_k1_IBM_s115
                                 + 1.09917 w_k1_INTC_s115
                                 + 1.15853 w_k1_KO_s115 + 1.22545 w_k1_MAT_s115
                                 + 1.076 w_k1_MSFT_s115 + 1.05196 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.23629 w_k1_ADM_s115
                                 + 1.09706 w_k1_AXP_s115
                                 + 1.41159 w_k1_DIS_s115
                                 + 1.15452 w_k1_FDX_s115
                                 + 0.928989 w_k1_IBM_s115
                                 + 1.13588 w_k1_INTC_s115
                                 + 1.08732 w_k1_KO_s115 + 1.13852 w_k1_MAT_s115
                                 + 1.32395 w_k1_MSFT_s115
                                 + 1.15318 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.0134 w_k1_ADM_s115
                                 + 0.746638 w_k1_AXP_s115
                                 + 0.935097 w_k1_DIS_s115
                                 + 0.877727 w_k1_FDX_s115
                                 + 0.957705 w_k1_IBM_s115
                                 + 1.05149 w_k1_INTC_s115
                                 + 0.924554 w_k1_KO_s115
                                 + 1.07433 w_k1_MAT_s115
                                 + 0.863167 w_k1_MSFT_s115
                                 + 0.925331 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.44246 w_k1_ADM_s115
                                 + 1.62091 w_k1_AXP_s115
                                 + 1.84782 w_k1_DIS_s115
                                 + 1.38768 w_k1_FDX_s115
                                 + 1.25638 w_k1_IBM_s115
                                 + 1.49416 w_k1_INTC_s115
                                 + 1.46017 w_k1_KO_s115 + 1.16821 w_k1_MAT_s115
                                 + 1.34951 w_k1_MSFT_s115
                                 + 1.31913 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.707989 w_k1_ADM_s115
                                 + 0.715895 w_k1_AXP_s115
                                 + 0.592238 w_k1_DIS_s115
                                 + 0.683739 w_k1_FDX_s115
                                 + 0.776836 w_k1_IBM_s115
                                 + 0.81158 w_k1_INTC_s115
                                 + 0.678366 w_k1_KO_s115
                                 + 1.06186 w_k1_MAT_s115
                                 + 0.70261 w_k1_MSFT_s115
                                 + 0.811593 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.13548 w_k1_ADM_s115
                                 + 1.41472 w_k1_AXP_s115 + 1.294 w_k1_DIS_s115
                                 + 1.1672 w_k1_FDX_s115 + 1.10226 w_k1_IBM_s115
                                 + 1.18136 w_k1_INTC_s115
                                 + 0.979668 w_k1_KO_s115
                                 + 0.920576 w_k1_MAT_s115
                                 + 1.02991 w_k1_MSFT_s115
                                 + 1.00935 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.12646 w_k1_ADM_s115
                                 + 0.893689 w_k1_AXP_s115
                                 + 0.9817 w_k1_DIS_s115
                                 + 0.976477 w_k1_FDX_s115
                                 + 0.719453 w_k1_IBM_s115
                                 + 1.01553 w_k1_INTC_s115
                                 + 0.790647 w_k1_KO_s115
                                 + 0.738943 w_k1_MAT_s115
                                 + 1.05254 w_k1_MSFT_s115
                                 + 0.944331 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.731758 w_k1_ADM_s116
                                 + 0.843895 w_k1_AXP_s116
                                 + 0.628051 w_k1_DIS_s116
                                 + 0.75992 w_k1_FDX_s116
                                 + 0.960048 w_k1_IBM_s116
                                 + 0.8399 w_k1_INTC_s116 + 0.69419 w_k1_KO_s116
                                 + 0.979604 w_k1_MAT_s116
                                 + 0.761435 w_k1_MSFT_s116
                                 + 0.838114 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.999293 w_k1_ADM_s116
                                 + 1.04463 w_k1_AXP_s116
                                 + 0.701241 w_k1_DIS_s116
                                 + 0.883995 w_k1_FDX_s116
                                 + 0.872516 w_k1_IBM_s116
                                 + 0.796152 w_k1_INTC_s116
                                 + 0.730284 w_k1_KO_s116
                                 + 0.714543 w_k1_MAT_s116
                                 + 0.744189 w_k1_MSFT_s116
                                 + 0.975114 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.95807 w_k1_ADM_s116
                                 + 0.935821 w_k1_AXP_s116
                                 + 0.954422 w_k1_DIS_s116
                                 + 1.08847 w_k1_FDX_s116
                                 + 0.746709 w_k1_IBM_s116
                                 + 0.96978 w_k1_INTC_s116
                                 + 0.807959 w_k1_KO_s116
                                 + 0.584208 w_k1_MAT_s116
                                 + 1.01679 w_k1_MSFT_s116
                                 + 0.916799 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.753313 w_k1_ADM_s116
                                 + 0.634465 w_k1_AXP_s116
                                 + 0.68103 w_k1_DIS_s116
                                 + 0.83449 w_k1_FDX_s116
                                 + 0.727081 w_k1_IBM_s116
                                 + 0.860021 w_k1_INTC_s116
                                 + 0.670209 w_k1_KO_s116
                                 + 0.355807 w_k1_MAT_s116
                                 + 0.918373 w_k1_MSFT_s116
                                 + 0.865493 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.970239 w_k1_ADM_s116
                                 + 0.891808 w_k1_AXP_s116
                                 + 1.33539 w_k1_DIS_s116
                                 + 1.16994 w_k1_FDX_s116
                                 + 0.944503 w_k1_IBM_s116
                                 + 1.07916 w_k1_INTC_s116
                                 + 1.10583 w_k1_KO_s116
                                 + 0.992518 w_k1_MAT_s116
                                 + 1.20016 w_k1_MSFT_s116
                                 + 1.09056 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.07919 w_k1_ADM_s116
                                 + 1.47954 w_k1_AXP_s116
                                 + 1.34559 w_k1_DIS_s116
                                 + 1.30582 w_k1_FDX_s116
                                 + 1.36227 w_k1_IBM_s116
                                 + 1.08373 w_k1_INTC_s116
                                 + 1.11908 w_k1_KO_s116 + 1.15763 w_k1_MAT_s116
                                 + 1.05254 w_k1_MSFT_s116 + 1.2035 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.973651 w_k1_ADM_s116
                                 + 1.00004 w_k1_AXP_s116
                                 + 0.853975 w_k1_DIS_s116
                                 + 0.862228 w_k1_FDX_s116
                                 + 1.23489 w_k1_IBM_s116
                                 + 0.82464 w_k1_INTC_s116
                                 + 0.956885 w_k1_KO_s116
                                 + 1.11388 w_k1_MAT_s116
                                 + 0.803598 w_k1_MSFT_s116
                                 + 0.95706 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.06365 w_k1_ADM_s116
                                 + 0.791926 w_k1_AXP_s116
                                 + 1.11334 w_k1_DIS_s116
                                 + 1.00654 w_k1_FDX_s116
                                 + 1.15206 w_k1_IBM_s116
                                 + 0.932773 w_k1_INTC_s116
                                 + 1.00259 w_k1_KO_s116 + 1.00228 w_k1_MAT_s116
                                 + 1.02453 w_k1_MSFT_s116
                                 + 1.04916 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.899996 w_k1_ADM_s117
                                 + 0.760859 w_k1_AXP_s117
                                 + 0.688796 w_k1_DIS_s117
                                 + 0.794825 w_k1_FDX_s117
                                 + 0.732996 w_k1_IBM_s117
                                 + 0.726928 w_k1_INTC_s117
                                 + 0.75512 w_k1_KO_s117
                                 + 0.576757 w_k1_MAT_s117
                                 + 0.727328 w_k1_MSFT_s117
                                 + 0.848384 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.850696 w_k1_ADM_s117
                                 + 0.800141 w_k1_AXP_s117
                                 + 0.878575 w_k1_DIS_s117
                                 + 0.869555 w_k1_FDX_s117
                                 + 0.950811 w_k1_IBM_s117
                                 + 0.873644 w_k1_INTC_s117
                                 + 0.863478 w_k1_KO_s117
                                 + 1.09575 w_k1_MAT_s117
                                 + 0.920318 w_k1_MSFT_s117
                                 + 1.00644 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.79759 w_k1_ADM_s117
                                 + 0.844943 w_k1_AXP_s117
                                 + 0.924734 w_k1_DIS_s117
                                 + 0.839292 w_k1_FDX_s117
                                 + 1.24038 w_k1_IBM_s117
                                 + 0.741442 w_k1_INTC_s117
                                 + 0.984067 w_k1_KO_s117
                                 + 1.29462 w_k1_MAT_s117
                                 + 0.899647 w_k1_MSFT_s117
                                 + 1.01424 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.02227 w_k1_ADM_s117
                                 + 1.06028 w_k1_AXP_s117
                                 + 1.08852 w_k1_DIS_s117
                                 + 1.02159 w_k1_FDX_s117
                                 + 1.47662 w_k1_IBM_s117
                                 + 0.843819 w_k1_INTC_s117
                                 + 1.13211 w_k1_KO_s117 + 1.44015 w_k1_MAT_s117
                                 + 0.969644 w_k1_MSFT_s117
                                 + 1.02864 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.866849 w_k1_ADM_s117
                                 + 1.00711 w_k1_AXP_s117
                                 + 0.760101 w_k1_DIS_s117
                                 + 0.86073 w_k1_FDX_s117
                                 + 1.06027 w_k1_IBM_s117
                                 + 0.785927 w_k1_INTC_s117
                                 + 0.924148 w_k1_KO_s117 + 1.2591 w_k1_MAT_s117
                                 + 0.848636 w_k1_MSFT_s117
                                 + 0.950301 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.14021 w_k1_ADM_s117
                                 + 1.02104 w_k1_AXP_s117
                                 + 0.838525 w_k1_DIS_s117
                                 + 1.06153 w_k1_FDX_s117
                                 + 0.680426 w_k1_IBM_s117
                                 + 0.934837 w_k1_INTC_s117
                                 + 0.750675 w_k1_KO_s117
                                 + 0.791823 w_k1_MAT_s117
                                 + 1.02443 w_k1_MSFT_s117
                                 + 0.965187 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.06658 w_k1_ADM_s117
                                 + 1.31806 w_k1_AXP_s117
                                 + 1.59382 w_k1_DIS_s117
                                 + 1.27842 w_k1_FDX_s117 + 1.3825 w_k1_IBM_s117
                                 + 0.969988 w_k1_INTC_s117
                                 + 1.21401 w_k1_KO_s117 + 2.04631 w_k1_MAT_s117
                                 + 1.21855 w_k1_MSFT_s117
                                 + 1.35955 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.987283 w_k1_ADM_s117
                                 + 1.07975 w_k1_AXP_s117
                                 + 1.34492 w_k1_DIS_s117
                                 + 1.15609 w_k1_FDX_s117
                                 + 0.993385 w_k1_IBM_s117
                                 + 1.10744 w_k1_INTC_s117
                                 + 1.12196 w_k1_KO_s117 + 1.18863 w_k1_MAT_s117
                                 + 1.13766 w_k1_MSFT_s117
                                 + 1.17779 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0172664542808576 w_k1_ADM_s000
                                 + 0.0128219018532534 w_k1_ADM_s001
                                 + 0.0382885547755524 w_k1_ADM_s002
                                 + 0.0235493151676367 w_k1_ADM_s003
                                 + 0.022281932168556 w_k1_ADM_s004
                                 + 0.0400422042109422 w_k1_ADM_s005
                                 + 0.0240671529605737 w_k1_ADM_s006
                                 + 0.0130618194784204 w_k1_ADM_s007
                                 + 0.00881503334562919 w_k1_ADM_s010
                                 + 0.00946247771720316 w_k1_ADM_s011
                                 + 0.00388952108579169 w_k1_ADM_s012
                                 + 0.0124031598439619 w_k1_ADM_s013
                                 + 0.0166574538302036 w_k1_ADM_s014
                                 + 0.00467256413400972 w_k1_ADM_s015
                                 + 0.00899423357518473 w_k1_ADM_s016
                                 + 0.00980154896950283 w_k1_ADM_s017
                                 + 0.0190597134268698 w_k1_ADM_s100
                                 + 0.0288335469405224 w_k1_ADM_s101
                                 + 0.0377834812268028 w_k1_ADM_s102
                                 + 0.0487817392697841 w_k1_ADM_s103
                                 + 0.0591413227259864 w_k1_ADM_s104
                                 + 0.032882648788871 w_k1_ADM_s105
                                 + 0.0936238502896016 w_k1_ADM_s106
                                 + 0.0478619769758487 w_k1_ADM_s107
                                 + 0.0577401123944069 w_k1_ADM_s110
                                 + 0.0225904237265322 w_k1_ADM_s111
                                 + 0.0879773311768991 w_k1_ADM_s112
                                 + 0.0403879010104709 w_k1_ADM_s113
                                 + 0.0305186321348639 w_k1_ADM_s114
                                 + 0.0641091765573038 w_k1_ADM_s115
                                 + 0.0487315013991456 w_k1_ADM_s116
                                 + 0.0236758360878656 w_k1_ADM_s117
                                 + 0.0176720995313912 w_k1_AXP_s000
                                 + 0.0116323829142098 w_k1_AXP_s001
                                 + 0.0425549689049125 w_k1_AXP_s002
                                 + 0.0252259326456304 w_k1_AXP_s003
                                 + 0.02106045224489 w_k1_AXP_s004
                                 + 0.0427588721451742 w_k1_AXP_s005
                                 + 0.0210084137940904 w_k1_AXP_s006
                                 + 0.0164474146801135 w_k1_AXP_s007
                                 + 0.00923122314965804 w_k1_AXP_s010
                                 + 0.00947962891768255 w_k1_AXP_s011
                                 + 0.00423196705839394 w_k1_AXP_s012
                                 + 0.0102811365575426 w_k1_AXP_s013
                                 + 0.0159099692869202 w_k1_AXP_s014
                                 + 0.00456937235548705 w_k1_AXP_s015
                                 + 0.00984635617673896 w_k1_AXP_s016
                                 + 0.0103695183980023 w_k1_AXP_s017
                                 + 0.0221596024767633 w_k1_AXP_s100
                                 + 0.0248817639325286 w_k1_AXP_s101
                                 + 0.034624950117357 w_k1_AXP_s102
                                 + 0.0532327503576057 w_k1_AXP_s103
                                 + 0.0648740194917409 w_k1_AXP_s104
                                 + 0.0324153750162751 w_k1_AXP_s105
                                 + 0.0938713141726013 w_k1_AXP_s106
                                 + 0.0462208793413812 w_k1_AXP_s107
                                 + 0.0585634468082764 w_k1_AXP_s110
                                 + 0.0202631711094683 w_k1_AXP_s111
                                 + 0.0865100558567494 w_k1_AXP_s112
                                 + 0.0437160544714788 w_k1_AXP_s113
                                 + 0.0301696273735656 w_k1_AXP_s114
                                 + 0.0604937055985264 w_k1_AXP_s115
                                 + 0.0527620008076508 w_k1_AXP_s116
                                 + 0.024392125921203 w_k1_AXP_s117
                                 + 0.0183368443804155 w_k1_DIS_s000
                                 + 0.0129923788372423 w_k1_DIS_s001
                                 + 0.0417195618235815 w_k1_DIS_s002
                                 + 0.0249683927302719 w_k1_DIS_s003
                                 + 0.0235969105739803 w_k1_DIS_s004
                                 + 0.0442924978026106 w_k1_DIS_s005
                                 + 0.0204102726956221 w_k1_DIS_s006
                                 + 0.016780303156505 w_k1_DIS_s007
                                 + 0.00962891057684299 w_k1_DIS_s010
                                 + 0.0100547453244893 w_k1_DIS_s011
                                 + 0.00368158797584722 w_k1_DIS_s012
                                 + 0.0108691864404649 w_k1_DIS_s013
                                 + 0.0161537197591672 w_k1_DIS_s014
                                 + 0.0043151168192707 w_k1_DIS_s015
                                 + 0.00920230304416045 w_k1_DIS_s016
                                 + 0.00881576236085891 w_k1_DIS_s017
                                 + 0.0203996225930893 w_k1_DIS_s100
                                 + 0.0280351919877403 w_k1_DIS_s101
                                 + 0.03734967641335 w_k1_DIS_s102
                                 + 0.0540365864088545 w_k1_DIS_s103
                                 + 0.0628640995680824 w_k1_DIS_s104
                                 + 0.0358886308431215 w_k1_DIS_s105
                                 + 0.0995399708884278 w_k1_DIS_s106
                                 + 0.0493193970915781 w_k1_DIS_s107
                                 + 0.0552084028222888 w_k1_DIS_s110
                                 + 0.02333244568305 w_k1_DIS_s111
                                 + 0.0835696818026885 w_k1_DIS_s112
                                 + 0.0413018050703742 w_k1_DIS_s113
                                 + 0.0322740372080434 w_k1_DIS_s114
                                 + 0.064756316628822 w_k1_DIS_s115
                                 + 0.0500867876134847 w_k1_DIS_s116
                                 + 0.0258275192114594 w_k1_DIS_s117
                                 + 0.0180459934807072 w_k1_FDX_s000
                                 + 0.0127191069036958 w_k1_FDX_s001
                                 + 0.0392916164308678 w_k1_FDX_s002
                                 + 0.0250181462724515 w_k1_FDX_s003
                                 + 0.0224681731992218 w_k1_FDX_s004
                                 + 0.0420503452020928 w_k1_FDX_s005
                                 + 0.0194188264058958 w_k1_FDX_s006
                                 + 0.0166384449853637 w_k1_FDX_s007
                                 + 0.00945609450734511 w_k1_FDX_s010
                                 + 0.00964951901134948 w_k1_FDX_s011
                                 + 0.00365609322686185 w_k1_FDX_s012
                                 + 0.0110740107466656 w_k1_FDX_s013
                                 + 0.0158144503115937 w_k1_FDX_s014
                                 + 0.00453962049767954 w_k1_FDX_s015
                                 + 0.00954567728324339 w_k1_FDX_s016
                                 + 0.00890889675019603 w_k1_FDX_s017
                                 + 0.0214044655915134 w_k1_FDX_s100
                                 + 0.0260352454852221 w_k1_FDX_s101
                                 + 0.035862167108674 w_k1_FDX_s102
                                 + 0.0537420162219924 w_k1_FDX_s103
                                 + 0.0607941934627378 w_k1_FDX_s104
                                 + 0.0349670882601684 w_k1_FDX_s105
                                 + 0.0987868949350359 w_k1_FDX_s106
                                 + 0.0450839531145638 w_k1_FDX_s107
                                 + 0.057980465469557 w_k1_FDX_s110
                                 + 0.0233391379514221 w_k1_FDX_s111
                                 + 0.0790772722478846 w_k1_FDX_s112
                                 + 0.042413273796534 w_k1_FDX_s113
                                 + 0.0309359944360466 w_k1_FDX_s114
                                 + 0.0588320577679764 w_k1_FDX_s115
                                 + 0.0513126699603978 w_k1_FDX_s116
                                 + 0.0247353855019729 w_k1_FDX_s117
                                 + 0.0188420907302943 w_k1_IBM_s000
                                 + 0.011632445640305 w_k1_IBM_s001
                                 + 0.0404730552907257 w_k1_IBM_s002
                                 + 0.0258746193911877 w_k1_IBM_s003
                                 + 0.0226045307555933 w_k1_IBM_s004
                                 + 0.0421796491981132 w_k1_IBM_s005
                                 + 0.0195058433992248 w_k1_IBM_s006
                                 + 0.0165660086627784 w_k1_IBM_s007
                                 + 0.00915118592493048 w_k1_IBM_s010
                                 + 0.0101111713085576 w_k1_IBM_s011
                                 + 0.00374209339225547 w_k1_IBM_s012
                                 + 0.0106573023323831 w_k1_IBM_s013
                                 + 0.0155984357556932 w_k1_IBM_s014
                                 + 0.00451955548856447 w_k1_IBM_s015
                                 + 0.00862368676941593 w_k1_IBM_s016
                                 + 0.00895328476360249 w_k1_IBM_s017
                                 + 0.0210824390749686 w_k1_IBM_s100
                                 + 0.0274335399558745 w_k1_IBM_s101
                                 + 0.0372787023470424 w_k1_IBM_s102
                                 + 0.0543700698051762 w_k1_IBM_s103
                                 + 0.0607897913979899 w_k1_IBM_s104
                                 + 0.0360916318946352 w_k1_IBM_s105
                                 + 0.0988860699648307 w_k1_IBM_s106
                                 + 0.0475981198415969 w_k1_IBM_s107
                                 + 0.0576868882521659 w_k1_IBM_s110
                                 + 0.0233921256402275 w_k1_IBM_s111
                                 + 0.0853881727560184 w_k1_IBM_s112
                                 + 0.0444398399803376 w_k1_IBM_s113
                                 + 0.0319750458259486 w_k1_IBM_s114
                                 + 0.060387304318045 w_k1_IBM_s115
                                 + 0.0551492556569383 w_k1_IBM_s116
                                 + 0.0256133194980326 w_k1_IBM_s117
                                 + 0.0187387397705051 w_k1_INTC_s000
                                 + 0.0114917167736302 w_k1_INTC_s001
                                 + 0.0434536992544225 w_k1_INTC_s002
                                 + 0.0233097963405472 w_k1_INTC_s003
                                 + 0.0233185078351743 w_k1_INTC_s004
                                 + 0.0443083799630475 w_k1_INTC_s005
                                 + 0.0195110645979632 w_k1_INTC_s006
                                 + 0.0171265774397662 w_k1_INTC_s007
                                 + 0.00933896998968197 w_k1_INTC_s010
                                 + 0.00891909747909816 w_k1_INTC_s011
                                 + 0.00358296317003164 w_k1_INTC_s012
                                 + 0.0109240919046555 w_k1_INTC_s013
                                 + 0.0152886158652815 w_k1_INTC_s014
                                 + 0.00455551040586542 w_k1_INTC_s015
                                 + 0.00864157136596179 w_k1_INTC_s016
                                 + 0.00879269272270129 w_k1_INTC_s017
                                 + 0.0218629056855694 w_k1_INTC_s100
                                 + 0.0242500761871516 w_k1_INTC_s101
                                 + 0.0387451269430693 w_k1_INTC_s102
                                 + 0.0543969670414954 w_k1_INTC_s103
                                 + 0.0593585707760866 w_k1_INTC_s104
                                 + 0.0346699740914466 w_k1_INTC_s105
                                 + 0.0963721011980453 w_k1_INTC_s106
                                 + 0.0467832840213376 w_k1_INTC_s107
                                 + 0.0620553513553681 w_k1_INTC_s110
                                 + 0.0200244174738714 w_k1_INTC_s111
                                 + 0.0818932927354427 w_k1_INTC_s112
                                 + 0.0478210680327748 w_k1_INTC_s113
                                 + 0.0305392952692465 w_k1_INTC_s114
                                 + 0.0634155425736715 w_k1_INTC_s115
                                 + 0.0469984181152173 w_k1_INTC_s116
                                 + 0.0222659517100967 w_k1_INTC_s117
                                 + 0.0185975818856712 w_k1_KO_s000
                                 + 0.0119492507566018 w_k1_KO_s001
                                 + 0.0423528268057096 w_k1_KO_s002
                                 + 0.0243851888072352 w_k1_KO_s003
                                 + 0.0245593237756808 w_k1_KO_s004
                                 + 0.0407875481274123 w_k1_KO_s005
                                 + 0.0195653844993992 w_k1_KO_s006
                                 + 0.0161015672212694 w_k1_KO_s007
                                 + 0.00948317641433399 w_k1_KO_s010
                                 + 0.0104556104470216 w_k1_KO_s011
                                 + 0.00377447712114571 w_k1_KO_s012
                                 + 0.0109196345829276 w_k1_KO_s013
                                 + 0.0153719577661796 w_k1_KO_s014
                                 + 0.00420567608898672 w_k1_KO_s015
                                 + 0.00964211458426425 w_k1_KO_s016
                                 + 0.00749276450852078 w_k1_KO_s017
                                 + 0.0206984446569222 w_k1_KO_s100
                                 + 0.0262248430125797 w_k1_KO_s101
                                 + 0.0417584788437888 w_k1_KO_s102
                                 + 0.0590481210231829 w_k1_KO_s103
                                 + 0.0570647607527878 w_k1_KO_s104
                                 + 0.0353761352612311 w_k1_KO_s105
                                 + 0.088802187009383 w_k1_KO_s106
                                 + 0.0470757042543904 w_k1_KO_s107
                                 + 0.0551006786265691 w_k1_KO_s110
                                 + 0.0233540382926945 w_k1_KO_s111
                                 + 0.0808933485922685 w_k1_KO_s112
                                 + 0.0449988908300417 w_k1_KO_s113
                                 + 0.0337687992355245 w_k1_KO_s114
                                 + 0.0570013356946574 w_k1_KO_s115
                                 + 0.0468534030692305 w_k1_KO_s116
                                 + 0.0241516516104649 w_k1_KO_s117
                                 + 0.0166507428255405 w_k1_MAT_s000
                                 + 0.0111513075617463 w_k1_MAT_s001
                                 + 0.0355759094105541 w_k1_MAT_s002
                                 + 0.0249390467308343 w_k1_MAT_s003
                                 + 0.0231075761577972 w_k1_MAT_s004
                                 + 0.0398031982963915 w_k1_MAT_s005
                                 + 0.0188619391948127 w_k1_MAT_s006
                                 + 0.0139980193540926 w_k1_MAT_s007
                                 + 0.00853338257501505 w_k1_MAT_s010
                                 + 0.0105293156790327 w_k1_MAT_s011
                                 + 0.00406316601197739 w_k1_MAT_s012
                                 + 0.0107130489998611 w_k1_MAT_s013
                                 + 0.0157474605301746 w_k1_MAT_s014
                                 + 0.00401942650747552 w_k1_MAT_s015
                                 + 0.0101819833383161 w_k1_MAT_s016
                                 + 0.00902342648114008 w_k1_MAT_s017
                                 + 0.0198287657743685 w_k1_MAT_s100
                                 + 0.0309305631984306 w_k1_MAT_s101
                                 + 0.0384482712206465 w_k1_MAT_s102
                                 + 0.0577087876955258 w_k1_MAT_s103
                                 + 0.0679850750333287 w_k1_MAT_s104
                                 + 0.0379255636509272 w_k1_MAT_s105
                                 + 0.0883989379038076 w_k1_MAT_s106
                                 + 0.0540674963119943 w_k1_MAT_s107
                                 + 0.0651152739926192 w_k1_MAT_s110
                                 + 0.0196097754879438 w_k1_MAT_s111
                                 + 0.0717070598857457 w_k1_MAT_s112
                                 + 0.0455307899271258 w_k1_MAT_s113
                                 + 0.0312559261509525 w_k1_MAT_s114
                                 + 0.0605450148793627 w_k1_MAT_s115
                                 + 0.0477936275985732 w_k1_MAT_s116
                                 + 0.0289739970198071 w_k1_MAT_s117
                                 + 0.019203426453528 w_k1_MSFT_s000
                                 + 0.0121650025953683 w_k1_MSFT_s001
                                 + 0.0445761152101244 w_k1_MSFT_s002
                                 + 0.0241624898554913 w_k1_MSFT_s003
                                 + 0.0211248080780727 w_k1_MSFT_s004
                                 + 0.0442169627138437 w_k1_MSFT_s005
                                 + 0.0175900563569912 w_k1_MSFT_s006
                                 + 0.0178244079618423 w_k1_MSFT_s007
                                 + 0.00856526914492427 w_k1_MSFT_s010
                                 + 0.00935056752223286 w_k1_MSFT_s011
                                 + 0.00352023414892083 w_k1_MSFT_s012
                                 + 0.0116622635499909 w_k1_MSFT_s013
                                 + 0.0171716861016173 w_k1_MSFT_s014
                                 + 0.00466282587000519 w_k1_MSFT_s015
                                 + 0.00911251692973065 w_k1_MSFT_s016
                                 + 0.00859655374021388 w_k1_MSFT_s017
                                 + 0.0191270215626377 w_k1_MSFT_s100
                                 + 0.0278713546355939 w_k1_MSFT_s101
                                 + 0.0366983884151482 w_k1_MSFT_s102
                                 + 0.0523844459428549 w_k1_MSFT_s103
                                 + 0.060496892947119 w_k1_MSFT_s104
                                 + 0.0344739751363929 w_k1_MSFT_s105
                                 + 0.0919354346055596 w_k1_MSFT_s106
                                 + 0.0453735748499851 w_k1_MSFT_s107
                                 + 0.0568441481085845 w_k1_MSFT_s110
                                 + 0.0231175049518622 w_k1_MSFT_s111
                                 + 0.081466368928218 w_k1_MSFT_s112
                                 + 0.0449376077596959 w_k1_MSFT_s113
                                 + 0.0303705851901329 w_k1_MSFT_s114
                                 + 0.0604620033420252 w_k1_MSFT_s115
                                 + 0.047038205148127 w_k1_MSFT_s116
                                 + 0.0242254510907282 w_k1_MSFT_s117
                                 + 0.017171364913688 w_k1_PG_s000
                                 + 0.0113901198740433 w_k1_PG_s001
                                 + 0.0416842480379447 w_k1_PG_s002
                                 + 0.0243698961922444 w_k1_PG_s003
                                 + 0.0215494066882782 w_k1_PG_s004
                                 + 0.0410211752830489 w_k1_PG_s005
                                 + 0.0198735838399541 w_k1_PG_s006
                                 + 0.0167537361044817 w_k1_PG_s007
                                 + 0.00843620194995061 w_k1_PG_s010
                                 + 0.0098302572020712 w_k1_PG_s011
                                 + 0.00379957039395953 w_k1_PG_s012
                                 + 0.0108809545596142 w_k1_PG_s013
                                 + 0.0144447969477645 w_k1_PG_s014
                                 + 0.00446504301164152 w_k1_PG_s015
                                 + 0.0096333260754255 w_k1_PG_s016
                                 + 0.00873153585837546 w_k1_PG_s017
                                 + 0.0210050542221271 w_k1_PG_s100
                                 + 0.0274223545465842 w_k1_PG_s101
                                 + 0.0360587854994535 w_k1_PG_s102
                                 + 0.0562461779391528 w_k1_PG_s103
                                 + 0.062389123483785 w_k1_PG_s104
                                 + 0.0348461043288435 w_k1_PG_s105
                                 + 0.103000127383286 w_k1_PG_s106
                                 + 0.0495048210792637 w_k1_PG_s107
                                 + 0.0578539193906973 w_k1_PG_s110
                                 + 0.0220101043563227 w_k1_PG_s111
                                 + 0.0805575744969445 w_k1_PG_s112
                                 + 0.0443716207423016 w_k1_PG_s113
                                 + 0.0328492534389287 w_k1_PG_s114
                                 + 0.0592764021378322 w_k1_PG_s115
                                 + 0.0511968470400225 w_k1_PG_s116
                                 + 0.0259432982397184 w_k1_PG_s117 >= 1.219
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
