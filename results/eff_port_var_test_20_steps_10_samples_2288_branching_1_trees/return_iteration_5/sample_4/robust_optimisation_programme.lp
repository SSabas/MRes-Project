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
 w_k1_ADM_s0_return_constraint:  0.976835 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.10621 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.28504 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.0154 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.03592 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.06416 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.993454 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.29264 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.36623 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.863013 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.987777 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.01349 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.870066 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.4089 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.9694 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.181 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.770228 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.29483 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.994176 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.22302 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.13247 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.818061 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.608528 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.2264 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.927823 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.59098 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.10498 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.756367 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.783757 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.26432 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.973001 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.843858 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.1788 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.0531 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.15876 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.23034 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.810301 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.764967 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.985412 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.22816 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.50118 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.930857 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.02942 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.986609 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.754527 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.61655 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.5087 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.939027 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.971343 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.04136 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.647327 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.04983 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.902594 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.34486 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.676012 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.65161 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.27347 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.20687 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.2789 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.88985 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.668644 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.4454 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.992587 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.89792 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.996611 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.842043 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.713064 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.42561 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.16801 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.935394 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.46689 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.964697 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.23806 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.83503 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.916535 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.665486 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.95848 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.2482 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.6468 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.932207 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.0761 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.03706 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.710815 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.6463 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.17087 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.82449 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.01389 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.29622 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.771175 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.38606 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.02701 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.24968 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.654671 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.78125 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.40385 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.25241 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.25833 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.636894 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.677035 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.41569 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.936503 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 2.32572 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.04607 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.618404 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.635866 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.52552 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.28628 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.1734 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.18085 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.884033 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.40523 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.92288 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.930498 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.62042 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.995249 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.12089 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.40506 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.909664 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.96374 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.06224 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.886243 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.35219 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.21382 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.781048 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.00828 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.0356 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.775187 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.21609 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.06202 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.05499 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.810123 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.49254 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.14666 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.29139 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.10568 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.767687 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.690102 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.15398 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.925594 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.50126 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.0599 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.725852 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.799556 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.33286 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.03869 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.02251 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.29245 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.93106 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.23963 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.49661 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.853445 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.683457 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.933741 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.40639 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.157 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.00885 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.28639 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.874608 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.61051 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.5973 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.974327 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.34981 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.957268 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.51997 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.935171 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.925689 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.961513 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.95718 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.649278 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.24241 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.48696 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.763659 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.27115 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.09653 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.00051 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.36951 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.24487 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.6414 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.839319 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.81445 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.623193 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.97463 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.28191 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.920733 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.832629 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.81461 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.07628 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.3986 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.34422 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.782527 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.06932 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.14137 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.1056 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.884779 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.947555 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.05965 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.860365 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.36946 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.01918 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.976699 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.06511 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.08253 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.948983 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.29936 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.995573 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.941784 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.837728 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.18953 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.09143 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.01332 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.14813 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.867471 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.852607 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.19221 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.815412 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.24367 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.07153 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.837949 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.871541 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.13494 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.02107 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.856019 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.11792 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.903694 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.08988 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.45079 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.879714 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.632491 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.942666 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.26101 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.28216 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.95566 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.15827 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.92446 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.668303 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.45022 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.04462 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.09891 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.954093 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.20726 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.884665 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.21476 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.00423 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.49834 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.796475 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.57333 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.26958 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.01608 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.31181 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.894313 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.869448 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.30141 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.992349 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.68671 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.922549 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.714019 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.741631 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.01701 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.36096 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.10124 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.04399 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.92289 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.34239 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.46109 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.07282 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.750773 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.926429 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.331 w_0_MAT - w_k1_MAT_s1 + 0.99 b_k1_MAT_s1
                                 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.4297 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.991856 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.41788 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.89674 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.526985 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.59868 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.04718 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.29023 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.01963 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.12129 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.769323 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.828354 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.90531 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 2.20699 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.646564 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.58513 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.46412 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.976759 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.42827 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.10197 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.756379 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.33635 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.18993 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.73079 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.770427 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.896026 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.647624 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.976901 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.21658 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.860048 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.03471 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.911711 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.98411 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.18779 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.42774 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.746471 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.99172 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.15804 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.29441 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.92313 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.878422 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.00914 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.900332 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.30094 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.988265 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.875187 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.03047 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.08214 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.945084 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.17342 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.02872 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.04483 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.812939 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.4054 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.09301 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.23917 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.10442 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.832051 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.920532 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.25327 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.994675 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.47534 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.10633 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.881916 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.929506 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.3855 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.06969 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.05784 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.05587 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.07419 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.21449 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.44547 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.899415 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.68144 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.967789 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.10146 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.18924 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.959554 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06006 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.04412 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.89434 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.33631 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.06568 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.966154 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.00908 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.06894 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.936237 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.13499 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.981326 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.27836 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.82259 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.31149 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.15284 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.1131 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.21473 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.946369 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.781745 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.18264 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.07379 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.28876 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.03777 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.874033 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.809566 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.13047 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.11868 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.977244 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.10102 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.991487 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.15976 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.26295 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.90959 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.825944 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.56999855122308 z_k1_s0000
                                 - 0.250999868214114 z_k1_s0001
                                 - 0.208000018002555 z_k1_s0002
                                 - 0.393998269659331 z_k1_s0003
                                 - 0.951001652551868 z_k1_s0004
                                 - 0.414999046425594 z_k1_s0005
                                 - 0.613000324934983 z_k1_s0006
                                 - 0.445001804806264 z_k1_s0007
                                 - 0.211000373485711 z_k1_s0010
                                 - 0.219000995950724 z_k1_s0011
                                 - 0.149000286190424 z_k1_s0012
                                 - 0.253000585421504 z_k1_s0013
                                 - 0.175999942205423 z_k1_s0014
                                 - 0.123000249582723 z_k1_s0015
                                 - 0.307000203052054 z_k1_s0016
                                 - 0.0900001267189647 z_k1_s0017
                                 - 0.211000038593526 z_k1_s0020
                                 - 0.118000113692183 z_k1_s0021
                                 - 0.0859999396399583 z_k1_s0022
                                 - 0.0620000626009799 z_k1_s0023
                                 - 0.0540000190879236 z_k1_s0024
                                 - 0.0610000043493082 z_k1_s0025
                                 - 0.0450000018232591 z_k1_s0026
                                 - 0.208000286338829 z_k1_s0027
                                 - 0.182999759260625 z_k1_s0030
                                 - 0.187999847377132 z_k1_s0031
                                 - 0.224999966238742 z_k1_s0032
                                 - 0.133999962119943 z_k1_s0033
                                 - 0.0830001297326591 z_k1_s0034
                                 - 0.152000812539916 z_k1_s0035
                                 - 0.137000548190394 z_k1_s0036
                                 - 0.230999805378278 z_k1_s0037
                                 - 0.318000391808442 z_k1_s0040
                                 - 0.676000443664102 z_k1_s0041
                                 - 0.925000462225145 z_k1_s0042
                                 - 0.521000683805067 z_k1_s0043
                                 - 0.508998983969124 z_k1_s0044
                                 - 0.367000543331845 z_k1_s0045
                                 - 0.192000170048427 z_k1_s0046
                                 - 0.409001792353008 z_k1_s0047
                                 - 0.0880000817262795 z_k1_s0050
                                 - 0.143999720169856 z_k1_s0051
                                 - 0.0780001271756215 z_k1_s0052
                                 - 0.116000168548339 z_k1_s0053
                                 - 0.341000350139367 z_k1_s0054
                                 - 0.236000693557322 z_k1_s0055
                                 - 0.232000069496407 z_k1_s0056
                                 - 0.103000147352402 z_k1_s0057
                                 - 0.0710000329657972 z_k1_s0060
                                 - 0.117000162787391 z_k1_s0061
                                 - 0.302000492215377 z_k1_s0062
                                 - 0.106000071727829 z_k1_s0063
                                 - 0.0940000252265328 z_k1_s0064
                                 - 0.234000034974537 z_k1_s0065
                                 - 0.305000939741068 z_k1_s0066
                                 - 0.224000480557198 z_k1_s0067
                                 - 0.126000224111899 z_k1_s0070
                                 - 0.378000672335698 z_k1_s0071
                                 - 0.142000364515228 z_k1_s0072
                                 - 0.31000032749707 z_k1_s0073
                                 - 0.234999858263106 z_k1_s0074
                                 - 0.174000443821471 z_k1_s0075
                                 - 0.302000761046442 z_k1_s0076
                                 - 0.34800129064377 z_k1_s0077
                                 - 1.53399546947475 z_k1_s0100
                                 - 3.61500773973828 z_k1_s0101
                                 - 2.04800922436302 z_k1_s0102
                                 - 2.257005355034 z_k1_s0103
                                 - 0.768001027332751 z_k1_s0104
                                 - 1.30600209078247 z_k1_s0105
                                 - 1.4609996799525 z_k1_s0106
                                 - 0.907001519054078 z_k1_s0107
                                 - 0.813000190618549 z_k1_s0110
                                 - 1.01900056505066 z_k1_s0111
                                 - 0.462000970638939 z_k1_s0112
                                 - 0.791999737953324 z_k1_s0113
                                 - 0.305999919181957 z_k1_s0114
                                 - 0.305999919181957 z_k1_s0115
                                 - 0.502999714414853 z_k1_s0116
                                 - 0.292999681760608 z_k1_s0117
                                 - 3.24400113670794 z_k1_s0120
                                 - 2.02400077206436 z_k1_s0121
                                 - 2.8249939508787 z_k1_s0122
                                 - 2.77499240404976 z_k1_s0123
                                 - 1.70999173757827 z_k1_s0124
                                 - 2.53300394618885 z_k1_s0125
                                 - 0.823999327153141 z_k1_s0126
                                 - 1.0550003571057 z_k1_s0127
                                 - 0.786999886348484 z_k1_s0130
                                 - 0.482998015302914 z_k1_s0131
                                 - 0.288999663352068 z_k1_s0132
                                 - 0.286999786379303 z_k1_s0133
                                 - 0.852000359960149 z_k1_s0134
                                 - 0.974000904893058 z_k1_s0135
                                 - 0.406999559940086 z_k1_s0136
                                 - 0.39299952673137 z_k1_s0137
                                 - 0.194000031384308 z_k1_s0140
                                 - 0.465000406756338 z_k1_s0141
                                 - 0.492001863833101 z_k1_s0142
                                 - 0.503002024345678 z_k1_s0143
                                 - 0.267000246131064 z_k1_s0144
                                 - 0.821998883207225 z_k1_s0145
                                 - 0.250000281557197 z_k1_s0146
                                 - 0.904999031256237 z_k1_s0147
                                 - 1.19599842440965 z_k1_s0150
                                 - 0.306999651003959 z_k1_s0151
                                 - 0.53299990041057 z_k1_s0152
                                 - 0.522000008514622 z_k1_s0153
                                 - 0.70800084866316 z_k1_s0154
                                 - 1.10600029707876 z_k1_s0155
                                 - 0.497999602060097 z_k1_s0156
                                 - 0.565000623516528 z_k1_s0157
                                 - 0.702001470124791 z_k1_s0160
                                 - 0.270999699262666 z_k1_s0161
                                 - 0.703997248248777 z_k1_s0162
                                 - 0.846997447829847 z_k1_s0163
                                 - 0.366999863423355 z_k1_s0164
                                 - 1.35299732419441 z_k1_s0165
                                 - 0.318999781343011 z_k1_s0166
                                 - 0.83099885886838 z_k1_s0167
                                 - 0.745000941596759 z_k1_s0170
                                 - 1.60500045956874 z_k1_s0171
                                 - 0.891000285682141 z_k1_s0172
                                 - 0.45800066334393 z_k1_s0173
                                 - 1.96400318008505 z_k1_s0174
                                 - 1.49200288196361 z_k1_s0175
                                 - 0.49200047700485 z_k1_s0176
                                 - 0.527000790050814 z_k1_s0177
                                 - 0.141999806826255 z_k1_s1000
                                 - 0.0399999872764141 z_k1_s1001
                                 - 0.0619999840734422 z_k1_s1002
                                 - 0.131999980782169 z_k1_s1003
                                 - 0.17900030168949 z_k1_s1004
                                 - 0.0780001611440269 z_k1_s1005
                                 - 0.0600000188646251 z_k1_s1006
                                 - 0.0659999903910844 z_k1_s1007
                                 - 0.0959997102202133 z_k1_s1010
                                 - 0.156000124831338 z_k1_s1011
                                 - 0.232999371782203 z_k1_s1012
                                 - 0.118999647158373 z_k1_s1013
                                 - 0.188998929241062 z_k1_s1014
                                 - 0.186999742305263 z_k1_s1015
                                 - 0.0479999162099519 z_k1_s1016
                                 - 0.193999340574368 z_k1_s1017
                                 - 0.341000281172409 z_k1_s1020
                                 - 0.162000645419401 z_k1_s1021
                                 - 0.109000053636324 z_k1_s1022
                                 - 0.0920000441022181 z_k1_s1023
                                 - 0.0780000512388439 z_k1_s1024
                                 - 0.0890000274314861 z_k1_s1025
                                 - 0.0899999905217093 z_k1_s1026
                                 - 0.313000295445661 z_k1_s1027
                                 - 0.0340000378546448 z_k1_s1030
                                 - 0.0290000395672018 z_k1_s1031
                                 - 0.0430000347720422 z_k1_s1032
                                 - 0.0300000392246904 z_k1_s1033
                                 - 0.0380000364845992 z_k1_s1034
                                 - 0.134000053103537 z_k1_s1035
                                 - 0.144000049678423 z_k1_s1036
                                 - 0.0430000347720422 z_k1_s1037
                                 - 0.270998671792196 z_k1_s1040
                                 - 0.0909997283603417 z_k1_s1041
                                 - 0.145999969893436 z_k1_s1042
                                 - 0.108999703973291 z_k1_s1043
                                 - 0.161000046320612 z_k1_s1044
                                 - 0.20299909932008 z_k1_s1045
                                 - 0.0959996764362922 z_k1_s1046
                                 - 0.0839997313942135 z_k1_s1047
                                 - 0.0850000348321372 z_k1_s1050
                                 - 0.121000003850439 z_k1_s1051
                                 - 0.138999674859489 z_k1_s1052
                                 - 0.0740000686821083 z_k1_s1053
                                 - 0.0910000505685276 z_k1_s1054
                                 - 0.205000224159905 z_k1_s1055
                                 - 0.341000330051341 z_k1_s1056
                                 - 0.198000393900844 z_k1_s1057
                                 - 0.0870001236676577 z_k1_s1060
                                 - 0.0580001118118192 z_k1_s1061
                                 - 0.163000239800337 z_k1_s1062
                                 - 0.14199980894198 z_k1_s1063
                                 - 0.0910003987171797 z_k1_s1064
                                 - 0.190000290331699 z_k1_s1065
                                 - 0.0910003987171797 z_k1_s1066
                                 - 0.0590001365241286 z_k1_s1067
                                 - 0.0919997891997701 z_k1_s1070
                                 - 0.0989999076713663 z_k1_s1071
                                 - 0.116999740884482 z_k1_s1072
                                 - 0.224999565162406 z_k1_s1073
                                 - 0.100999705806328 z_k1_s1074
                                 - 0.0809999094584046 z_k1_s1075
                                 - 0.0479999402346158 z_k1_s1076
                                 - 0.0619999296780391 z_k1_s1077
                                 - 0.256999402814998 z_k1_s1100
                                 - 0.142000299889555 z_k1_s1101
                                 - 0.262999534584896 z_k1_s1102
                                 - 0.134999944824817 z_k1_s1103
                                 - 0.0949999522248774 z_k1_s1104
                                 - 0.121999457990181 z_k1_s1105
                                 - 0.0579999047099718 z_k1_s1106
                                 - 0.136000168119658 z_k1_s1107
                                 - 0.143000502202132 z_k1_s1110
                                 - 0.17400051530186 z_k1_s1111
                                 - 0.103999867237736 z_k1_s1112
                                 - 0.0790000009863933 z_k1_s1113
                                 - 0.151999946424433 z_k1_s1114
                                 - 0.0710000182219546 z_k1_s1115
                                 - 0.0950004230154341 z_k1_s1116
                                 - 0.0950004230154341 z_k1_s1117
                                 - 0.0579999701633184 z_k1_s1120
                                 - 0.0479999649489639 z_k1_s1121
                                 - 0.132999708871289 z_k1_s1122
                                 - 0.118999581411317 z_k1_s1123
                                 - 0.157000141945303 z_k1_s1124
                                 - 0.084999593722481 z_k1_s1125
                                 - 0.092999988413559 z_k1_s1126
                                 - 0.0579999701633184 z_k1_s1127
                                 - 0.207999527579366 z_k1_s1130
                                 - 0.0999997450324665 z_k1_s1131
                                 - 0.485998538862417 z_k1_s1132
                                 - 0.326000201566912 z_k1_s1133
                                 - 0.13799956127705 z_k1_s1134
                                 - 0.472999744722871 z_k1_s1135
                                 - 0.300000200294663 z_k1_s1136
                                 - 0.381998533773418 z_k1_s1137
                                 - 0.0469999971857989 z_k1_s1140
                                 - 0.124999904717574 z_k1_s1141
                                 - 0.0729999663630572 z_k1_s1142
                                 - 0.175999763138371 z_k1_s1143
                                 - 0.220999692156707 z_k1_s1144
                                 - 0.118999975315113 z_k1_s1145
                                 - 0.0620000040919029 z_k1_s1146
                                 - 0.0939998109716467 z_k1_s1147
                                 - 0.192000248755188 z_k1_s1150
                                 - 0.208999269213535 z_k1_s1151
                                 - 0.15299995901762 z_k1_s1152
                                 - 0.193999804044687 z_k1_s1153
                                 - 0.319999792259179 z_k1_s1154
                                 - 0.470998889088418 z_k1_s1155
                                 - 0.140999826783021 z_k1_s1156
                                 - 0.187000427032239 z_k1_s1157
                                 - 0.274000405268758 z_k1_s1160
                                 - 0.352000866039585 z_k1_s1161
                                 - 0.277000162990044 z_k1_s1162
                                 - 0.0720002693265161 z_k1_s1163
                                 - 0.239000415179842 z_k1_s1164
                                 - 0.212000905683921 z_k1_s1165
                                 - 0.111000330711495 z_k1_s1166
                                 - 0.153000318818194 z_k1_s1167
                                 - 0.309999743929068 z_k1_s1170
                                 - 0.833999694762363 z_k1_s1171
                                 - 0.408998159780708 z_k1_s1172
                                 - 0.38400082089689 z_k1_s1173
                                 - 0.513000052605404 z_k1_s1174
                                 - 0.22699969178301 z_k1_s1175
                                 - 0.277999675520994 z_k1_s1176
                                 - 0.385998737049325 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.812301 w_k1_ADM_s000
                                 + 0.802259 w_k1_AXP_s000
                                 + 0.708404 w_k1_DIS_s000
                                 + 0.75651 w_k1_FDX_s000
                                 + 1.04367 w_k1_IBM_s000
                                 + 1.03518 w_k1_INTC_s000
                                 + 0.85949 w_k1_KO_s000 + 1.07197 w_k1_MAT_s000
                                 + 0.788028 w_k1_MSFT_s000
                                 + 0.861058 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.901922 w_k1_ADM_s000
                                 + 0.955942 w_k1_AXP_s000
                                 + 1.00952 w_k1_DIS_s000
                                 + 0.879442 w_k1_FDX_s000
                                 + 1.5105 w_k1_IBM_s000
                                 + 1.01789 w_k1_INTC_s000 + 1.3551 w_k1_KO_s000
                                 + 1.22778 w_k1_MAT_s000
                                 + 0.860232 w_k1_MSFT_s000
                                 + 0.973739 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.828572 w_k1_ADM_s000
                                 + 0.880513 w_k1_AXP_s000
                                 + 1.10474 w_k1_DIS_s000
                                 + 1.17473 w_k1_FDX_s000
                                 + 0.568397 w_k1_IBM_s000
                                 + 1.30688 w_k1_INTC_s000
                                 + 0.94707 w_k1_KO_s000
                                 + 0.853068 w_k1_MAT_s000
                                 + 1.11468 w_k1_MSFT_s000
                                 + 0.911061 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.05803 w_k1_ADM_s000
                                 + 1.43232 w_k1_AXP_s000
                                 + 0.932688 w_k1_DIS_s000
                                 + 1.11127 w_k1_FDX_s000
                                 + 0.965213 w_k1_IBM_s000
                                 + 0.991472 w_k1_INTC_s000
                                 + 0.921877 w_k1_KO_s000
                                 + 0.920463 w_k1_MAT_s000
                                 + 0.878223 w_k1_MSFT_s000
                                 + 0.971317 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.08781 w_k1_ADM_s000
                                 + 1.18369 w_k1_AXP_s000
                                 + 1.40772 w_k1_DIS_s000
                                 + 1.15339 w_k1_FDX_s000
                                 + 1.29529 w_k1_IBM_s000
                                 + 1.27334 w_k1_INTC_s000
                                 + 1.30361 w_k1_KO_s000 + 1.00587 w_k1_MAT_s000
                                 + 1.27504 w_k1_MSFT_s000
                                 + 1.16279 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.902383 w_k1_ADM_s000
                                 + 0.658334 w_k1_AXP_s000
                                 + 0.898636 w_k1_DIS_s000
                                 + 0.812134 w_k1_FDX_s000
                                 + 0.930842 w_k1_IBM_s000
                                 + 1.05406 w_k1_INTC_s000
                                 + 1.02136 w_k1_KO_s000
                                 + 0.747278 w_k1_MAT_s000
                                 + 0.86552 w_k1_MSFT_s000
                                 + 0.841869 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.02936 w_k1_ADM_s000
                                 + 0.873602 w_k1_AXP_s000
                                 + 1.11512 w_k1_DIS_s000
                                 + 1.08642 w_k1_FDX_s000
                                 + 0.86226 w_k1_IBM_s000
                                 + 1.26049 w_k1_INTC_s000 + 1.0952 w_k1_KO_s000
                                 + 0.979055 w_k1_MAT_s000
                                 + 1.17059 w_k1_MSFT_s000
                                 + 1.01609 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.875464 w_k1_ADM_s000
                                 + 0.83338 w_k1_AXP_s000
                                 + 0.668096 w_k1_DIS_s000
                                 + 0.821958 w_k1_FDX_s000
                                 + 0.692332 w_k1_IBM_s000
                                 + 0.939362 w_k1_INTC_s000
                                 + 0.850557 w_k1_KO_s000
                                 + 0.746387 w_k1_MAT_s000
                                 + 0.867221 w_k1_MSFT_s000
                                 + 0.907905 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.17195 w_k1_ADM_s001
                                 + 1.05053 w_k1_AXP_s001
                                 + 1.15887 w_k1_DIS_s001
                                 + 1.19371 w_k1_FDX_s001
                                 + 0.834236 w_k1_IBM_s001
                                 + 0.993932 w_k1_INTC_s001
                                 + 0.979366 w_k1_KO_s001
                                 + 0.580342 w_k1_MAT_s001
                                 + 1.09389 w_k1_MSFT_s001
                                 + 0.976957 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.15009 w_k1_ADM_s001
                                 + 0.751824 w_k1_AXP_s001
                                 + 0.854603 w_k1_DIS_s001
                                 + 0.886015 w_k1_FDX_s001
                                 + 0.774246 w_k1_IBM_s001
                                 + 0.941888 w_k1_INTC_s001
                                 + 0.937368 w_k1_KO_s001
                                 + 0.749368 w_k1_MAT_s001
                                 + 0.977407 w_k1_MSFT_s001
                                 + 0.919862 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.08159 w_k1_ADM_s001
                                 + 1.42956 w_k1_AXP_s001
                                 + 1.77584 w_k1_DIS_s001
                                 + 1.41888 w_k1_FDX_s001
                                 + 0.881819 w_k1_IBM_s001
                                 + 1.52187 w_k1_INTC_s001
                                 + 1.28951 w_k1_KO_s001
                                 + 0.799252 w_k1_MAT_s001
                                 + 1.42991 w_k1_MSFT_s001
                                 + 1.23669 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.14914 w_k1_ADM_s001
                                 + 1.23701 w_k1_AXP_s001
                                 + 1.36605 w_k1_DIS_s001 + 1.1543 w_k1_FDX_s001
                                 + 1.59962 w_k1_IBM_s001
                                 + 1.08787 w_k1_INTC_s001
                                 + 1.28551 w_k1_KO_s001 + 1.46517 w_k1_MAT_s001
                                 + 1.13834 w_k1_MSFT_s001
                                 + 1.01205 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.827833 w_k1_ADM_s001
                                 + 0.742331 w_k1_AXP_s001
                                 + 0.844423 w_k1_DIS_s001
                                 + 0.831324 w_k1_FDX_s001
                                 + 1.01362 w_k1_IBM_s001
                                 + 0.953434 w_k1_INTC_s001
                                 + 0.997434 w_k1_KO_s001
                                 + 0.916142 w_k1_MAT_s001
                                 + 0.879262 w_k1_MSFT_s001
                                 + 0.905729 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.877783 w_k1_ADM_s001
                                 + 1.03121 w_k1_AXP_s001
                                 + 0.906889 w_k1_DIS_s001
                                 + 0.752686 w_k1_FDX_s001
                                 + 1.57298 w_k1_IBM_s001
                                 + 0.916391 w_k1_INTC_s001
                                 + 1.21995 w_k1_KO_s001 + 1.61313 w_k1_MAT_s001
                                 + 0.81602 w_k1_MSFT_s001
                                 + 1.04274 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.900016 w_k1_ADM_s001
                                 + 1.14757 w_k1_AXP_s001
                                 + 1.07219 w_k1_DIS_s001
                                 + 1.00434 w_k1_FDX_s001
                                 + 1.05086 w_k1_IBM_s001
                                 + 0.988001 w_k1_INTC_s001
                                 + 1.1133 w_k1_KO_s001 + 1.1951 w_k1_MAT_s001
                                 + 0.960562 w_k1_MSFT_s001
                                 + 1.08375 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.786837 w_k1_ADM_s001
                                 + 0.665961 w_k1_AXP_s001
                                 + 0.586757 w_k1_DIS_s001
                                 + 0.671495 w_k1_FDX_s001
                                 + 0.717523 w_k1_IBM_s001
                                 + 0.729739 w_k1_INTC_s001
                                 + 0.702728 w_k1_KO_s001
                                 + 0.938736 w_k1_MAT_s001
                                 + 0.704269 w_k1_MSFT_s001
                                 + 0.775093 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.907256 w_k1_ADM_s002
                                 + 0.746766 w_k1_AXP_s002
                                 + 0.78844 w_k1_DIS_s002
                                 + 0.779224 w_k1_FDX_s002
                                 + 0.809581 w_k1_IBM_s002
                                 + 0.830968 w_k1_INTC_s002
                                 + 0.775374 w_k1_KO_s002
                                 + 0.627455 w_k1_MAT_s002
                                 + 0.891958 w_k1_MSFT_s002
                                 + 0.91195 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.681061 w_k1_ADM_s002
                                 + 1.04614 w_k1_AXP_s002
                                 + 1.06068 w_k1_DIS_s002
                                 + 0.965689 w_k1_FDX_s002
                                 + 1.52233 w_k1_IBM_s002
                                 + 1.04667 w_k1_INTC_s002
                                 + 1.28452 w_k1_KO_s002 + 1.83883 w_k1_MAT_s002
                                 + 1.00705 w_k1_MSFT_s002
                                 + 1.07112 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.29037 w_k1_ADM_s002
                                 + 1.275 w_k1_AXP_s002 + 1.37853 w_k1_DIS_s002
                                 + 1.40498 w_k1_FDX_s002 + 0.812 w_k1_IBM_s002
                                 + 1.15514 w_k1_INTC_s002 + 1.1425 w_k1_KO_s002
                                 + 1.20363 w_k1_MAT_s002
                                 + 1.28202 w_k1_MSFT_s002
                                 + 1.22349 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.964531 w_k1_ADM_s002
                                 + 1.24159 w_k1_AXP_s002
                                 + 1.34176 w_k1_DIS_s002
                                 + 1.17408 w_k1_FDX_s002
                                 + 1.68014 w_k1_IBM_s002
                                 + 1.40412 w_k1_INTC_s002
                                 + 1.40357 w_k1_KO_s002 + 1.87933 w_k1_MAT_s002
                                 + 1.1912 w_k1_MSFT_s002 + 1.33076 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.25575 w_k1_ADM_s002
                                 + 1.11769 w_k1_AXP_s002
                                 + 1.71287 w_k1_DIS_s002
                                 + 1.19548 w_k1_FDX_s002
                                 + 1.53052 w_k1_IBM_s002
                                 + 1.39702 w_k1_INTC_s002
                                 + 1.44864 w_k1_KO_s002 + 1.91514 w_k1_MAT_s002
                                 + 1.39311 w_k1_MSFT_s002
                                 + 1.32205 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.00575 w_k1_ADM_s002
                                 + 0.977717 w_k1_AXP_s002
                                 + 0.840497 w_k1_DIS_s002
                                 + 1.04288 w_k1_FDX_s002
                                 + 0.784833 w_k1_IBM_s002
                                 + 0.876943 w_k1_INTC_s002
                                 + 0.745837 w_k1_KO_s002
                                 + 0.903677 w_k1_MAT_s002
                                 + 0.928118 w_k1_MSFT_s002 + 1.001 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.40917 w_k1_ADM_s002
                                 + 1.08388 w_k1_AXP_s002
                                 + 1.21444 w_k1_DIS_s002
                                 + 1.22968 w_k1_FDX_s002
                                 + 1.03464 w_k1_IBM_s002
                                 + 1.07317 w_k1_INTC_s002 + 1.1661 w_k1_KO_s002
                                 + 0.991619 w_k1_MAT_s002
                                 + 1.25499 w_k1_MSFT_s002
                                 + 1.27712 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.825974 w_k1_ADM_s002
                                 + 0.975325 w_k1_AXP_s002
                                 + 0.998581 w_k1_DIS_s002
                                 + 1.02945 w_k1_FDX_s002
                                 + 0.942591 w_k1_IBM_s002
                                 + 1.15642 w_k1_INTC_s002
                                 + 0.990943 w_k1_KO_s002
                                 + 1.08294 w_k1_MAT_s002
                                 + 1.09198 w_k1_MSFT_s002
                                 + 1.02311 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.29704 w_k1_ADM_s003
                                 + 1.15496 w_k1_AXP_s003
                                 + 1.34813 w_k1_DIS_s003
                                 + 1.29593 w_k1_FDX_s003
                                 + 1.10476 w_k1_IBM_s003
                                 + 1.15872 w_k1_INTC_s003 + 1.2029 w_k1_KO_s003
                                 + 1.0328 w_k1_MAT_s003
                                 + 1.18015 w_k1_MSFT_s003
                                 + 1.33126 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.3427 w_k1_ADM_s003
                                 + 1.1153 w_k1_AXP_s003 + 1.36108 w_k1_DIS_s003
                                 + 1.22488 w_k1_FDX_s003
                                 + 1.58118 w_k1_IBM_s003
                                 + 1.20891 w_k1_INTC_s003
                                 + 1.54512 w_k1_KO_s003 + 1.13472 w_k1_MAT_s003
                                 + 1.24335 w_k1_MSFT_s003
                                 + 1.28055 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.00705 w_k1_ADM_s003
                                 + 0.929948 w_k1_AXP_s003
                                 + 1.13076 w_k1_DIS_s003 + 1.1283 w_k1_FDX_s003
                                 + 0.869743 w_k1_IBM_s003
                                 + 1.10443 w_k1_INTC_s003
                                 + 1.03834 w_k1_KO_s003
                                 + 0.860045 w_k1_MAT_s003
                                 + 1.16581 w_k1_MSFT_s003
                                 + 1.00315 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.03958 w_k1_ADM_s003
                                 + 1.01078 w_k1_AXP_s003
                                 + 0.955076 w_k1_DIS_s003
                                 + 1.12524 w_k1_FDX_s003
                                 + 0.714057 w_k1_IBM_s003
                                 + 0.97767 w_k1_INTC_s003
                                 + 0.819228 w_k1_KO_s003
                                 + 0.893157 w_k1_MAT_s003
                                 + 1.08981 w_k1_MSFT_s003
                                 + 1.02476 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.09886 w_k1_ADM_s003
                                 + 1.00137 w_k1_AXP_s003
                                 + 0.987235 w_k1_DIS_s003
                                 + 1.02854 w_k1_FDX_s003 + 1.0225 w_k1_IBM_s003
                                 + 0.89885 w_k1_INTC_s003
                                 + 1.08326 w_k1_KO_s003 + 1.26547 w_k1_MAT_s003
                                 + 1.07367 w_k1_MSFT_s003
                                 + 1.06791 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.924227 w_k1_ADM_s003
                                 + 0.610226 w_k1_AXP_s003
                                 + 0.635208 w_k1_DIS_s003
                                 + 0.816291 w_k1_FDX_s003
                                 + 0.67648 w_k1_IBM_s003
                                 + 0.635781 w_k1_INTC_s003
                                 + 0.620663 w_k1_KO_s003
                                 + 0.590913 w_k1_MAT_s003
                                 + 0.689379 w_k1_MSFT_s003
                                 + 0.850263 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.01304 w_k1_ADM_s003
                                 + 0.927354 w_k1_AXP_s003
                                 + 0.725405 w_k1_DIS_s003
                                 + 0.970178 w_k1_FDX_s003
                                 + 0.767952 w_k1_IBM_s003
                                 + 0.892425 w_k1_INTC_s003
                                 + 0.658966 w_k1_KO_s003
                                 + 0.62718 w_k1_MAT_s003
                                 + 0.861071 w_k1_MSFT_s003
                                 + 1.03649 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.03845 w_k1_ADM_s003
                                 + 0.848443 w_k1_AXP_s003
                                 + 0.887423 w_k1_DIS_s003
                                 + 0.879124 w_k1_FDX_s003
                                 + 1.11681 w_k1_IBM_s003
                                 + 0.827993 w_k1_INTC_s003
                                 + 0.951771 w_k1_KO_s003
                                 + 0.853718 w_k1_MAT_s003
                                 + 1.04189 w_k1_MSFT_s003
                                 + 1.01651 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.16394 w_k1_ADM_s004
                                 + 1.52197 w_k1_AXP_s004
                                 + 1.20483 w_k1_DIS_s004
                                 + 1.36114 w_k1_FDX_s004
                                 + 0.971423 w_k1_IBM_s004
                                 + 1.41936 w_k1_INTC_s004
                                 + 0.993042 w_k1_KO_s004
                                 + 0.978036 w_k1_MAT_s004
                                 + 1.21246 w_k1_MSFT_s004 + 1.1689 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.717638 w_k1_ADM_s004
                                 + 0.786358 w_k1_AXP_s004
                                 + 0.700602 w_k1_DIS_s004
                                 + 0.751411 w_k1_FDX_s004
                                 + 0.973669 w_k1_IBM_s004
                                 + 0.75321 w_k1_INTC_s004
                                 + 0.920209 w_k1_KO_s004
                                 + 0.747906 w_k1_MAT_s004
                                 + 0.779592 w_k1_MSFT_s004
                                 + 0.934937 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.955993 w_k1_ADM_s004
                                 + 1.02101 w_k1_AXP_s004
                                 + 0.981201 w_k1_DIS_s004
                                 + 1.0198 w_k1_FDX_s004 + 1.31552 w_k1_IBM_s004
                                 + 1.09695 w_k1_INTC_s004
                                 + 1.09699 w_k1_KO_s004 + 1.02985 w_k1_MAT_s004
                                 + 0.944673 w_k1_MSFT_s004
                                 + 0.961097 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.02258 w_k1_ADM_s004
                                 + 0.966356 w_k1_AXP_s004
                                 + 0.720195 w_k1_DIS_s004
                                 + 0.861752 w_k1_FDX_s004
                                 + 0.668981 w_k1_IBM_s004
                                 + 1.05887 w_k1_INTC_s004
                                 + 0.78941 w_k1_KO_s004 + 0.72634 w_k1_MAT_s004
                                 + 0.857829 w_k1_MSFT_s004
                                 + 0.864505 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.09232 w_k1_ADM_s004
                                 + 0.777284 w_k1_AXP_s004
                                 + 1.03305 w_k1_DIS_s004
                                 + 1.07321 w_k1_FDX_s004
                                 + 0.761973 w_k1_IBM_s004
                                 + 0.934644 w_k1_INTC_s004
                                 + 0.980922 w_k1_KO_s004
                                 + 0.510529 w_k1_MAT_s004
                                 + 1.04712 w_k1_MSFT_s004
                                 + 1.03755 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.04824 w_k1_ADM_s004
                                 + 1.19631 w_k1_AXP_s004
                                 + 0.884968 w_k1_DIS_s004
                                 + 1.05227 w_k1_FDX_s004
                                 + 0.902308 w_k1_IBM_s004
                                 + 0.993941 w_k1_INTC_s004
                                 + 0.899468 w_k1_KO_s004
                                 + 0.912426 w_k1_MAT_s004
                                 + 0.842142 w_k1_MSFT_s004
                                 + 0.969637 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.42385 w_k1_ADM_s004
                                 + 1.34895 w_k1_AXP_s004
                                 + 1.65599 w_k1_DIS_s004 + 1.2715 w_k1_FDX_s004
                                 + 1.3948 w_k1_IBM_s004 + 1.3189 w_k1_INTC_s004
                                 + 1.60208 w_k1_KO_s004 + 1.15545 w_k1_MAT_s004
                                 + 1.32486 w_k1_MSFT_s004
                                 + 1.27397 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.38671 w_k1_ADM_s004
                                 + 1.262 w_k1_AXP_s004 + 1.23167 w_k1_DIS_s004
                                 + 1.20885 w_k1_FDX_s004
                                 + 1.19795 w_k1_IBM_s004
                                 + 1.07436 w_k1_INTC_s004
                                 + 1.12186 w_k1_KO_s004
                                 + 0.940123 w_k1_MAT_s004
                                 + 1.09665 w_k1_MSFT_s004
                                 + 1.20984 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.934904 w_k1_ADM_s005
                                 + 0.776406 w_k1_AXP_s005
                                 + 0.894373 w_k1_DIS_s005
                                 + 0.882651 w_k1_FDX_s005
                                 + 0.801581 w_k1_IBM_s005
                                 + 0.822337 w_k1_INTC_s005
                                 + 0.947017 w_k1_KO_s005
                                 + 1.02443 w_k1_MAT_s005
                                 + 0.896018 w_k1_MSFT_s005
                                 + 0.938944 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.10408 w_k1_ADM_s005
                                 + 0.908382 w_k1_AXP_s005
                                 + 1.10062 w_k1_DIS_s005
                                 + 0.971459 w_k1_FDX_s005
                                 + 1.15377 w_k1_IBM_s005
                                 + 1.09865 w_k1_INTC_s005
                                 + 1.13975 w_k1_KO_s005 + 1.13407 w_k1_MAT_s005
                                 + 1.01553 w_k1_MSFT_s005
                                 + 0.947261 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.702685 w_k1_ADM_s005
                                 + 0.941062 w_k1_AXP_s005
                                 + 0.867838 w_k1_DIS_s005
                                 + 0.84697 w_k1_FDX_s005
                                 + 1.48679 w_k1_IBM_s005
                                 + 0.862933 w_k1_INTC_s005
                                 + 1.22393 w_k1_KO_s005 + 1.765 w_k1_MAT_s005
                                 + 0.854513 w_k1_MSFT_s005
                                 + 1.00839 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.722092 w_k1_ADM_s005
                                 + 0.744953 w_k1_AXP_s005
                                 + 0.64849 w_k1_DIS_s005
                                 + 0.727792 w_k1_FDX_s005
                                 + 0.982044 w_k1_IBM_s005
                                 + 0.92718 w_k1_INTC_s005
                                 + 0.941729 w_k1_KO_s005
                                 + 1.38762 w_k1_MAT_s005
                                 + 0.85251 w_k1_MSFT_s005
                                 + 1.05371 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.01927 w_k1_ADM_s005
                                 + 0.885418 w_k1_AXP_s005
                                 + 1.10132 w_k1_DIS_s005
                                 + 1.12598 w_k1_FDX_s005
                                 + 0.795734 w_k1_IBM_s005
                                 + 1.19088 w_k1_INTC_s005
                                 + 0.993503 w_k1_KO_s005
                                 + 1.14166 w_k1_MAT_s005
                                 + 1.19399 w_k1_MSFT_s005
                                 + 1.13061 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.04856 w_k1_ADM_s005
                                 + 1.16727 w_k1_AXP_s005
                                 + 1.39502 w_k1_DIS_s005
                                 + 1.13181 w_k1_FDX_s005
                                 + 1.40141 w_k1_IBM_s005 + 1.324 w_k1_INTC_s005
                                 + 1.41668 w_k1_KO_s005 + 1.55368 w_k1_MAT_s005
                                 + 1.12862 w_k1_MSFT_s005
                                 + 1.13502 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.977835 w_k1_ADM_s005
                                 + 0.764495 w_k1_AXP_s005
                                 + 0.721318 w_k1_DIS_s005
                                 + 0.783213 w_k1_FDX_s005
                                 + 0.69517 w_k1_IBM_s005
                                 + 0.712542 w_k1_INTC_s005
                                 + 0.839713 w_k1_KO_s005
                                 + 0.952838 w_k1_MAT_s005
                                 + 0.820729 w_k1_MSFT_s005
                                 + 0.918659 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.18454 w_k1_ADM_s005
                                 + 1.30149 w_k1_AXP_s005
                                 + 0.756726 w_k1_DIS_s005
                                 + 0.983684 w_k1_FDX_s005
                                 + 1.13455 w_k1_IBM_s005
                                 + 1.11111 w_k1_INTC_s005
                                 + 1.10864 w_k1_KO_s005 + 1.71229 w_k1_MAT_s005
                                 + 0.97039 w_k1_MSFT_s005
                                 + 1.03249 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.982159 w_k1_ADM_s006
                                 + 1.60153 w_k1_AXP_s006
                                 + 1.32359 w_k1_DIS_s006
                                 + 1.16516 w_k1_FDX_s006
                                 + 1.40077 w_k1_IBM_s006
                                 + 1.19344 w_k1_INTC_s006
                                 + 1.35386 w_k1_KO_s006 + 1.69308 w_k1_MAT_s006
                                 + 1.14411 w_k1_MSFT_s006
                                 + 1.16852 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.45722 w_k1_ADM_s006
                                 + 1.57257 w_k1_AXP_s006
                                 + 1.35859 w_k1_DIS_s006
                                 + 1.22908 w_k1_FDX_s006
                                 + 0.988875 w_k1_IBM_s006
                                 + 1.13703 w_k1_INTC_s006
                                 + 1.12998 w_k1_KO_s006 + 1.07871 w_k1_MAT_s006
                                 + 1.18553 w_k1_MSFT_s006
                                 + 1.13922 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.916247 w_k1_ADM_s006
                                 + 1.18218 w_k1_AXP_s006
                                 + 0.872463 w_k1_DIS_s006
                                 + 0.980188 w_k1_FDX_s006
                                 + 0.83737 w_k1_IBM_s006
                                 + 0.87797 w_k1_INTC_s006
                                 + 0.919921 w_k1_KO_s006 + 1.0102 w_k1_MAT_s006
                                 + 0.912805 w_k1_MSFT_s006
                                 + 0.918136 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.829979 w_k1_ADM_s006
                                 + 0.793801 w_k1_AXP_s006
                                 + 0.625047 w_k1_DIS_s006
                                 + 0.705871 w_k1_FDX_s006
                                 + 1.08182 w_k1_IBM_s006
                                 + 0.836382 w_k1_INTC_s006
                                 + 0.772613 w_k1_KO_s006
                                 + 0.678207 w_k1_MAT_s006
                                 + 0.663856 w_k1_MSFT_s006
                                 + 0.7956 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.12377 w_k1_ADM_s006
                                 + 1.55715 w_k1_AXP_s006
                                 + 1.88323 w_k1_DIS_s006
                                 + 1.27281 w_k1_FDX_s006
                                 + 1.45157 w_k1_IBM_s006
                                 + 1.09149 w_k1_INTC_s006
                                 + 1.51924 w_k1_KO_s006 + 1.69721 w_k1_MAT_s006
                                 + 1.27709 w_k1_MSFT_s006
                                 + 1.11804 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.924306 w_k1_ADM_s006
                                 + 1.02043 w_k1_AXP_s006 + 1.2763 w_k1_DIS_s006
                                 + 1.06606 w_k1_FDX_s006
                                 + 1.00816 w_k1_IBM_s006
                                 + 0.91525 w_k1_INTC_s006
                                 + 1.08105 w_k1_KO_s006 + 1.21961 w_k1_MAT_s006
                                 + 1.02636 w_k1_MSFT_s006 + 1.0227 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.08879 w_k1_ADM_s006
                                 + 1.04149 w_k1_AXP_s006
                                 + 1.11832 w_k1_DIS_s006
                                 + 0.943205 w_k1_FDX_s006
                                 + 1.4074 w_k1_IBM_s006
                                 + 0.813563 w_k1_INTC_s006
                                 + 1.1869 w_k1_KO_s006 + 1.07099 w_k1_MAT_s006
                                 + 0.857263 w_k1_MSFT_s006
                                 + 1.02834 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.689317 w_k1_ADM_s006
                                 + 0.640776 w_k1_AXP_s006
                                 + 0.690999 w_k1_DIS_s006
                                 + 0.758123 w_k1_FDX_s006
                                 + 0.710225 w_k1_IBM_s006
                                 + 0.808977 w_k1_INTC_s006
                                 + 0.745466 w_k1_KO_s006
                                 + 0.558392 w_k1_MAT_s006
                                 + 0.836032 w_k1_MSFT_s006
                                 + 0.832211 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.884359 w_k1_ADM_s007
                                 + 0.905041 w_k1_AXP_s007
                                 + 1.20402 w_k1_DIS_s007
                                 + 1.00531 w_k1_FDX_s007 + 1.0315 w_k1_IBM_s007
                                 + 1.10518 w_k1_INTC_s007
                                 + 1.06293 w_k1_KO_s007 + 0.90734 w_k1_MAT_s007
                                 + 1.12163 w_k1_MSFT_s007
                                 + 0.945027 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.868466 w_k1_ADM_s007
                                 + 0.848031 w_k1_AXP_s007
                                 + 0.783968 w_k1_DIS_s007
                                 + 0.993969 w_k1_FDX_s007
                                 + 0.66542 w_k1_IBM_s007
                                 + 0.724491 w_k1_INTC_s007
                                 + 0.776868 w_k1_KO_s007
                                 + 0.771424 w_k1_MAT_s007
                                 + 0.997256 w_k1_MSFT_s007
                                 + 0.897202 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.893581 w_k1_ADM_s007
                                 + 1.25814 w_k1_AXP_s007
                                 + 0.955643 w_k1_DIS_s007
                                 + 1.03565 w_k1_FDX_s007
                                 + 1.29937 w_k1_IBM_s007
                                 + 1.03372 w_k1_INTC_s007
                                 + 1.13163 w_k1_KO_s007 + 1.53225 w_k1_MAT_s007
                                 + 1.11782 w_k1_MSFT_s007
                                 + 1.14541 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.88798 w_k1_ADM_s007
                                 + 0.841324 w_k1_AXP_s007
                                 + 0.891189 w_k1_DIS_s007
                                 + 0.862327 w_k1_FDX_s007
                                 + 1.29115 w_k1_IBM_s007
                                 + 0.891149 w_k1_INTC_s007
                                 + 0.984036 w_k1_KO_s007
                                 + 1.06861 w_k1_MAT_s007
                                 + 0.926646 w_k1_MSFT_s007
                                 + 1.03224 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.645346 w_k1_ADM_s007
                                 + 0.627406 w_k1_AXP_s007 + 0.659 w_k1_DIS_s007
                                 + 0.733725 w_k1_FDX_s007
                                 + 0.789416 w_k1_IBM_s007
                                 + 0.808143 w_k1_INTC_s007
                                 + 0.75446 w_k1_KO_s007
                                 + 0.636639 w_k1_MAT_s007
                                 + 0.808687 w_k1_MSFT_s007
                                 + 0.77686 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.08652 w_k1_ADM_s007
                                 + 1.36393 w_k1_AXP_s007
                                 + 1.31299 w_k1_DIS_s007
                                 + 1.37533 w_k1_FDX_s007
                                 + 1.04027 w_k1_IBM_s007
                                 + 1.0069 w_k1_INTC_s007 + 1.0501 w_k1_KO_s007
                                 + 0.945782 w_k1_MAT_s007
                                 + 1.23887 w_k1_MSFT_s007
                                 + 1.22746 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.98063 w_k1_ADM_s007
                                 + 1.1597 w_k1_AXP_s007 + 1.53019 w_k1_DIS_s007
                                 + 1.26448 w_k1_FDX_s007
                                 + 1.33878 w_k1_IBM_s007
                                 + 1.10185 w_k1_INTC_s007 + 1.2316 w_k1_KO_s007
                                 + 1.27599 w_k1_MAT_s007
                                 + 1.45972 w_k1_MSFT_s007
                                 + 1.22964 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.771101 w_k1_ADM_s007
                                 + 1.02653 w_k1_AXP_s007
                                 + 1.01978 w_k1_DIS_s007
                                 + 1.13891 w_k1_FDX_s007
                                 + 0.820947 w_k1_IBM_s007
                                 + 0.990841 w_k1_INTC_s007
                                 + 1.02764 w_k1_KO_s007 + 1.05632 w_k1_MAT_s007
                                 + 1.19486 w_k1_MSFT_s007
                                 + 0.989109 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.24538 w_k1_ADM_s010
                                 + 1.18893 w_k1_AXP_s010 + 1.1945 w_k1_DIS_s010
                                 + 1.02526 w_k1_FDX_s010
                                 + 1.40844 w_k1_IBM_s010
                                 + 1.05051 w_k1_INTC_s010
                                 + 1.22545 w_k1_KO_s010 + 1.19136 w_k1_MAT_s010
                                 + 1.06908 w_k1_MSFT_s010
                                 + 1.21671 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.853503 w_k1_ADM_s010
                                 + 0.839064 w_k1_AXP_s010
                                 + 0.727435 w_k1_DIS_s010
                                 + 0.799224 w_k1_FDX_s010
                                 + 0.802757 w_k1_IBM_s010
                                 + 0.903346 w_k1_INTC_s010
                                 + 0.717487 w_k1_KO_s010
                                 + 0.864135 w_k1_MAT_s010
                                 + 0.85335 w_k1_MSFT_s010
                                 + 0.820657 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.18959 w_k1_ADM_s010
                                 + 1.16539 w_k1_AXP_s010
                                 + 0.948071 w_k1_DIS_s010
                                 + 1.03698 w_k1_FDX_s010
                                 + 0.859095 w_k1_IBM_s010
                                 + 1.07979 w_k1_INTC_s010
                                 + 1.01869 w_k1_KO_s010
                                 + 0.928003 w_k1_MAT_s010
                                 + 1.06348 w_k1_MSFT_s010
                                 + 0.944403 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.814761 w_k1_ADM_s010
                                 + 0.852014 w_k1_AXP_s010
                                 + 0.849586 w_k1_DIS_s010
                                 + 0.809682 w_k1_FDX_s010
                                 + 1.29474 w_k1_IBM_s010
                                 + 0.92388 w_k1_INTC_s010
                                 + 1.08089 w_k1_KO_s010 + 1.1879 w_k1_MAT_s010
                                 + 0.93008 w_k1_MSFT_s010
                                 + 1.08237 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.3857 w_k1_ADM_s010
                                 + 1.76785 w_k1_AXP_s010
                                 + 1.50862 w_k1_DIS_s010
                                 + 1.38148 w_k1_FDX_s010
                                 + 1.32218 w_k1_IBM_s010
                                 + 1.39312 w_k1_INTC_s010
                                 + 1.33133 w_k1_KO_s010 + 1.36383 w_k1_MAT_s010
                                 + 1.18444 w_k1_MSFT_s010
                                 + 1.29234 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.32468 w_k1_ADM_s010
                                 + 1.28428 w_k1_AXP_s010
                                 + 1.57625 w_k1_DIS_s010
                                 + 1.23136 w_k1_FDX_s010
                                 + 1.21525 w_k1_IBM_s010
                                 + 1.13125 w_k1_INTC_s010
                                 + 1.36425 w_k1_KO_s010 + 1.09933 w_k1_MAT_s010
                                 + 1.24832 w_k1_MSFT_s010
                                 + 1.27433 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.10366 w_k1_ADM_s010
                                 + 1.15491 w_k1_AXP_s010
                                 + 1.18557 w_k1_DIS_s010
                                 + 1.16667 w_k1_FDX_s010
                                 + 0.808616 w_k1_IBM_s010
                                 + 1.16137 w_k1_INTC_s010
                                 + 1.13297 w_k1_KO_s010
                                 + 0.992284 w_k1_MAT_s010
                                 + 1.21104 w_k1_MSFT_s010
                                 + 1.07048 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.05965 w_k1_ADM_s010
                                 + 1.52714 w_k1_AXP_s010
                                 + 0.844244 w_k1_DIS_s010
                                 + 1.0087 w_k1_FDX_s010 + 1.08047 w_k1_IBM_s010
                                 + 0.935419 w_k1_INTC_s010
                                 + 0.964492 w_k1_KO_s010
                                 + 1.28329 w_k1_MAT_s010
                                 + 0.911284 w_k1_MSFT_s010
                                 + 1.05495 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.04957 w_k1_ADM_s011
                                 + 0.887651 w_k1_AXP_s011
                                 + 1.10118 w_k1_DIS_s011
                                 + 1.09442 w_k1_FDX_s011
                                 + 0.768332 w_k1_IBM_s011
                                 + 1.18066 w_k1_INTC_s011
                                 + 0.976779 w_k1_KO_s011
                                 + 0.791437 w_k1_MAT_s011
                                 + 1.19285 w_k1_MSFT_s011
                                 + 0.97067 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.738257 w_k1_ADM_s011
                                 + 0.67204 w_k1_AXP_s011
                                 + 0.718837 w_k1_DIS_s011
                                 + 0.752782 w_k1_FDX_s011
                                 + 0.777639 w_k1_IBM_s011
                                 + 0.892389 w_k1_INTC_s011
                                 + 0.76269 w_k1_KO_s011 + 0.54901 w_k1_MAT_s011
                                 + 0.804123 w_k1_MSFT_s011
                                 + 0.764717 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.27415 w_k1_ADM_s011
                                 + 1.84501 w_k1_AXP_s011
                                 + 2.12192 w_k1_DIS_s011
                                 + 1.50997 w_k1_FDX_s011
                                 + 1.61121 w_k1_IBM_s011
                                 + 1.68032 w_k1_INTC_s011
                                 + 1.79215 w_k1_KO_s011 + 1.767 w_k1_MAT_s011
                                 + 1.52699 w_k1_MSFT_s011
                                 + 1.21987 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.05075 w_k1_ADM_s011
                                 + 1.37524 w_k1_AXP_s011
                                 + 1.47759 w_k1_DIS_s011 + 1.2188 w_k1_FDX_s011
                                 + 1.14912 w_k1_IBM_s011
                                 + 1.51809 w_k1_INTC_s011
                                 + 1.20303 w_k1_KO_s011 + 1.0567 w_k1_MAT_s011
                                 + 1.29655 w_k1_MSFT_s011 + 1.1607 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.75914 w_k1_ADM_s011
                                 + 0.732012 w_k1_AXP_s011
                                 + 0.879017 w_k1_DIS_s011
                                 + 0.871756 w_k1_FDX_s011
                                 + 1.00551 w_k1_IBM_s011
                                 + 1.13885 w_k1_INTC_s011
                                 + 1.17433 w_k1_KO_s011
                                 + 0.900018 w_k1_MAT_s011
                                 + 1.09553 w_k1_MSFT_s011
                                 + 0.871466 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.755241 w_k1_ADM_s011
                                 + 0.825841 w_k1_AXP_s011
                                 + 0.854571 w_k1_DIS_s011
                                 + 0.717253 w_k1_FDX_s011
                                 + 1.38427 w_k1_IBM_s011
                                 + 0.903328 w_k1_INTC_s011
                                 + 1.18158 w_k1_KO_s011 + 1.32247 w_k1_MAT_s011
                                 + 0.798935 w_k1_MSFT_s011
                                 + 0.873187 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.955158 w_k1_ADM_s011
                                 + 0.957652 w_k1_AXP_s011
                                 + 0.948921 w_k1_DIS_s011
                                 + 0.936801 w_k1_FDX_s011
                                 + 1.28229 w_k1_IBM_s011
                                 + 1.17259 w_k1_INTC_s011
                                 + 1.03665 w_k1_KO_s011 + 1.07066 w_k1_MAT_s011
                                 + 1.08765 w_k1_MSFT_s011
                                 + 0.866124 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.946679 w_k1_ADM_s011
                                 + 1.12252 w_k1_AXP_s011
                                 + 0.832893 w_k1_DIS_s011
                                 + 1.02232 w_k1_FDX_s011
                                 + 0.974061 w_k1_IBM_s011
                                 + 1.19364 w_k1_INTC_s011
                                 + 0.879285 w_k1_KO_s011
                                 + 1.06764 w_k1_MAT_s011
                                 + 1.00476 w_k1_MSFT_s011
                                 + 0.960355 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.929776 w_k1_ADM_s012
                                 + 0.67841 w_k1_AXP_s012
                                 + 0.645008 w_k1_DIS_s012
                                 + 0.762765 w_k1_FDX_s012
                                 + 0.657768 w_k1_IBM_s012
                                 + 0.731479 w_k1_INTC_s012
                                 + 0.649346 w_k1_KO_s012
                                 + 0.486128 w_k1_MAT_s012
                                 + 0.811651 w_k1_MSFT_s012
                                 + 0.848247 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.09504 w_k1_ADM_s012
                                 + 1.03715 w_k1_AXP_s012
                                 + 1.11219 w_k1_DIS_s012
                                 + 1.06758 w_k1_FDX_s012
                                 + 0.770666 w_k1_IBM_s012
                                 + 0.989094 w_k1_INTC_s012
                                 + 0.916232 w_k1_KO_s012
                                 + 0.759708 w_k1_MAT_s012
                                 + 1.10873 w_k1_MSFT_s012
                                 + 1.05766 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.884958 w_k1_ADM_s012
                                 + 0.806982 w_k1_AXP_s012
                                 + 0.80012 w_k1_DIS_s012
                                 + 0.773196 w_k1_FDX_s012
                                 + 1.15379 w_k1_IBM_s012
                                 + 0.848688 w_k1_INTC_s012
                                 + 0.880969 w_k1_KO_s012
                                 + 0.842142 w_k1_MAT_s012
                                 + 0.853161 w_k1_MSFT_s012
                                 + 0.936894 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.01167 w_k1_ADM_s012
                                 + 1.0806 w_k1_AXP_s012
                                 + 0.909114 w_k1_DIS_s012
                                 + 1.07181 w_k1_FDX_s012
                                 + 0.95066 w_k1_IBM_s012
                                 + 1.05699 w_k1_INTC_s012
                                 + 0.920453 w_k1_KO_s012
                                 + 0.970114 w_k1_MAT_s012
                                 + 1.06823 w_k1_MSFT_s012
                                 + 1.01295 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.2233 w_k1_ADM_s012
                                 + 1.37722 w_k1_AXP_s012
                                 + 1.38739 w_k1_DIS_s012
                                 + 1.31095 w_k1_FDX_s012
                                 + 1.39014 w_k1_IBM_s012
                                 + 1.17425 w_k1_INTC_s012
                                 + 1.19143 w_k1_KO_s012 + 1.42837 w_k1_MAT_s012
                                 + 1.31101 w_k1_MSFT_s012
                                 + 1.24708 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.20593 w_k1_ADM_s012
                                 + 1.32078 w_k1_AXP_s012
                                 + 1.59616 w_k1_DIS_s012
                                 + 1.42092 w_k1_FDX_s012
                                 + 0.87783 w_k1_IBM_s012
                                 + 1.4889 w_k1_INTC_s012 + 1.1089 w_k1_KO_s012
                                 + 1.05089 w_k1_MAT_s012
                                 + 1.38301 w_k1_MSFT_s012
                                 + 1.14157 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.963046 w_k1_ADM_s012
                                 + 0.811478 w_k1_AXP_s012
                                 + 1.31121 w_k1_DIS_s012
                                 + 0.981612 w_k1_FDX_s012
                                 + 1.30779 w_k1_IBM_s012
                                 + 0.992246 w_k1_INTC_s012
                                 + 1.18867 w_k1_KO_s012 + 1.18127 w_k1_MAT_s012
                                 + 1.12644 w_k1_MSFT_s012
                                 + 1.08023 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.087 w_k1_ADM_s012
                                 + 1.16924 w_k1_AXP_s012
                                 + 1.15152 w_k1_DIS_s012
                                 + 0.968522 w_k1_FDX_s012
                                 + 1.63106 w_k1_IBM_s012
                                 + 1.06169 w_k1_INTC_s012
                                 + 1.23791 w_k1_KO_s012 + 1.34454 w_k1_MAT_s012
                                 + 1.05604 w_k1_MSFT_s012
                                 + 1.12364 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.856671 w_k1_ADM_s013
                                 + 0.878269 w_k1_AXP_s013
                                 + 0.912808 w_k1_DIS_s013
                                 + 0.894078 w_k1_FDX_s013
                                 + 1.13747 w_k1_IBM_s013
                                 + 1.19727 w_k1_INTC_s013
                                 + 0.917132 w_k1_KO_s013
                                 + 0.862037 w_k1_MAT_s013
                                 + 0.942774 w_k1_MSFT_s013
                                 + 0.917791 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.626685 w_k1_ADM_s013
                                 + 0.736715 w_k1_AXP_s013
                                 + 0.562811 w_k1_DIS_s013
                                 + 0.70005 w_k1_FDX_s013
                                 + 0.816645 w_k1_IBM_s013
                                 + 0.949639 w_k1_INTC_s013
                                 + 0.695598 w_k1_KO_s013
                                 + 0.769841 w_k1_MAT_s013
                                 + 0.845745 w_k1_MSFT_s013
                                 + 0.782919 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.43221 w_k1_ADM_s013
                                 + 1.45101 w_k1_AXP_s013
                                 + 1.77957 w_k1_DIS_s013
                                 + 1.50643 w_k1_FDX_s013
                                 + 1.17974 w_k1_IBM_s013
                                 + 1.72951 w_k1_INTC_s013
                                 + 1.23669 w_k1_KO_s013 + 0.71486 w_k1_MAT_s013
                                 + 1.40956 w_k1_MSFT_s013
                                 + 1.17767 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.00712 w_k1_ADM_s013
                                 + 1.31767 w_k1_AXP_s013
                                 + 0.918533 w_k1_DIS_s013
                                 + 1.20646 w_k1_FDX_s013
                                 + 0.73966 w_k1_IBM_s013
                                 + 1.32917 w_k1_INTC_s013 + 0.7252 w_k1_KO_s013
                                 + 0.771737 w_k1_MAT_s013
                                 + 1.07048 w_k1_MSFT_s013
                                 + 0.888166 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.06815 w_k1_ADM_s013
                                 + 0.905297 w_k1_AXP_s013
                                 + 0.860772 w_k1_DIS_s013
                                 + 0.987125 w_k1_FDX_s013
                                 + 0.783607 w_k1_IBM_s013
                                 + 1.08399 w_k1_INTC_s013
                                 + 0.757875 w_k1_KO_s013
                                 + 0.668073 w_k1_MAT_s013
                                 + 0.96607 w_k1_MSFT_s013
                                 + 0.835744 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.08192 w_k1_ADM_s013
                                 + 0.994595 w_k1_AXP_s013
                                 + 1.21924 w_k1_DIS_s013
                                 + 1.15668 w_k1_FDX_s013 + 1.0087 w_k1_IBM_s013
                                 + 1.18644 w_k1_INTC_s013
                                 + 1.04977 w_k1_KO_s013 + 1.00299 w_k1_MAT_s013
                                 + 1.19939 w_k1_MSFT_s013
                                 + 1.18893 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.820177 w_k1_ADM_s013
                                 + 0.750916 w_k1_AXP_s013
                                 + 0.751383 w_k1_DIS_s013
                                 + 0.827448 w_k1_FDX_s013
                                 + 0.782707 w_k1_IBM_s013
                                 + 0.932222 w_k1_INTC_s013
                                 + 0.75848 w_k1_KO_s013
                                 + 0.512376 w_k1_MAT_s013
                                 + 0.826939 w_k1_MSFT_s013
                                 + 0.899655 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.19744 w_k1_ADM_s013
                                 + 1.37957 w_k1_AXP_s013
                                 + 1.50323 w_k1_DIS_s013
                                 + 1.25581 w_k1_FDX_s013
                                 + 1.61403 w_k1_IBM_s013
                                 + 1.21625 w_k1_INTC_s013
                                 + 1.23555 w_k1_KO_s013 + 1.88727 w_k1_MAT_s013
                                 + 1.07615 w_k1_MSFT_s013
                                 + 1.22634 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.791233 w_k1_ADM_s014
                                 + 1.05541 w_k1_AXP_s014
                                 + 1.12461 w_k1_DIS_s014
                                 + 0.939478 w_k1_FDX_s014
                                 + 1.63191 w_k1_IBM_s014
                                 + 1.03151 w_k1_INTC_s014
                                 + 1.28608 w_k1_KO_s014 + 2.11171 w_k1_MAT_s014
                                 + 1.09964 w_k1_MSFT_s014 + 1.0722 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.762084 w_k1_ADM_s014
                                 + 0.740814 w_k1_AXP_s014
                                 + 0.616516 w_k1_DIS_s014
                                 + 0.783533 w_k1_FDX_s014
                                 + 0.743155 w_k1_IBM_s014
                                 + 0.904253 w_k1_INTC_s014
                                 + 0.728936 w_k1_KO_s014
                                 + 0.75254 w_k1_MAT_s014
                                 + 0.89664 w_k1_MSFT_s014
                                 + 0.879084 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.15252 w_k1_ADM_s014
                                 + 0.854519 w_k1_AXP_s014
                                 + 0.975358 w_k1_DIS_s014
                                 + 1.05185 w_k1_FDX_s014
                                 + 0.84442 w_k1_IBM_s014
                                 + 0.994325 w_k1_INTC_s014
                                 + 0.975643 w_k1_KO_s014
                                 + 0.899511 w_k1_MAT_s014
                                 + 1.03152 w_k1_MSFT_s014
                                 + 0.989572 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.31655 w_k1_ADM_s014
                                 + 1.17788 w_k1_AXP_s014
                                 + 1.13252 w_k1_DIS_s014
                                 + 1.27751 w_k1_FDX_s014
                                 + 0.852402 w_k1_IBM_s014
                                 + 1.15124 w_k1_INTC_s014
                                 + 0.986594 w_k1_KO_s014
                                 + 0.874107 w_k1_MAT_s014
                                 + 1.22881 w_k1_MSFT_s014
                                 + 1.02304 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.977769 w_k1_ADM_s014
                                 + 1.12771 w_k1_AXP_s014
                                 + 1.41791 w_k1_DIS_s014
                                 + 1.20498 w_k1_FDX_s014
                                 + 1.42259 w_k1_IBM_s014
                                 + 1.37222 w_k1_INTC_s014 + 1.4104 w_k1_KO_s014
                                 + 1.23027 w_k1_MAT_s014
                                 + 1.33998 w_k1_MSFT_s014
                                 + 1.09947 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.898998 w_k1_ADM_s014
                                 + 0.719262 w_k1_AXP_s014
                                 + 0.747725 w_k1_DIS_s014
                                 + 0.80674 w_k1_FDX_s014 + 1.065 w_k1_IBM_s014
                                 + 0.913914 w_k1_INTC_s014
                                 + 0.994235 w_k1_KO_s014
                                 + 0.806095 w_k1_MAT_s014
                                 + 0.855806 w_k1_MSFT_s014
                                 + 0.932011 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.43203 w_k1_ADM_s014
                                 + 1.67076 w_k1_AXP_s014
                                 + 1.60989 w_k1_DIS_s014
                                 + 1.48607 w_k1_FDX_s014
                                 + 1.48846 w_k1_IBM_s014
                                 + 1.45599 w_k1_INTC_s014
                                 + 1.30915 w_k1_KO_s014 + 1.29989 w_k1_MAT_s014
                                 + 1.42516 w_k1_MSFT_s014
                                 + 1.17262 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.909545 w_k1_ADM_s014
                                 + 1.08056 w_k1_AXP_s014
                                 + 0.993653 w_k1_DIS_s014
                                 + 1.11148 w_k1_FDX_s014
                                 + 1.16527 w_k1_IBM_s014
                                 + 1.1849 w_k1_INTC_s014 + 1.11418 w_k1_KO_s014
                                 + 1.24301 w_k1_MAT_s014
                                 + 1.13099 w_k1_MSFT_s014
                                 + 1.01694 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.964906 w_k1_ADM_s015
                                 + 1.01513 w_k1_AXP_s015
                                 + 1.07163 w_k1_DIS_s015
                                 + 1.06745 w_k1_FDX_s015
                                 + 0.959678 w_k1_IBM_s015
                                 + 1.16758 w_k1_INTC_s015
                                 + 1.03107 w_k1_KO_s015
                                 + 0.782287 w_k1_MAT_s015
                                 + 1.07857 w_k1_MSFT_s015
                                 + 0.924026 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.1 w_k1_ADM_s015
                                 + 1.20499 w_k1_AXP_s015
                                 + 1.08052 w_k1_DIS_s015
                                 + 1.19072 w_k1_FDX_s015
                                 + 0.706319 w_k1_IBM_s015
                                 + 1.41308 w_k1_INTC_s015
                                 + 0.930778 w_k1_KO_s015
                                 + 0.535765 w_k1_MAT_s015
                                 + 1.03004 w_k1_MSFT_s015
                                 + 1.01666 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.919137 w_k1_ADM_s015
                                 + 0.899034 w_k1_AXP_s015
                                 + 0.810914 w_k1_DIS_s015
                                 + 1.02579 w_k1_FDX_s015
                                 + 0.687323 w_k1_IBM_s015
                                 + 0.886631 w_k1_INTC_s015
                                 + 0.790387 w_k1_KO_s015
                                 + 0.754458 w_k1_MAT_s015
                                 + 0.881153 w_k1_MSFT_s015
                                 + 0.900938 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.927079 w_k1_ADM_s015
                                 + 1.33915 w_k1_AXP_s015
                                 + 1.24172 w_k1_DIS_s015
                                 + 1.06473 w_k1_FDX_s015
                                 + 1.28044 w_k1_IBM_s015
                                 + 1.26817 w_k1_INTC_s015
                                 + 1.13758 w_k1_KO_s015 + 1.01062 w_k1_MAT_s015
                                 + 1.02227 w_k1_MSFT_s015
                                 + 1.01091 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.732431 w_k1_ADM_s015
                                 + 0.753253 w_k1_AXP_s015
                                 + 0.899988 w_k1_DIS_s015
                                 + 0.886505 w_k1_FDX_s015
                                 + 1.16617 w_k1_IBM_s015
                                 + 1.10397 w_k1_INTC_s015
                                 + 0.877792 w_k1_KO_s015
                                 + 0.89239 w_k1_MAT_s015
                                 + 0.945898 w_k1_MSFT_s015
                                 + 0.870593 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.794527 w_k1_ADM_s015
                                 + 0.826101 w_k1_AXP_s015
                                 + 0.723852 w_k1_DIS_s015
                                 + 0.851571 w_k1_FDX_s015
                                 + 0.793844 w_k1_IBM_s015
                                 + 1.03636 w_k1_INTC_s015
                                 + 0.716926 w_k1_KO_s015
                                 + 0.570975 w_k1_MAT_s015
                                 + 0.815473 w_k1_MSFT_s015
                                 + 0.818564 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.03398 w_k1_ADM_s015
                                 + 1.49574 w_k1_AXP_s015
                                 + 1.72239 w_k1_DIS_s015
                                 + 1.31899 w_k1_FDX_s015
                                 + 1.38262 w_k1_IBM_s015
                                 + 1.47563 w_k1_INTC_s015
                                 + 1.48905 w_k1_KO_s015 + 1.04119 w_k1_MAT_s015
                                 + 1.27118 w_k1_MSFT_s015
                                 + 1.29278 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.12639 w_k1_ADM_s015
                                 + 1.21965 w_k1_AXP_s015
                                 + 1.43978 w_k1_DIS_s015
                                 + 1.34518 w_k1_FDX_s015
                                 + 0.889553 w_k1_IBM_s015
                                 + 1.20137 w_k1_INTC_s015
                                 + 1.20166 w_k1_KO_s015 + 1.15704 w_k1_MAT_s015
                                 + 1.21815 w_k1_MSFT_s015
                                 + 1.11891 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.67699 w_k1_ADM_s016
                                 + 0.653143 w_k1_AXP_s016
                                 + 0.702958 w_k1_DIS_s016
                                 + 0.77162 w_k1_FDX_s016
                                 + 0.835422 w_k1_IBM_s016
                                 + 0.788224 w_k1_INTC_s016
                                 + 0.727545 w_k1_KO_s016
                                 + 0.817251 w_k1_MAT_s016
                                 + 0.945402 w_k1_MSFT_s016
                                 + 0.770484 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.821935 w_k1_ADM_s016
                                 + 1.20521 w_k1_AXP_s016
                                 + 1.21848 w_k1_DIS_s016
                                 + 1.08818 w_k1_FDX_s016
                                 + 1.48555 w_k1_IBM_s016
                                 + 1.18362 w_k1_INTC_s016
                                 + 1.21692 w_k1_KO_s016 + 1.67806 w_k1_MAT_s016
                                 + 1.15487 w_k1_MSFT_s016
                                 + 1.09211 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.965908 w_k1_ADM_s016
                                 + 1.20056 w_k1_AXP_s016
                                 + 0.853519 w_k1_DIS_s016
                                 + 0.955919 w_k1_FDX_s016
                                 + 0.981028 w_k1_IBM_s016
                                 + 0.871149 w_k1_INTC_s016
                                 + 0.858615 w_k1_KO_s016
                                 + 1.17865 w_k1_MAT_s016
                                 + 0.822344 w_k1_MSFT_s016
                                 + 0.954265 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.91389 w_k1_ADM_s016
                                 + 0.835205 w_k1_AXP_s016
                                 + 0.905347 w_k1_DIS_s016
                                 + 0.92757 w_k1_FDX_s016
                                 + 0.970099 w_k1_IBM_s016
                                 + 0.846444 w_k1_INTC_s016
                                 + 0.969121 w_k1_KO_s016
                                 + 0.941807 w_k1_MAT_s016
                                 + 0.932392 w_k1_MSFT_s016
                                 + 0.86478 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.775132 w_k1_ADM_s016
                                 + 0.915338 w_k1_AXP_s016
                                 + 0.925041 w_k1_DIS_s016
                                 + 0.832583 w_k1_FDX_s016
                                 + 1.3734 w_k1_IBM_s016
                                 + 0.91412 w_k1_INTC_s016
                                 + 1.10659 w_k1_KO_s016 + 1.12249 w_k1_MAT_s016
                                 + 0.974664 w_k1_MSFT_s016
                                 + 0.923906 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.973524 w_k1_ADM_s016
                                 + 0.967463 w_k1_AXP_s016
                                 + 1.07796 w_k1_DIS_s016
                                 + 1.14421 w_k1_FDX_s016
                                 + 0.805432 w_k1_IBM_s016
                                 + 1.02876 w_k1_INTC_s016
                                 + 1.07946 w_k1_KO_s016
                                 + 0.977292 w_k1_MAT_s016
                                 + 1.06068 w_k1_MSFT_s016
                                 + 0.942886 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.04585 w_k1_ADM_s016
                                 + 1.20906 w_k1_AXP_s016
                                 + 1.56968 w_k1_DIS_s016
                                 + 1.38524 w_k1_FDX_s016
                                 + 0.886631 w_k1_IBM_s016
                                 + 1.32168 w_k1_INTC_s016
                                 + 1.05554 w_k1_KO_s016
                                 + 0.843447 w_k1_MAT_s016
                                 + 1.4018 w_k1_MSFT_s016 + 1.14107 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.977288 w_k1_ADM_s016
                                 + 1.26861 w_k1_AXP_s016
                                 + 1.32952 w_k1_DIS_s016
                                 + 1.24061 w_k1_FDX_s016
                                 + 1.21827 w_k1_IBM_s016
                                 + 1.11023 w_k1_INTC_s016
                                 + 1.10885 w_k1_KO_s016 + 1.37208 w_k1_MAT_s016
                                 + 1.1349 w_k1_MSFT_s016 + 1.07279 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.709937 w_k1_ADM_s017
                                 + 0.617197 w_k1_AXP_s017
                                 + 0.788745 w_k1_DIS_s017
                                 + 0.774071 w_k1_FDX_s017
                                 + 1.0622 w_k1_IBM_s017
                                 + 0.773659 w_k1_INTC_s017
                                 + 0.985659 w_k1_KO_s017
                                 + 0.767917 w_k1_MAT_s017
                                 + 0.830356 w_k1_MSFT_s017
                                 + 0.885443 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.721024 w_k1_ADM_s017
                                 + 0.78105 w_k1_AXP_s017
                                 + 0.721194 w_k1_DIS_s017
                                 + 0.831163 w_k1_FDX_s017
                                 + 0.823831 w_k1_IBM_s017
                                 + 0.784007 w_k1_INTC_s017
                                 + 0.789301 w_k1_KO_s017
                                 + 0.811652 w_k1_MAT_s017
                                 + 0.759219 w_k1_MSFT_s017
                                 + 0.877733 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.946384 w_k1_ADM_s017
                                 + 0.876782 w_k1_AXP_s017
                                 + 1.18953 w_k1_DIS_s017
                                 + 1.04374 w_k1_FDX_s017
                                 + 1.40376 w_k1_IBM_s017
                                 + 1.0199 w_k1_INTC_s017 + 1.24189 w_k1_KO_s017
                                 + 1.35355 w_k1_MAT_s017
                                 + 1.11213 w_k1_MSFT_s017
                                 + 0.939183 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.781177 w_k1_ADM_s017
                                 + 1.08944 w_k1_AXP_s017 + 1.3417 w_k1_DIS_s017
                                 + 1.05466 w_k1_FDX_s017
                                 + 1.72269 w_k1_IBM_s017
                                 + 1.23647 w_k1_INTC_s017
                                 + 1.54332 w_k1_KO_s017 + 2.12181 w_k1_MAT_s017
                                 + 1.08156 w_k1_MSFT_s017
                                 + 1.08817 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.05972 w_k1_ADM_s017
                                 + 1.05372 w_k1_AXP_s017
                                 + 1.09871 w_k1_DIS_s017
                                 + 1.02248 w_k1_FDX_s017
                                 + 1.11008 w_k1_IBM_s017
                                 + 0.952479 w_k1_INTC_s017
                                 + 1.04841 w_k1_KO_s017 + 1.17745 w_k1_MAT_s017
                                 + 1.04765 w_k1_MSFT_s017
                                 + 0.968878 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.09816 w_k1_ADM_s017
                                 + 1.24735 w_k1_AXP_s017
                                 + 1.37833 w_k1_DIS_s017
                                 + 1.29988 w_k1_FDX_s017
                                 + 0.833998 w_k1_IBM_s017
                                 + 0.965495 w_k1_INTC_s017
                                 + 1.05663 w_k1_KO_s017 + 1.09828 w_k1_MAT_s017
                                 + 1.03965 w_k1_MSFT_s017
                                 + 0.970443 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.294 w_k1_ADM_s017
                                 + 1.67473 w_k1_AXP_s017
                                 + 1.68361 w_k1_DIS_s017
                                 + 1.41665 w_k1_FDX_s017
                                 + 1.25787 w_k1_IBM_s017
                                 + 1.23686 w_k1_INTC_s017
                                 + 1.30563 w_k1_KO_s017 + 1.51953 w_k1_MAT_s017
                                 + 1.20028 w_k1_MSFT_s017
                                 + 1.20929 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.777047 w_k1_ADM_s017
                                 + 0.745348 w_k1_AXP_s017
                                 + 0.807791 w_k1_DIS_s017
                                 + 0.963991 w_k1_FDX_s017
                                 + 0.604363 w_k1_IBM_s017
                                 + 0.821197 w_k1_INTC_s017
                                 + 0.749416 w_k1_KO_s017
                                 + 0.628792 w_k1_MAT_s017
                                 + 0.797859 w_k1_MSFT_s017
                                 + 0.797533 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.51968 w_k1_ADM_s100
                                 + 0.960634 w_k1_AXP_s100
                                 + 1.01496 w_k1_DIS_s100
                                 + 1.14555 w_k1_FDX_s100
                                 + 0.813267 w_k1_IBM_s100
                                 + 1.12347 w_k1_INTC_s100
                                 + 0.81566 w_k1_KO_s100
                                 + 0.581672 w_k1_MAT_s100
                                 + 0.939003 w_k1_MSFT_s100
                                 + 0.946896 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.59393 w_k1_ADM_s100
                                 + 1.54245 w_k1_AXP_s100
                                 + 1.65708 w_k1_DIS_s100
                                 + 1.48622 w_k1_FDX_s100
                                 + 1.01392 w_k1_IBM_s100
                                 + 1.50983 w_k1_INTC_s100
                                 + 1.17538 w_k1_KO_s100
                                 + 0.863423 w_k1_MAT_s100
                                 + 1.21387 w_k1_MSFT_s100
                                 + 1.01469 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.02845 w_k1_ADM_s100
                                 + 1.03116 w_k1_AXP_s100
                                 + 1.04919 w_k1_DIS_s100
                                 + 1.01101 w_k1_FDX_s100
                                 + 1.12229 w_k1_IBM_s100
                                 + 1.23814 w_k1_INTC_s100 + 1.0823 w_k1_KO_s100
                                 + 1.05405 w_k1_MAT_s100
                                 + 0.941416 w_k1_MSFT_s100
                                 + 1.01169 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.19229 w_k1_ADM_s100
                                 + 1.44918 w_k1_AXP_s100
                                 + 1.32712 w_k1_DIS_s100
                                 + 1.16243 w_k1_FDX_s100
                                 + 1.41467 w_k1_IBM_s100
                                 + 1.12369 w_k1_INTC_s100
                                 + 1.19946 w_k1_KO_s100 + 1.79339 w_k1_MAT_s100
                                 + 1.08109 w_k1_MSFT_s100
                                 + 1.28219 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.712327 w_k1_ADM_s100
                                 + 0.814704 w_k1_AXP_s100
                                 + 0.679112 w_k1_DIS_s100
                                 + 0.785291 w_k1_FDX_s100
                                 + 0.958919 w_k1_IBM_s100
                                 + 0.834771 w_k1_INTC_s100
                                 + 0.792975 w_k1_KO_s100
                                 + 0.964055 w_k1_MAT_s100
                                 + 0.671808 w_k1_MSFT_s100
                                 + 1.00547 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.19578 w_k1_ADM_s100
                                 + 0.962474 w_k1_AXP_s100
                                 + 0.761642 w_k1_DIS_s100
                                 + 0.926372 w_k1_FDX_s100
                                 + 0.87094 w_k1_IBM_s100
                                 + 0.842649 w_k1_INTC_s100
                                 + 0.741959 w_k1_KO_s100
                                 + 0.776687 w_k1_MAT_s100
                                 + 0.779061 w_k1_MSFT_s100
                                 + 1.02334 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.73266 w_k1_ADM_s100
                                 + 1.14835 w_k1_AXP_s100
                                 + 1.54049 w_k1_DIS_s100
                                 + 1.31916 w_k1_FDX_s100
                                 + 1.07362 w_k1_IBM_s100 + 1.019 w_k1_INTC_s100
                                 + 1.29448 w_k1_KO_s100 + 1.34462 w_k1_MAT_s100
                                 + 1.12381 w_k1_MSFT_s100
                                 + 1.30927 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.09298 w_k1_ADM_s100
                                 + 0.865509 w_k1_AXP_s100
                                 + 0.75482 w_k1_DIS_s100
                                 + 0.813267 w_k1_FDX_s100
                                 + 1.29785 w_k1_IBM_s100
                                 + 0.718831 w_k1_INTC_s100
                                 + 0.900123 w_k1_KO_s100
                                 + 1.54895 w_k1_MAT_s100
                                 + 0.719559 w_k1_MSFT_s100
                                 + 0.998411 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.804655 w_k1_ADM_s101
                                 + 1.00758 w_k1_AXP_s101
                                 + 1.12253 w_k1_DIS_s101
                                 + 1.09061 w_k1_FDX_s101 + 1.5772 w_k1_IBM_s101
                                 + 0.979101 w_k1_INTC_s101
                                 + 1.10484 w_k1_KO_s101 + 1.71834 w_k1_MAT_s101
                                 + 0.917811 w_k1_MSFT_s101
                                 + 0.919348 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.728165 w_k1_ADM_s101
                                 + 0.807157 w_k1_AXP_s101
                                 + 0.694534 w_k1_DIS_s101
                                 + 0.866396 w_k1_FDX_s101
                                 + 0.832341 w_k1_IBM_s101
                                 + 0.967705 w_k1_INTC_s101
                                 + 0.73346 w_k1_KO_s101
                                 + 0.758122 w_k1_MAT_s101
                                 + 0.798707 w_k1_MSFT_s101
                                 + 0.798854 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.717407 w_k1_ADM_s101
                                 + 0.905991 w_k1_AXP_s101 + 0.904 w_k1_DIS_s101
                                 + 1.07977 w_k1_FDX_s101
                                 + 0.738171 w_k1_IBM_s101
                                 + 0.933468 w_k1_INTC_s101
                                 + 0.732729 w_k1_KO_s101
                                 + 0.798984 w_k1_MAT_s101
                                 + 0.972831 w_k1_MSFT_s101
                                 + 0.851795 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.779555 w_k1_ADM_s101
                                 + 0.730418 w_k1_AXP_s101
                                 + 0.828249 w_k1_DIS_s101
                                 + 0.82735 w_k1_FDX_s101
                                 + 1.22642 w_k1_IBM_s101
                                 + 0.915406 w_k1_INTC_s101
                                 + 0.896934 w_k1_KO_s101
                                 + 1.17637 w_k1_MAT_s101
                                 + 0.905966 w_k1_MSFT_s101
                                 + 0.916177 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.869574 w_k1_ADM_s101
                                 + 1.04167 w_k1_AXP_s101
                                 + 0.837589 w_k1_DIS_s101 + 0.892 w_k1_FDX_s101
                                 + 1.01371 w_k1_IBM_s101
                                 + 0.81819 w_k1_INTC_s101
                                 + 0.905296 w_k1_KO_s101
                                 + 1.14998 w_k1_MAT_s101
                                 + 0.877124 w_k1_MSFT_s101
                                 + 0.964869 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.14333 w_k1_ADM_s101
                                 + 0.907377 w_k1_AXP_s101
                                 + 1.19034 w_k1_DIS_s101
                                 + 1.10493 w_k1_FDX_s101
                                 + 1.10269 w_k1_IBM_s101
                                 + 0.879613 w_k1_INTC_s101
                                 + 1.06384 w_k1_KO_s101 + 1.01637 w_k1_MAT_s101
                                 + 1.30571 w_k1_MSFT_s101 + 1.1559 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.11924 w_k1_ADM_s101
                                 + 1.43246 w_k1_AXP_s101
                                 + 1.90962 w_k1_DIS_s101
                                 + 1.47696 w_k1_FDX_s101
                                 + 1.74391 w_k1_IBM_s101
                                 + 1.23922 w_k1_INTC_s101
                                 + 1.45907 w_k1_KO_s101 + 2.08806 w_k1_MAT_s101
                                 + 1.33367 w_k1_MSFT_s101
                                 + 1.22026 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.842392 w_k1_ADM_s101
                                 + 1.39671 w_k1_AXP_s101
                                 + 1.21707 w_k1_DIS_s101
                                 + 1.34828 w_k1_FDX_s101
                                 + 1.11072 w_k1_IBM_s101
                                 + 1.1264 w_k1_INTC_s101 + 1.0027 w_k1_KO_s101
                                 + 1.78668 w_k1_MAT_s101
                                 + 1.12688 w_k1_MSFT_s101
                                 + 1.13812 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.04308 w_k1_ADM_s102
                                 + 0.909453 w_k1_AXP_s102
                                 + 0.89673 w_k1_DIS_s102
                                 + 0.915654 w_k1_FDX_s102
                                 + 0.983735 w_k1_IBM_s102
                                 + 1.41024 w_k1_INTC_s102
                                 + 1.02646 w_k1_KO_s102
                                 + 0.772491 w_k1_MAT_s102
                                 + 1.11675 w_k1_MSFT_s102
                                 + 0.923993 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.30194 w_k1_ADM_s102
                                 + 1.07241 w_k1_AXP_s102
                                 + 1.14522 w_k1_DIS_s102
                                 + 1.04901 w_k1_FDX_s102
                                 + 1.54214 w_k1_IBM_s102
                                 + 1.39673 w_k1_INTC_s102
                                 + 1.39387 w_k1_KO_s102 + 1.50667 w_k1_MAT_s102
                                 + 1.24086 w_k1_MSFT_s102
                                 + 1.03583 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.15323 w_k1_ADM_s102
                                 + 1.44818 w_k1_AXP_s102
                                 + 1.17573 w_k1_DIS_s102
                                 + 1.17237 w_k1_FDX_s102
                                 + 0.744128 w_k1_IBM_s102
                                 + 1.1892 w_k1_INTC_s102 + 1.05873 w_k1_KO_s102
                                 + 0.989558 w_k1_MAT_s102
                                 + 1.11947 w_k1_MSFT_s102
                                 + 1.07305 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.09225 w_k1_ADM_s102
                                 + 1.0546 w_k1_AXP_s102 + 1.28911 w_k1_DIS_s102
                                 + 1.0861 w_k1_FDX_s102 + 1.11098 w_k1_IBM_s102
                                 + 1.13595 w_k1_INTC_s102
                                 + 1.24757 w_k1_KO_s102 + 1.08005 w_k1_MAT_s102
                                 + 1.23161 w_k1_MSFT_s102
                                 + 1.11128 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.983466 w_k1_ADM_s102
                                 + 1.15367 w_k1_AXP_s102
                                 + 1.53505 w_k1_DIS_s102 + 1.2627 w_k1_FDX_s102
                                 + 0.909422 w_k1_IBM_s102
                                 + 1.70288 w_k1_INTC_s102
                                 + 1.17292 w_k1_KO_s102
                                 + 0.858953 w_k1_MAT_s102
                                 + 1.41909 w_k1_MSFT_s102
                                 + 1.02747 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.35983 w_k1_ADM_s102
                                 + 1.75056 w_k1_AXP_s102
                                 + 1.52508 w_k1_DIS_s102
                                 + 1.42897 w_k1_FDX_s102
                                 + 1.11733 w_k1_IBM_s102
                                 + 1.29653 w_k1_INTC_s102
                                 + 1.32879 w_k1_KO_s102 + 1.36051 w_k1_MAT_s102
                                 + 1.20198 w_k1_MSFT_s102
                                 + 1.15033 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.871999 w_k1_ADM_s102
                                 + 0.771223 w_k1_AXP_s102
                                 + 0.706297 w_k1_DIS_s102
                                 + 0.703479 w_k1_FDX_s102
                                 + 1.26831 w_k1_IBM_s102
                                 + 1.1368 w_k1_INTC_s102 + 1.08473 w_k1_KO_s102
                                 + 1.08595 w_k1_MAT_s102
                                 + 0.79528 w_k1_MSFT_s102
                                 + 0.933677 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.871904 w_k1_ADM_s102
                                 + 0.852188 w_k1_AXP_s102
                                 + 0.761058 w_k1_DIS_s102
                                 + 0.919982 w_k1_FDX_s102
                                 + 0.724275 w_k1_IBM_s102
                                 + 1.38956 w_k1_INTC_s102
                                 + 0.844167 w_k1_KO_s102
                                 + 0.607881 w_k1_MAT_s102
                                 + 0.928778 w_k1_MSFT_s102
                                 + 0.774501 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.930396 w_k1_ADM_s103
                                 + 1.45124 w_k1_AXP_s103
                                 + 1.45218 w_k1_DIS_s103
                                 + 1.53213 w_k1_FDX_s103
                                 + 1.01603 w_k1_IBM_s103
                                 + 1.59692 w_k1_INTC_s103
                                 + 1.33914 w_k1_KO_s103 + 1.03659 w_k1_MAT_s103
                                 + 1.43656 w_k1_MSFT_s103
                                 + 1.10677 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.922118 w_k1_ADM_s103
                                 + 0.546095 w_k1_AXP_s103
                                 + 0.668934 w_k1_DIS_s103
                                 + 0.809492 w_k1_FDX_s103
                                 + 0.805073 w_k1_IBM_s103
                                 + 1.01498 w_k1_INTC_s103
                                 + 0.833023 w_k1_KO_s103
                                 + 0.816092 w_k1_MAT_s103
                                 + 0.990809 w_k1_MSFT_s103
                                 + 1.01994 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.802324 w_k1_ADM_s103
                                 + 1.57993 w_k1_AXP_s103
                                 + 0.90226 w_k1_DIS_s103 + 1.0474 w_k1_FDX_s103
                                 + 1.32633 w_k1_IBM_s103
                                 + 1.53119 w_k1_INTC_s103
                                 + 1.17916 w_k1_KO_s103 + 1.79274 w_k1_MAT_s103
                                 + 1.15409 w_k1_MSFT_s103 + 1.1059 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.703929 w_k1_ADM_s103
                                 + 0.700852 w_k1_AXP_s103
                                 + 0.580281 w_k1_DIS_s103
                                 + 0.842229 w_k1_FDX_s103
                                 + 0.596501 w_k1_IBM_s103
                                 + 0.93482 w_k1_INTC_s103
                                 + 0.568919 w_k1_KO_s103
                                 + 0.562281 w_k1_MAT_s103
                                 + 0.971081 w_k1_MSFT_s103
                                 + 0.825204 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.10399 w_k1_ADM_s103
                                 + 1.22569 w_k1_AXP_s103
                                 + 1.45941 w_k1_DIS_s103 + 1.3744 w_k1_FDX_s103
                                 + 1.79517 w_k1_IBM_s103
                                 + 1.47638 w_k1_INTC_s103
                                 + 1.51707 w_k1_KO_s103 + 1.50931 w_k1_MAT_s103
                                 + 1.37916 w_k1_MSFT_s103
                                 + 1.24201 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.02123 w_k1_ADM_s103
                                 + 1.04839 w_k1_AXP_s103
                                 + 0.887468 w_k1_DIS_s103
                                 + 1.08745 w_k1_FDX_s103
                                 + 0.784102 w_k1_IBM_s103
                                 + 0.999699 w_k1_INTC_s103 + 0.811 w_k1_KO_s103
                                 + 0.863585 w_k1_MAT_s103
                                 + 0.970288 w_k1_MSFT_s103
                                 + 1.09159 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.639338 w_k1_ADM_s103
                                 + 0.956851 w_k1_AXP_s103
                                 + 0.833654 w_k1_DIS_s103
                                 + 1.01293 w_k1_FDX_s103
                                 + 1.11444 w_k1_IBM_s103
                                 + 1.36114 w_k1_INTC_s103
                                 + 0.949732 w_k1_KO_s103
                                 + 1.06886 w_k1_MAT_s103
                                 + 1.05942 w_k1_MSFT_s103
                                 + 1.10156 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.14174 w_k1_ADM_s103
                                 + 1.00464 w_k1_AXP_s103
                                 + 1.66827 w_k1_DIS_s103
                                 + 1.48018 w_k1_FDX_s103
                                 + 1.02423 w_k1_IBM_s103
                                 + 1.46879 w_k1_INTC_s103
                                 + 1.23348 w_k1_KO_s103 + 0.86844 w_k1_MAT_s103
                                 + 1.59955 w_k1_MSFT_s103 + 1.27 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.922447 w_k1_ADM_s104
                                 + 0.872162 w_k1_AXP_s104
                                 + 0.878106 w_k1_DIS_s104
                                 + 1.0051 w_k1_FDX_s104
                                 + 0.830385 w_k1_IBM_s104
                                 + 1.07621 w_k1_INTC_s104
                                 + 0.899021 w_k1_KO_s104
                                 + 1.07289 w_k1_MAT_s104
                                 + 1.10331 w_k1_MSFT_s104 + 1.0112 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.686289 w_k1_ADM_s104
                                 + 0.637733 w_k1_AXP_s104
                                 + 0.634065 w_k1_DIS_s104
                                 + 0.73992 w_k1_FDX_s104
                                 + 0.79322 w_k1_IBM_s104
                                 + 0.85431 w_k1_INTC_s104
                                 + 0.762819 w_k1_KO_s104
                                 + 0.617757 w_k1_MAT_s104
                                 + 0.87539 w_k1_MSFT_s104
                                 + 0.801626 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.643019 w_k1_ADM_s104
                                 + 0.699204 w_k1_AXP_s104
                                 + 0.891238 w_k1_DIS_s104
                                 + 0.811227 w_k1_FDX_s104
                                 + 1.12406 w_k1_IBM_s104
                                 + 1.02204 w_k1_INTC_s104
                                 + 1.09117 w_k1_KO_s104 + 1.05503 w_k1_MAT_s104
                                 + 1.03869 w_k1_MSFT_s104
                                 + 0.880074 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.995729 w_k1_ADM_s104
                                 + 1.24415 w_k1_AXP_s104
                                 + 0.823312 w_k1_DIS_s104
                                 + 1.00067 w_k1_FDX_s104
                                 + 1.00757 w_k1_IBM_s104
                                 + 1.14818 w_k1_INTC_s104
                                 + 1.05946 w_k1_KO_s104 + 0.93633 w_k1_MAT_s104
                                 + 1.03107 w_k1_MSFT_s104
                                 + 1.02733 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.90936 w_k1_ADM_s104
                                 + 1.27127 w_k1_AXP_s104 + 0.9352 w_k1_DIS_s104
                                 + 1.02379 w_k1_FDX_s104
                                 + 1.13237 w_k1_IBM_s104
                                 + 0.881346 w_k1_INTC_s104
                                 + 1.0556 w_k1_KO_s104 + 1.547 w_k1_MAT_s104
                                 + 0.943579 w_k1_MSFT_s104
                                 + 1.03156 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.13468 w_k1_ADM_s104
                                 + 1.14489 w_k1_AXP_s104
                                 + 1.37174 w_k1_DIS_s104
                                 + 1.30556 w_k1_FDX_s104
                                 + 1.05705 w_k1_IBM_s104
                                 + 1.01488 w_k1_INTC_s104
                                 + 1.17411 w_k1_KO_s104 + 1.09857 w_k1_MAT_s104
                                 + 1.1642 w_k1_MSFT_s104 + 1.18207 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.15859 w_k1_ADM_s104
                                 + 1.34662 w_k1_AXP_s104
                                 + 1.15881 w_k1_DIS_s104
                                 + 1.11811 w_k1_FDX_s104
                                 + 1.46791 w_k1_IBM_s104
                                 + 0.954095 w_k1_INTC_s104
                                 + 1.13835 w_k1_KO_s104 + 1.47858 w_k1_MAT_s104
                                 + 0.904166 w_k1_MSFT_s104
                                 + 0.947118 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.809868 w_k1_ADM_s104
                                 + 0.829887 w_k1_AXP_s104
                                 + 0.662201 w_k1_DIS_s104
                                 + 0.848227 w_k1_FDX_s104
                                 + 0.66906 w_k1_IBM_s104
                                 + 1.00529 w_k1_INTC_s104
                                 + 0.729939 w_k1_KO_s104
                                 + 0.63639 w_k1_MAT_s104
                                 + 0.873897 w_k1_MSFT_s104
                                 + 0.846438 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.840723 w_k1_ADM_s105
                                 + 0.627744 w_k1_AXP_s105
                                 + 0.791328 w_k1_DIS_s105
                                 + 0.772553 w_k1_FDX_s105
                                 + 0.860855 w_k1_IBM_s105
                                 + 0.687336 w_k1_INTC_s105
                                 + 0.858252 w_k1_KO_s105
                                 + 0.70306 w_k1_MAT_s105
                                 + 0.825152 w_k1_MSFT_s105
                                 + 0.787198 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.933201 w_k1_ADM_s105
                                 + 0.892105 w_k1_AXP_s105
                                 + 0.747671 w_k1_DIS_s105
                                 + 0.956686 w_k1_FDX_s105
                                 + 0.734098 w_k1_IBM_s105
                                 + 0.917344 w_k1_INTC_s105
                                 + 0.78812 w_k1_KO_s105
                                 + 0.658469 w_k1_MAT_s105
                                 + 0.885177 w_k1_MSFT_s105
                                 + 0.938654 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.795495 w_k1_ADM_s105
                                 + 0.755712 w_k1_AXP_s105
                                 + 0.909349 w_k1_DIS_s105
                                 + 0.807321 w_k1_FDX_s105
                                 + 1.17722 w_k1_IBM_s105
                                 + 1.00311 w_k1_INTC_s105 + 1.1762 w_k1_KO_s105
                                 + 1.18621 w_k1_MAT_s105
                                 + 0.896649 w_k1_MSFT_s105
                                 + 0.838873 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.794505 w_k1_ADM_s105
                                 + 0.628051 w_k1_AXP_s105
                                 + 0.675915 w_k1_DIS_s105
                                 + 0.820009 w_k1_FDX_s105
                                 + 0.619042 w_k1_IBM_s105
                                 + 0.811171 w_k1_INTC_s105
                                 + 0.644072 w_k1_KO_s105
                                 + 0.521986 w_k1_MAT_s105
                                 + 0.716192 w_k1_MSFT_s105
                                 + 0.705052 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.64971 w_k1_ADM_s105
                                 + 0.590764 w_k1_AXP_s105
                                 + 0.49613 w_k1_DIS_s105
                                 + 0.698892 w_k1_FDX_s105
                                 + 0.768176 w_k1_IBM_s105
                                 + 0.687343 w_k1_INTC_s105
                                 + 0.665794 w_k1_KO_s105
                                 + 0.587247 w_k1_MAT_s105
                                 + 0.587001 w_k1_MSFT_s105
                                 + 0.717822 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.872799 w_k1_ADM_s105
                                 + 0.921084 w_k1_AXP_s105
                                 + 1.10926 w_k1_DIS_s105
                                 + 0.998318 w_k1_FDX_s105
                                 + 0.976374 w_k1_IBM_s105
                                 + 1.1185 w_k1_INTC_s105 + 1.09724 w_k1_KO_s105
                                 + 0.843481 w_k1_MAT_s105
                                 + 0.951771 w_k1_MSFT_s105
                                 + 0.980133 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.962462 w_k1_ADM_s105
                                 + 1.25162 w_k1_AXP_s105
                                 + 1.32575 w_k1_DIS_s105
                                 + 1.12708 w_k1_FDX_s105
                                 + 1.40719 w_k1_IBM_s105
                                 + 1.37964 w_k1_INTC_s105
                                 + 1.36404 w_k1_KO_s105 + 1.38138 w_k1_MAT_s105
                                 + 0.968502 w_k1_MSFT_s105
                                 + 1.03092 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.41138 w_k1_ADM_s105
                                 + 1.33094 w_k1_AXP_s105
                                 + 1.23706 w_k1_DIS_s105
                                 + 1.30812 w_k1_FDX_s105
                                 + 0.935415 w_k1_IBM_s105
                                 + 0.979151 w_k1_INTC_s105
                                 + 1.09291 w_k1_KO_s105 + 1.05776 w_k1_MAT_s105
                                 + 1.24039 w_k1_MSFT_s105
                                 + 1.09819 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.78754 w_k1_ADM_s106
                                 + 0.63961 w_k1_AXP_s106
                                 + 0.686293 w_k1_DIS_s106
                                 + 0.789172 w_k1_FDX_s106
                                 + 0.659147 w_k1_IBM_s106
                                 + 0.81838 w_k1_INTC_s106
                                 + 0.650154 w_k1_KO_s106
                                 + 0.514957 w_k1_MAT_s106
                                 + 0.914646 w_k1_MSFT_s106
                                 + 0.861537 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.78606 w_k1_ADM_s106
                                 + 1.7205 w_k1_AXP_s106 + 1.53435 w_k1_DIS_s106
                                 + 1.59317 w_k1_FDX_s106
                                 + 1.06537 w_k1_IBM_s106
                                 + 1.03042 w_k1_INTC_s106
                                 + 1.27404 w_k1_KO_s106 + 1.04054 w_k1_MAT_s106
                                 + 1.16671 w_k1_MSFT_s106
                                 + 1.34345 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.23415 w_k1_ADM_s106
                                 + 1.30898 w_k1_AXP_s106
                                 + 1.64735 w_k1_DIS_s106 + 1.3106 w_k1_FDX_s106
                                 + 1.12759 w_k1_IBM_s106
                                 + 1.2921 w_k1_INTC_s106 + 1.20423 w_k1_KO_s106
                                 + 0.802134 w_k1_MAT_s106
                                 + 1.2586 w_k1_MSFT_s106 + 1.14372 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.08535 w_k1_ADM_s106
                                 + 1.05463 w_k1_AXP_s106
                                 + 0.838433 w_k1_DIS_s106
                                 + 0.997144 w_k1_FDX_s106
                                 + 1.10462 w_k1_IBM_s106
                                 + 0.935682 w_k1_INTC_s106
                                 + 0.810244 w_k1_KO_s106
                                 + 0.896827 w_k1_MAT_s106
                                 + 0.845725 w_k1_MSFT_s106
                                 + 1.07969 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.12083 w_k1_ADM_s106
                                 + 1.21088 w_k1_AXP_s106
                                 + 1.16506 w_k1_DIS_s106
                                 + 1.03351 w_k1_FDX_s106
                                 + 1.48459 w_k1_IBM_s106
                                 + 1.20837 w_k1_INTC_s106 + 1.2827 w_k1_KO_s106
                                 + 1.17657 w_k1_MAT_s106
                                 + 1.02914 w_k1_MSFT_s106
                                 + 1.30611 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.13776 w_k1_ADM_s106
                                 + 0.897327 w_k1_AXP_s106
                                 + 0.899979 w_k1_DIS_s106
                                 + 0.933154 w_k1_FDX_s106
                                 + 0.97458 w_k1_IBM_s106
                                 + 1.01514 w_k1_INTC_s106
                                 + 0.965575 w_k1_KO_s106
                                 + 0.830279 w_k1_MAT_s106
                                 + 0.920989 w_k1_MSFT_s106
                                 + 0.933316 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.7906 w_k1_ADM_s106
                                 + 0.603312 w_k1_AXP_s106
                                 + 0.514303 w_k1_DIS_s106
                                 + 0.662565 w_k1_FDX_s106
                                 + 0.819907 w_k1_IBM_s106
                                 + 0.722163 w_k1_INTC_s106
                                 + 0.6718 w_k1_KO_s106 + 0.352449 w_k1_MAT_s106
                                 + 0.67326 w_k1_MSFT_s106
                                 + 0.806922 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.18288 w_k1_ADM_s106
                                 + 0.890879 w_k1_AXP_s106
                                 + 1.06017 w_k1_DIS_s106
                                 + 1.13093 w_k1_FDX_s106
                                 + 0.71994 w_k1_IBM_s106
                                 + 1.22794 w_k1_INTC_s106
                                 + 0.967296 w_k1_KO_s106
                                 + 0.654179 w_k1_MAT_s106
                                 + 1.16032 w_k1_MSFT_s106
                                 + 1.06108 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.888938 w_k1_ADM_s107
                                 + 0.803595 w_k1_AXP_s107
                                 + 0.833861 w_k1_DIS_s107
                                 + 1.02127 w_k1_FDX_s107
                                 + 0.746462 w_k1_IBM_s107
                                 + 1.11243 w_k1_INTC_s107 + 1.1007 w_k1_KO_s107
                                 + 0.80574 w_k1_MAT_s107
                                 + 0.962069 w_k1_MSFT_s107
                                 + 0.987392 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.689758 w_k1_ADM_s107
                                 + 0.822328 w_k1_AXP_s107
                                 + 0.607439 w_k1_DIS_s107
                                 + 0.769303 w_k1_FDX_s107
                                 + 0.710537 w_k1_IBM_s107
                                 + 0.952271 w_k1_INTC_s107
                                 + 0.75775 w_k1_KO_s107
                                 + 0.533515 w_k1_MAT_s107
                                 + 0.761196 w_k1_MSFT_s107
                                 + 0.805465 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.791131 w_k1_ADM_s107
                                 + 0.603666 w_k1_AXP_s107
                                 + 0.810678 w_k1_DIS_s107
                                 + 0.792569 w_k1_FDX_s107
                                 + 0.857483 w_k1_IBM_s107
                                 + 1.10706 w_k1_INTC_s107
                                 + 0.992259 w_k1_KO_s107
                                 + 0.918852 w_k1_MAT_s107
                                 + 0.896112 w_k1_MSFT_s107
                                 + 0.888465 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.828148 w_k1_ADM_s107
                                 + 1.02981 w_k1_AXP_s107
                                 + 1.26126 w_k1_DIS_s107
                                 + 1.03679 w_k1_FDX_s107
                                 + 1.31025 w_k1_IBM_s107
                                 + 1.13605 w_k1_INTC_s107
                                 + 1.26437 w_k1_KO_s107 + 1.33289 w_k1_MAT_s107
                                 + 1.1614 w_k1_MSFT_s107 + 1.06512 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.805817 w_k1_ADM_s107
                                 + 1.01566 w_k1_AXP_s107
                                 + 0.892378 w_k1_DIS_s107
                                 + 0.944114 w_k1_FDX_s107
                                 + 1.21622 w_k1_IBM_s107
                                 + 0.939547 w_k1_INTC_s107
                                 + 1.2281 w_k1_KO_s107 + 1.8648 w_k1_MAT_s107
                                 + 0.906037 w_k1_MSFT_s107
                                 + 1.04463 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.00317 w_k1_ADM_s107
                                 + 0.952931 w_k1_AXP_s107
                                 + 0.845098 w_k1_DIS_s107
                                 + 1.08477 w_k1_FDX_s107
                                 + 0.973785 w_k1_IBM_s107
                                 + 1.2385 w_k1_INTC_s107 + 1.01132 w_k1_KO_s107
                                 + 1.14482 w_k1_MAT_s107
                                 + 1.04372 w_k1_MSFT_s107
                                 + 0.972306 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.746373 w_k1_ADM_s107
                                 + 1.37524 w_k1_AXP_s107
                                 + 1.65615 w_k1_DIS_s107
                                 + 1.45405 w_k1_FDX_s107
                                 + 1.01781 w_k1_IBM_s107
                                 + 1.57937 w_k1_INTC_s107
                                 + 1.31317 w_k1_KO_s107 + 1.60634 w_k1_MAT_s107
                                 + 1.30329 w_k1_MSFT_s107
                                 + 1.13283 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.0656 w_k1_ADM_s107
                                 + 0.955171 w_k1_AXP_s107
                                 + 1.45921 w_k1_DIS_s107
                                 + 1.20354 w_k1_FDX_s107
                                 + 0.934112 w_k1_IBM_s107
                                 + 1.06784 w_k1_INTC_s107
                                 + 1.31816 w_k1_KO_s107 + 1.06442 w_k1_MAT_s107
                                 + 1.15666 w_k1_MSFT_s107
                                 + 1.13236 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.2942 w_k1_ADM_s110
                                 + 0.9436 w_k1_AXP_s110 + 1.0009 w_k1_DIS_s110
                                 + 0.979538 w_k1_FDX_s110
                                 + 1.02326 w_k1_IBM_s110
                                 + 0.952326 w_k1_INTC_s110
                                 + 0.970859 w_k1_KO_s110
                                 + 1.20668 w_k1_MAT_s110
                                 + 1.08845 w_k1_MSFT_s110
                                 + 1.07493 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.98597 w_k1_ADM_s110
                                 + 1.16934 w_k1_AXP_s110
                                 + 1.35523 w_k1_DIS_s110
                                 + 1.19125 w_k1_FDX_s110
                                 + 0.978506 w_k1_IBM_s110
                                 + 0.997467 w_k1_INTC_s110
                                 + 0.885787 w_k1_KO_s110
                                 + 1.07659 w_k1_MAT_s110
                                 + 1.12547 w_k1_MSFT_s110
                                 + 1.04932 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.313 w_k1_ADM_s110
                                 + 1.43211 w_k1_AXP_s110
                                 + 1.45005 w_k1_DIS_s110
                                 + 1.10206 w_k1_FDX_s110
                                 + 1.53451 w_k1_IBM_s110
                                 + 1.23107 w_k1_INTC_s110
                                 + 1.21934 w_k1_KO_s110 + 1.8677 w_k1_MAT_s110
                                 + 1.17408 w_k1_MSFT_s110
                                 + 1.13008 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.856398 w_k1_ADM_s110
                                 + 0.744787 w_k1_AXP_s110
                                 + 0.541617 w_k1_DIS_s110
                                 + 0.751395 w_k1_FDX_s110
                                 + 0.752531 w_k1_IBM_s110
                                 + 0.846375 w_k1_INTC_s110
                                 + 0.610906 w_k1_KO_s110
                                 + 0.780231 w_k1_MAT_s110
                                 + 0.753377 w_k1_MSFT_s110
                                 + 0.815291 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.954234 w_k1_ADM_s110
                                 + 0.804345 w_k1_AXP_s110
                                 + 0.708219 w_k1_DIS_s110
                                 + 0.848877 w_k1_FDX_s110
                                 + 1.04333 w_k1_IBM_s110
                                 + 0.911446 w_k1_INTC_s110
                                 + 0.789975 w_k1_KO_s110
                                 + 0.971803 w_k1_MAT_s110
                                 + 0.797721 w_k1_MSFT_s110
                                 + 0.838328 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.20097 w_k1_ADM_s110
                                 + 1.0849 w_k1_AXP_s110
                                 + 0.808894 w_k1_DIS_s110
                                 + 0.853427 w_k1_FDX_s110
                                 + 1.25073 w_k1_IBM_s110
                                 + 0.93539 w_k1_INTC_s110
                                 + 0.904337 w_k1_KO_s110
                                 + 1.42287 w_k1_MAT_s110
                                 + 0.891856 w_k1_MSFT_s110
                                 + 1.08472 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.06988 w_k1_ADM_s110
                                 + 1.3115 w_k1_AXP_s110
                                 + 0.748622 w_k1_DIS_s110
                                 + 1.02555 w_k1_FDX_s110
                                 + 0.868673 w_k1_IBM_s110
                                 + 0.963379 w_k1_INTC_s110
                                 + 0.713834 w_k1_KO_s110
                                 + 1.10823 w_k1_MAT_s110
                                 + 0.887134 w_k1_MSFT_s110
                                 + 0.874974 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.0654 w_k1_ADM_s110
                                 + 0.834333 w_k1_AXP_s110
                                 + 0.923183 w_k1_DIS_s110
                                 + 0.958791 w_k1_FDX_s110
                                 + 0.770554 w_k1_IBM_s110
                                 + 0.902407 w_k1_INTC_s110
                                 + 0.832678 w_k1_KO_s110 + 0.8741 w_k1_MAT_s110
                                 + 0.930934 w_k1_MSFT_s110
                                 + 0.763572 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.27616 w_k1_ADM_s111
                                 + 0.958867 w_k1_AXP_s111
                                 + 1.16995 w_k1_DIS_s111
                                 + 1.20607 w_k1_FDX_s111
                                 + 0.787737 w_k1_IBM_s111
                                 + 1.28306 w_k1_INTC_s111
                                 + 0.879574 w_k1_KO_s111
                                 + 1.04322 w_k1_MAT_s111
                                 + 1.38588 w_k1_MSFT_s111
                                 + 1.10914 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.03057 w_k1_ADM_s111
                                 + 0.949631 w_k1_AXP_s111
                                 + 1.19284 w_k1_DIS_s111
                                 + 0.937563 w_k1_FDX_s111
                                 + 1.21412 w_k1_IBM_s111
                                 + 1.06202 w_k1_INTC_s111
                                 + 1.14615 w_k1_KO_s111 + 1.22252 w_k1_MAT_s111
                                 + 1.12191 w_k1_MSFT_s111
                                 + 1.03211 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.960019 w_k1_ADM_s111
                                 + 0.925119 w_k1_AXP_s111
                                 + 0.742533 w_k1_DIS_s111
                                 + 0.906175 w_k1_FDX_s111
                                 + 1.01758 w_k1_IBM_s111
                                 + 1.04944 w_k1_INTC_s111
                                 + 0.973575 w_k1_KO_s111
                                 + 1.26506 w_k1_MAT_s111
                                 + 0.917789 w_k1_MSFT_s111
                                 + 0.969079 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.47134 w_k1_ADM_s111
                                 + 1.33484 w_k1_AXP_s111
                                 + 0.952053 w_k1_DIS_s111
                                 + 1.17011 w_k1_FDX_s111
                                 + 0.908371 w_k1_IBM_s111
                                 + 1.05501 w_k1_INTC_s111
                                 + 0.975706 w_k1_KO_s111
                                 + 1.04491 w_k1_MAT_s111
                                 + 0.976973 w_k1_MSFT_s111
                                 + 0.954728 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.64121 w_k1_ADM_s111
                                 + 1.34812 w_k1_AXP_s111
                                 + 1.55925 w_k1_DIS_s111 + 1.2688 w_k1_FDX_s111
                                 + 1.41562 w_k1_IBM_s111
                                 + 1.40126 w_k1_INTC_s111 + 1.3234 w_k1_KO_s111
                                 + 2.00033 w_k1_MAT_s111
                                 + 1.28382 w_k1_MSFT_s111
                                 + 1.24507 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.940097 w_k1_ADM_s111
                                 + 1.21877 w_k1_AXP_s111
                                 + 0.896123 w_k1_DIS_s111
                                 + 0.879732 w_k1_FDX_s111
                                 + 1.52484 w_k1_IBM_s111
                                 + 1.27013 w_k1_INTC_s111
                                 + 1.18457 w_k1_KO_s111 + 1.89168 w_k1_MAT_s111
                                 + 0.97702 w_k1_MSFT_s111
                                 + 1.09836 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.921656 w_k1_ADM_s111
                                 + 0.614584 w_k1_AXP_s111
                                 + 0.708395 w_k1_DIS_s111
                                 + 0.764565 w_k1_FDX_s111
                                 + 0.894195 w_k1_IBM_s111
                                 + 0.996404 w_k1_INTC_s111
                                 + 0.858094 w_k1_KO_s111
                                 + 0.671511 w_k1_MAT_s111
                                 + 0.829343 w_k1_MSFT_s111
                                 + 0.919275 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.0305 w_k1_ADM_s111
                                 + 0.843622 w_k1_AXP_s111
                                 + 0.810447 w_k1_DIS_s111
                                 + 0.897328 w_k1_FDX_s111
                                 + 0.675367 w_k1_IBM_s111
                                 + 1.0613 w_k1_INTC_s111
                                 + 0.711687 w_k1_KO_s111
                                 + 0.719587 w_k1_MAT_s111
                                 + 1.00462 w_k1_MSFT_s111
                                 + 0.757522 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.15067 w_k1_ADM_s112
                                 + 0.824719 w_k1_AXP_s112
                                 + 1.06822 w_k1_DIS_s112
                                 + 1.04277 w_k1_FDX_s112
                                 + 0.651072 w_k1_IBM_s112
                                 + 0.972208 w_k1_INTC_s112
                                 + 0.822583 w_k1_KO_s112
                                 + 0.69608 w_k1_MAT_s112
                                 + 1.06804 w_k1_MSFT_s112
                                 + 1.01014 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.621217 w_k1_ADM_s112
                                 + 0.835173 w_k1_AXP_s112
                                 + 0.606765 w_k1_DIS_s112
                                 + 0.664835 w_k1_FDX_s112
                                 + 0.891442 w_k1_IBM_s112
                                 + 0.653402 w_k1_INTC_s112
                                 + 0.817058 w_k1_KO_s112
                                 + 1.22153 w_k1_MAT_s112
                                 + 0.700942 w_k1_MSFT_s112
                                 + 0.76266 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.02023 w_k1_ADM_s112
                                 + 0.875947 w_k1_AXP_s112
                                 + 0.890408 w_k1_DIS_s112
                                 + 0.868381 w_k1_FDX_s112
                                 + 0.99677 w_k1_IBM_s112
                                 + 0.814365 w_k1_INTC_s112
                                 + 0.892706 w_k1_KO_s112
                                 + 1.19665 w_k1_MAT_s112
                                 + 0.872077 w_k1_MSFT_s112
                                 + 0.86575 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.810036 w_k1_ADM_s112
                                 + 1.00923 w_k1_AXP_s112
                                 + 0.967489 w_k1_DIS_s112
                                 + 1.03193 w_k1_FDX_s112
                                 + 0.965662 w_k1_IBM_s112
                                 + 0.895867 w_k1_INTC_s112
                                 + 0.875567 w_k1_KO_s112
                                 + 1.03074 w_k1_MAT_s112
                                 + 1.03676 w_k1_MSFT_s112
                                 + 0.940768 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.03812 w_k1_ADM_s112
                                 + 1.22429 w_k1_AXP_s112
                                 + 1.49352 w_k1_DIS_s112
                                 + 1.28923 w_k1_FDX_s112 + 1.0509 w_k1_IBM_s112
                                 + 1.20702 w_k1_INTC_s112 + 1.1928 w_k1_KO_s112
                                 + 1.40954 w_k1_MAT_s112
                                 + 1.23961 w_k1_MSFT_s112
                                 + 1.07881 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.906885 w_k1_ADM_s112
                                 + 0.695297 w_k1_AXP_s112
                                 + 0.897677 w_k1_DIS_s112
                                 + 0.850596 w_k1_FDX_s112
                                 + 0.802884 w_k1_IBM_s112
                                 + 0.813123 w_k1_INTC_s112
                                 + 0.858837 w_k1_KO_s112
                                 + 0.642463 w_k1_MAT_s112
                                 + 0.981971 w_k1_MSFT_s112
                                 + 0.86584 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.057 w_k1_ADM_s112
                                 + 1.22519 w_k1_AXP_s112
                                 + 1.35177 w_k1_DIS_s112
                                 + 1.05287 w_k1_FDX_s112
                                 + 1.28268 w_k1_IBM_s112
                                 + 0.946849 w_k1_INTC_s112
                                 + 1.07609 w_k1_KO_s112 + 1.26178 w_k1_MAT_s112
                                 + 0.962886 w_k1_MSFT_s112
                                 + 0.997928 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.753562 w_k1_ADM_s112
                                 + 1.34498 w_k1_AXP_s112
                                 + 1.09219 w_k1_DIS_s112
                                 + 0.972836 w_k1_FDX_s112
                                 + 1.69126 w_k1_IBM_s112
                                 + 0.687226 w_k1_INTC_s112
                                 + 1.05492 w_k1_KO_s112 + 2.80881 w_k1_MAT_s112
                                 + 0.979533 w_k1_MSFT_s112
                                 + 0.906306 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.647712 w_k1_ADM_s113
                                 + 0.649319 w_k1_AXP_s113
                                 + 0.616362 w_k1_DIS_s113
                                 + 0.737805 w_k1_FDX_s113
                                 + 0.933889 w_k1_IBM_s113
                                 + 0.703428 w_k1_INTC_s113
                                 + 0.725946 w_k1_KO_s113
                                 + 0.927635 w_k1_MAT_s113
                                 + 0.687738 w_k1_MSFT_s113
                                 + 0.78395 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.990142 w_k1_ADM_s113
                                 + 1.82421 w_k1_AXP_s113 + 1.3969 w_k1_DIS_s113
                                 + 1.25221 w_k1_FDX_s113
                                 + 1.44901 w_k1_IBM_s113
                                 + 1.22011 w_k1_INTC_s113
                                 + 1.30393 w_k1_KO_s113 + 2.44083 w_k1_MAT_s113
                                 + 0.996401 w_k1_MSFT_s113
                                 + 1.04621 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.06876 w_k1_ADM_s113
                                 + 0.974258 w_k1_AXP_s113
                                 + 1.0635 w_k1_DIS_s113 + 1.11561 w_k1_FDX_s113
                                 + 0.857804 w_k1_IBM_s113
                                 + 1.07255 w_k1_INTC_s113
                                 + 1.01081 w_k1_KO_s113 + 1.06681 w_k1_MAT_s113
                                 + 0.955014 w_k1_MSFT_s113
                                 + 0.914261 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.921769 w_k1_ADM_s113
                                 + 0.809406 w_k1_AXP_s113
                                 + 0.847891 w_k1_DIS_s113
                                 + 0.977543 w_k1_FDX_s113
                                 + 0.674805 w_k1_IBM_s113
                                 + 0.991093 w_k1_INTC_s113
                                 + 0.801359 w_k1_KO_s113
                                 + 0.660826 w_k1_MAT_s113
                                 + 0.892442 w_k1_MSFT_s113
                                 + 0.800717 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.785024 w_k1_ADM_s113
                                 + 0.740445 w_k1_AXP_s113
                                 + 1.15976 w_k1_DIS_s113
                                 + 0.899347 w_k1_FDX_s113
                                 + 1.29907 w_k1_IBM_s113
                                 + 0.915781 w_k1_INTC_s113
                                 + 1.3048 w_k1_KO_s113 + 1.10775 w_k1_MAT_s113
                                 + 0.967911 w_k1_MSFT_s113
                                 + 1.03606 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.10312 w_k1_ADM_s113
                                 + 1.29215 w_k1_AXP_s113
                                 + 1.02971 w_k1_DIS_s113
                                 + 1.03108 w_k1_FDX_s113
                                 + 1.33314 w_k1_IBM_s113
                                 + 1.00286 w_k1_INTC_s113
                                 + 1.13825 w_k1_KO_s113 + 1.55924 w_k1_MAT_s113
                                 + 0.80267 w_k1_MSFT_s113
                                 + 1.00629 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.684818 w_k1_ADM_s113
                                 + 0.756687 w_k1_AXP_s113
                                 + 0.852704 w_k1_DIS_s113
                                 + 0.844269 w_k1_FDX_s113
                                 + 0.855506 w_k1_IBM_s113
                                 + 0.865849 w_k1_INTC_s113
                                 + 0.842694 w_k1_KO_s113
                                 + 0.928699 w_k1_MAT_s113
                                 + 0.747353 w_k1_MSFT_s113
                                 + 0.783603 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.15826 w_k1_ADM_s113
                                 + 1.34486 w_k1_AXP_s113 + 1.4754 w_k1_DIS_s113
                                 + 1.34501 w_k1_FDX_s113
                                 + 1.09018 w_k1_IBM_s113
                                 + 1.10751 w_k1_INTC_s113
                                 + 1.09596 w_k1_KO_s113 + 1.15021 w_k1_MAT_s113
                                 + 1.06635 w_k1_MSFT_s113
                                 + 1.09138 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.678893 w_k1_ADM_s114
                                 + 0.939591 w_k1_AXP_s114
                                 + 0.931131 w_k1_DIS_s114
                                 + 0.767557 w_k1_FDX_s114
                                 + 1.31617 w_k1_IBM_s114
                                 + 0.823766 w_k1_INTC_s114
                                 + 0.819754 w_k1_KO_s114
                                 + 1.18296 w_k1_MAT_s114
                                 + 0.785692 w_k1_MSFT_s114
                                 + 0.874015 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.883205 w_k1_ADM_s114
                                 + 0.808589 w_k1_AXP_s114
                                 + 0.915744 w_k1_DIS_s114
                                 + 0.908797 w_k1_FDX_s114
                                 + 0.957319 w_k1_IBM_s114
                                 + 0.884753 w_k1_INTC_s114
                                 + 0.838417 w_k1_KO_s114
                                 + 0.894447 w_k1_MAT_s114
                                 + 1.07665 w_k1_MSFT_s114
                                 + 0.936333 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.04905 w_k1_ADM_s114
                                 + 1.221 w_k1_AXP_s114 + 1.00675 w_k1_DIS_s114
                                 + 0.983672 w_k1_FDX_s114
                                 + 1.08964 w_k1_IBM_s114
                                 + 0.794102 w_k1_INTC_s114
                                 + 0.867489 w_k1_KO_s114
                                 + 1.17411 w_k1_MAT_s114
                                 + 0.815807 w_k1_MSFT_s114
                                 + 0.90699 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.05925 w_k1_ADM_s114
                                 + 1.1757 w_k1_AXP_s114 + 1.32958 w_k1_DIS_s114
                                 + 1.16353 w_k1_FDX_s114
                                 + 0.90673 w_k1_IBM_s114
                                 + 1.10844 w_k1_INTC_s114
                                 + 0.953618 w_k1_KO_s114
                                 + 1.38897 w_k1_MAT_s114
                                 + 1.14018 w_k1_MSFT_s114
                                 + 1.11918 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.15387 w_k1_ADM_s114
                                 + 0.946409 w_k1_AXP_s114
                                 + 0.833413 w_k1_DIS_s114
                                 + 0.928622 w_k1_FDX_s114
                                 + 0.761683 w_k1_IBM_s114
                                 + 0.737224 w_k1_INTC_s114
                                 + 0.783884 w_k1_KO_s114
                                 + 0.84064 w_k1_MAT_s114
                                 + 0.856533 w_k1_MSFT_s114
                                 + 0.91568 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.36197 w_k1_ADM_s114
                                 + 1.29691 w_k1_AXP_s114
                                 + 1.30117 w_k1_DIS_s114
                                 + 1.13493 w_k1_FDX_s114
                                 + 1.34357 w_k1_IBM_s114
                                 + 1.0391 w_k1_INTC_s114 + 1.14806 w_k1_KO_s114
                                 + 1.80814 w_k1_MAT_s114
                                 + 1.03079 w_k1_MSFT_s114
                                 + 1.01731 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.90955 w_k1_ADM_s114
                                 + 0.987537 w_k1_AXP_s114
                                 + 1.19945 w_k1_DIS_s114
                                 + 0.977996 w_k1_FDX_s114
                                 + 1.21418 w_k1_IBM_s114
                                 + 0.997838 w_k1_INTC_s114
                                 + 1.09839 w_k1_KO_s114 + 1.74776 w_k1_MAT_s114
                                 + 1.05521 w_k1_MSFT_s114
                                 + 1.08484 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.813275 w_k1_ADM_s114
                                 + 0.771844 w_k1_AXP_s114
                                 + 0.611627 w_k1_DIS_s114
                                 + 0.687188 w_k1_FDX_s114
                                 + 1.02268 w_k1_IBM_s114
                                 + 0.61574 w_k1_INTC_s114
                                 + 0.712542 w_k1_KO_s114
                                 + 1.43702 w_k1_MAT_s114
                                 + 0.643789 w_k1_MSFT_s114
                                 + 0.747121 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.912414 w_k1_ADM_s115
                                 + 0.706803 w_k1_AXP_s115
                                 + 0.708021 w_k1_DIS_s115
                                 + 0.833205 w_k1_FDX_s115
                                 + 0.674577 w_k1_IBM_s115
                                 + 0.844509 w_k1_INTC_s115
                                 + 0.756671 w_k1_KO_s115
                                 + 0.789617 w_k1_MAT_s115
                                 + 0.784024 w_k1_MSFT_s115
                                 + 0.802607 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.908321 w_k1_ADM_s115
                                 + 0.849936 w_k1_AXP_s115
                                 + 0.650725 w_k1_DIS_s115
                                 + 0.865851 w_k1_FDX_s115
                                 + 0.908173 w_k1_IBM_s115
                                 + 0.71148 w_k1_INTC_s115
                                 + 0.785841 w_k1_KO_s115
                                 + 1.76436 w_k1_MAT_s115
                                 + 0.865781 w_k1_MSFT_s115
                                 + 0.845908 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.16823 w_k1_ADM_s115
                                 + 0.957124 w_k1_AXP_s115
                                 + 1.60664 w_k1_DIS_s115 + 1.2127 w_k1_FDX_s115
                                 + 1.1068 w_k1_IBM_s115
                                 + 0.999714 w_k1_INTC_s115
                                 + 1.4479 w_k1_KO_s115 + 2.18028 w_k1_MAT_s115
                                 + 1.21983 w_k1_MSFT_s115
                                 + 1.06956 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.67314 w_k1_ADM_s115
                                 + 1.57597 w_k1_AXP_s115
                                 + 1.71545 w_k1_DIS_s115
                                 + 1.53147 w_k1_FDX_s115
                                 + 1.07639 w_k1_IBM_s115
                                 + 1.05153 w_k1_INTC_s115
                                 + 1.26746 w_k1_KO_s115 + 1.48144 w_k1_MAT_s115
                                 + 1.23742 w_k1_MSFT_s115
                                 + 1.17739 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.08539 w_k1_ADM_s115
                                 + 0.786597 w_k1_AXP_s115
                                 + 1.18542 w_k1_DIS_s115
                                 + 1.02075 w_k1_FDX_s115
                                 + 0.80414 w_k1_IBM_s115
                                 + 0.819541 w_k1_INTC_s115
                                 + 1.16384 w_k1_KO_s115 + 1.42654 w_k1_MAT_s115
                                 + 1.12069 w_k1_MSFT_s115
                                 + 1.09334 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.742725 w_k1_ADM_s115
                                 + 1.05633 w_k1_AXP_s115
                                 + 1.13859 w_k1_DIS_s115
                                 + 0.97121 w_k1_FDX_s115
                                 + 1.49225 w_k1_IBM_s115
                                 + 0.920557 w_k1_INTC_s115
                                 + 1.16245 w_k1_KO_s115 + 2.72253 w_k1_MAT_s115
                                 + 1.01052 w_k1_MSFT_s115
                                 + 1.00232 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.24071 w_k1_ADM_s115
                                 + 1.11982 w_k1_AXP_s115
                                 + 0.811396 w_k1_DIS_s115
                                 + 0.936765 w_k1_FDX_s115
                                 + 1.04022 w_k1_IBM_s115
                                 + 0.796809 w_k1_INTC_s115
                                 + 0.934934 w_k1_KO_s115
                                 + 1.86297 w_k1_MAT_s115
                                 + 0.811517 w_k1_MSFT_s115
                                 + 0.981334 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.691665 w_k1_ADM_s115
                                 + 0.618112 w_k1_AXP_s115
                                 + 0.724131 w_k1_DIS_s115
                                 + 0.700054 w_k1_FDX_s115
                                 + 0.856035 w_k1_IBM_s115
                                 + 0.755906 w_k1_INTC_s115
                                 + 1.03078 w_k1_KO_s115 + 1.41904 w_k1_MAT_s115
                                 + 0.771413 w_k1_MSFT_s115
                                 + 0.96648 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.3452 w_k1_ADM_s116
                                 + 1.45075 w_k1_AXP_s116
                                 + 1.34775 w_k1_DIS_s116 + 1.2543 w_k1_FDX_s116
                                 + 1.53749 w_k1_IBM_s116
                                 + 1.35083 w_k1_INTC_s116
                                 + 1.37419 w_k1_KO_s116 + 1.56346 w_k1_MAT_s116
                                 + 1.25808 w_k1_MSFT_s116
                                 + 1.32725 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.04646 w_k1_ADM_s116
                                 + 1.12276 w_k1_AXP_s116
                                 + 1.09032 w_k1_DIS_s116
                                 + 1.16148 w_k1_FDX_s116
                                 + 0.861804 w_k1_IBM_s116
                                 + 1.15922 w_k1_INTC_s116
                                 + 0.996463 w_k1_KO_s116
                                 + 0.881339 w_k1_MAT_s116
                                 + 1.13916 w_k1_MSFT_s116
                                 + 1.17703 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.15938 w_k1_ADM_s116
                                 + 0.971019 w_k1_AXP_s116 + 0.749 w_k1_DIS_s116
                                 + 0.870974 w_k1_FDX_s116
                                 + 0.768223 w_k1_IBM_s116
                                 + 0.937832 w_k1_INTC_s116
                                 + 0.809075 w_k1_KO_s116
                                 + 0.707096 w_k1_MAT_s116
                                 + 0.813076 w_k1_MSFT_s116
                                 + 0.963148 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.778202 w_k1_ADM_s116
                                 + 0.622756 w_k1_AXP_s116
                                 + 0.426843 w_k1_DIS_s116
                                 + 0.593102 w_k1_FDX_s116
                                 + 0.932595 w_k1_IBM_s116
                                 + 0.64536 w_k1_INTC_s116
                                 + 0.730087 w_k1_KO_s116
                                 + 0.714761 w_k1_MAT_s116
                                 + 0.524932 w_k1_MSFT_s116
                                 + 0.719306 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.29477 w_k1_ADM_s116
                                 + 1.23015 w_k1_AXP_s116
                                 + 0.94972 w_k1_DIS_s116
                                 + 0.973755 w_k1_FDX_s116
                                 + 1.05182 w_k1_IBM_s116
                                 + 0.866128 w_k1_INTC_s116
                                 + 0.93273 w_k1_KO_s116 + 1.2588 w_k1_MAT_s116
                                 + 0.962237 w_k1_MSFT_s116
                                 + 1.07402 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.16563 w_k1_ADM_s116
                                 + 0.947758 w_k1_AXP_s116
                                 + 0.92535 w_k1_DIS_s116
                                 + 0.873158 w_k1_FDX_s116
                                 + 1.25951 w_k1_IBM_s116
                                 + 1.12203 w_k1_INTC_s116
                                 + 1.08825 w_k1_KO_s116
                                 + 0.895012 w_k1_MAT_s116
                                 + 1.04742 w_k1_MSFT_s116
                                 + 0.946699 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.843429 w_k1_ADM_s116
                                 + 0.735662 w_k1_AXP_s116
                                 + 0.76267 w_k1_DIS_s116
                                 + 0.814302 w_k1_FDX_s116
                                 + 0.601392 w_k1_IBM_s116
                                 + 0.963266 w_k1_INTC_s116
                                 + 0.772387 w_k1_KO_s116
                                 + 0.503006 w_k1_MAT_s116
                                 + 0.884023 w_k1_MSFT_s116
                                 + 0.976475 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.838747 w_k1_ADM_s116
                                 + 0.698039 w_k1_AXP_s116
                                 + 0.717595 w_k1_DIS_s116
                                 + 0.744025 w_k1_FDX_s116
                                 + 0.96435 w_k1_IBM_s116
                                 + 0.717094 w_k1_INTC_s116
                                 + 0.89009 w_k1_KO_s116
                                 + 0.760842 w_k1_MAT_s116
                                 + 0.729464 w_k1_MSFT_s116
                                 + 0.94536 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.984708 w_k1_ADM_s117
                                 + 0.885019 w_k1_AXP_s117
                                 + 0.873274 w_k1_DIS_s117
                                 + 0.926968 w_k1_FDX_s117
                                 + 0.827537 w_k1_IBM_s117
                                 + 1.11426 w_k1_INTC_s117
                                 + 0.825999 w_k1_KO_s117
                                 + 0.726626 w_k1_MAT_s117
                                 + 1.02078 w_k1_MSFT_s117
                                 + 0.92501 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.811521 w_k1_ADM_s117
                                 + 0.819882 w_k1_AXP_s117
                                 + 0.726732 w_k1_DIS_s117
                                 + 0.835633 w_k1_FDX_s117
                                 + 1.07448 w_k1_IBM_s117
                                 + 0.972067 w_k1_INTC_s117
                                 + 0.932813 w_k1_KO_s117
                                 + 1.01922 w_k1_MAT_s117
                                 + 0.973441 w_k1_MSFT_s117
                                 + 0.874185 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.03617 w_k1_ADM_s117
                                 + 1.2866 w_k1_AXP_s117 + 1.48949 w_k1_DIS_s117
                                 + 1.1849 w_k1_FDX_s117 + 1.49099 w_k1_IBM_s117
                                 + 1.27612 w_k1_INTC_s117
                                 + 1.57384 w_k1_KO_s117 + 1.80251 w_k1_MAT_s117
                                 + 1.41062 w_k1_MSFT_s117
                                 + 1.18526 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.719761 w_k1_ADM_s117
                                 + 0.676468 w_k1_AXP_s117
                                 + 0.521891 w_k1_DIS_s117
                                 + 0.719635 w_k1_FDX_s117
                                 + 0.628436 w_k1_IBM_s117
                                 + 0.851992 w_k1_INTC_s117
                                 + 0.711815 w_k1_KO_s117
                                 + 0.605645 w_k1_MAT_s117
                                 + 0.774212 w_k1_MSFT_s117
                                 + 0.731944 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.14826 w_k1_ADM_s117
                                 + 1.15176 w_k1_AXP_s117 + 1.3121 w_k1_DIS_s117
                                 + 1.21336 w_k1_FDX_s117
                                 + 1.09759 w_k1_IBM_s117
                                 + 1.13026 w_k1_INTC_s117
                                 + 1.22019 w_k1_KO_s117 + 1.22921 w_k1_MAT_s117
                                 + 1.25073 w_k1_MSFT_s117
                                 + 0.967116 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.27175 w_k1_ADM_s117
                                 + 1.77709 w_k1_AXP_s117
                                 + 1.31381 w_k1_DIS_s117
                                 + 1.42405 w_k1_FDX_s117
                                 + 0.937461 w_k1_IBM_s117
                                 + 1.73722 w_k1_INTC_s117
                                 + 1.10124 w_k1_KO_s117 + 1.19045 w_k1_MAT_s117
                                 + 1.45778 w_k1_MSFT_s117
                                 + 1.18453 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.08709 w_k1_ADM_s117
                                 + 1.22529 w_k1_AXP_s117
                                 + 0.959061 w_k1_DIS_s117
                                 + 0.885146 w_k1_FDX_s117
                                 + 1.69538 w_k1_IBM_s117
                                 + 1.05197 w_k1_INTC_s117
                                 + 1.15413 w_k1_KO_s117 + 1.63887 w_k1_MAT_s117
                                 + 1.09506 w_k1_MSFT_s117
                                 + 0.892182 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.93234 w_k1_ADM_s117
                                 + 0.993215 w_k1_AXP_s117
                                 + 0.989566 w_k1_DIS_s117
                                 + 1.13029 w_k1_FDX_s117
                                 + 0.699035 w_k1_IBM_s117
                                 + 1.3421 w_k1_INTC_s117
                                 + 0.977018 w_k1_KO_s117
                                 + 0.985043 w_k1_MAT_s117
                                 + 1.03512 w_k1_MSFT_s117
                                 + 0.889642 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0370817594315165 w_k1_ADM_s000
                                 + 0.01551829668104 w_k1_ADM_s001
                                 + 0.00806945810361006 w_k1_ADM_s002
                                 + 0.0146603286020928 w_k1_ADM_s003
                                 + 0.0405355656586975 w_k1_ADM_s004
                                 + 0.0132372904914702 w_k1_ADM_s005
                                 + 0.0141332453423107 w_k1_ADM_s006
                                 + 0.0174707941290404 w_k1_ADM_s007
                                 + 0.146388434015819 w_k1_ADM_s010
                                 + 0.0424765848217538 w_k1_ADM_s011
                                 + 0.176267267541103 w_k1_ADM_s012
                                 + 0.0444809929844146 w_k1_ADM_s013
                                 + 0.0391832306281919 w_k1_ADM_s014
                                 + 0.0501420239565036 w_k1_ADM_s015
                                 + 0.0489945102400941 w_k1_ADM_s016
                                 + 0.0765305991839628 w_k1_ADM_s017
                                 + 0.00897571669642325 w_k1_ADM_s100
                                 + 0.0104606051696745 w_k1_ADM_s101
                                 + 0.0134191629063733 w_k1_ADM_s102
                                 + 0.00433948913186773 w_k1_ADM_s103
                                 + 0.0107084792497963 w_k1_ADM_s104
                                 + 0.0119944706890511 w_k1_ADM_s105
                                 + 0.00987299699620044 w_k1_ADM_s106
                                 + 0.00693500092385824 w_k1_ADM_s107
                                 + 0.0137766612510294 w_k1_ADM_s110
                                 + 0.010795553239357 w_k1_ADM_s111
                                 + 0.00710718862981992 w_k1_ADM_s112
                                 + 0.0233165802063059 w_k1_ADM_s113
                                 + 0.00955235984911569 w_k1_ADM_s114
                                 + 0.0186977888431111 w_k1_ADM_s115
                                 + 0.0189263401359537 w_k1_ADM_s116
                                 + 0.0322208496040011 w_k1_ADM_s117
                                 + 0.0374997914746792 w_k1_AXP_s000
                                 + 0.0158201162831207 w_k1_AXP_s001
                                 + 0.00839279378360358 w_k1_AXP_s002
                                 + 0.0126462324403524 w_k1_AXP_s003
                                 + 0.0407331434952024 w_k1_AXP_s004
                                 + 0.0124776910699653 w_k1_AXP_s005
                                 + 0.0158520388924609 w_k1_AXP_s006
                                 + 0.0196628750501926 w_k1_AXP_s007
                                 + 0.152741901397053 w_k1_AXP_s010
                                 + 0.0463535527977697 w_k1_AXP_s011
                                 + 0.171811129886015 w_k1_AXP_s012
                                 + 0.0443238453458391 w_k1_AXP_s013
                                 + 0.0385005698149779 w_k1_AXP_s014
                                 + 0.0564319938383474 w_k1_AXP_s015
                                 + 0.0542253344274526 w_k1_AXP_s016
                                 + 0.0814090137970489 w_k1_AXP_s017
                                 + 0.00800261454763472 w_k1_AXP_s100
                                 + 0.0122693135764462 w_k1_AXP_s101
                                 + 0.0132066115065111 w_k1_AXP_s102
                                 + 0.0052218835083198 w_k1_AXP_s103
                                 + 0.0116815555507855 w_k1_AXP_s104
                                 + 0.0124573396742536 w_k1_AXP_s105
                                 + 0.00906703222368102 w_k1_AXP_s106
                                 + 0.00762677193016356 w_k1_AXP_s107
                                 + 0.0128404798639047 w_k1_AXP_s110
                                 + 0.00933996407213179 w_k1_AXP_s111
                                 + 0.00767785717627209 w_k1_AXP_s112
                                 + 0.025089415687428 w_k1_AXP_s113
                                 + 0.00938559046543321 w_k1_AXP_s114
                                 + 0.0178824292573865 w_k1_AXP_s115
                                 + 0.0178992043123147 w_k1_AXP_s116
                                 + 0.0346238008169489 w_k1_AXP_s117
                                 + 0.0394699201619969 w_k1_DIS_s000
                                 + 0.0168402975351169 w_k1_DIS_s001
                                 + 0.00899384201304533 w_k1_DIS_s002
                                 + 0.0136785959552262 w_k1_DIS_s003
                                 + 0.0381188787730981 w_k1_DIS_s004
                                 + 0.013301752015531 w_k1_DIS_s005
                                 + 0.0155422013820863 w_k1_DIS_s006
                                 + 0.0206034680717999 w_k1_DIS_s007
                                 + 0.140362745348213 w_k1_DIS_s010
                                 + 0.0503015701534299 w_k1_DIS_s011
                                 + 0.178373938927787 w_k1_DIS_s012
                                 + 0.0458563043998271 w_k1_DIS_s013
                                 + 0.0384933176220218 w_k1_DIS_s014
                                 + 0.0580278091341386 w_k1_DIS_s015
                                 + 0.0559491098459281 w_k1_DIS_s016
                                 + 0.0888788921352821 w_k1_DIS_s017
                                 + 0.007738538740608 w_k1_DIS_s100
                                 + 0.0123397179108893 w_k1_DIS_s101
                                 + 0.0129530861384358 w_k1_DIS_s102
                                 + 0.0049113922937182 w_k1_DIS_s103
                                 + 0.0111142610690159 w_k1_DIS_s104
                                 + 0.013037140317843 w_k1_DIS_s105
                                 + 0.00922644966898547 w_k1_DIS_s106
                                 + 0.00844032303991952 w_k1_DIS_s107
                                 + 0.0123909295754312 w_k1_DIS_s110
                                 + 0.00972215031867809 w_k1_DIS_s111
                                 + 0.00824482831699475 w_k1_DIS_s112
                                 + 0.025276756774873 w_k1_DIS_s113
                                 + 0.00936611489948175 w_k1_DIS_s114
                                 + 0.0201598235613418 w_k1_DIS_s115
                                 + 0.0160889096188622 w_k1_DIS_s116
                                 + 0.033063459311012 w_k1_DIS_s117
                                 + 0.0379979098645761 w_k1_FDX_s000
                                 + 0.0155702428591868 w_k1_FDX_s001
                                 + 0.0086962205892677 w_k1_FDX_s002
                                 + 0.0141751994980682 w_k1_FDX_s003
                                 + 0.0400408256954763 w_k1_FDX_s004
                                 + 0.0130214353327807 w_k1_FDX_s005
                                 + 0.0142396917066683 w_k1_FDX_s006
                                 + 0.0210672393916272 w_k1_FDX_s007
                                 + 0.137016800787985 w_k1_FDX_s010
                                 + 0.0457673253393037 w_k1_FDX_s011
                                 + 0.174652802216968 w_k1_FDX_s012
                                 + 0.0462131549967331 w_k1_FDX_s013
                                 + 0.0406897977908257 w_k1_FDX_s014
                                 + 0.0573112228467845 w_k1_FDX_s015
                                 + 0.0562169626236698 w_k1_FDX_s016
                                 + 0.0847630341075049 w_k1_FDX_s017
                                 + 0.00783889731622677 w_k1_FDX_s100
                                 + 0.0129756427516583 w_k1_FDX_s101
                                 + 0.0128682498432406 w_k1_FDX_s102
                                 + 0.005533282871655 w_k1_FDX_s103
                                 + 0.0117567191529082 w_k1_FDX_s104
                                 + 0.0126592166880916 w_k1_FDX_s105
                                 + 0.00914652528077913 w_k1_FDX_s106
                                 + 0.00823760433978838 w_k1_FDX_s107
                                 + 0.0118681653425021 w_k1_FDX_s110
                                 + 0.00935484889727373 w_k1_FDX_s111
                                 + 0.00759737308016084 w_k1_FDX_s112
                                 + 0.0251843060239127 w_k1_FDX_s113
                                 + 0.00891776858782403 w_k1_FDX_s114
                                 + 0.01870658442415 w_k1_FDX_s115
                                 + 0.0165854610751079 w_k1_FDX_s116
                                 + 0.0337331332803517 w_k1_FDX_s117
                                 + 0.0392732064958808 w_k1_IBM_s000
                                 + 0.0164074622081614 w_k1_IBM_s001
                                 + 0.00897598396902413 w_k1_IBM_s002
                                 + 0.0134169404925147 w_k1_IBM_s003
                                 + 0.0400926628051015 w_k1_IBM_s004
                                 + 0.0134678643932239 w_k1_IBM_s005
                                 + 0.0154341855769019 w_k1_IBM_s006
                                 + 0.0202278965927372 w_k1_IBM_s007
                                 + 0.14508113232776 w_k1_IBM_s010
                                 + 0.0473321551931895 w_k1_IBM_s011
                                 + 0.169902144987906 w_k1_IBM_s012
                                 + 0.0444583598929332 w_k1_IBM_s013
                                 + 0.0418831844491149 w_k1_IBM_s014
                                 + 0.0529412910286147 w_k1_IBM_s015
                                 + 0.0539036449318324 w_k1_IBM_s016
                                 + 0.0851523334572445 w_k1_IBM_s017
                                 + 0.00802014477361435 w_k1_IBM_s100
                                 + 0.0129617230124138 w_k1_IBM_s101
                                 + 0.0127982532804544 w_k1_IBM_s102
                                 + 0.00510627072945549 w_k1_IBM_s103
                                 + 0.0116516516020951 w_k1_IBM_s104
                                 + 0.0130656665018236 w_k1_IBM_s105
                                 + 0.00897147906069269 w_k1_IBM_s106
                                 + 0.00842632469618951 w_k1_IBM_s107
                                 + 0.0131397552067928 w_k1_IBM_s110
                                 + 0.00974040672933153 w_k1_IBM_s111
                                 + 0.00778653352577974 w_k1_IBM_s112
                                 + 0.0245897226066018 w_k1_IBM_s113
                                 + 0.00920280115666567 w_k1_IBM_s114
                                 + 0.0196440872440062 w_k1_IBM_s115
                                 + 0.017372768705903 w_k1_IBM_s116
                                 + 0.0352079315908851 w_k1_IBM_s117
                                 + 0.0434709157613391 w_k1_INTC_s000
                                 + 0.015674990064 w_k1_INTC_s001
                                 + 0.00902999933808095 w_k1_INTC_s002
                                 + 0.0130359766350305 w_k1_INTC_s003
                                 + 0.040600325047099 w_k1_INTC_s004
                                 + 0.0140374263827755 w_k1_INTC_s005
                                 + 0.0131768823442228 w_k1_INTC_s006
                                 + 0.0187884413416247 w_k1_INTC_s007
                                 + 0.142662224587373 w_k1_INTC_s010
                                 + 0.0541231293169189 w_k1_INTC_s011
                                 + 0.174225641532999 w_k1_INTC_s012
                                 + 0.0521877754195653 w_k1_INTC_s013
                                 + 0.0424282741612083 w_k1_INTC_s014
                                 + 0.0630626200935301 w_k1_INTC_s015
                                 + 0.0527592441385216 w_k1_INTC_s016
                                 + 0.0766224383561327 w_k1_INTC_s017
                                 + 0.00768751299395806 w_k1_INTC_s100
                                 + 0.0116851371329445 w_k1_INTC_s101
                                 + 0.0172675432446335 w_k1_INTC_s102
                                 + 0.00626840122166188 w_k1_INTC_s103
                                 + 0.0116771592184817 w_k1_INTC_s104
                                 + 0.0132505186255601 w_k1_INTC_s105
                                 + 0.00915447706604171 w_k1_INTC_s106
                                 + 0.00918982584695141 w_k1_INTC_s107
                                 + 0.012037270351346 w_k1_INTC_s110
                                 + 0.0105941111184248 w_k1_INTC_s111
                                 + 0.00689202883507374 w_k1_INTC_s112
                                 + 0.0239622960100907 w_k1_INTC_s113
                                 + 0.00808690192213485 w_k1_INTC_s114
                                 + 0.0161733650297832 w_k1_INTC_s115
                                 + 0.0174593490503861 w_k1_INTC_s116
                                 + 0.037898883226847 w_k1_INTC_s117
                                 + 0.0410370254239587 w_k1_KO_s000
                                 + 0.0165993467490114 w_k1_KO_s001
                                 + 0.00882767386634776 w_k1_KO_s002
                                 + 0.0134840534035708 w_k1_KO_s003
                                 + 0.0395968477346776 w_k1_KO_s004
                                 + 0.0143429456397179 w_k1_KO_s005
                                 + 0.0151281027054679 w_k1_KO_s006
                                 + 0.019829094420407 w_k1_KO_s007
                                 + 0.143336575177787 w_k1_KO_s010
                                 + 0.0485205068715624 w_k1_KO_s011
                                 + 0.161355579208068 w_k1_KO_s012
                                 + 0.0408574876617906 w_k1_KO_s013
                                 + 0.0409418894653956 w_k1_KO_s014
                                 + 0.0536888238522577 w_k1_KO_s015
                                 + 0.0539062320884766 w_k1_KO_s016
                                 + 0.0848740026543988 w_k1_KO_s017
                                 + 0.0072516264675768 w_k1_KO_s100
                                 + 0.0113254055464807 w_k1_KO_s101
                                 + 0.013776094867238 w_k1_KO_s102
                                 + 0.00493583920458997 w_k1_KO_s103
                                 + 0.0116673226522572 w_k1_KO_s104
                                 + 0.0134652319706215 w_k1_KO_s105
                                 + 0.00860192618515901 w_k1_KO_s106
                                 + 0.00927570570061372 w_k1_KO_s107
                                 + 0.0111847319471733 w_k1_KO_s110
                                 + 0.0093793373977044 w_k1_KO_s111
                                 + 0.00731382540866166 w_k1_KO_s112
                                 + 0.0242380174002982 w_k1_KO_s113
                                 + 0.00819429922902152 w_k1_KO_s114
                                 + 0.0202145989631356 w_k1_KO_s115
                                 + 0.01679516932471 w_k1_KO_s116
                                 + 0.0352497459822323 w_k1_KO_s117
                                 + 0.036583012386737 w_k1_MAT_s000
                                 + 0.0158738099484785 w_k1_MAT_s001
                                 + 0.00997822447655207 w_k1_MAT_s002
                                 + 0.0119350856428629 w_k1_MAT_s003
                                 + 0.0334871415027096 w_k1_MAT_s004
                                 + 0.0170549068655701 w_k1_MAT_s005
                                 + 0.0152004846755901 w_k1_MAT_s006
                                 + 0.0202190057944029 w_k1_MAT_s007
                                 + 0.146298736265497 w_k1_MAT_s010
                                 + 0.0438758277314849 w_k1_MAT_s011
                                 + 0.156820196805164 w_k1_MAT_s012
                                 + 0.0397468197702839 w_k1_MAT_s013
                                 + 0.0408282918580987 w_k1_MAT_s014
                                 + 0.0446531529899087 w_k1_MAT_s015
                                 + 0.0579942310130361 w_k1_MAT_s016
                                 + 0.0908274038044785 w_k1_MAT_s017
                                 + 0.00835268506883705 w_k1_MAT_s100
                                 + 0.0146362501728311 w_k1_MAT_s101
                                 + 0.0119081391098148 w_k1_MAT_s102
                                 + 0.00517200248145452 w_k1_MAT_s103
                                 + 0.01270537108075 w_k1_MAT_s104
                                 + 0.0124978591354037 w_k1_MAT_s105
                                 + 0.00698741296047884 w_k1_MAT_s106
                                 + 0.00958523719089802 w_k1_MAT_s107
                                 + 0.0150859310365433 w_k1_MAT_s110
                                 + 0.0114652832270982 w_k1_MAT_s111
                                 + 0.0093698124221432 w_k1_MAT_s112
                                 + 0.0277930508538947 w_k1_MAT_s113
                                 + 0.0114196382005369 w_k1_MAT_s114
                                 + 0.0340817907383891 w_k1_MAT_s115
                                 + 0.0164878943479615 w_k1_MAT_s116
                                 + 0.0378172175289287 w_k1_MAT_s117
                                 + 0.0391820682166245 w_k1_MSFT_s000
                                 + 0.01559319743773 w_k1_MSFT_s001
                                 + 0.0090659317769772 w_k1_MSFT_s002
                                 + 0.0141060298918095 w_k1_MSFT_s003
                                 + 0.0377827449869908 w_k1_MSFT_s004
                                 + 0.0135449472826726 w_k1_MSFT_s005
                                 + 0.013749277520989 w_k1_MSFT_s006
                                 + 0.0222653511406735 w_k1_MSFT_s007
                                 + 0.141378678132263 w_k1_MSFT_s010
                                 + 0.0494271532767105 w_k1_MSFT_s011
                                 + 0.180388663772222 w_k1_MSFT_s012
                                 + 0.0461583526695818 w_k1_MSFT_s013
                                 + 0.0419695366102007 w_k1_MSFT_s014
                                 + 0.0550238196177826 w_k1_MSFT_s015
                                 + 0.055283778291638 w_k1_MSFT_s016
                                 + 0.0794318557467389 w_k1_MSFT_s017
                                 + 0.00678904891899286 w_k1_MSFT_s100
                                 + 0.0123975958342354 w_k1_MSFT_s101
                                 + 0.0139711097578556 w_k1_MSFT_s102
                                 + 0.00560098984898186 w_k1_MSFT_s103
                                 + 0.0119114540113526 w_k1_MSFT_s104
                                 + 0.0117926414508152 w_k1_MSFT_s105
                                 + 0.00870854810374362 w_k1_MSFT_s106
                                 + 0.00840349222417888 w_k1_MSFT_s107
                                 + 0.0121268669797576 w_k1_MSFT_s110
                                 + 0.0100476152604662 w_k1_MSFT_s111
                                 + 0.00759398826323472 w_k1_MSFT_s112
                                 + 0.0214254489016044 w_k1_MSFT_s113
                                 + 0.00869630545197504 w_k1_MSFT_s114
                                 + 0.0185142625829671 w_k1_MSFT_s115
                                 + 0.0167048126737539 w_k1_MSFT_s116
                                 + 0.0367905156702828 w_k1_MSFT_s117
                                 + 0.0378948100242798 w_k1_PG_s000
                                 + 0.0153804086260169 w_k1_PG_s001
                                 + 0.00909270341303798 w_k1_PG_s002
                                 + 0.0144208062716088 w_k1_PG_s003
                                 + 0.0396654973692776 w_k1_PG_s004
                                 + 0.013927970405638 w_k1_PG_s005
                                 + 0.0142233301923021 w_k1_PG_s006
                                 + 0.0205256394765159 w_k1_PG_s007
                                 + 0.14387778257911 w_k1_PG_s010
                                 + 0.0430216468105292 w_k1_PG_s011
                                 + 0.174497145324663 w_k1_PG_s012
                                 + 0.0441388273504248 w_k1_PG_s013
                                 + 0.0389139951736278 w_k1_PG_s014
                                 + 0.0522284418141517 w_k1_PG_s015
                                 + 0.0511139272644267 w_k1_PG_s016
                                 + 0.0776966764407233 w_k1_PG_s017
                                 + 0.00811271914770509 w_k1_PG_s100
                                 + 0.0119825002217279 w_k1_PG_s101
                                 + 0.0121105939508542 w_k1_PG_s102
                                 + 0.00546222695241114 w_k1_PG_s103
                                 + 0.0115551498785436 w_k1_PG_s104
                                 + 0.011845014015471 w_k1_PG_s105
                                 + 0.00924837414075262 w_k1_PG_s106
                                 + 0.00823028561429302 w_k1_PG_s107
                                 + 0.0119910557782296 w_k1_PG_s110
                                 + 0.00940933426765236 w_k1_PG_s111
                                 + 0.0071063362404632 w_k1_PG_s112
                                 + 0.0224398273174822 w_k1_PG_s113
                                 + 0.00882220072938976 w_k1_PG_s114
                                 + 0.0186401123903976 w_k1_PG_s115
                                 + 0.0180698669661024 w_k1_PG_s116
                                 + 0.0313810464072486 w_k1_PG_s117 >= 1.08
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
