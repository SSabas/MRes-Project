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
 w_k1_ADM_s0_return_constraint:  1.06275 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.882753 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.89528 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.03567 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.1729 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.959918 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.06091 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.02907 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.991481 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.798227 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.988699 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.820543 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.11308 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.896187 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.958447 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.74606 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.02538 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.42372 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.936725 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.32393 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.38876 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.15718 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.845125 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.01099 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.908663 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.29547 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.684809 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.32801 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.42697 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.99326 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.06672 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.820405 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.19431 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.03979 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.07442 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.754517 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.755216 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.14235 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.06334 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.922188 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.74221 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.1902 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.39443 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.845983 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.0232 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.47904 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.91927 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.918024 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.07257 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.801286 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.41443 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.695685 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.717776 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.590723 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.962015 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.26771 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.831168 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.1451 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.81317 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.33316 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.683499 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.03661 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.986488 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.60898 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.64842 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.25397 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.42375 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.820204 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.47442 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.812052 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.42127 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.05837 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.04368 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.7192 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.803505 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.42882 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.15702 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.899029 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.673112 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.21377 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.44514 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.846037 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.21954 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.22626 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.685688 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.06522 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.16197 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.687836 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.5627 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.685343 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.760585 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.517524 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.970652 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.49664 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.744431 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.04246 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 2.16017 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.23812 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.537019 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.03194 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.92159 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.55723 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.686883 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.2132 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.85114 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.933618 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.31072 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.901594 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.51045 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.905902 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.02823 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.637954 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.698852 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.60993 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.10461 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.889911 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.840033 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.09322 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.23438 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.940194 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.993224 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.12161 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.86041 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.954072 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.0898 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.751349 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.39014 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.829066 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.895713 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.65992 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.930725 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.39581 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.876521 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.14515 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.56925 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.09821 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.744932 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.992017 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.04681 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.23401 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.739397 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.14345 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.53044 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.896323 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.05374 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.849378 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.32469 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.959561 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.1257 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.719926 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.731242 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.33688 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.967035 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.10486 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.728516 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.1911 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.47902 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.867151 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.25152 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.45536 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.735789 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.858958 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.970056 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.04407 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.953765 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.602015 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.598952 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.947842 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.16095 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.973737 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.912846 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.989282 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.70691 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.58687 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.68133 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.33714 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.653079 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.75497 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.88957 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.943789 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.935946 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.05199 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.59719 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.919097 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.24984 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.01362 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.688966 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.699366 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.39264 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.993333 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11545 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.932146 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.905214 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.0001 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.1799 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.895583 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.993121 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.06434 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.784976 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.905127 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.951279 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.777664 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.12682 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.793238 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.971578 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.762875 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.904883 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.21747 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.964017 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.01507 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.6706 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.17287 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.746383 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.898712 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.00573 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.23415 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.803227 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.13983 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.32389 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.920232 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.12508 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.825193 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.15974 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.09384 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.15686 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.783174 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.964852 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.35254 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.02625 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.01853 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.777614 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.14236 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.27224 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.85089 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.13326 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.09173 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.80262 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.931543 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.10035 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.787876 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.1877 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.75456 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.758112 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.821438 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.08703 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.18394 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.925027 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.08998 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.88596 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.40048 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.672637 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.08721 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.826206 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.7427 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.852121 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.14183 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.29023 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.063 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.33425 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.962896 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.28796 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.953971 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.940316 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.648072 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.0318 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.22402 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.949522 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.05296 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.782446 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.08559 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.40847 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.862016 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.873876 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.29621 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.781303 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.676393 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.867568 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.834558 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.0899 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.579062 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.580873 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.924458 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.2862 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.19683 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.08369 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.37039 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.21993 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 2.02404 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.667711 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.64034 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.849048 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 2.87506 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.964366 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.07698 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.05484 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.12102 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.4978 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.896858 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.5873 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.985666 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.642668 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.639719 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.24142 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.10199 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.05951 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.915098 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.869745 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.11417 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.16052 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.886927 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.02286 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.05002 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.858759 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.950653 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.14246 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.769097 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.3692 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.871018 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.894753 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.726962 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.999416 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.24542 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.914887 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.08266 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.56363 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.18696 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.7569 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.940295 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06798 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.27031 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.798329 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.15414 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.52603 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.959398 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.14546 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.0073 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.27149 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.00774 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15914 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.776992 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.866488 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.39904 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.02536 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.955306 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.836276 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.11043 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.19962 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.907438 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.05534 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.1162 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.987627 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.986565 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.17628 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.899087 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.26408 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.889635 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.879691 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.794296 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.03205 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.13263 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.930692 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.00665 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.35096 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.17578 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.820999 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.09449 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.945335 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.36621 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.83741 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.08656 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.2839 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.937864 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.16175 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.943666 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.27899 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.99422 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.06975 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.821316 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.97096 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.21598 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.250998926610834 z_k1_s0000
                                 - 0.308999988773677 z_k1_s0001
                                 - 0.227000307092105 z_k1_s0002
                                 - 0.142999657785798 z_k1_s0003
                                 - 0.115999891329453 z_k1_s0004
                                 - 0.141999681472575 z_k1_s0005
                                 - 0.423999294790935 z_k1_s0006
                                 - 0.417998624912964 z_k1_s0007
                                 - 0.328999760270808 z_k1_s0010
                                 - 0.341999071338994 z_k1_s0011
                                 - 0.1130000734566 z_k1_s0012
                                 - 0.099000016398241 z_k1_s0013
                                 - 0.364001031917144 z_k1_s0014
                                 - 0.412999651020782 z_k1_s0015
                                 - 0.196999964206574 z_k1_s0016
                                 - 0.401000634342592 z_k1_s0017
                                 - 0.073000040819159 z_k1_s0020
                                 - 0.196000155292559 z_k1_s0021
                                 - 0.102000088670621 z_k1_s0022
                                 - 0.151999649124615 z_k1_s0023
                                 - 0.108000041057103 z_k1_s0024
                                 - 0.12400008515447 z_k1_s0025
                                 - 0.175000001189718 z_k1_s0026
                                 - 0.352999657822523 z_k1_s0027
                                 - 0.401000327543879 z_k1_s0030
                                 - 0.195999551125489 z_k1_s0031
                                 - 0.231001161149831 z_k1_s0032
                                 - 0.374000437669505 z_k1_s0033
                                 - 0.147000136594658 z_k1_s0034
                                 - 0.214000108709127 z_k1_s0035
                                 - 0.170000112895131 z_k1_s0036
                                 - 0.160000139660162 z_k1_s0037
                                 - 0.243000869660656 z_k1_s0040
                                 - 0.111000067730974 z_k1_s0041
                                 - 0.176999766294885 z_k1_s0042
                                 - 0.136000097223555 z_k1_s0043
                                 - 0.4030007774528 z_k1_s0044
                                 - 0.483999896671468 z_k1_s0045
                                 - 0.105000158988885 z_k1_s0046
                                 - 0.0970001635992779 z_k1_s0047
                                 - 0.524999922785885 z_k1_s0050
                                 - 0.484000895641229 z_k1_s0051
                                 - 0.157000489979165 z_k1_s0052
                                 - 0.474001816485716 z_k1_s0053
                                 - 0.412001920507431 z_k1_s0054
                                 - 0.29000069877883 z_k1_s0055
                                 - 0.223000640421222 z_k1_s0056
                                 - 0.549002695171652 z_k1_s0057
                                 - 0.521998216093087 z_k1_s0060
                                 - 0.31399976029184 z_k1_s0061
                                 - 0.310999823306236 z_k1_s0062
                                 - 0.699000147842523 z_k1_s0063
                                 - 0.547001221972391 z_k1_s0064
                                 - 0.530998733249753 z_k1_s0065
                                 - 0.276999831269544 z_k1_s0066
                                 - 0.329999777314985 z_k1_s0067
                                 - 0.205999566557271 z_k1_s0070
                                 - 0.293998443092379 z_k1_s0071
                                 - 0.195999557260378 z_k1_s0072
                                 - 0.183999680023834 z_k1_s0073
                                 - 0.530998417909249 z_k1_s0074
                                 - 0.272999829726048 z_k1_s0075
                                 - 0.183999680023834 z_k1_s0076
                                 - 0.363998284971085 z_k1_s0077
                                 - 0.695000239439242 z_k1_s0100
                                 - 0.93399709740878 z_k1_s0101
                                 - 0.494999938975416 z_k1_s0102
                                 - 1.80100292734127 z_k1_s0103
                                 - 0.649999306432323 z_k1_s0104
                                 - 0.417000143663545 z_k1_s0105
                                 - 0.36200001548849 z_k1_s0106
                                 - 0.719000737042932 z_k1_s0107
                                 - 0.220999817669538 z_k1_s0110
                                 - 0.570000361889674 z_k1_s0111
                                 - 0.294999978525379 z_k1_s0112
                                 - 0.71900026669945 z_k1_s0113
                                 - 0.174999883359134 z_k1_s0114
                                 - 0.608000386015652 z_k1_s0115
                                 - 0.294999978525379 z_k1_s0116
                                 - 0.722999320291755 z_k1_s0117
                                 - 1.58500129796656 z_k1_s0120
                                 - 0.450000340555774 z_k1_s0121
                                 - 0.716001184081488 z_k1_s0122
                                 - 0.450000340555774 z_k1_s0123
                                 - 0.330000431501112 z_k1_s0124
                                 - 1.21200208049557 z_k1_s0125
                                 - 0.83299796004614 z_k1_s0126
                                 - 1.2400022410352 z_k1_s0127
                                 - 0.948996799917301 z_k1_s0130
                                 - 0.410999842050765 z_k1_s0131
                                 - 0.760998659061783 z_k1_s0132
                                 - 1.05900123287326 z_k1_s0133
                                 - 0.403000116835563 z_k1_s0134
                                 - 1.5460004351677 z_k1_s0135
                                 - 0.772001729956398 z_k1_s0136
                                 - 0.66800136076025 z_k1_s0137
                                 - 1.42899736962882 z_k1_s0140
                                 - 0.553999046030933 z_k1_s0141
                                 - 0.946995996139979 z_k1_s0142
                                 - 0.449999024364821 z_k1_s0143
                                 - 1.16399922403401 z_k1_s0144
                                 - 0.676999759115638 z_k1_s0145
                                 - 0.988999824113975 z_k1_s0146
                                 - 0.40999906695469 z_k1_s0147
                                 - 0.470998872666809 z_k1_s0150
                                 - 0.892996925820479 z_k1_s0151
                                 - 0.598996056723433 z_k1_s0152
                                 - 0.313998708713634 z_k1_s0153
                                 - 0.343998301725767 z_k1_s0154
                                 - 0.274998832999683 z_k1_s0155
                                 - 0.778997662778017 z_k1_s0156
                                 - 0.372998448035067 z_k1_s0157
                                 - 0.179000186240488 z_k1_s0160
                                 - 0.108000036513767 z_k1_s0161
                                 - 0.324999860157016 z_k1_s0162
                                 - 0.190000977015523 z_k1_s0163
                                 - 0.260000260125842 z_k1_s0164
                                 - 0.447999875908791 z_k1_s0165
                                 - 0.205000026252958 z_k1_s0166
                                 - 0.145000109300928 z_k1_s0167
                                 - 0.365999888444517 z_k1_s0170
                                 - 0.940998962550709 z_k1_s0171
                                 - 0.370999787982855 z_k1_s0172
                                 - 0.31799998695669 z_k1_s0173
                                 - 0.549001013466477 z_k1_s0174
                                 - 0.510998943055823 z_k1_s0175
                                 - 0.233000120405341 z_k1_s0176
                                 - 0.646999516737784 z_k1_s0177
                                 - 0.106999378571344 z_k1_s1000
                                 - 0.140999701230165 z_k1_s1001
                                 - 0.0759995090210632 z_k1_s1002
                                 - 0.299998419133646 z_k1_s1003
                                 - 0.108999310445911 z_k1_s1004
                                 - 0.100999336149112 z_k1_s1005
                                 - 0.296998583021427 z_k1_s1006
                                 - 0.102999391422944 z_k1_s1007
                                 - 0.292998334053365 z_k1_s1010
                                 - 0.267999244032765 z_k1_s1011
                                 - 0.368998704986204 z_k1_s1012
                                 - 0.308998412144489 z_k1_s1013
                                 - 0.179999433729651 z_k1_s1014
                                 - 0.174999409326175 z_k1_s1015
                                 - 0.294998343814755 z_k1_s1016
                                 - 0.174999409326175 z_k1_s1017
                                 - 0.0800001966559577 z_k1_s1020
                                 - 0.225001106002571 z_k1_s1021
                                 - 0.09700021973155 z_k1_s1022
                                 - 0.118000368325241 z_k1_s1023
                                 - 0.0870002461871882 z_k1_s1024
                                 - 0.0840001860737024 z_k1_s1025
                                 - 0.392001684946071 z_k1_s1026
                                 - 0.278001306038575 z_k1_s1027
                                 - 0.0969997059107652 z_k1_s1030
                                 - 0.0569999204629557 z_k1_s1031
                                 - 0.162999595499465 z_k1_s1032
                                 - 0.101999592091808 z_k1_s1033
                                 - 0.0729999042420263 z_k1_s1034
                                 - 0.0619999806438655 z_k1_s1035
                                 - 0.162999595499465 z_k1_s1036
                                 - 0.152999823137379 z_k1_s1037
                                 - 0.168999377495026 z_k1_s1040
                                 - 0.100999827332214 z_k1_s1041
                                 - 0.112999596467068 z_k1_s1042
                                 - 0.251000053212952 z_k1_s1043
                                 - 0.114999665289373 z_k1_s1044
                                 - 0.0669999449509778 z_k1_s1045
                                 - 0.134999280514131 z_k1_s1046
                                 - 0.121999906167442 z_k1_s1047
                                 - 0.120999864204681 z_k1_s1050
                                 - 0.292000428823921 z_k1_s1051
                                 - 0.331000417947718 z_k1_s1052
                                 - 0.308000016157084 z_k1_s1053
                                 - 0.224000192659554 z_k1_s1054
                                 - 0.167999845994902 z_k1_s1055
                                 - 0.261999212575816 z_k1_s1056
                                 - 0.284000635157339 z_k1_s1057
                                 - 0.108999490460509 z_k1_s1060
                                 - 0.0819999484828534 z_k1_s1061
                                 - 0.0869995311382571 z_k1_s1062
                                 - 0.0769999423278093 z_k1_s1063
                                 - 0.100999582252352 z_k1_s1064
                                 - 0.117999704819416 z_k1_s1065
                                 - 0.101999888403102 z_k1_s1066
                                 - 0.170999837822826 z_k1_s1067
                                 - 0.490998435013835 z_k1_s1070
                                 - 0.316999959857954 z_k1_s1071
                                 - 0.165999582934758 z_k1_s1072
                                 - 0.297999596939889 z_k1_s1073
                                 - 0.248999506001824 z_k1_s1074
                                 - 0.528999160849963 z_k1_s1075
                                 - 0.288998178297449 z_k1_s1076
                                 - 0.2929988088052 z_k1_s1077
                                 - 0.38599942558439 z_k1_s1100
                                 - 0.521000911068928 z_k1_s1101
                                 - 0.329999499634335 z_k1_s1102
                                 - 1.47399869089881 z_k1_s1103
                                 - 0.677997823468426 z_k1_s1104
                                 - 0.29999966730378 z_k1_s1105
                                 - 0.384999490906626 z_k1_s1106
                                 - 0.405999463137896 z_k1_s1107
                                 - 0.243001233034436 z_k1_s1110
                                 - 0.193000015625976 z_k1_s1111
                                 - 0.401001125434907 z_k1_s1112
                                 - 0.18300004550444 z_k1_s1113
                                 - 0.55400080497448 z_k1_s1114
                                 - 0.280000530203779 z_k1_s1115
                                 - 0.162000062689189 z_k1_s1116
                                 - 0.261999763904545 z_k1_s1117
                                 - 0.498998663856566 z_k1_s1120
                                 - 0.626998825190378 z_k1_s1121
                                 - 0.292999930047048 z_k1_s1122
                                 - 0.77199958591993 z_k1_s1123
                                 - 0.393998878293579 z_k1_s1124
                                 - 0.35599980138484 z_k1_s1125
                                 - 0.29099988128872 z_k1_s1126
                                 - 0.466998623523113 z_k1_s1127
                                 - 0.59500055070219 z_k1_s1130
                                 - 0.364998690146498 z_k1_s1131
                                 - 0.353999674675494 z_k1_s1132
                                 - 0.640998018015898 z_k1_s1133
                                 - 0.907999735843983 z_k1_s1134
                                 - 0.283999866192657 z_k1_s1135
                                 - 0.215999647641115 z_k1_s1136
                                 - 0.267999878845179 z_k1_s1137
                                 - 0.686003520456299 z_k1_s1140
                                 - 0.433001429189042 z_k1_s1141
                                 - 0.752003888855966 z_k1_s1142
                                 - 1.24500255084527 z_k1_s1143
                                 - 0.352001115962559 z_k1_s1144
                                 - 0.418000975060827 z_k1_s1145
                                 - 0.601999347573815 z_k1_s1146
                                 - 0.604999132818618 z_k1_s1147
                                 - 0.531999289007093 z_k1_s1150
                                 - 0.941995985292802 z_k1_s1151
                                 - 0.360999228383836 z_k1_s1152
                                 - 1.23199612452517 z_k1_s1153
                                 - 0.393999480461633 z_k1_s1154
                                 - 1.44199772865661 z_k1_s1155
                                 - 0.826996144838252 z_k1_s1156
                                 - 0.497999444489927 z_k1_s1157
                                 - 0.180999709874365 z_k1_s1160
                                 - 0.151000280259927 z_k1_s1161
                                 - 0.151000280259927 z_k1_s1162
                                 - 0.161000090131406 z_k1_s1163
                                 - 0.14900058088568 z_k1_s1164
                                 - 0.271999817905171 z_k1_s1165
                                 - 0.151999473446928 z_k1_s1166
                                 - 0.0960000129665452 z_k1_s1167
                                 - 0.643999752263434 z_k1_s1170
                                 - 0.299999619838601 z_k1_s1171
                                 - 0.838999988727876 z_k1_s1172
                                 - 0.392998602907773 z_k1_s1173
                                 - 0.222999618311493 z_k1_s1174
                                 - 0.609000077205221 z_k1_s1175
                                 - 0.218999604276337 z_k1_s1176
                                 - 0.354999365072592 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.762473 w_k1_ADM_s000
                                 + 0.919385 w_k1_AXP_s000
                                 + 0.781043 w_k1_DIS_s000
                                 + 0.887908 w_k1_FDX_s000
                                 + 0.752343 w_k1_IBM_s000
                                 + 0.864344 w_k1_INTC_s000
                                 + 0.806059 w_k1_KO_s000
                                 + 0.912765 w_k1_MAT_s000
                                 + 0.945505 w_k1_MSFT_s000
                                 + 0.847222 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.00725 w_k1_ADM_s000
                                 + 1.49115 w_k1_AXP_s000
                                 + 1.72122 w_k1_DIS_s000
                                 + 1.37602 w_k1_FDX_s000
                                 + 1.29409 w_k1_IBM_s000
                                 + 1.20681 w_k1_INTC_s000
                                 + 1.39046 w_k1_KO_s000 + 1.63428 w_k1_MAT_s000
                                 + 1.34931 w_k1_MSFT_s000
                                 + 1.31083 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.738784 w_k1_ADM_s000
                                 + 0.902103 w_k1_AXP_s000
                                 + 0.722975 w_k1_DIS_s000
                                 + 0.817882 w_k1_FDX_s000
                                 + 0.941962 w_k1_IBM_s000
                                 + 0.940683 w_k1_INTC_s000
                                 + 0.943717 w_k1_KO_s000
                                 + 1.13858 w_k1_MAT_s000
                                 + 0.919243 w_k1_MSFT_s000
                                 + 0.920838 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.00138 w_k1_ADM_s000
                                 + 1.00178 w_k1_AXP_s000
                                 + 1.12312 w_k1_DIS_s000
                                 + 0.905117 w_k1_FDX_s000
                                 + 1.61773 w_k1_IBM_s000
                                 + 0.993891 w_k1_INTC_s000
                                 + 1.39233 w_k1_KO_s000 + 1.41925 w_k1_MAT_s000
                                 + 1.05815 w_k1_MSFT_s000
                                 + 1.26631 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.677424 w_k1_ADM_s000
                                 + 0.712985 w_k1_AXP_s000
                                 + 0.521706 w_k1_DIS_s000
                                 + 0.746573 w_k1_FDX_s000
                                 + 0.723979 w_k1_IBM_s000
                                 + 0.847891 w_k1_INTC_s000
                                 + 0.679501 w_k1_KO_s000
                                 + 0.982742 w_k1_MAT_s000
                                 + 0.775632 w_k1_MSFT_s000
                                 + 0.782709 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.929837 w_k1_ADM_s000
                                 + 0.8151 w_k1_AXP_s000
                                 + 0.900031 w_k1_DIS_s000
                                 + 0.905605 w_k1_FDX_s000
                                 + 0.885468 w_k1_IBM_s000
                                 + 0.816288 w_k1_INTC_s000
                                 + 0.975409 w_k1_KO_s000
                                 + 1.02912 w_k1_MAT_s000
                                 + 0.898969 w_k1_MSFT_s000
                                 + 1.0137 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.01157 w_k1_ADM_s000
                                 + 1.07982 w_k1_AXP_s000 + 1.1182 w_k1_DIS_s000
                                 + 1.14118 w_k1_FDX_s000
                                 + 0.899065 w_k1_IBM_s000
                                 + 1.0492 w_k1_INTC_s000
                                 + 0.950258 w_k1_KO_s000
                                 + 0.880959 w_k1_MAT_s000
                                 + 1.06132 w_k1_MSFT_s000
                                 + 1.08472 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.740163 w_k1_ADM_s000
                                 + 0.900541 w_k1_AXP_s000
                                 + 0.968321 w_k1_DIS_s000
                                 + 0.945652 w_k1_FDX_s000
                                 + 1.15632 w_k1_IBM_s000
                                 + 0.907814 w_k1_INTC_s000
                                 + 0.999656 w_k1_KO_s000
                                 + 1.27973 w_k1_MAT_s000
                                 + 1.0067 w_k1_MSFT_s000 + 1.00592 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.42069 w_k1_ADM_s001
                                 + 1.3092 w_k1_AXP_s001 + 1.36101 w_k1_DIS_s001
                                 + 1.21135 w_k1_FDX_s001
                                 + 1.34646 w_k1_IBM_s001
                                 + 1.07275 w_k1_INTC_s001
                                 + 1.36488 w_k1_KO_s001 + 1.21568 w_k1_MAT_s001
                                 + 1.18993 w_k1_MSFT_s001
                                 + 1.13483 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.950463 w_k1_ADM_s001
                                 + 0.823362 w_k1_AXP_s001
                                 + 0.870828 w_k1_DIS_s001
                                 + 0.897425 w_k1_FDX_s001
                                 + 0.947449 w_k1_IBM_s001
                                 + 0.874431 w_k1_INTC_s001
                                 + 1.02388 w_k1_KO_s001
                                 + 0.858486 w_k1_MAT_s001
                                 + 0.889561 w_k1_MSFT_s001
                                 + 0.836354 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.26785 w_k1_ADM_s001
                                 + 1.25164 w_k1_AXP_s001
                                 + 1.66643 w_k1_DIS_s001
                                 + 1.11018 w_k1_FDX_s001
                                 + 1.84427 w_k1_IBM_s001
                                 + 1.16913 w_k1_INTC_s001
                                 + 2.00613 w_k1_KO_s001 + 2.18519 w_k1_MAT_s001
                                 + 1.38546 w_k1_MSFT_s001
                                 + 1.50953 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.17387 w_k1_ADM_s001
                                 + 0.966129 w_k1_AXP_s001
                                 + 1.41254 w_k1_DIS_s001
                                 + 1.30739 w_k1_FDX_s001
                                 + 0.80819 w_k1_IBM_s001
                                 + 1.02771 w_k1_INTC_s001
                                 + 1.31853 w_k1_KO_s001 + 1.18891 w_k1_MAT_s001
                                 + 1.2888 w_k1_MSFT_s001 + 1.1114 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.898998 w_k1_ADM_s001
                                 + 0.786655 w_k1_AXP_s001
                                 + 0.567742 w_k1_DIS_s001
                                 + 0.775578 w_k1_FDX_s001
                                 + 0.759889 w_k1_IBM_s001
                                 + 0.879076 w_k1_INTC_s001
                                 + 0.771201 w_k1_KO_s001
                                 + 0.958906 w_k1_MAT_s001
                                 + 0.878882 w_k1_MSFT_s001
                                 + 0.863659 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.893508 w_k1_ADM_s001
                                 + 0.815242 w_k1_AXP_s001
                                 + 0.855192 w_k1_DIS_s001
                                 + 0.792305 w_k1_FDX_s001
                                 + 1.27043 w_k1_IBM_s001
                                 + 0.868732 w_k1_INTC_s001
                                 + 1.14389 w_k1_KO_s001 + 1.36708 w_k1_MAT_s001
                                 + 0.931303 w_k1_MSFT_s001
                                 + 0.942699 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.35664 w_k1_ADM_s001
                                 + 1.43791 w_k1_AXP_s001
                                 + 1.51876 w_k1_DIS_s001
                                 + 1.26735 w_k1_FDX_s001
                                 + 0.952601 w_k1_IBM_s001
                                 + 0.906035 w_k1_INTC_s001
                                 + 1.18105 w_k1_KO_s001 + 1.57263 w_k1_MAT_s001
                                 + 1.20321 w_k1_MSFT_s001
                                 + 1.29431 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.33011 w_k1_ADM_s001
                                 + 1.01234 w_k1_AXP_s001
                                 + 0.910816 w_k1_DIS_s001
                                 + 1.07338 w_k1_FDX_s001
                                 + 0.773276 w_k1_IBM_s001
                                 + 0.969347 w_k1_INTC_s001
                                 + 0.989037 w_k1_KO_s001
                                 + 0.870619 w_k1_MAT_s001
                                 + 0.956701 w_k1_MSFT_s001
                                 + 1.00576 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.43149 w_k1_ADM_s002
                                 + 1.33418 w_k1_AXP_s002
                                 + 1.63645 w_k1_DIS_s002
                                 + 1.45824 w_k1_FDX_s002
                                 + 1.08179 w_k1_IBM_s002
                                 + 1.19546 w_k1_INTC_s002
                                 + 1.23345 w_k1_KO_s002 + 1.00155 w_k1_MAT_s002
                                 + 1.31956 w_k1_MSFT_s002 + 1.2772 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.888004 w_k1_ADM_s002
                                 + 1.00903 w_k1_AXP_s002
                                 + 1.07037 w_k1_DIS_s002
                                 + 0.950213 w_k1_FDX_s002
                                 + 1.23215 w_k1_IBM_s002
                                 + 0.959426 w_k1_INTC_s002
                                 + 1.08263 w_k1_KO_s002 + 1.36057 w_k1_MAT_s002
                                 + 0.908609 w_k1_MSFT_s002
                                 + 1.01244 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.31879 w_k1_ADM_s002
                                 + 1.53818 w_k1_AXP_s002
                                 + 1.36865 w_k1_DIS_s002 + 1.3164 w_k1_FDX_s002
                                 + 1.3962 w_k1_IBM_s002
                                 + 1.22304 w_k1_INTC_s002
                                 + 1.29964 w_k1_KO_s002 + 1.22086 w_k1_MAT_s002
                                 + 1.16927 w_k1_MSFT_s002 + 1.1422 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.978008 w_k1_ADM_s002
                                 + 1.14633 w_k1_AXP_s002
                                 + 1.17106 w_k1_DIS_s002
                                 + 1.10923 w_k1_FDX_s002
                                 + 0.904083 w_k1_IBM_s002
                                 + 1.11221 w_k1_INTC_s002
                                 + 0.97792 w_k1_KO_s002 + 1.1628 w_k1_MAT_s002
                                 + 0.992711 w_k1_MSFT_s002
                                 + 0.910915 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.970727 w_k1_ADM_s002
                                 + 1.12805 w_k1_AXP_s002
                                 + 1.40976 w_k1_DIS_s002 + 1.129 w_k1_FDX_s002
                                 + 1.2743 w_k1_IBM_s002
                                 + 1.15417 w_k1_INTC_s002
                                 + 1.12164 w_k1_KO_s002 + 1.53671 w_k1_MAT_s002
                                 + 1.12312 w_k1_MSFT_s002
                                 + 1.07201 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.744394 w_k1_ADM_s002
                                 + 0.695022 w_k1_AXP_s002
                                 + 0.687199 w_k1_DIS_s002
                                 + 0.741225 w_k1_FDX_s002
                                 + 0.669483 w_k1_IBM_s002
                                 + 0.756342 w_k1_INTC_s002
                                 + 0.713826 w_k1_KO_s002
                                 + 0.559179 w_k1_MAT_s002
                                 + 0.749841 w_k1_MSFT_s002
                                 + 0.762567 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.11477 w_k1_ADM_s002
                                 + 0.974894 w_k1_AXP_s002
                                 + 1.16379 w_k1_DIS_s002
                                 + 1.06426 w_k1_FDX_s002
                                 + 1.01585 w_k1_IBM_s002
                                 + 1.05122 w_k1_INTC_s002
                                 + 0.956199 w_k1_KO_s002
                                 + 0.802327 w_k1_MAT_s002
                                 + 1.12618 w_k1_MSFT_s002
                                 + 0.989067 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.675157 w_k1_ADM_s002
                                 + 0.73316 w_k1_AXP_s002
                                 + 0.842536 w_k1_DIS_s002
                                 + 0.819467 w_k1_FDX_s002
                                 + 0.982461 w_k1_IBM_s002
                                 + 0.952115 w_k1_INTC_s002
                                 + 0.911587 w_k1_KO_s002
                                 + 0.79367 w_k1_MAT_s002
                                 + 0.954937 w_k1_MSFT_s002
                                 + 0.943039 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.17019 w_k1_ADM_s003
                                 + 1.04677 w_k1_AXP_s003
                                 + 1.15444 w_k1_DIS_s003
                                 + 1.12884 w_k1_FDX_s003
                                 + 0.887181 w_k1_IBM_s003
                                 + 1.11154 w_k1_INTC_s003
                                 + 1.04899 w_k1_KO_s003
                                 + 0.996636 w_k1_MAT_s003
                                 + 1.21408 w_k1_MSFT_s003
                                 + 1.02236 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.0155 w_k1_ADM_s003
                                 + 0.926059 w_k1_AXP_s003
                                 + 1.02659 w_k1_DIS_s003
                                 + 0.903598 w_k1_FDX_s003
                                 + 1.40242 w_k1_IBM_s003
                                 + 1.00319 w_k1_INTC_s003
                                 + 1.19762 w_k1_KO_s003 + 1.45224 w_k1_MAT_s003
                                 + 0.875478 w_k1_MSFT_s003
                                 + 0.88874 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.52566 w_k1_ADM_s003
                                 + 1.66585 w_k1_AXP_s003
                                 + 1.33233 w_k1_DIS_s003
                                 + 1.48196 w_k1_FDX_s003
                                 + 1.05103 w_k1_IBM_s003
                                 + 1.18931 w_k1_INTC_s003
                                 + 1.14878 w_k1_KO_s003 + 1.51753 w_k1_MAT_s003
                                 + 1.33017 w_k1_MSFT_s003
                                 + 1.19368 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.927489 w_k1_ADM_s003
                                 + 0.814872 w_k1_AXP_s003
                                 + 0.8168 w_k1_DIS_s003
                                 + 0.884633 w_k1_FDX_s003
                                 + 0.899834 w_k1_IBM_s003
                                 + 1.01779 w_k1_INTC_s003
                                 + 0.929111 w_k1_KO_s003
                                 + 0.936964 w_k1_MAT_s003
                                 + 0.934679 w_k1_MSFT_s003
                                 + 0.888699 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.11557 w_k1_ADM_s003
                                 + 1.19116 w_k1_AXP_s003
                                 + 0.799808 w_k1_DIS_s003
                                 + 0.96412 w_k1_FDX_s003
                                 + 1.00547 w_k1_IBM_s003
                                 + 1.09961 w_k1_INTC_s003
                                 + 0.895246 w_k1_KO_s003
                                 + 1.79566 w_k1_MAT_s003
                                 + 1.04867 w_k1_MSFT_s003
                                 + 0.962915 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.18693 w_k1_ADM_s003
                                 + 0.908786 w_k1_AXP_s003
                                 + 0.803222 w_k1_DIS_s003
                                 + 1.08476 w_k1_FDX_s003
                                 + 0.608975 w_k1_IBM_s003
                                 + 1.02465 w_k1_INTC_s003
                                 + 0.762011 w_k1_KO_s003
                                 + 0.801316 w_k1_MAT_s003
                                 + 1.07038 w_k1_MSFT_s003
                                 + 0.93121 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.38543 w_k1_ADM_s003
                                 + 1.13536 w_k1_AXP_s003
                                 + 1.40333 w_k1_DIS_s003
                                 + 1.15236 w_k1_FDX_s003
                                 + 1.35709 w_k1_IBM_s003
                                 + 1.01128 w_k1_INTC_s003
                                 + 1.41741 w_k1_KO_s003 + 1.37313 w_k1_MAT_s003
                                 + 1.23375 w_k1_MSFT_s003
                                 + 1.20579 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.809496 w_k1_ADM_s003
                                 + 0.832561 w_k1_AXP_s003
                                 + 0.658376 w_k1_DIS_s003
                                 + 0.79209 w_k1_FDX_s003
                                 + 0.934377 w_k1_IBM_s003
                                 + 0.733117 w_k1_INTC_s003
                                 + 0.836167 w_k1_KO_s003
                                 + 1.18896 w_k1_MAT_s003
                                 + 0.809262 w_k1_MSFT_s003
                                 + 0.854468 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.12897 w_k1_ADM_s004
                                 + 1.46636 w_k1_AXP_s004
                                 + 1.61305 w_k1_DIS_s004
                                 + 1.29722 w_k1_FDX_s004
                                 + 1.39145 w_k1_IBM_s004
                                 + 1.23518 w_k1_INTC_s004
                                 + 1.30962 w_k1_KO_s004 + 1.63467 w_k1_MAT_s004
                                 + 1.14451 w_k1_MSFT_s004
                                 + 1.15318 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.07188 w_k1_ADM_s004
                                 + 1.29556 w_k1_AXP_s004
                                 + 0.979971 w_k1_DIS_s004
                                 + 0.946869 w_k1_FDX_s004
                                 + 1.06704 w_k1_IBM_s004
                                 + 1.06882 w_k1_INTC_s004
                                 + 0.847396 w_k1_KO_s004
                                 + 1.05517 w_k1_MAT_s004
                                 + 0.823664 w_k1_MSFT_s004
                                 + 0.974713 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.15504 w_k1_ADM_s004
                                 + 1.41721 w_k1_AXP_s004
                                 + 1.29301 w_k1_DIS_s004
                                 + 1.26053 w_k1_FDX_s004
                                 + 0.853565 w_k1_IBM_s004
                                 + 1.19124 w_k1_INTC_s004
                                 + 0.893007 w_k1_KO_s004
                                 + 0.811323 w_k1_MAT_s004
                                 + 1.1431 w_k1_MSFT_s004 + 0.99871 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.31036 w_k1_ADM_s004
                                 + 1.25835 w_k1_AXP_s004
                                 + 0.995891 w_k1_DIS_s004
                                 + 1.0828 w_k1_FDX_s004 + 1.21414 w_k1_IBM_s004
                                 + 1.18641 w_k1_INTC_s004
                                 + 1.12074 w_k1_KO_s004 + 1.6003 w_k1_MAT_s004
                                 + 0.991103 w_k1_MSFT_s004
                                 + 0.945562 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.05994 w_k1_ADM_s004
                                 + 0.936629 w_k1_AXP_s004
                                 + 0.979112 w_k1_DIS_s004
                                 + 0.958375 w_k1_FDX_s004
                                 + 0.729431 w_k1_IBM_s004
                                 + 0.845597 w_k1_INTC_s004
                                 + 0.795571 w_k1_KO_s004
                                 + 0.626518 w_k1_MAT_s004
                                 + 0.917415 w_k1_MSFT_s004
                                 + 0.889986 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.961649 w_k1_ADM_s004
                                 + 0.858889 w_k1_AXP_s004
                                 + 0.772961 w_k1_DIS_s004
                                 + 0.841922 w_k1_FDX_s004
                                 + 0.9986 w_k1_IBM_s004
                                 + 1.02052 w_k1_INTC_s004
                                 + 0.828588 w_k1_KO_s004
                                 + 0.932893 w_k1_MAT_s004
                                 + 0.834811 w_k1_MSFT_s004
                                 + 0.809358 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.49663 w_k1_ADM_s004
                                 + 1.47905 w_k1_AXP_s004
                                 + 0.822198 w_k1_DIS_s004
                                 + 1.11042 w_k1_FDX_s004
                                 + 0.847002 w_k1_IBM_s004
                                 + 1.00064 w_k1_INTC_s004
                                 + 0.783191 w_k1_KO_s004
                                 + 0.727604 w_k1_MAT_s004
                                 + 0.938438 w_k1_MSFT_s004
                                 + 0.91418 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.986425 w_k1_ADM_s004
                                 + 1.01862 w_k1_AXP_s004
                                 + 1.22707 w_k1_DIS_s004 + 0.9511 w_k1_FDX_s004
                                 + 1.20672 w_k1_IBM_s004
                                 + 1.10818 w_k1_INTC_s004
                                 + 1.13592 w_k1_KO_s004 + 1.20253 w_k1_MAT_s004
                                 + 1.13545 w_k1_MSFT_s004
                                 + 1.04015 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.16576 w_k1_ADM_s005
                                 + 1.254 w_k1_AXP_s005 + 1.04323 w_k1_DIS_s005
                                 + 1.11211 w_k1_FDX_s005
                                 + 1.01152 w_k1_IBM_s005
                                 + 0.953605 w_k1_INTC_s005
                                 + 1.12246 w_k1_KO_s005 + 1.20203 w_k1_MAT_s005
                                 + 1.09705 w_k1_MSFT_s005
                                 + 1.05842 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.802239 w_k1_ADM_s005
                                 + 0.706507 w_k1_AXP_s005
                                 + 0.844547 w_k1_DIS_s005
                                 + 0.843168 w_k1_FDX_s005
                                 + 0.649435 w_k1_IBM_s005
                                 + 0.910625 w_k1_INTC_s005
                                 + 0.901301 w_k1_KO_s005
                                 + 0.671628 w_k1_MAT_s005
                                 + 0.993293 w_k1_MSFT_s005
                                 + 0.888955 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.856142 w_k1_ADM_s005
                                 + 0.622091 w_k1_AXP_s005
                                 + 0.544274 w_k1_DIS_s005
                                 + 0.670244 w_k1_FDX_s005
                                 + 0.786895 w_k1_IBM_s005
                                 + 0.794457 w_k1_INTC_s005
                                 + 0.812427 w_k1_KO_s005
                                 + 0.456287 w_k1_MAT_s005
                                 + 0.657551 w_k1_MSFT_s005
                                 + 0.826344 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.65494 w_k1_ADM_s005
                                 + 1.60394 w_k1_AXP_s005 + 1.6573 w_k1_DIS_s005
                                 + 1.37102 w_k1_FDX_s005
                                 + 1.15154 w_k1_IBM_s005
                                 + 1.06414 w_k1_INTC_s005
                                 + 1.43895 w_k1_KO_s005 + 1.44153 w_k1_MAT_s005
                                 + 1.31032 w_k1_MSFT_s005
                                 + 1.25989 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.937834 w_k1_ADM_s005
                                 + 1.19668 w_k1_AXP_s005
                                 + 1.00604 w_k1_DIS_s005
                                 + 0.917171 w_k1_FDX_s005
                                 + 1.56718 w_k1_IBM_s005
                                 + 0.93948 w_k1_INTC_s005
                                 + 1.26086 w_k1_KO_s005 + 1.47483 w_k1_MAT_s005
                                 + 0.897638 w_k1_MSFT_s005
                                 + 1.00992 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.645784 w_k1_ADM_s005
                                 + 0.77799 w_k1_AXP_s005
                                 + 0.695436 w_k1_DIS_s005
                                 + 0.767921 w_k1_FDX_s005
                                 + 0.994706 w_k1_IBM_s005
                                 + 0.784962 w_k1_INTC_s005
                                 + 0.957999 w_k1_KO_s005
                                 + 1.15638 w_k1_MAT_s005
                                 + 0.8032 w_k1_MSFT_s005
                                 + 0.860088 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.771362 w_k1_ADM_s005
                                 + 1.0336 w_k1_AXP_s005
                                 + 0.897989 w_k1_DIS_s005
                                 + 0.914381 w_k1_FDX_s005
                                 + 0.883881 w_k1_IBM_s005
                                 + 0.899497 w_k1_INTC_s005
                                 + 1.00383 w_k1_KO_s005
                                 + 0.974164 w_k1_MAT_s005
                                 + 0.994871 w_k1_MSFT_s005
                                 + 0.916297 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.09225 w_k1_ADM_s005
                                 + 1.09276 w_k1_AXP_s005
                                 + 1.16758 w_k1_DIS_s005
                                 + 1.02663 w_k1_FDX_s005
                                 + 1.13713 w_k1_IBM_s005
                                 + 1.07442 w_k1_INTC_s005
                                 + 1.21952 w_k1_KO_s005 + 1.15899 w_k1_MAT_s005
                                 + 1.18811 w_k1_MSFT_s005
                                 + 1.17435 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.06271 w_k1_ADM_s006
                                 + 0.930245 w_k1_AXP_s006
                                 + 0.813333 w_k1_DIS_s006
                                 + 0.97387 w_k1_FDX_s006
                                 + 0.698452 w_k1_IBM_s006
                                 + 1.03398 w_k1_INTC_s006
                                 + 0.816489 w_k1_KO_s006
                                 + 0.410869 w_k1_MAT_s006
                                 + 0.95022 w_k1_MSFT_s006
                                 + 0.844091 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.917925 w_k1_ADM_s006
                                 + 0.727481 w_k1_AXP_s006
                                 + 0.782934 w_k1_DIS_s006
                                 + 0.829294 w_k1_FDX_s006
                                 + 0.770886 w_k1_IBM_s006
                                 + 0.806145 w_k1_INTC_s006
                                 + 0.781789 w_k1_KO_s006
                                 + 0.554471 w_k1_MAT_s006
                                 + 0.85125 w_k1_MSFT_s006
                                 + 0.883961 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.76819 w_k1_ADM_s006
                                 + 1.50985 w_k1_AXP_s006
                                 + 1.91632 w_k1_DIS_s006
                                 + 1.37365 w_k1_FDX_s006
                                 + 1.41268 w_k1_IBM_s006
                                 + 1.3694 w_k1_INTC_s006 + 1.54653 w_k1_KO_s006
                                 + 1.59092 w_k1_MAT_s006
                                 + 1.41309 w_k1_MSFT_s006
                                 + 1.25464 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.35474 w_k1_ADM_s006
                                 + 1.2633 w_k1_AXP_s006 + 1.28597 w_k1_DIS_s006
                                 + 1.14958 w_k1_FDX_s006
                                 + 1.23344 w_k1_IBM_s006
                                 + 1.0507 w_k1_INTC_s006 + 1.07153 w_k1_KO_s006
                                 + 1.35236 w_k1_MAT_s006
                                 + 1.07484 w_k1_MSFT_s006
                                 + 1.06754 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.979856 w_k1_ADM_s006
                                 + 0.784111 w_k1_AXP_s006
                                 + 0.625323 w_k1_DIS_s006
                                 + 0.749925 w_k1_FDX_s006
                                 + 0.76296 w_k1_IBM_s006
                                 + 0.85398 w_k1_INTC_s006
                                 + 0.668212 w_k1_KO_s006
                                 + 0.566929 w_k1_MAT_s006
                                 + 0.716142 w_k1_MSFT_s006
                                 + 0.80446 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.99267 w_k1_ADM_s006
                                 + 1.09861 w_k1_AXP_s006
                                 + 0.869662 w_k1_DIS_s006
                                 + 0.901009 w_k1_FDX_s006
                                 + 1.20274 w_k1_IBM_s006
                                 + 1.0874 w_k1_INTC_s006 + 1.0443 w_k1_KO_s006
                                 + 1.22436 w_k1_MAT_s006
                                 + 0.847192 w_k1_MSFT_s006
                                 + 0.929383 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.02213 w_k1_ADM_s006
                                 + 0.825724 w_k1_AXP_s006
                                 + 1.12144 w_k1_DIS_s006
                                 + 0.973482 w_k1_FDX_s006
                                 + 0.91684 w_k1_IBM_s006
                                 + 1.06736 w_k1_INTC_s006
                                 + 1.03421 w_k1_KO_s006
                                 + 0.706498 w_k1_MAT_s006
                                 + 0.950911 w_k1_MSFT_s006
                                 + 0.927382 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.42143 w_k1_ADM_s006
                                 + 1.07891 w_k1_AXP_s006
                                 + 1.11514 w_k1_DIS_s006
                                 + 1.12526 w_k1_FDX_s006
                                 + 1.00118 w_k1_IBM_s006
                                 + 1.26586 w_k1_INTC_s006
                                 + 1.06432 w_k1_KO_s006
                                 + 0.868558 w_k1_MAT_s006
                                 + 1.00335 w_k1_MSFT_s006
                                 + 0.957283 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.25658 w_k1_ADM_s007
                                 + 1.65728 w_k1_AXP_s007
                                 + 1.76659 w_k1_DIS_s007
                                 + 1.51509 w_k1_FDX_s007
                                 + 1.18429 w_k1_IBM_s007
                                 + 1.34212 w_k1_INTC_s007
                                 + 1.17069 w_k1_KO_s007
                                 + 0.847424 w_k1_MAT_s007
                                 + 1.2712 w_k1_MSFT_s007 + 1.14804 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.01919 w_k1_ADM_s007
                                 + 1.45609 w_k1_AXP_s007
                                 + 1.39385 w_k1_DIS_s007
                                 + 1.12712 w_k1_FDX_s007
                                 + 1.35927 w_k1_IBM_s007
                                 + 0.934262 w_k1_INTC_s007
                                 + 1.16846 w_k1_KO_s007 + 1.66803 w_k1_MAT_s007
                                 + 0.943939 w_k1_MSFT_s007
                                 + 1.02382 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.987245 w_k1_ADM_s007
                                 + 1.57883 w_k1_AXP_s007
                                 + 1.15061 w_k1_DIS_s007
                                 + 1.32964 w_k1_FDX_s007
                                 + 0.846452 w_k1_IBM_s007
                                 + 1.17834 w_k1_INTC_s007
                                 + 0.810115 w_k1_KO_s007
                                 + 1.21629 w_k1_MAT_s007
                                 + 1.02772 w_k1_MSFT_s007
                                 + 0.967652 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.561667 w_k1_ADM_s007
                                 + 0.690262 w_k1_AXP_s007
                                 + 0.510969 w_k1_DIS_s007
                                 + 0.623672 w_k1_FDX_s007
                                 + 0.927482 w_k1_IBM_s007
                                 + 0.703849 w_k1_INTC_s007
                                 + 0.576399 w_k1_KO_s007
                                 + 0.665909 w_k1_MAT_s007
                                 + 0.668566 w_k1_MSFT_s007
                                 + 0.738309 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.910349 w_k1_ADM_s007
                                 + 1.05449 w_k1_AXP_s007
                                 + 0.95347 w_k1_DIS_s007
                                 + 1.00984 w_k1_FDX_s007
                                 + 0.772228 w_k1_IBM_s007
                                 + 0.823255 w_k1_INTC_s007
                                 + 0.763984 w_k1_KO_s007
                                 + 0.901876 w_k1_MAT_s007
                                 + 0.974198 w_k1_MSFT_s007
                                 + 0.971866 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.743025 w_k1_ADM_s007
                                 + 0.767236 w_k1_AXP_s007
                                 + 0.990799 w_k1_DIS_s007
                                 + 0.84657 w_k1_FDX_s007
                                 + 0.992838 w_k1_IBM_s007
                                 + 0.831754 w_k1_INTC_s007
                                 + 0.964283 w_k1_KO_s007
                                 + 0.809959 w_k1_MAT_s007
                                 + 0.906027 w_k1_MSFT_s007
                                 + 0.842623 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.0187 w_k1_ADM_s007
                                 + 1.33476 w_k1_AXP_s007
                                 + 1.48214 w_k1_DIS_s007
                                 + 1.26496 w_k1_FDX_s007
                                 + 0.977233 w_k1_IBM_s007
                                 + 1.19136 w_k1_INTC_s007
                                 + 1.17348 w_k1_KO_s007
                                 + 0.960351 w_k1_MAT_s007
                                 + 1.23958 w_k1_MSFT_s007
                                 + 1.13405 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.759691 w_k1_ADM_s007
                                 + 1.0812 w_k1_AXP_s007
                                 + 0.712031 w_k1_DIS_s007
                                 + 0.8293 w_k1_FDX_s007 + 1.02602 w_k1_IBM_s007
                                 + 0.807954 w_k1_INTC_s007
                                 + 0.925354 w_k1_KO_s007
                                 + 1.12614 w_k1_MAT_s007
                                 + 0.812136 w_k1_MSFT_s007
                                 + 0.85268 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.774232 w_k1_ADM_s010
                                 + 0.755989 w_k1_AXP_s010
                                 + 0.654363 w_k1_DIS_s010
                                 + 0.708546 w_k1_FDX_s010
                                 + 1.20962 w_k1_IBM_s010
                                 + 0.77078 w_k1_INTC_s010
                                 + 0.971207 w_k1_KO_s010 + 1.29 w_k1_MAT_s010
                                 + 0.830903 w_k1_MSFT_s010
                                 + 0.894039 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.918789 w_k1_ADM_s010
                                 + 0.922473 w_k1_AXP_s010
                                 + 0.678392 w_k1_DIS_s010
                                 + 0.860419 w_k1_FDX_s010
                                 + 0.856525 w_k1_IBM_s010
                                 + 0.785539 w_k1_INTC_s010
                                 + 0.788545 w_k1_KO_s010
                                 + 0.996267 w_k1_MAT_s010
                                 + 0.781559 w_k1_MSFT_s010
                                 + 0.779113 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.10592 w_k1_ADM_s010
                                 + 1.47235 w_k1_AXP_s010
                                 + 1.26775 w_k1_DIS_s010
                                 + 1.14712 w_k1_FDX_s010
                                 + 1.53041 w_k1_IBM_s010
                                 + 1.10093 w_k1_INTC_s010
                                 + 1.18807 w_k1_KO_s010 + 1.69539 w_k1_MAT_s010
                                 + 1.13032 w_k1_MSFT_s010
                                 + 1.14695 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.02672 w_k1_ADM_s010
                                 + 1.00616 w_k1_AXP_s010
                                 + 0.927626 w_k1_DIS_s010 + 1.013 w_k1_FDX_s010
                                 + 0.965372 w_k1_IBM_s010
                                 + 1.01227 w_k1_INTC_s010
                                 + 0.978788 w_k1_KO_s010
                                 + 1.32573 w_k1_MAT_s010
                                 + 1.15123 w_k1_MSFT_s010
                                 + 0.952808 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.52325 w_k1_ADM_s010
                                 + 1.68802 w_k1_AXP_s010
                                 + 1.68886 w_k1_DIS_s010 + 1.5046 w_k1_FDX_s010
                                 + 1.18755 w_k1_IBM_s010
                                 + 1.30029 w_k1_INTC_s010
                                 + 1.25053 w_k1_KO_s010 + 1.18728 w_k1_MAT_s010
                                 + 1.21235 w_k1_MSFT_s010 + 1.1575 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.752805 w_k1_ADM_s010
                                 + 0.617394 w_k1_AXP_s010
                                 + 0.508329 w_k1_DIS_s010
                                 + 0.648413 w_k1_FDX_s010
                                 + 0.644963 w_k1_IBM_s010
                                 + 0.70038 w_k1_INTC_s010
                                 + 0.679937 w_k1_KO_s010
                                 + 0.743204 w_k1_MAT_s010
                                 + 0.725417 w_k1_MSFT_s010
                                 + 0.690005 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.829107 w_k1_ADM_s010
                                 + 0.858853 w_k1_AXP_s010
                                 + 0.92259 w_k1_DIS_s010
                                 + 0.831157 w_k1_FDX_s010
                                 + 1.19399 w_k1_IBM_s010
                                 + 0.863927 w_k1_INTC_s010
                                 + 1.09344 w_k1_KO_s010 + 1.1819 w_k1_MAT_s010
                                 + 1.02353 w_k1_MSFT_s010
                                 + 0.950186 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.21621 w_k1_ADM_s010
                                 + 1.10858 w_k1_AXP_s010 + 1.2291 w_k1_DIS_s010
                                 + 1.20344 w_k1_FDX_s010
                                 + 0.88103 w_k1_IBM_s010
                                 + 1.17856 w_k1_INTC_s010
                                 + 1.09691 w_k1_KO_s010
                                 + 0.717222 w_k1_MAT_s010
                                 + 1.21095 w_k1_MSFT_s010
                                 + 1.03292 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.986064 w_k1_ADM_s011
                                 + 1.19997 w_k1_AXP_s011
                                 + 0.711181 w_k1_DIS_s011
                                 + 0.938212 w_k1_FDX_s011
                                 + 1.07963 w_k1_IBM_s011
                                 + 0.922036 w_k1_INTC_s011
                                 + 0.810351 w_k1_KO_s011
                                 + 1.24363 w_k1_MAT_s011
                                 + 0.879416 w_k1_MSFT_s011
                                 + 0.947485 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.22362 w_k1_ADM_s011
                                 + 1.2676 w_k1_AXP_s011 + 1.27251 w_k1_DIS_s011
                                 + 1.15999 w_k1_FDX_s011
                                 + 1.02399 w_k1_IBM_s011
                                 + 1.12522 w_k1_INTC_s011
                                 + 0.98891 w_k1_KO_s011 + 1.43916 w_k1_MAT_s011
                                 + 1.06956 w_k1_MSFT_s011
                                 + 1.07006 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.13127 w_k1_ADM_s011
                                 + 1.1614 w_k1_AXP_s011 + 1.17364 w_k1_DIS_s011
                                 + 0.957887 w_k1_FDX_s011
                                 + 1.49288 w_k1_IBM_s011
                                 + 1.02572 w_k1_INTC_s011
                                 + 1.18005 w_k1_KO_s011 + 1.36925 w_k1_MAT_s011
                                 + 0.991262 w_k1_MSFT_s011
                                 + 1.13507 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.955339 w_k1_ADM_s011
                                 + 0.794889 w_k1_AXP_s011
                                 + 0.714953 w_k1_DIS_s011
                                 + 0.845222 w_k1_FDX_s011
                                 + 0.64656 w_k1_IBM_s011
                                 + 0.823517 w_k1_INTC_s011
                                 + 0.765304 w_k1_KO_s011
                                 + 0.568884 w_k1_MAT_s011
                                 + 0.823429 w_k1_MSFT_s011
                                 + 0.787835 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.02611 w_k1_ADM_s011
                                 + 1.5157 w_k1_AXP_s011 + 1.42659 w_k1_DIS_s011
                                 + 1.08045 w_k1_FDX_s011 + 1.8396 w_k1_IBM_s011
                                 + 1.39788 w_k1_INTC_s011
                                 + 1.40973 w_k1_KO_s011 + 2.20728 w_k1_MAT_s011
                                 + 1.20327 w_k1_MSFT_s011 + 1.1072 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.20632 w_k1_ADM_s011
                                 + 1.25453 w_k1_AXP_s011
                                 + 1.03427 w_k1_DIS_s011
                                 + 1.12859 w_k1_FDX_s011
                                 + 0.754282 w_k1_IBM_s011
                                 + 1.01846 w_k1_INTC_s011
                                 + 0.86338 w_k1_KO_s011
                                 + 0.885264 w_k1_MAT_s011
                                 + 1.0515 w_k1_MSFT_s011 + 1.01052 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.91435 w_k1_ADM_s011
                                 + 1.58966 w_k1_AXP_s011 + 1.4611 w_k1_DIS_s011
                                 + 1.305 w_k1_FDX_s011 + 1.22057 w_k1_IBM_s011
                                 + 1.3605 w_k1_INTC_s011 + 1.14807 w_k1_KO_s011
                                 + 1.83844 w_k1_MAT_s011
                                 + 1.23439 w_k1_MSFT_s011
                                 + 1.02706 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.859234 w_k1_ADM_s011
                                 + 0.839118 w_k1_AXP_s011
                                 + 0.756453 w_k1_DIS_s011
                                 + 0.809841 w_k1_FDX_s011
                                 + 1.00402 w_k1_IBM_s011
                                 + 0.926569 w_k1_INTC_s011
                                 + 0.933628 w_k1_KO_s011
                                 + 0.863536 w_k1_MAT_s011
                                 + 0.764022 w_k1_MSFT_s011
                                 + 0.889009 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.11815 w_k1_ADM_s012
                                 + 1.06954 w_k1_AXP_s012
                                 + 1.32405 w_k1_DIS_s012
                                 + 1.26801 w_k1_FDX_s012
                                 + 0.87265 w_k1_IBM_s012
                                 + 1.28814 w_k1_INTC_s012 + 1.0869 w_k1_KO_s012
                                 + 0.896504 w_k1_MAT_s012
                                 + 1.17996 w_k1_MSFT_s012
                                 + 0.966941 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.710256 w_k1_ADM_s012
                                 + 0.875498 w_k1_AXP_s012
                                 + 0.824581 w_k1_DIS_s012
                                 + 0.744897 w_k1_FDX_s012
                                 + 1.6137 w_k1_IBM_s012
                                 + 1.02639 w_k1_INTC_s012
                                 + 1.10812 w_k1_KO_s012 + 1.41686 w_k1_MAT_s012
                                 + 0.891474 w_k1_MSFT_s012
                                 + 0.938928 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.11139 w_k1_ADM_s012
                                 + 1.287 w_k1_AXP_s012 + 1.22632 w_k1_DIS_s012
                                 + 1.09299 w_k1_FDX_s012
                                 + 1.31617 w_k1_IBM_s012
                                 + 1.34193 w_k1_INTC_s012
                                 + 1.11274 w_k1_KO_s012 + 1.45599 w_k1_MAT_s012
                                 + 1.03385 w_k1_MSFT_s012
                                 + 1.01083 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.05354 w_k1_ADM_s012
                                 + 1.6047 w_k1_AXP_s012 + 1.333 w_k1_DIS_s012
                                 + 1.16788 w_k1_FDX_s012
                                 + 1.58864 w_k1_IBM_s012
                                 + 1.24151 w_k1_INTC_s012
                                 + 1.28671 w_k1_KO_s012 + 2.2656 w_k1_MAT_s012
                                 + 1.04225 w_k1_MSFT_s012 + 1.1874 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.13951 w_k1_ADM_s012
                                 + 1.41016 w_k1_AXP_s012
                                 + 0.841992 w_k1_DIS_s012
                                 + 0.973867 w_k1_FDX_s012
                                 + 1.10563 w_k1_IBM_s012
                                 + 1.17102 w_k1_INTC_s012
                                 + 1.00131 w_k1_KO_s012 + 1.27867 w_k1_MAT_s012
                                 + 0.886056 w_k1_MSFT_s012
                                 + 0.962247 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.826874 w_k1_ADM_s012
                                 + 0.728859 w_k1_AXP_s012
                                 + 0.684657 w_k1_DIS_s012
                                 + 0.788446 w_k1_FDX_s012
                                 + 0.887662 w_k1_IBM_s012
                                 + 1.0425 w_k1_INTC_s012
                                 + 0.831543 w_k1_KO_s012
                                 + 0.933341 w_k1_MAT_s012
                                 + 0.808418 w_k1_MSFT_s012
                                 + 0.838497 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.843446 w_k1_ADM_s012
                                 + 0.744997 w_k1_AXP_s012
                                 + 0.765467 w_k1_DIS_s012
                                 + 0.90745 w_k1_FDX_s012
                                 + 0.576008 w_k1_IBM_s012
                                 + 1.0344 w_k1_INTC_s012
                                 + 0.710682 w_k1_KO_s012
                                 + 0.557448 w_k1_MAT_s012
                                 + 0.902746 w_k1_MSFT_s012
                                 + 0.796052 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.844715 w_k1_ADM_s012
                                 + 0.903511 w_k1_AXP_s012
                                 + 1.0482 w_k1_DIS_s012
                                 + 0.938338 w_k1_FDX_s012
                                 + 1.04982 w_k1_IBM_s012
                                 + 1.18749 w_k1_INTC_s012
                                 + 1.11505 w_k1_KO_s012 + 1.06295 w_k1_MAT_s012
                                 + 1.03305 w_k1_MSFT_s012
                                 + 0.938995 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.855571 w_k1_ADM_s013
                                 + 0.689199 w_k1_AXP_s013
                                 + 0.765477 w_k1_DIS_s013
                                 + 0.79459 w_k1_FDX_s013
                                 + 0.889937 w_k1_IBM_s013
                                 + 0.804542 w_k1_INTC_s013
                                 + 0.831187 w_k1_KO_s013
                                 + 0.991206 w_k1_MAT_s013
                                 + 0.90574 w_k1_MSFT_s013
                                 + 0.82761 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.42269 w_k1_ADM_s013
                                 + 1.7077 w_k1_AXP_s013 + 1.74812 w_k1_DIS_s013
                                 + 1.54444 w_k1_FDX_s013 + 1.2624 w_k1_IBM_s013
                                 + 1.72705 w_k1_INTC_s013
                                 + 1.51413 w_k1_KO_s013 + 1.65107 w_k1_MAT_s013
                                 + 1.57709 w_k1_MSFT_s013
                                 + 1.42899 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.03693 w_k1_ADM_s013
                                 + 1.05688 w_k1_AXP_s013
                                 + 0.837605 w_k1_DIS_s013
                                 + 1.0218 w_k1_FDX_s013
                                 + 0.722083 w_k1_IBM_s013
                                 + 0.84006 w_k1_INTC_s013
                                 + 0.833676 w_k1_KO_s013
                                 + 0.925025 w_k1_MAT_s013
                                 + 0.999051 w_k1_MSFT_s013
                                 + 0.982449 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.20143 w_k1_ADM_s013
                                 + 1.42051 w_k1_AXP_s013 + 0.9968 w_k1_DIS_s013
                                 + 1.02262 w_k1_FDX_s013
                                 + 1.37761 w_k1_IBM_s013
                                 + 1.1628 w_k1_INTC_s013 + 1.19129 w_k1_KO_s013
                                 + 1.38723 w_k1_MAT_s013
                                 + 1.07417 w_k1_MSFT_s013
                                 + 1.12306 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.2499 w_k1_ADM_s013
                                 + 1.22707 w_k1_AXP_s013
                                 + 1.45803 w_k1_DIS_s013
                                 + 1.11111 w_k1_FDX_s013
                                 + 1.54565 w_k1_IBM_s013
                                 + 1.31095 w_k1_INTC_s013
                                 + 1.44146 w_k1_KO_s013 + 1.39619 w_k1_MAT_s013
                                 + 1.28724 w_k1_MSFT_s013
                                 + 1.19417 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.773929 w_k1_ADM_s013
                                 + 0.91256 w_k1_AXP_s013
                                 + 1.01265 w_k1_DIS_s013
                                 + 0.961295 w_k1_FDX_s013
                                 + 1.03694 w_k1_IBM_s013
                                 + 1.08465 w_k1_INTC_s013
                                 + 1.06193 w_k1_KO_s013
                                 + 0.999605 w_k1_MAT_s013
                                 + 1.06686 w_k1_MSFT_s013
                                 + 1.03521 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.20273 w_k1_ADM_s013
                                 + 1.14401 w_k1_AXP_s013
                                 + 1.22977 w_k1_DIS_s013
                                 + 1.22795 w_k1_FDX_s013
                                 + 0.904941 w_k1_IBM_s013
                                 + 1.24256 w_k1_INTC_s013
                                 + 1.08714 w_k1_KO_s013 + 1.05311 w_k1_MAT_s013
                                 + 1.2519 w_k1_MSFT_s013 + 1.07877 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.76945 w_k1_ADM_s013
                                 + 0.947257 w_k1_AXP_s013
                                 + 0.58255 w_k1_DIS_s013
                                 + 0.759036 w_k1_FDX_s013
                                 + 0.689983 w_k1_IBM_s013
                                 + 0.961312 w_k1_INTC_s013
                                 + 0.740412 w_k1_KO_s013
                                 + 0.878938 w_k1_MAT_s013
                                 + 0.797279 w_k1_MSFT_s013
                                 + 0.831661 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.997618 w_k1_ADM_s014
                                 + 1.27889 w_k1_AXP_s014
                                 + 1.22161 w_k1_DIS_s014
                                 + 1.11255 w_k1_FDX_s014
                                 + 1.07879 w_k1_IBM_s014
                                 + 1.11441 w_k1_INTC_s014
                                 + 1.09311 w_k1_KO_s014 + 1.07463 w_k1_MAT_s014
                                 + 1.08884 w_k1_MSFT_s014
                                 + 1.03735 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.869997 w_k1_ADM_s014
                                 + 0.982279 w_k1_AXP_s014
                                 + 0.83152 w_k1_DIS_s014
                                 + 0.752294 w_k1_FDX_s014
                                 + 1.42215 w_k1_IBM_s014
                                 + 0.870063 w_k1_INTC_s014
                                 + 1.06002 w_k1_KO_s014 + 1.27298 w_k1_MAT_s014
                                 + 0.90216 w_k1_MSFT_s014
                                 + 0.987565 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.09134 w_k1_ADM_s014
                                 + 0.842684 w_k1_AXP_s014
                                 + 0.82589 w_k1_DIS_s014
                                 + 0.844926 w_k1_FDX_s014
                                 + 0.992924 w_k1_IBM_s014
                                 + 0.699302 w_k1_INTC_s014
                                 + 0.782569 w_k1_KO_s014
                                 + 0.925192 w_k1_MAT_s014
                                 + 0.814475 w_k1_MSFT_s014
                                 + 0.909487 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.904633 w_k1_ADM_s014
                                 + 0.779572 w_k1_AXP_s014
                                 + 0.721894 w_k1_DIS_s014
                                 + 0.833565 w_k1_FDX_s014
                                 + 0.581992 w_k1_IBM_s014
                                 + 0.794928 w_k1_INTC_s014
                                 + 0.60833 w_k1_KO_s014
                                 + 0.527427 w_k1_MAT_s014
                                 + 0.791032 w_k1_MSFT_s014
                                 + 0.888052 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.27985 w_k1_ADM_s014
                                 + 1.52392 w_k1_AXP_s014
                                 + 1.44161 w_k1_DIS_s014
                                 + 1.28606 w_k1_FDX_s014
                                 + 1.15039 w_k1_IBM_s014
                                 + 1.25026 w_k1_INTC_s014
                                 + 1.11131 w_k1_KO_s014 + 1.04172 w_k1_MAT_s014
                                 + 1.26965 w_k1_MSFT_s014
                                 + 1.20911 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.874865 w_k1_ADM_s014
                                 + 0.893115 w_k1_AXP_s014
                                 + 1.04468 w_k1_DIS_s014
                                 + 0.943351 w_k1_FDX_s014
                                 + 1.02242 w_k1_IBM_s014
                                 + 0.913023 w_k1_INTC_s014
                                 + 1.06498 w_k1_KO_s014
                                 + 0.953486 w_k1_MAT_s014
                                 + 0.890847 w_k1_MSFT_s014
                                 + 0.96739 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.12259 w_k1_ADM_s014
                                 + 0.916729 w_k1_AXP_s014
                                 + 0.947814 w_k1_DIS_s014
                                 + 0.926819 w_k1_FDX_s014
                                 + 0.736235 w_k1_IBM_s014
                                 + 0.845041 w_k1_INTC_s014
                                 + 0.866241 w_k1_KO_s014
                                 + 0.610817 w_k1_MAT_s014
                                 + 0.928079 w_k1_MSFT_s014
                                 + 0.921088 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.940878 w_k1_ADM_s014
                                 + 1.2477 w_k1_AXP_s014
                                 + 0.867051 w_k1_DIS_s014
                                 + 1.03226 w_k1_FDX_s014
                                 + 0.790286 w_k1_IBM_s014
                                 + 0.975627 w_k1_INTC_s014
                                 + 0.831807 w_k1_KO_s014
                                 + 1.03781 w_k1_MAT_s014
                                 + 0.898989 w_k1_MSFT_s014
                                 + 0.904346 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.759478 w_k1_ADM_s015
                                 + 0.812938 w_k1_AXP_s015
                                 + 0.694518 w_k1_DIS_s015
                                 + 0.770091 w_k1_FDX_s015
                                 + 0.867131 w_k1_IBM_s015
                                 + 0.77186 w_k1_INTC_s015
                                 + 0.808725 w_k1_KO_s015
                                 + 0.797966 w_k1_MAT_s015
                                 + 0.756348 w_k1_MSFT_s015
                                 + 0.878001 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.987772 w_k1_ADM_s015
                                 + 1.28961 w_k1_AXP_s015
                                 + 1.06417 w_k1_DIS_s015
                                 + 1.02217 w_k1_FDX_s015 + 1.0612 w_k1_IBM_s015
                                 + 0.986473 w_k1_INTC_s015
                                 + 1.07033 w_k1_KO_s015 + 1.1235 w_k1_MAT_s015
                                 + 0.995526 w_k1_MSFT_s015
                                 + 1.12267 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.991478 w_k1_ADM_s015
                                 + 1.01964 w_k1_AXP_s015
                                 + 1.08032 w_k1_DIS_s015
                                 + 1.01481 w_k1_FDX_s015
                                 + 0.880963 w_k1_IBM_s015
                                 + 1.16891 w_k1_INTC_s015
                                 + 0.944515 w_k1_KO_s015
                                 + 0.850755 w_k1_MAT_s015
                                 + 1.08248 w_k1_MSFT_s015
                                 + 1.15148 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.757934 w_k1_ADM_s015
                                 + 0.865247 w_k1_AXP_s015
                                 + 1.05235 w_k1_DIS_s015
                                 + 0.862034 w_k1_FDX_s015
                                 + 1.08613 w_k1_IBM_s015
                                 + 1.26254 w_k1_INTC_s015
                                 + 1.07233 w_k1_KO_s015
                                 + 0.883624 w_k1_MAT_s015
                                 + 1.09628 w_k1_MSFT_s015
                                 + 1.04487 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.953576 w_k1_ADM_s015
                                 + 1.09635 w_k1_AXP_s015
                                 + 0.787312 w_k1_DIS_s015
                                 + 0.999153 w_k1_FDX_s015
                                 + 0.75216 w_k1_IBM_s015
                                 + 0.907242 w_k1_INTC_s015
                                 + 0.784798 w_k1_KO_s015
                                 + 0.982374 w_k1_MAT_s015
                                 + 0.866528 w_k1_MSFT_s015
                                 + 0.900686 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.696835 w_k1_ADM_s015
                                 + 1.06498 w_k1_AXP_s015
                                 + 0.980951 w_k1_DIS_s015
                                 + 0.78272 w_k1_FDX_s015
                                 + 1.58509 w_k1_IBM_s015
                                 + 1.25233 w_k1_INTC_s015
                                 + 1.23772 w_k1_KO_s015 + 1.5586 w_k1_MAT_s015
                                 + 0.955134 w_k1_MSFT_s015
                                 + 1.25801 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.02455 w_k1_ADM_s015
                                 + 1.5261 w_k1_AXP_s015 + 1.47503 w_k1_DIS_s015
                                 + 1.25009 w_k1_FDX_s015
                                 + 1.27999 w_k1_IBM_s015
                                 + 1.43729 w_k1_INTC_s015
                                 + 1.24993 w_k1_KO_s015 + 1.16305 w_k1_MAT_s015
                                 + 1.15624 w_k1_MSFT_s015
                                 + 1.32758 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.840112 w_k1_ADM_s015
                                 + 0.683609 w_k1_AXP_s015
                                 + 0.751507 w_k1_DIS_s015
                                 + 0.863786 w_k1_FDX_s015
                                 + 0.613898 w_k1_IBM_s015
                                 + 1.04248 w_k1_INTC_s015
                                 + 0.689595 w_k1_KO_s015
                                 + 0.546611 w_k1_MAT_s015
                                 + 0.972058 w_k1_MSFT_s015
                                 + 0.950948 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.16317 w_k1_ADM_s016
                                 + 1.13475 w_k1_AXP_s016
                                 + 1.34618 w_k1_DIS_s016
                                 + 1.16019 w_k1_FDX_s016
                                 + 1.31036 w_k1_IBM_s016
                                 + 1.10678 w_k1_INTC_s016
                                 + 1.24195 w_k1_KO_s016 + 1.43778 w_k1_MAT_s016
                                 + 1.21924 w_k1_MSFT_s016 + 1.0806 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.904937 w_k1_ADM_s016
                                 + 0.915541 w_k1_AXP_s016
                                 + 1.60538 w_k1_DIS_s016
                                 + 1.22062 w_k1_FDX_s016
                                 + 0.922532 w_k1_IBM_s016
                                 + 0.911204 w_k1_INTC_s016
                                 + 1.03456 w_k1_KO_s016
                                 + 0.699763 w_k1_MAT_s016
                                 + 1.12522 w_k1_MSFT_s016
                                 + 1.02911 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.22446 w_k1_ADM_s016
                                 + 1.08059 w_k1_AXP_s016
                                 + 0.854395 w_k1_DIS_s016
                                 + 0.999249 w_k1_FDX_s016
                                 + 1.05412 w_k1_IBM_s016
                                 + 0.800227 w_k1_INTC_s016
                                 + 0.941881 w_k1_KO_s016
                                 + 1.15768 w_k1_MAT_s016
                                 + 0.8825 w_k1_MSFT_s016 + 1.01224 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.32113 w_k1_ADM_s016
                                 + 1.74161 w_k1_AXP_s016
                                 + 1.56943 w_k1_DIS_s016
                                 + 1.50561 w_k1_FDX_s016
                                 + 1.07422 w_k1_IBM_s016
                                 + 1.12525 w_k1_INTC_s016
                                 + 1.14342 w_k1_KO_s016 + 1.38458 w_k1_MAT_s016
                                 + 1.15329 w_k1_MSFT_s016
                                 + 1.24434 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.09914 w_k1_ADM_s016
                                 + 1.2066 w_k1_AXP_s016 + 1.04827 w_k1_DIS_s016
                                 + 0.968911 w_k1_FDX_s016
                                 + 1.89969 w_k1_IBM_s016
                                 + 0.731008 w_k1_INTC_s016
                                 + 1.09894 w_k1_KO_s016 + 2.35119 w_k1_MAT_s016
                                 + 0.843688 w_k1_MSFT_s016
                                 + 1.12767 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.856138 w_k1_ADM_s016
                                 + 0.66797 w_k1_AXP_s016
                                 + 0.707287 w_k1_DIS_s016
                                 + 0.7452 w_k1_FDX_s016
                                 + 0.984821 w_k1_IBM_s016
                                 + 0.761367 w_k1_INTC_s016
                                 + 0.85503 w_k1_KO_s016 + 0.92958 w_k1_MAT_s016
                                 + 0.825178 w_k1_MSFT_s016
                                 + 0.944264 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.955685 w_k1_ADM_s016
                                 + 0.847228 w_k1_AXP_s016
                                 + 0.961631 w_k1_DIS_s016
                                 + 1.0545 w_k1_FDX_s016
                                 + 0.631974 w_k1_IBM_s016
                                 + 0.814642 w_k1_INTC_s016
                                 + 0.727419 w_k1_KO_s016
                                 + 0.566573 w_k1_MAT_s016
                                 + 0.993077 w_k1_MSFT_s016
                                 + 0.999971 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.40843 w_k1_ADM_s016
                                 + 1.38849 w_k1_AXP_s016
                                 + 1.20762 w_k1_DIS_s016
                                 + 1.27031 w_k1_FDX_s016
                                 + 0.777325 w_k1_IBM_s016
                                 + 1.19468 w_k1_INTC_s016
                                 + 0.94959 w_k1_KO_s016
                                 + 0.721153 w_k1_MAT_s016
                                 + 1.11752 w_k1_MSFT_s016
                                 + 1.16084 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.897381 w_k1_ADM_s017
                                 + 1.22034 w_k1_AXP_s017
                                 + 1.22057 w_k1_DIS_s017
                                 + 1.11304 w_k1_FDX_s017
                                 + 1.03761 w_k1_IBM_s017
                                 + 1.17162 w_k1_INTC_s017
                                 + 1.06228 w_k1_KO_s017 + 1.41406 w_k1_MAT_s017
                                 + 1.03937 w_k1_MSFT_s017
                                 + 1.02999 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.01636 w_k1_ADM_s017
                                 + 0.788053 w_k1_AXP_s017
                                 + 0.807589 w_k1_DIS_s017
                                 + 0.855442 w_k1_FDX_s017
                                 + 0.736736 w_k1_IBM_s017
                                 + 0.78611 w_k1_INTC_s017
                                 + 0.84042 w_k1_KO_s017 + 0.7178 w_k1_MAT_s017
                                 + 0.86788 w_k1_MSFT_s017
                                 + 0.873422 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.761102 w_k1_ADM_s017
                                 + 0.713435 w_k1_AXP_s017
                                 + 0.668659 w_k1_DIS_s017
                                 + 0.705095 w_k1_FDX_s017
                                 + 1.1572 w_k1_IBM_s017
                                 + 0.76994 w_k1_INTC_s017
                                 + 0.926936 w_k1_KO_s017
                                 + 0.801989 w_k1_MAT_s017
                                 + 0.677232 w_k1_MSFT_s017
                                 + 0.855651 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.01294 w_k1_ADM_s017
                                 + 1.14551 w_k1_AXP_s017
                                 + 0.775181 w_k1_DIS_s017
                                 + 0.918298 w_k1_FDX_s017
                                 + 0.965689 w_k1_IBM_s017
                                 + 0.732764 w_k1_INTC_s017
                                 + 0.851291 w_k1_KO_s017
                                 + 1.39154 w_k1_MAT_s017
                                 + 0.859303 w_k1_MSFT_s017
                                 + 1.03357 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.711071 w_k1_ADM_s017
                                 + 1.00621 w_k1_AXP_s017
                                 + 0.98654 w_k1_DIS_s017
                                 + 0.898116 w_k1_FDX_s017
                                 + 1.43205 w_k1_IBM_s017
                                 + 1.17689 w_k1_INTC_s017
                                 + 1.16099 w_k1_KO_s017 + 1.38908 w_k1_MAT_s017
                                 + 0.857493 w_k1_MSFT_s017
                                 + 1.00646 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.05096 w_k1_ADM_s017
                                 + 0.831335 w_k1_AXP_s017
                                 + 1.15221 w_k1_DIS_s017
                                 + 1.10638 w_k1_FDX_s017
                                 + 0.952163 w_k1_IBM_s017
                                 + 1.08052 w_k1_INTC_s017
                                 + 0.998871 w_k1_KO_s017
                                 + 0.721386 w_k1_MAT_s017 + 1.02 w_k1_MSFT_s017
                                 + 1.02266 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.27008 w_k1_ADM_s017
                                 + 0.893558 w_k1_AXP_s017
                                 + 1.24646 w_k1_DIS_s017 + 1.0869 w_k1_FDX_s017
                                 + 0.664647 w_k1_IBM_s017
                                 + 0.961049 w_k1_INTC_s017
                                 + 0.919794 w_k1_KO_s017
                                 + 0.496083 w_k1_MAT_s017
                                 + 1.06671 w_k1_MSFT_s017
                                 + 1.01997 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.29146 w_k1_ADM_s017
                                 + 1.43098 w_k1_AXP_s017
                                 + 1.54881 w_k1_DIS_s017
                                 + 1.28387 w_k1_FDX_s017
                                 + 1.36561 w_k1_IBM_s017
                                 + 1.22986 w_k1_INTC_s017
                                 + 1.28053 w_k1_KO_s017 + 1.41685 w_k1_MAT_s017
                                 + 1.19275 w_k1_MSFT_s017
                                 + 1.25171 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.30452 w_k1_ADM_s100
                                 + 0.990195 w_k1_AXP_s100
                                 + 1.29216 w_k1_DIS_s100
                                 + 1.10788 w_k1_FDX_s100
                                 + 0.999207 w_k1_IBM_s100
                                 + 1.01926 w_k1_INTC_s100
                                 + 1.13942 w_k1_KO_s100 + 1.12047 w_k1_MAT_s100
                                 + 1.18309 w_k1_MSFT_s100
                                 + 1.05909 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.860489 w_k1_ADM_s100
                                 + 0.940369 w_k1_AXP_s100
                                 + 0.857465 w_k1_DIS_s100
                                 + 0.96248 w_k1_FDX_s100
                                 + 0.860318 w_k1_IBM_s100
                                 + 1.00168 w_k1_INTC_s100
                                 + 0.946594 w_k1_KO_s100 + 1.0439 w_k1_MAT_s100
                                 + 1.00527 w_k1_MSFT_s100
                                 + 0.951111 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.06666 w_k1_ADM_s100
                                 + 1.1523 w_k1_AXP_s100 + 1.02146 w_k1_DIS_s100
                                 + 1.11241 w_k1_FDX_s100
                                 + 0.677997 w_k1_IBM_s100
                                 + 0.824636 w_k1_INTC_s100
                                 + 0.8764 w_k1_KO_s100 + 0.895779 w_k1_MAT_s100
                                 + 0.988658 w_k1_MSFT_s100
                                 + 0.949029 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.74211 w_k1_ADM_s100
                                 + 0.973431 w_k1_AXP_s100
                                 + 0.957834 w_k1_DIS_s100
                                 + 0.856502 w_k1_FDX_s100
                                 + 1.14317 w_k1_IBM_s100
                                 + 0.946405 w_k1_INTC_s100
                                 + 1.07377 w_k1_KO_s100 + 1.03348 w_k1_MAT_s100
                                 + 0.842141 w_k1_MSFT_s100
                                 + 0.997012 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.710052 w_k1_ADM_s100
                                 + 0.721569 w_k1_AXP_s100
                                 + 0.539012 w_k1_DIS_s100
                                 + 0.650504 w_k1_FDX_s100
                                 + 0.991577 w_k1_IBM_s100
                                 + 0.714408 w_k1_INTC_s100
                                 + 0.794631 w_k1_KO_s100
                                 + 1.24322 w_k1_MAT_s100
                                 + 0.825227 w_k1_MSFT_s100
                                 + 0.805549 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.13168 w_k1_ADM_s100
                                 + 1.11025 w_k1_AXP_s100
                                 + 1.44165 w_k1_DIS_s100
                                 + 1.23236 w_k1_FDX_s100
                                 + 0.930857 w_k1_IBM_s100
                                 + 1.16031 w_k1_INTC_s100
                                 + 1.08146 w_k1_KO_s100 + 0.64792 w_k1_MAT_s100
                                 + 1.12647 w_k1_MSFT_s100
                                 + 0.966517 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.907836 w_k1_ADM_s100
                                 + 1.36609 w_k1_AXP_s100
                                 + 1.36107 w_k1_DIS_s100
                                 + 1.16168 w_k1_FDX_s100
                                 + 1.42825 w_k1_IBM_s100
                                 + 1.20865 w_k1_INTC_s100
                                 + 1.30054 w_k1_KO_s100 + 1.95457 w_k1_MAT_s100
                                 + 1.08682 w_k1_MSFT_s100
                                 + 1.01036 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.948841 w_k1_ADM_s100
                                 + 0.647163 w_k1_AXP_s100
                                 + 0.690722 w_k1_DIS_s100
                                 + 0.859178 w_k1_FDX_s100
                                 + 0.67927 w_k1_IBM_s100
                                 + 0.842483 w_k1_INTC_s100
                                 + 0.730237 w_k1_KO_s100
                                 + 0.583419 w_k1_MAT_s100
                                 + 0.936533 w_k1_MSFT_s100
                                 + 0.8076 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.766387 w_k1_ADM_s101
                                 + 0.992559 w_k1_AXP_s101
                                 + 0.654968 w_k1_DIS_s101
                                 + 0.746248 w_k1_FDX_s101
                                 + 0.921409 w_k1_IBM_s101
                                 + 0.820409 w_k1_INTC_s101
                                 + 0.910348 w_k1_KO_s101
                                 + 1.06038 w_k1_MAT_s101
                                 + 0.802615 w_k1_MSFT_s101
                                 + 0.864392 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.14695 w_k1_ADM_s101
                                 + 1.30654 w_k1_AXP_s101
                                 + 1.35495 w_k1_DIS_s101
                                 + 1.06918 w_k1_FDX_s101 + 1.0343 w_k1_IBM_s101
                                 + 1.28408 w_k1_INTC_s101
                                 + 1.29194 w_k1_KO_s101 + 1.23001 w_k1_MAT_s101
                                 + 1.27787 w_k1_MSFT_s101
                                 + 1.18513 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.25442 w_k1_ADM_s101
                                 + 1.16412 w_k1_AXP_s101 + 1.3339 w_k1_DIS_s101
                                 + 1.14726 w_k1_FDX_s101
                                 + 1.41648 w_k1_IBM_s101
                                 + 1.35474 w_k1_INTC_s101
                                 + 1.50994 w_k1_KO_s101 + 1.24054 w_k1_MAT_s101
                                 + 1.31729 w_k1_MSFT_s101
                                 + 1.11416 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.869816 w_k1_ADM_s101
                                 + 0.881882 w_k1_AXP_s101
                                 + 0.906741 w_k1_DIS_s101
                                 + 0.897688 w_k1_FDX_s101
                                 + 0.923639 w_k1_IBM_s101
                                 + 0.886748 w_k1_INTC_s101
                                 + 1.00282 w_k1_KO_s101
                                 + 0.952703 w_k1_MAT_s101
                                 + 0.976289 w_k1_MSFT_s101
                                 + 0.889228 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.00551 w_k1_ADM_s101
                                 + 0.937149 w_k1_AXP_s101
                                 + 1.11295 w_k1_DIS_s101 + 1.1069 w_k1_FDX_s101
                                 + 0.739084 w_k1_IBM_s101
                                 + 1.0973 w_k1_INTC_s101
                                 + 0.993274 w_k1_KO_s101
                                 + 0.825808 w_k1_MAT_s101
                                 + 1.05284 w_k1_MSFT_s101
                                 + 0.944575 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.654231 w_k1_ADM_s101
                                 + 0.600424 w_k1_AXP_s101
                                 + 0.591003 w_k1_DIS_s101
                                 + 0.610069 w_k1_FDX_s101
                                 + 0.71725 w_k1_IBM_s101
                                 + 0.905089 w_k1_INTC_s101
                                 + 0.784037 w_k1_KO_s101
                                 + 0.613853 w_k1_MAT_s101
                                 + 0.737633 w_k1_MSFT_s101
                                 + 0.767191 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.818798 w_k1_ADM_s101
                                 + 0.798112 w_k1_AXP_s101
                                 + 0.708845 w_k1_DIS_s101
                                 + 0.656725 w_k1_FDX_s101
                                 + 1.29808 w_k1_IBM_s101
                                 + 0.817114 w_k1_INTC_s101
                                 + 1.18632 w_k1_KO_s101 + 1.1525 w_k1_MAT_s101
                                 + 0.804764 w_k1_MSFT_s101
                                 + 0.830426 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.46193 w_k1_ADM_s101
                                 + 1.33517 w_k1_AXP_s101
                                 + 1.43052 w_k1_DIS_s101
                                 + 1.23405 w_k1_FDX_s101
                                 + 0.903135 w_k1_IBM_s101
                                 + 1.59569 w_k1_INTC_s101
                                 + 1.23288 w_k1_KO_s101
                                 + 0.793153 w_k1_MAT_s101
                                 + 1.38151 w_k1_MSFT_s101
                                 + 0.923413 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.17151 w_k1_ADM_s102
                                 + 1.02027 w_k1_AXP_s102
                                 + 0.743031 w_k1_DIS_s102
                                 + 0.930516 w_k1_FDX_s102
                                 + 0.841812 w_k1_IBM_s102
                                 + 0.763852 w_k1_INTC_s102
                                 + 0.831056 w_k1_KO_s102
                                 + 0.693422 w_k1_MAT_s102
                                 + 0.759717 w_k1_MSFT_s102
                                 + 0.822562 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.898309 w_k1_ADM_s102
                                 + 0.922139 w_k1_AXP_s102
                                 + 0.971604 w_k1_DIS_s102
                                 + 0.983705 w_k1_FDX_s102
                                 + 0.869309 w_k1_IBM_s102
                                 + 1.095 w_k1_INTC_s102 + 0.922572 w_k1_KO_s102
                                 + 0.738142 w_k1_MAT_s102
                                 + 1.00551 w_k1_MSFT_s102
                                 + 1.06545 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.41348 w_k1_ADM_s102
                                 + 1.0805 w_k1_AXP_s102 + 1.47903 w_k1_DIS_s102
                                 + 1.26008 w_k1_FDX_s102
                                 + 0.851292 w_k1_IBM_s102
                                 + 1.0278 w_k1_INTC_s102 + 1.12209 w_k1_KO_s102
                                 + 0.752093 w_k1_MAT_s102
                                 + 1.12182 w_k1_MSFT_s102
                                 + 1.03421 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.31921 w_k1_ADM_s102
                                 + 1.20419 w_k1_AXP_s102
                                 + 1.01063 w_k1_DIS_s102
                                 + 1.13509 w_k1_FDX_s102
                                 + 0.816907 w_k1_IBM_s102
                                 + 1.19649 w_k1_INTC_s102
                                 + 1.03785 w_k1_KO_s102
                                 + 0.893618 w_k1_MAT_s102
                                 + 1.19114 w_k1_MSFT_s102
                                 + 1.05779 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.09279 w_k1_ADM_s102
                                 + 0.820757 w_k1_AXP_s102
                                 + 1.00421 w_k1_DIS_s102
                                 + 0.796542 w_k1_FDX_s102
                                 + 1.4766 w_k1_IBM_s102
                                 + 1.03083 w_k1_INTC_s102
                                 + 1.21787 w_k1_KO_s102
                                 + 0.917738 w_k1_MAT_s102
                                 + 0.93662 w_k1_MSFT_s102
                                 + 0.932653 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05488 w_k1_ADM_s102
                                 + 0.655861 w_k1_AXP_s102
                                 + 0.72671 w_k1_DIS_s102
                                 + 0.851003 w_k1_FDX_s102
                                 + 0.649198 w_k1_IBM_s102
                                 + 0.757458 w_k1_INTC_s102 + 0.663 w_k1_KO_s102
                                 + 0.455996 w_k1_MAT_s102
                                 + 0.816831 w_k1_MSFT_s102
                                 + 0.814453 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.61382 w_k1_ADM_s102
                                 + 1.11936 w_k1_AXP_s102
                                 + 1.39221 w_k1_DIS_s102
                                 + 1.22689 w_k1_FDX_s102 + 1.3299 w_k1_IBM_s102
                                 + 1.27368 w_k1_INTC_s102
                                 + 1.47794 w_k1_KO_s102 + 1.32705 w_k1_MAT_s102
                                 + 1.18953 w_k1_MSFT_s102
                                 + 1.23555 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.0713 w_k1_ADM_s102
                                 + 0.56644 w_k1_AXP_s102
                                 + 0.732346 w_k1_DIS_s102
                                 + 0.754597 w_k1_FDX_s102
                                 + 0.947954 w_k1_IBM_s102
                                 + 0.743948 w_k1_INTC_s102
                                 + 0.856157 w_k1_KO_s102
                                 + 0.577196 w_k1_MAT_s102
                                 + 0.733821 w_k1_MSFT_s102
                                 + 0.866211 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.771257 w_k1_ADM_s103
                                 + 1.04499 w_k1_AXP_s103
                                 + 1.30481 w_k1_DIS_s103 + 1.1775 w_k1_FDX_s103
                                 + 0.894755 w_k1_IBM_s103
                                 + 1.34011 w_k1_INTC_s103
                                 + 0.902009 w_k1_KO_s103
                                 + 0.754131 w_k1_MAT_s103
                                 + 1.16883 w_k1_MSFT_s103
                                 + 0.96536 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.29938 w_k1_ADM_s103
                                 + 1.84555 w_k1_AXP_s103
                                 + 2.04612 w_k1_DIS_s103
                                 + 1.68227 w_k1_FDX_s103
                                 + 1.18558 w_k1_IBM_s103
                                 + 1.64506 w_k1_INTC_s103
                                 + 1.24488 w_k1_KO_s103
                                 + 0.876194 w_k1_MAT_s103
                                 + 1.53076 w_k1_MSFT_s103
                                 + 1.09828 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.03351 w_k1_ADM_s103
                                 + 0.697119 w_k1_AXP_s103
                                 + 0.697148 w_k1_DIS_s103
                                 + 0.794705 w_k1_FDX_s103
                                 + 0.821939 w_k1_IBM_s103
                                 + 0.750004 w_k1_INTC_s103
                                 + 0.712936 w_k1_KO_s103
                                 + 0.61774 w_k1_MAT_s103
                                 + 0.878945 w_k1_MSFT_s103
                                 + 0.83758 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.05643 w_k1_ADM_s103
                                 + 0.800972 w_k1_AXP_s103
                                 + 1.05461 w_k1_DIS_s103
                                 + 1.13473 w_k1_FDX_s103
                                 + 0.616772 w_k1_IBM_s103
                                 + 1.03426 w_k1_INTC_s103
                                 + 0.753719 w_k1_KO_s103
                                 + 0.398535 w_k1_MAT_s103
                                 + 1.12636 w_k1_MSFT_s103
                                 + 0.87942 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.15139 w_k1_ADM_s103
                                 + 1.00641 w_k1_AXP_s103
                                 + 1.46418 w_k1_DIS_s103
                                 + 1.26621 w_k1_FDX_s103
                                 + 1.01203 w_k1_IBM_s103
                                 + 1.10911 w_k1_INTC_s103
                                 + 1.23763 w_k1_KO_s103
                                 + 0.800397 w_k1_MAT_s103
                                 + 1.33701 w_k1_MSFT_s103
                                 + 1.14343 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.941664 w_k1_ADM_s103
                                 + 1.57848 w_k1_AXP_s103
                                 + 0.986638 w_k1_DIS_s103
                                 + 1.22886 w_k1_FDX_s103
                                 + 1.02949 w_k1_IBM_s103
                                 + 1.23157 w_k1_INTC_s103
                                 + 0.912493 w_k1_KO_s103
                                 + 0.927987 w_k1_MAT_s103
                                 + 1.16997 w_k1_MSFT_s103
                                 + 0.925958 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.752145 w_k1_ADM_s103
                                 + 1.12077 w_k1_AXP_s103
                                 + 1.14623 w_k1_DIS_s103
                                 + 0.97861 w_k1_FDX_s103
                                 + 1.52714 w_k1_IBM_s103
                                 + 0.998447 w_k1_INTC_s103
                                 + 1.06039 w_k1_KO_s103 + 1.37007 w_k1_MAT_s103
                                 + 1.11185 w_k1_MSFT_s103
                                 + 0.969452 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.806095 w_k1_ADM_s103
                                 + 0.896261 w_k1_AXP_s103
                                 + 0.886385 w_k1_DIS_s103
                                 + 0.816309 w_k1_FDX_s103
                                 + 1.1138 w_k1_IBM_s103
                                 + 1.03731 w_k1_INTC_s103
                                 + 0.78439 w_k1_KO_s103 + 0.60641 w_k1_MAT_s103
                                 + 0.882029 w_k1_MSFT_s103
                                 + 0.813161 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.942779 w_k1_ADM_s104
                                 + 1.00247 w_k1_AXP_s104
                                 + 0.864725 w_k1_DIS_s104
                                 + 0.863564 w_k1_FDX_s104
                                 + 1.19485 w_k1_IBM_s104
                                 + 1.16256 w_k1_INTC_s104
                                 + 1.12972 w_k1_KO_s104 + 1.09829 w_k1_MAT_s104
                                 + 0.990496 w_k1_MSFT_s104
                                 + 0.997103 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.920445 w_k1_ADM_s104
                                 + 0.959484 w_k1_AXP_s104
                                 + 1.07879 w_k1_DIS_s104
                                 + 1.06172 w_k1_FDX_s104
                                 + 0.939809 w_k1_IBM_s104
                                 + 1.05197 w_k1_INTC_s104
                                 + 1.05481 w_k1_KO_s104 + 1.09379 w_k1_MAT_s104
                                 + 1.0919 w_k1_MSFT_s104
                                 + 0.977002 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.3806 w_k1_ADM_s104
                                 + 1.11453 w_k1_AXP_s104
                                 + 0.831385 w_k1_DIS_s104
                                 + 1.02158 w_k1_FDX_s104
                                 + 0.909326 w_k1_IBM_s104
                                 + 0.994768 w_k1_INTC_s104
                                 + 0.870861 w_k1_KO_s104
                                 + 0.859211 w_k1_MAT_s104
                                 + 1.09009 w_k1_MSFT_s104
                                 + 1.06875 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.38856 w_k1_ADM_s104
                                 + 1.33071 w_k1_AXP_s104
                                 + 1.45257 w_k1_DIS_s104 + 1.3618 w_k1_FDX_s104
                                 + 0.926781 w_k1_IBM_s104
                                 + 1.35016 w_k1_INTC_s104
                                 + 1.17659 w_k1_KO_s104 + 0.99932 w_k1_MAT_s104
                                 + 1.23718 w_k1_MSFT_s104
                                 + 1.20868 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.868569 w_k1_ADM_s104
                                 + 0.638587 w_k1_AXP_s104
                                 + 0.774353 w_k1_DIS_s104
                                 + 0.722114 w_k1_FDX_s104
                                 + 0.826225 w_k1_IBM_s104
                                 + 0.88634 w_k1_INTC_s104
                                 + 0.852686 w_k1_KO_s104
                                 + 0.583278 w_k1_MAT_s104
                                 + 0.92253 w_k1_MSFT_s104
                                 + 0.820328 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.964612 w_k1_ADM_s104
                                 + 0.980087 w_k1_AXP_s104
                                 + 0.697007 w_k1_DIS_s104
                                 + 0.86298 w_k1_FDX_s104
                                 + 1.00504 w_k1_IBM_s104
                                 + 0.699784 w_k1_INTC_s104
                                 + 0.826296 w_k1_KO_s104
                                 + 1.01369 w_k1_MAT_s104
                                 + 0.85111 w_k1_MSFT_s104
                                 + 0.898425 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.17948 w_k1_ADM_s104
                                 + 0.908637 w_k1_AXP_s104
                                 + 0.842406 w_k1_DIS_s104
                                 + 0.886397 w_k1_FDX_s104
                                 + 0.687043 w_k1_IBM_s104
                                 + 1.00641 w_k1_INTC_s104
                                 + 0.871053 w_k1_KO_s104
                                 + 0.586403 w_k1_MAT_s104
                                 + 1.04478 w_k1_MSFT_s104
                                 + 0.905845 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.4508 w_k1_ADM_s104
                                 + 1.1943 w_k1_AXP_s104
                                 + 0.994997 w_k1_DIS_s104
                                 + 0.942037 w_k1_FDX_s104
                                 + 1.46794 w_k1_IBM_s104
                                 + 0.920516 w_k1_INTC_s104
                                 + 1.27948 w_k1_KO_s104 + 1.28668 w_k1_MAT_s104
                                 + 0.944483 w_k1_MSFT_s104
                                 + 1.1203 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.46182 w_k1_ADM_s105
                                 + 1.79084 w_k1_AXP_s105
                                 + 1.46391 w_k1_DIS_s105
                                 + 1.54273 w_k1_FDX_s105 + 1.2683 w_k1_IBM_s105
                                 + 0.974064 w_k1_INTC_s105
                                 + 1.09695 w_k1_KO_s105 + 1.67599 w_k1_MAT_s105
                                 + 1.08616 w_k1_MSFT_s105 + 1.2572 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.963314 w_k1_ADM_s105
                                 + 0.98875 w_k1_AXP_s105 + 1.1001 w_k1_DIS_s105
                                 + 1.156 w_k1_FDX_s105 + 0.832047 w_k1_IBM_s105
                                 + 1.04699 w_k1_INTC_s105
                                 + 0.905915 w_k1_KO_s105
                                 + 0.754338 w_k1_MAT_s105
                                 + 1.11806 w_k1_MSFT_s105
                                 + 0.975911 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.11277 w_k1_ADM_s105
                                 + 1.06374 w_k1_AXP_s105
                                 + 1.16265 w_k1_DIS_s105
                                 + 1.07628 w_k1_FDX_s105
                                 + 1.36894 w_k1_IBM_s105
                                 + 0.91653 w_k1_INTC_s105 + 1.1241 w_k1_KO_s105
                                 + 1.37121 w_k1_MAT_s105
                                 + 1.16522 w_k1_MSFT_s105
                                 + 1.09912 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.811775 w_k1_ADM_s105
                                 + 0.841197 w_k1_AXP_s105
                                 + 1.1459 w_k1_DIS_s105 + 1.03317 w_k1_FDX_s105
                                 + 0.975075 w_k1_IBM_s105
                                 + 1.0696 w_k1_INTC_s105 + 1.07182 w_k1_KO_s105
                                 + 0.975039 w_k1_MAT_s105
                                 + 1.17274 w_k1_MSFT_s105
                                 + 1.00138 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.08836 w_k1_ADM_s105
                                 + 1.2477 w_k1_AXP_s105 + 1.3421 w_k1_DIS_s105
                                 + 1.38247 w_k1_FDX_s105
                                 + 1.06299 w_k1_IBM_s105
                                 + 1.33207 w_k1_INTC_s105 + 1.1431 w_k1_KO_s105
                                 + 1.44877 w_k1_MAT_s105
                                 + 1.27326 w_k1_MSFT_s105
                                 + 1.08767 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.753048 w_k1_ADM_s105
                                 + 0.661966 w_k1_AXP_s105
                                 + 0.653698 w_k1_DIS_s105
                                 + 0.720189 w_k1_FDX_s105
                                 + 0.795783 w_k1_IBM_s105
                                 + 0.695364 w_k1_INTC_s105
                                 + 0.758842 w_k1_KO_s105
                                 + 0.722036 w_k1_MAT_s105
                                 + 0.761927 w_k1_MSFT_s105
                                 + 0.823748 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.688886 w_k1_ADM_s105
                                 + 0.663764 w_k1_AXP_s105
                                 + 0.719973 w_k1_DIS_s105
                                 + 0.898534 w_k1_FDX_s105
                                 + 0.783914 w_k1_IBM_s105
                                 + 0.833161 w_k1_INTC_s105
                                 + 0.689499 w_k1_KO_s105
                                 + 0.612384 w_k1_MAT_s105
                                 + 0.756116 w_k1_MSFT_s105
                                 + 0.823016 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.863358 w_k1_ADM_s105
                                 + 1.04593 w_k1_AXP_s105
                                 + 0.743023 w_k1_DIS_s105
                                 + 0.985046 w_k1_FDX_s105
                                 + 0.927139 w_k1_IBM_s105
                                 + 0.900207 w_k1_INTC_s105
                                 + 0.74012 w_k1_KO_s105 + 1.18456 w_k1_MAT_s105
                                 + 0.879234 w_k1_MSFT_s105
                                 + 0.975885 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.09977 w_k1_ADM_s106
                                 + 0.760086 w_k1_AXP_s106
                                 + 0.80599 w_k1_DIS_s106
                                 + 0.969859 w_k1_FDX_s106
                                 + 0.661821 w_k1_IBM_s106
                                 + 1.02901 w_k1_INTC_s106
                                 + 0.845218 w_k1_KO_s106
                                 + 0.672413 w_k1_MAT_s106
                                 + 0.916207 w_k1_MSFT_s106
                                 + 0.847652 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.806385 w_k1_ADM_s106
                                 + 0.939709 w_k1_AXP_s106
                                 + 1.11849 w_k1_DIS_s106
                                 + 0.829819 w_k1_FDX_s106
                                 + 1.51204 w_k1_IBM_s106
                                 + 0.821857 w_k1_INTC_s106
                                 + 1.25502 w_k1_KO_s106 + 1.34117 w_k1_MAT_s106
                                 + 0.967591 w_k1_MSFT_s106
                                 + 1.00877 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.12748 w_k1_ADM_s106
                                 + 1.1613 w_k1_AXP_s106
                                 + 0.862388 w_k1_DIS_s106
                                 + 0.937603 w_k1_FDX_s106
                                 + 1.28458 w_k1_IBM_s106
                                 + 1.09249 w_k1_INTC_s106
                                 + 1.08947 w_k1_KO_s106 + 1.52251 w_k1_MAT_s106
                                 + 0.90423 w_k1_MSFT_s106
                                 + 0.93067 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.729032 w_k1_ADM_s106
                                 + 0.669278 w_k1_AXP_s106
                                 + 0.5873 w_k1_DIS_s106
                                 + 0.703406 w_k1_FDX_s106
                                 + 0.797612 w_k1_IBM_s106
                                 + 0.595563 w_k1_INTC_s106
                                 + 0.727351 w_k1_KO_s106
                                 + 0.994554 w_k1_MAT_s106
                                 + 0.632019 w_k1_MSFT_s106
                                 + 0.771145 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.52603 w_k1_ADM_s106
                                 + 1.53613 w_k1_AXP_s106
                                 + 1.14504 w_k1_DIS_s106
                                 + 1.19015 w_k1_FDX_s106
                                 + 1.27835 w_k1_IBM_s106
                                 + 1.06423 w_k1_INTC_s106
                                 + 1.13034 w_k1_KO_s106 + 1.61309 w_k1_MAT_s106
                                 + 0.986561 w_k1_MSFT_s106
                                 + 1.06158 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.91857 w_k1_ADM_s106
                                 + 0.800036 w_k1_AXP_s106
                                 + 0.889326 w_k1_DIS_s106
                                 + 0.869468 w_k1_FDX_s106
                                 + 1.07191 w_k1_IBM_s106
                                 + 0.807425 w_k1_INTC_s106
                                 + 1.04124 w_k1_KO_s106 + 1.27825 w_k1_MAT_s106
                                 + 0.881881 w_k1_MSFT_s106
                                 + 1.01379 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.919083 w_k1_ADM_s106
                                 + 0.669779 w_k1_AXP_s106
                                 + 0.902516 w_k1_DIS_s106
                                 + 0.858146 w_k1_FDX_s106
                                 + 0.847805 w_k1_IBM_s106
                                 + 0.888642 w_k1_INTC_s106
                                 + 0.926355 w_k1_KO_s106 + 0.7404 w_k1_MAT_s106
                                 + 0.834312 w_k1_MSFT_s106
                                 + 0.87212 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.3571 w_k1_ADM_s106
                                 + 1.29923 w_k1_AXP_s106
                                 + 1.37367 w_k1_DIS_s106
                                 + 1.29747 w_k1_FDX_s106
                                 + 1.03522 w_k1_IBM_s106
                                 + 1.19819 w_k1_INTC_s106
                                 + 1.18426 w_k1_KO_s106 + 1.12394 w_k1_MAT_s106
                                 + 1.36439 w_k1_MSFT_s106
                                 + 1.05247 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.885444 w_k1_ADM_s107
                                 + 1.09594 w_k1_AXP_s107
                                 + 0.949344 w_k1_DIS_s107
                                 + 0.996129 w_k1_FDX_s107
                                 + 1.13444 w_k1_IBM_s107
                                 + 0.977538 w_k1_INTC_s107
                                 + 0.870512 w_k1_KO_s107
                                 + 1.26311 w_k1_MAT_s107
                                 + 1.01944 w_k1_MSFT_s107
                                 + 1.14747 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.3501 w_k1_ADM_s107
                                 + 1.25932 w_k1_AXP_s107
                                 + 0.834148 w_k1_DIS_s107
                                 + 0.959735 w_k1_FDX_s107
                                 + 0.95165 w_k1_IBM_s107
                                 + 0.96884 w_k1_INTC_s107
                                 + 0.789624 w_k1_KO_s107
                                 + 0.923599 w_k1_MAT_s107
                                 + 0.86831 w_k1_MSFT_s107
                                 + 0.957239 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.864839 w_k1_ADM_s107
                                 + 0.996855 w_k1_AXP_s107
                                 + 0.778388 w_k1_DIS_s107
                                 + 0.787612 w_k1_FDX_s107
                                 + 1.34212 w_k1_IBM_s107
                                 + 0.899109 w_k1_INTC_s107
                                 + 1.08414 w_k1_KO_s107 + 1.28417 w_k1_MAT_s107
                                 + 0.97573 w_k1_MSFT_s107
                                 + 1.07678 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.18194 w_k1_ADM_s107
                                 + 1.46674 w_k1_AXP_s107
                                 + 1.62214 w_k1_DIS_s107
                                 + 1.42572 w_k1_FDX_s107
                                 + 1.15851 w_k1_IBM_s107
                                 + 1.46994 w_k1_INTC_s107
                                 + 1.28225 w_k1_KO_s107 + 1.03669 w_k1_MAT_s107
                                 + 1.31029 w_k1_MSFT_s107
                                 + 1.29632 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.835629 w_k1_ADM_s107
                                 + 1.01763 w_k1_AXP_s107
                                 + 0.680594 w_k1_DIS_s107
                                 + 0.865983 w_k1_FDX_s107
                                 + 0.603947 w_k1_IBM_s107
                                 + 1.07202 w_k1_INTC_s107
                                 + 0.744124 w_k1_KO_s107
                                 + 0.530579 w_k1_MAT_s107
                                 + 0.892685 w_k1_MSFT_s107
                                 + 0.890614 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.21914 w_k1_ADM_s107
                                 + 0.96525 w_k1_AXP_s107
                                 + 0.945179 w_k1_DIS_s107
                                 + 0.992967 w_k1_FDX_s107
                                 + 0.776412 w_k1_IBM_s107
                                 + 1.03563 w_k1_INTC_s107
                                 + 0.895429 w_k1_KO_s107
                                 + 0.924994 w_k1_MAT_s107
                                 + 1.03341 w_k1_MSFT_s107
                                 + 1.14579 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.962516 w_k1_ADM_s107
                                 + 1.32882 w_k1_AXP_s107
                                 + 1.21628 w_k1_DIS_s107 + 1.1788 w_k1_FDX_s107
                                 + 1.36338 w_k1_IBM_s107
                                 + 1.17245 w_k1_INTC_s107
                                 + 1.23364 w_k1_KO_s107 + 1.62005 w_k1_MAT_s107
                                 + 1.12323 w_k1_MSFT_s107
                                 + 1.24186 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.855429 w_k1_ADM_s107
                                 + 0.856328 w_k1_AXP_s107
                                 + 0.78182 w_k1_DIS_s107
                                 + 0.777898 w_k1_FDX_s107
                                 + 0.920013 w_k1_IBM_s107
                                 + 0.889186 w_k1_INTC_s107
                                 + 0.951274 w_k1_KO_s107
                                 + 0.916429 w_k1_MAT_s107
                                 + 0.964314 w_k1_MSFT_s107
                                 + 0.960121 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.949123 w_k1_ADM_s110
                                 + 0.853653 w_k1_AXP_s110
                                 + 0.899287 w_k1_DIS_s110
                                 + 0.982819 w_k1_FDX_s110
                                 + 0.697475 w_k1_IBM_s110
                                 + 1.03925 w_k1_INTC_s110
                                 + 0.839094 w_k1_KO_s110
                                 + 0.653975 w_k1_MAT_s110
                                 + 0.923006 w_k1_MSFT_s110
                                 + 0.876769 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.32741 w_k1_ADM_s110
                                 + 1.32916 w_k1_AXP_s110
                                 + 1.36425 w_k1_DIS_s110
                                 + 1.16889 w_k1_FDX_s110
                                 + 1.60873 w_k1_IBM_s110
                                 + 1.27039 w_k1_INTC_s110
                                 + 1.37877 w_k1_KO_s110 + 1.47226 w_k1_MAT_s110
                                 + 1.05319 w_k1_MSFT_s110
                                 + 1.11805 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.20346 w_k1_ADM_s110
                                 + 1.44986 w_k1_AXP_s110
                                 + 1.34169 w_k1_DIS_s110
                                 + 1.24589 w_k1_FDX_s110
                                 + 1.24029 w_k1_IBM_s110
                                 + 1.02569 w_k1_INTC_s110
                                 + 1.12234 w_k1_KO_s110 + 2.25405 w_k1_MAT_s110
                                 + 1.16827 w_k1_MSFT_s110 + 1.1645 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.903497 w_k1_ADM_s110
                                 + 0.790049 w_k1_AXP_s110
                                 + 0.849857 w_k1_DIS_s110
                                 + 0.859918 w_k1_FDX_s110
                                 + 1.06922 w_k1_IBM_s110
                                 + 0.987569 w_k1_INTC_s110
                                 + 0.939687 w_k1_KO_s110
                                 + 1.04082 w_k1_MAT_s110
                                 + 0.914777 w_k1_MSFT_s110
                                 + 0.898849 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.739567 w_k1_ADM_s110
                                 + 0.630867 w_k1_AXP_s110
                                 + 0.649774 w_k1_DIS_s110
                                 + 0.762006 w_k1_FDX_s110
                                 + 0.677041 w_k1_IBM_s110
                                 + 0.936769 w_k1_INTC_s110
                                 + 0.757921 w_k1_KO_s110
                                 + 0.743433 w_k1_MAT_s110
                                 + 0.877868 w_k1_MSFT_s110
                                 + 0.851413 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.13424 w_k1_ADM_s110
                                 + 0.943756 w_k1_AXP_s110
                                 + 1.20369 w_k1_DIS_s110
                                 + 1.16301 w_k1_FDX_s110
                                 + 0.792949 w_k1_IBM_s110
                                 + 1.14937 w_k1_INTC_s110
                                 + 1.11303 w_k1_KO_s110
                                 + 0.919817 w_k1_MAT_s110
                                 + 1.22186 w_k1_MSFT_s110
                                 + 1.13621 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.52113 w_k1_ADM_s110
                                 + 1.21989 w_k1_AXP_s110
                                 + 1.85125 w_k1_DIS_s110
                                 + 1.54038 w_k1_FDX_s110
                                 + 1.06298 w_k1_IBM_s110
                                 + 1.24067 w_k1_INTC_s110
                                 + 1.37087 w_k1_KO_s110
                                 + 0.908844 w_k1_MAT_s110
                                 + 1.30639 w_k1_MSFT_s110
                                 + 1.34714 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.98812 w_k1_ADM_s110
                                 + 0.943977 w_k1_AXP_s110
                                 + 1.22848 w_k1_DIS_s110
                                 + 1.10912 w_k1_FDX_s110
                                 + 0.93111 w_k1_IBM_s110
                                 + 1.21196 w_k1_INTC_s110
                                 + 1.08749 w_k1_KO_s110 + 1.07236 w_k1_MAT_s110
                                 + 1.28553 w_k1_MSFT_s110
                                 + 1.02548 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.42036 w_k1_ADM_s111
                                 + 1.14233 w_k1_AXP_s111
                                 + 1.64233 w_k1_DIS_s111
                                 + 1.30664 w_k1_FDX_s111
                                 + 1.04988 w_k1_IBM_s111
                                 + 1.41191 w_k1_INTC_s111
                                 + 1.19054 w_k1_KO_s111
                                 + 0.901118 w_k1_MAT_s111
                                 + 1.35791 w_k1_MSFT_s111
                                 + 1.33182 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.909134 w_k1_ADM_s111
                                 + 0.745052 w_k1_AXP_s111
                                 + 0.852827 w_k1_DIS_s111
                                 + 0.827418 w_k1_FDX_s111
                                 + 1.1834 w_k1_IBM_s111 + 1.2382 w_k1_INTC_s111
                                 + 1.02088 w_k1_KO_s111
                                 + 0.797366 w_k1_MAT_s111
                                 + 0.842658 w_k1_MSFT_s111
                                 + 0.925297 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.919026 w_k1_ADM_s111
                                 + 0.846037 w_k1_AXP_s111
                                 + 0.837886 w_k1_DIS_s111
                                 + 0.850925 w_k1_FDX_s111
                                 + 0.859624 w_k1_IBM_s111
                                 + 0.923927 w_k1_INTC_s111
                                 + 0.87503 w_k1_KO_s111 + 1.12918 w_k1_MAT_s111
                                 + 0.880938 w_k1_MSFT_s111
                                 + 0.96038 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.868836 w_k1_ADM_s111
                                 + 0.63689 w_k1_AXP_s111
                                 + 0.69989 w_k1_DIS_s111
                                 + 0.767349 w_k1_FDX_s111
                                 + 0.630296 w_k1_IBM_s111
                                 + 1.08359 w_k1_INTC_s111
                                 + 0.801651 w_k1_KO_s111
                                 + 0.504506 w_k1_MAT_s111
                                 + 0.905651 w_k1_MSFT_s111
                                 + 0.933858 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.08716 w_k1_ADM_s111
                                 + 0.95715 w_k1_AXP_s111
                                 + 1.08941 w_k1_DIS_s111
                                 + 1.06923 w_k1_FDX_s111
                                 + 0.938614 w_k1_IBM_s111
                                 + 1.34497 w_k1_INTC_s111
                                 + 0.996288 w_k1_KO_s111
                                 + 0.976175 w_k1_MAT_s111
                                 + 1.08533 w_k1_MSFT_s111
                                 + 1.12943 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.03894 w_k1_ADM_s111
                                 + 0.966762 w_k1_AXP_s111
                                 + 1.15874 w_k1_DIS_s111
                                 + 1.02169 w_k1_FDX_s111
                                 + 0.895654 w_k1_IBM_s111
                                 + 1.3132 w_k1_INTC_s111
                                 + 0.840096 w_k1_KO_s111
                                 + 0.598083 w_k1_MAT_s111
                                 + 1.02974 w_k1_MSFT_s111
                                 + 0.989505 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.24069 w_k1_ADM_s111
                                 + 1.01433 w_k1_AXP_s111 + 1.1687 w_k1_DIS_s111
                                 + 0.999136 w_k1_FDX_s111
                                 + 1.2685 w_k1_IBM_s111
                                 + 1.05556 w_k1_INTC_s111
                                 + 1.22696 w_k1_KO_s111 + 1.39871 w_k1_MAT_s111
                                 + 0.98522 w_k1_MSFT_s111
                                 + 1.22458 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.25212 w_k1_ADM_s111
                                 + 1.25673 w_k1_AXP_s111
                                 + 1.25656 w_k1_DIS_s111
                                 + 1.04377 w_k1_FDX_s111
                                 + 1.55198 w_k1_IBM_s111
                                 + 1.25948 w_k1_INTC_s111 + 1.341 w_k1_KO_s111
                                 + 1.23555 w_k1_MAT_s111
                                 + 1.1312 w_k1_MSFT_s111 + 1.08253 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.29434 w_k1_ADM_s112
                                 + 1.6821 w_k1_AXP_s112 + 1.54751 w_k1_DIS_s112
                                 + 1.36778 w_k1_FDX_s112 + 1.2935 w_k1_IBM_s112
                                 + 1.19617 w_k1_INTC_s112
                                 + 1.36597 w_k1_KO_s112 + 1.3088 w_k1_MAT_s112
                                 + 1.16849 w_k1_MSFT_s112
                                 + 1.21179 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17714 w_k1_ADM_s112
                                 + 1.06001 w_k1_AXP_s112
                                 + 1.30206 w_k1_DIS_s112
                                 + 1.14771 w_k1_FDX_s112
                                 + 1.00341 w_k1_IBM_s112
                                 + 1.34937 w_k1_INTC_s112
                                 + 1.23794 w_k1_KO_s112
                                 + 0.807686 w_k1_MAT_s112
                                 + 1.32533 w_k1_MSFT_s112
                                 + 1.06805 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.92203 w_k1_ADM_s112
                                 + 0.801226 w_k1_AXP_s112
                                 + 1.03467 w_k1_DIS_s112
                                 + 0.843147 w_k1_FDX_s112
                                 + 1.32965 w_k1_IBM_s112
                                 + 1.14927 w_k1_INTC_s112 + 1.2879 w_k1_KO_s112
                                 + 0.925312 w_k1_MAT_s112
                                 + 1.07888 w_k1_MSFT_s112
                                 + 1.07063 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.09995 w_k1_ADM_s112
                                 + 0.954671 w_k1_AXP_s112
                                 + 0.848356 w_k1_DIS_s112
                                 + 0.901326 w_k1_FDX_s112
                                 + 0.828142 w_k1_IBM_s112
                                 + 1.14275 w_k1_INTC_s112
                                 + 0.909936 w_k1_KO_s112
                                 + 0.777442 w_k1_MAT_s112
                                 + 0.979778 w_k1_MSFT_s112
                                 + 1.01616 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.17775 w_k1_ADM_s112
                                 + 1.34088 w_k1_AXP_s112
                                 + 1.03706 w_k1_DIS_s112
                                 + 1.01222 w_k1_FDX_s112
                                 + 1.25182 w_k1_IBM_s112
                                 + 1.14004 w_k1_INTC_s112
                                 + 1.18613 w_k1_KO_s112 + 1.17977 w_k1_MAT_s112
                                 + 1.09163 w_k1_MSFT_s112
                                 + 1.10613 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.931042 w_k1_ADM_s112
                                 + 0.760667 w_k1_AXP_s112
                                 + 0.698431 w_k1_DIS_s112
                                 + 0.786918 w_k1_FDX_s112
                                 + 1.09473 w_k1_IBM_s112
                                 + 0.886775 w_k1_INTC_s112
                                 + 0.880232 w_k1_KO_s112
                                 + 0.889623 w_k1_MAT_s112
                                 + 0.845386 w_k1_MSFT_s112
                                 + 0.838527 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.14483 w_k1_ADM_s112
                                 + 1.43244 w_k1_AXP_s112
                                 + 0.832694 w_k1_DIS_s112
                                 + 1.03404 w_k1_FDX_s112
                                 + 0.807368 w_k1_IBM_s112
                                 + 1.03441 w_k1_INTC_s112
                                 + 0.906468 w_k1_KO_s112
                                 + 1.05918 w_k1_MAT_s112
                                 + 0.999301 w_k1_MSFT_s112
                                 + 1.08015 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.809758 w_k1_ADM_s112
                                 + 0.751239 w_k1_AXP_s112
                                 + 0.644198 w_k1_DIS_s112
                                 + 0.697033 w_k1_FDX_s112
                                 + 0.763278 w_k1_IBM_s112
                                 + 0.777149 w_k1_INTC_s112
                                 + 0.777607 w_k1_KO_s112
                                 + 0.637414 w_k1_MAT_s112
                                 + 0.80283 w_k1_MSFT_s112
                                 + 0.766222 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.02685 w_k1_ADM_s113
                                 + 0.87493 w_k1_AXP_s113
                                 + 1.08703 w_k1_DIS_s113
                                 + 1.10423 w_k1_FDX_s113 + 1.0174 w_k1_IBM_s113
                                 + 0.907191 w_k1_INTC_s113
                                 + 1.02458 w_k1_KO_s113
                                 + 0.816634 w_k1_MAT_s113
                                 + 1.08355 w_k1_MSFT_s113
                                 + 0.985581 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.01344 w_k1_ADM_s113
                                 + 1.16881 w_k1_AXP_s113
                                 + 1.12694 w_k1_DIS_s113
                                 + 1.04574 w_k1_FDX_s113
                                 + 1.36971 w_k1_IBM_s113
                                 + 0.7414 w_k1_INTC_s113 + 1.22269 w_k1_KO_s113
                                 + 1.6106 w_k1_MAT_s113
                                 + 0.862776 w_k1_MSFT_s113
                                 + 1.12477 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.04723 w_k1_ADM_s113
                                 + 1.41558 w_k1_AXP_s113
                                 + 1.36531 w_k1_DIS_s113
                                 + 1.18264 w_k1_FDX_s113
                                 + 1.19017 w_k1_IBM_s113
                                 + 1.00382 w_k1_INTC_s113
                                 + 1.18232 w_k1_KO_s113 + 1.18649 w_k1_MAT_s113
                                 + 1.08198 w_k1_MSFT_s113 + 1.1014 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.759042 w_k1_ADM_s113
                                 + 0.737135 w_k1_AXP_s113
                                 + 0.70808 w_k1_DIS_s113
                                 + 0.774191 w_k1_FDX_s113
                                 + 0.763242 w_k1_IBM_s113
                                 + 0.873273 w_k1_INTC_s113
                                 + 0.817477 w_k1_KO_s113
                                 + 0.724281 w_k1_MAT_s113
                                 + 0.785243 w_k1_MSFT_s113
                                 + 0.804095 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.00994 w_k1_ADM_s113
                                 + 0.977363 w_k1_AXP_s113
                                 + 0.855551 w_k1_DIS_s113
                                 + 0.948666 w_k1_FDX_s113
                                 + 0.993816 w_k1_IBM_s113
                                 + 0.733345 w_k1_INTC_s113
                                 + 0.997252 w_k1_KO_s113
                                 + 0.996138 w_k1_MAT_s113
                                 + 0.914914 w_k1_MSFT_s113
                                 + 1.01584 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.13962 w_k1_ADM_s113
                                 + 1.20278 w_k1_AXP_s113
                                 + 1.29991 w_k1_DIS_s113
                                 + 1.27756 w_k1_FDX_s113
                                 + 0.877817 w_k1_IBM_s113
                                 + 1.18347 w_k1_INTC_s113
                                 + 1.17421 w_k1_KO_s113 + 1.21552 w_k1_MAT_s113
                                 + 1.27011 w_k1_MSFT_s113
                                 + 1.15079 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.27737 w_k1_ADM_s113
                                 + 1.55522 w_k1_AXP_s113
                                 + 1.70114 w_k1_DIS_s113
                                 + 1.50091 w_k1_FDX_s113
                                 + 1.22332 w_k1_IBM_s113
                                 + 1.31009 w_k1_INTC_s113
                                 + 1.33034 w_k1_KO_s113 + 1.58635 w_k1_MAT_s113
                                 + 1.26736 w_k1_MSFT_s113
                                 + 1.28512 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.932526 w_k1_ADM_s113
                                 + 1.10286 w_k1_AXP_s113
                                 + 1.04767 w_k1_DIS_s113
                                 + 0.892322 w_k1_FDX_s113
                                 + 1.43807 w_k1_IBM_s113
                                 + 0.865809 w_k1_INTC_s113
                                 + 1.14185 w_k1_KO_s113 + 1.53198 w_k1_MAT_s113
                                 + 0.946134 w_k1_MSFT_s113
                                 + 1.08628 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.01536 w_k1_ADM_s114
                                 + 1.36201 w_k1_AXP_s114
                                 + 1.04575 w_k1_DIS_s114
                                 + 1.07314 w_k1_FDX_s114
                                 + 1.22114 w_k1_IBM_s114
                                 + 0.852954 w_k1_INTC_s114
                                 + 0.981926 w_k1_KO_s114
                                 + 1.56599 w_k1_MAT_s114
                                 + 0.852497 w_k1_MSFT_s114
                                 + 1.01451 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.887004 w_k1_ADM_s114
                                 + 1.19317 w_k1_AXP_s114
                                 + 1.07092 w_k1_DIS_s114
                                 + 1.21124 w_k1_FDX_s114
                                 + 0.685638 w_k1_IBM_s114
                                 + 1.05627 w_k1_INTC_s114
                                 + 0.724427 w_k1_KO_s114
                                 + 0.986892 w_k1_MAT_s114
                                 + 1.0236 w_k1_MSFT_s114
                                 + 0.919394 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.12618 w_k1_ADM_s114
                                 + 1.23169 w_k1_AXP_s114
                                 + 0.973846 w_k1_DIS_s114
                                 + 1.08496 w_k1_FDX_s114
                                 + 0.916972 w_k1_IBM_s114 + 0.98 w_k1_INTC_s114
                                 + 0.907543 w_k1_KO_s114 + 1.1495 w_k1_MAT_s114
                                 + 1.02098 w_k1_MSFT_s114
                                 + 0.933487 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.10292 w_k1_ADM_s114
                                 + 1.3514 w_k1_AXP_s114 + 1.6052 w_k1_DIS_s114
                                 + 1.23314 w_k1_FDX_s114
                                 + 1.12656 w_k1_IBM_s114
                                 + 0.932403 w_k1_INTC_s114
                                 + 1.25302 w_k1_KO_s114 + 1.49781 w_k1_MAT_s114
                                 + 1.32826 w_k1_MSFT_s114
                                 + 1.22489 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.712066 w_k1_ADM_s114
                                 + 0.808607 w_k1_AXP_s114
                                 + 0.731036 w_k1_DIS_s114
                                 + 0.722511 w_k1_FDX_s114
                                 + 1.00424 w_k1_IBM_s114
                                 + 0.752872 w_k1_INTC_s114
                                 + 0.805487 w_k1_KO_s114
                                 + 1.24618 w_k1_MAT_s114
                                 + 0.760219 w_k1_MSFT_s114
                                 + 0.806692 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.0293 w_k1_ADM_s114
                                 + 0.997519 w_k1_AXP_s114
                                 + 1.17887 w_k1_DIS_s114
                                 + 0.948988 w_k1_FDX_s114
                                 + 1.62352 w_k1_IBM_s114
                                 + 0.968686 w_k1_INTC_s114
                                 + 1.23572 w_k1_KO_s114 + 1.53197 w_k1_MAT_s114
                                 + 0.944038 w_k1_MSFT_s114
                                 + 1.08175 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.741936 w_k1_ADM_s114
                                 + 0.69388 w_k1_AXP_s114
                                 + 0.61067 w_k1_DIS_s114
                                 + 0.765586 w_k1_FDX_s114
                                 + 0.636117 w_k1_IBM_s114
                                 + 0.907945 w_k1_INTC_s114
                                 + 0.555431 w_k1_KO_s114
                                 + 0.609051 w_k1_MAT_s114
                                 + 0.772833 w_k1_MSFT_s114
                                 + 0.768411 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.978857 w_k1_ADM_s114
                                 + 0.870832 w_k1_AXP_s114
                                 + 0.914987 w_k1_DIS_s114
                                 + 0.887255 w_k1_FDX_s114
                                 + 0.94535 w_k1_IBM_s114
                                 + 0.941389 w_k1_INTC_s114
                                 + 0.921458 w_k1_KO_s114
                                 + 1.14802 w_k1_MAT_s114
                                 + 0.961186 w_k1_MSFT_s114
                                 + 0.86702 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.25159 w_k1_ADM_s115
                                 + 1.19432 w_k1_AXP_s115
                                 + 1.29363 w_k1_DIS_s115
                                 + 1.04984 w_k1_FDX_s115
                                 + 1.90009 w_k1_IBM_s115
                                 + 1.06219 w_k1_INTC_s115
                                 + 1.47855 w_k1_KO_s115 + 1.66812 w_k1_MAT_s115
                                 + 0.987041 w_k1_MSFT_s115
                                 + 1.05564 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.11253 w_k1_ADM_s115
                                 + 1.00865 w_k1_AXP_s115
                                 + 0.930443 w_k1_DIS_s115
                                 + 0.995462 w_k1_FDX_s115
                                 + 0.981679 w_k1_IBM_s115
                                 + 0.990376 w_k1_INTC_s115
                                 + 1.15281 w_k1_KO_s115 + 1.13392 w_k1_MAT_s115
                                 + 1.03833 w_k1_MSFT_s115
                                 + 0.963933 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.28117 w_k1_ADM_s115
                                 + 1.25964 w_k1_AXP_s115
                                 + 1.16063 w_k1_DIS_s115
                                 + 1.30798 w_k1_FDX_s115
                                 + 0.715376 w_k1_IBM_s115
                                 + 1.25056 w_k1_INTC_s115
                                 + 1.03908 w_k1_KO_s115
                                 + 0.768633 w_k1_MAT_s115
                                 + 1.12614 w_k1_MSFT_s115
                                 + 1.03403 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.926962 w_k1_ADM_s115
                                 + 0.739699 w_k1_AXP_s115
                                 + 0.626086 w_k1_DIS_s115
                                 + 0.754325 w_k1_FDX_s115
                                 + 0.716434 w_k1_IBM_s115
                                 + 0.898693 w_k1_INTC_s115
                                 + 0.825708 w_k1_KO_s115
                                 + 0.654681 w_k1_MAT_s115
                                 + 0.848691 w_k1_MSFT_s115
                                 + 0.888035 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.02733 w_k1_ADM_s115
                                 + 0.797484 w_k1_AXP_s115
                                 + 1.16752 w_k1_DIS_s115
                                 + 1.06587 w_k1_FDX_s115
                                 + 0.833197 w_k1_IBM_s115
                                 + 1.26618 w_k1_INTC_s115
                                 + 1.13428 w_k1_KO_s115
                                 + 0.610655 w_k1_MAT_s115
                                 + 1.12102 w_k1_MSFT_s115
                                 + 1.04408 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.934947 w_k1_ADM_s115
                                 + 0.951222 w_k1_AXP_s115
                                 + 0.939829 w_k1_DIS_s115
                                 + 0.890102 w_k1_FDX_s115
                                 + 1.27889 w_k1_IBM_s115
                                 + 0.828692 w_k1_INTC_s115
                                 + 1.11328 w_k1_KO_s115 + 1.03533 w_k1_MAT_s115
                                 + 0.950066 w_k1_MSFT_s115
                                 + 0.958767 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.44099 w_k1_ADM_s115
                                 + 1.53339 w_k1_AXP_s115
                                 + 1.56645 w_k1_DIS_s115
                                 + 1.34058 w_k1_FDX_s115
                                 + 1.24768 w_k1_IBM_s115
                                 + 1.33905 w_k1_INTC_s115
                                 + 1.39325 w_k1_KO_s115 + 1.27197 w_k1_MAT_s115
                                 + 1.32291 w_k1_MSFT_s115
                                 + 1.18934 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.12414 w_k1_ADM_s115
                                 + 0.929819 w_k1_AXP_s115
                                 + 0.938022 w_k1_DIS_s115
                                 + 1.1212 w_k1_FDX_s115
                                 + 0.721105 w_k1_IBM_s115
                                 + 1.09091 w_k1_INTC_s115
                                 + 1.05303 w_k1_KO_s115
                                 + 0.821487 w_k1_MAT_s115
                                 + 1.13348 w_k1_MSFT_s115
                                 + 0.979155 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.765404 w_k1_ADM_s116
                                 + 0.702075 w_k1_AXP_s116
                                 + 0.869478 w_k1_DIS_s116
                                 + 0.74441 w_k1_FDX_s116
                                 + 0.79017 w_k1_IBM_s116
                                 + 0.63738 w_k1_INTC_s116
                                 + 0.788656 w_k1_KO_s116
                                 + 0.796846 w_k1_MAT_s116
                                 + 0.761456 w_k1_MSFT_s116
                                 + 0.777818 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.997817 w_k1_ADM_s116
                                 + 1.01732 w_k1_AXP_s116
                                 + 0.957419 w_k1_DIS_s116
                                 + 1.00318 w_k1_FDX_s116
                                 + 0.689172 w_k1_IBM_s116
                                 + 0.709656 w_k1_INTC_s116
                                 + 0.800364 w_k1_KO_s116
                                 + 0.673022 w_k1_MAT_s116
                                 + 0.835012 w_k1_MSFT_s116
                                 + 0.839034 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.27056 w_k1_ADM_s116
                                 + 1.3746 w_k1_AXP_s116 + 1.56735 w_k1_DIS_s116
                                 + 1.17436 w_k1_FDX_s116
                                 + 1.28998 w_k1_IBM_s116
                                 + 1.00273 w_k1_INTC_s116
                                 + 1.34417 w_k1_KO_s116 + 1.1744 w_k1_MAT_s116
                                 + 1.04519 w_k1_MSFT_s116
                                 + 1.12847 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.917957 w_k1_ADM_s116
                                 + 0.98584 w_k1_AXP_s116 + 0.9675 w_k1_DIS_s116
                                 + 0.907757 w_k1_FDX_s116
                                 + 0.896217 w_k1_IBM_s116
                                 + 0.826225 w_k1_INTC_s116
                                 + 0.777442 w_k1_KO_s116
                                 + 0.925302 w_k1_MAT_s116
                                 + 0.847236 w_k1_MSFT_s116
                                 + 0.872197 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.00518 w_k1_ADM_s116
                                 + 1.25891 w_k1_AXP_s116
                                 + 0.975678 w_k1_DIS_s116
                                 + 0.909909 w_k1_FDX_s116
                                 + 1.23498 w_k1_IBM_s116
                                 + 0.813559 w_k1_INTC_s116
                                 + 1.03683 w_k1_KO_s116 + 1.09711 w_k1_MAT_s116
                                 + 0.807161 w_k1_MSFT_s116
                                 + 0.975154 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.00104 w_k1_ADM_s116
                                 + 1.1365 w_k1_AXP_s116 + 1.00563 w_k1_DIS_s116
                                 + 1.04544 w_k1_FDX_s116
                                 + 0.967948 w_k1_IBM_s116
                                 + 1.02395 w_k1_INTC_s116
                                 + 0.908751 w_k1_KO_s116
                                 + 0.924801 w_k1_MAT_s116
                                 + 0.966654 w_k1_MSFT_s116
                                 + 0.937773 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.19049 w_k1_ADM_s116
                                 + 1.15859 w_k1_AXP_s116
                                 + 1.56487 w_k1_DIS_s116
                                 + 1.30509 w_k1_FDX_s116
                                 + 0.937985 w_k1_IBM_s116
                                 + 1.10926 w_k1_INTC_s116
                                 + 1.12243 w_k1_KO_s116 + 1.03461 w_k1_MAT_s116
                                 + 1.28706 w_k1_MSFT_s116
                                 + 1.04845 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.18248 w_k1_ADM_s116
                                 + 1.28449 w_k1_AXP_s116
                                 + 1.41753 w_k1_DIS_s116
                                 + 1.09167 w_k1_FDX_s116
                                 + 1.72304 w_k1_IBM_s116
                                 + 0.847441 w_k1_INTC_s116
                                 + 1.34167 w_k1_KO_s116 + 2.08305 w_k1_MAT_s116
                                 + 0.954996 w_k1_MSFT_s116
                                 + 1.10268 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.11265 w_k1_ADM_s117
                                 + 0.870848 w_k1_AXP_s117
                                 + 0.847706 w_k1_DIS_s117
                                 + 0.912585 w_k1_FDX_s117
                                 + 0.94832 w_k1_IBM_s117
                                 + 0.946049 w_k1_INTC_s117
                                 + 0.873354 w_k1_KO_s117
                                 + 0.866374 w_k1_MAT_s117
                                 + 0.854491 w_k1_MSFT_s117
                                 + 0.991007 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.48523 w_k1_ADM_s117
                                 + 1.55133 w_k1_AXP_s117
                                 + 1.84595 w_k1_DIS_s117
                                 + 1.61134 w_k1_FDX_s117
                                 + 1.22151 w_k1_IBM_s117
                                 + 1.28773 w_k1_INTC_s117
                                 + 1.40718 w_k1_KO_s117 + 1.25812 w_k1_MAT_s117
                                 + 1.48745 w_k1_MSFT_s117
                                 + 1.35839 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.25128 w_k1_ADM_s117
                                 + 1.03707 w_k1_AXP_s117
                                 + 1.12112 w_k1_DIS_s117
                                 + 1.11111 w_k1_FDX_s117 + 1.011 w_k1_IBM_s117
                                 + 1.01988 w_k1_INTC_s117
                                 + 1.12822 w_k1_KO_s117
                                 + 0.904709 w_k1_MAT_s117
                                 + 0.907873 w_k1_MSFT_s117
                                 + 1.06741 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.13495 w_k1_ADM_s117
                                 + 0.765677 w_k1_AXP_s117
                                 + 0.943118 w_k1_DIS_s117
                                 + 0.845089 w_k1_FDX_s117
                                 + 1.25251 w_k1_IBM_s117
                                 + 0.921159 w_k1_INTC_s117
                                 + 0.982568 w_k1_KO_s117
                                 + 0.950602 w_k1_MAT_s117
                                 + 0.864178 w_k1_MSFT_s117
                                 + 0.936532 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.22685 w_k1_ADM_s117
                                 + 0.949832 w_k1_AXP_s117
                                 + 1.52679 w_k1_DIS_s117
                                 + 1.25207 w_k1_FDX_s117
                                 + 1.11523 w_k1_IBM_s117
                                 + 1.00994 w_k1_INTC_s117 + 1.3536 w_k1_KO_s117
                                 + 1.02889 w_k1_MAT_s117
                                 + 1.17913 w_k1_MSFT_s117
                                 + 1.14237 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.98617 w_k1_ADM_s117
                                 + 0.828452 w_k1_AXP_s117
                                 + 0.666039 w_k1_DIS_s117
                                 + 0.834164 w_k1_FDX_s117
                                 + 0.72785 w_k1_IBM_s117
                                 + 0.780468 w_k1_INTC_s117
                                 + 0.790067 w_k1_KO_s117
                                 + 0.789798 w_k1_MAT_s117
                                 + 0.768964 w_k1_MSFT_s117
                                 + 0.855945 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.46951 w_k1_ADM_s117
                                 + 1.18835 w_k1_AXP_s117
                                 + 1.35118 w_k1_DIS_s117
                                 + 1.02455 w_k1_FDX_s117
                                 + 1.65317 w_k1_IBM_s117
                                 + 0.879655 w_k1_INTC_s117
                                 + 1.28472 w_k1_KO_s117 + 1.36127 w_k1_MAT_s117
                                 + 0.914872 w_k1_MSFT_s117
                                 + 1.1081 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.881436 w_k1_ADM_s117
                                 + 0.681419 w_k1_AXP_s117
                                 + 0.654018 w_k1_DIS_s117
                                 + 0.69731 w_k1_FDX_s117
                                 + 0.971461 w_k1_IBM_s117
                                 + 0.787352 w_k1_INTC_s117
                                 + 0.772817 w_k1_KO_s117
                                 + 0.902934 w_k1_MAT_s117
                                 + 0.726929 w_k1_MSFT_s117
                                 + 0.775556 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.017624310864988 w_k1_ADM_s000
                                 + 0.0254883210220492 w_k1_ADM_s001
                                 + 0.0119227973665808 w_k1_ADM_s002
                                 + 0.0215062946178368 w_k1_ADM_s003
                                 + 0.0192139477689286 w_k1_ADM_s004
                                 + 0.03264497218347 w_k1_ADM_s005
                                 + 0.041551215549466 w_k1_ADM_s006
                                 + 0.0200554761212304 w_k1_ADM_s007
                                 + 0.0627141761249843 w_k1_ADM_s010
                                 + 0.0373996841282902 w_k1_ADM_s011
                                 + 0.064899836484212 w_k1_ADM_s012
                                 + 0.0660078540167021 w_k1_ADM_s013
                                 + 0.0692617779958869 w_k1_ADM_s014
                                 + 0.0370281221477963 w_k1_ADM_s015
                                 + 0.0202437055291217 w_k1_ADM_s016
                                 + 0.0394823957654267 w_k1_ADM_s017
                                 + 0.0112365914264836 w_k1_ADM_s100
                                 + 0.0205644517511763 w_k1_ADM_s101
                                 + 0.017027427590761 w_k1_ADM_s102
                                 + 0.00813459632558098 w_k1_ADM_s103
                                 + 0.0125757023738431 w_k1_ADM_s104
                                 + 0.0187250877841177 w_k1_ADM_s105
                                 + 0.00928553176107063 w_k1_ADM_s106
                                 + 0.0274031411476845 w_k1_ADM_s107
                                 + 0.046153442493022 w_k1_ADM_s110
                                 + 0.0247037786354271 w_k1_ADM_s111
                                 + 0.0401003920916842 w_k1_ADM_s112
                                 + 0.0360464959058791 w_k1_ADM_s113
                                 + 0.0502039354152105 w_k1_ADM_s114
                                 + 0.0682283196361315 w_k1_ADM_s115
                                 + 0.0134538167449558 w_k1_ADM_s116
                                 + 0.0416686844668956 w_k1_ADM_s117
                                 + 0.0207227977767406 w_k1_AXP_s000
                                 + 0.0226165281826931 w_k1_AXP_s001
                                 + 0.0126372539911506 w_k1_AXP_s002
                                 + 0.0198664022007654 w_k1_AXP_s003
                                 + 0.0196938675835228 w_k1_AXP_s004
                                 + 0.0340731273828695 w_k1_AXP_s005
                                 + 0.0366366072205039 w_k1_AXP_s006
                                 + 0.026144833004686 w_k1_AXP_s007
                                 + 0.0639054975828984 w_k1_AXP_s010
                                 + 0.0400549744497899 w_k1_AXP_s011
                                 + 0.0682247443569976 w_k1_AXP_s012
                                 + 0.0708579235443345 w_k1_AXP_s013
                                 + 0.0731721808136619 w_k1_AXP_s014
                                 + 0.0453078165154457 w_k1_AXP_s015
                                 + 0.0197207787821744 w_k1_AXP_s016
                                 + 0.0392842186423367 w_k1_AXP_s017
                                 + 0.012813132189211 w_k1_AXP_s100
                                 + 0.0208588459008987 w_k1_AXP_s101
                                 + 0.0125876826943586 w_k1_AXP_s102
                                 + 0.00893035252389862 w_k1_AXP_s103
                                 + 0.0113374714083652 w_k1_AXP_s104
                                 + 0.0197823955109244 w_k1_AXP_s105
                                 + 0.00852510001886674 w_k1_AXP_s106
                                 + 0.0293881010791424 w_k1_AXP_s107
                                 + 0.0422875152798112 w_k1_AXP_s110
                                 + 0.0217173556425676 w_k1_AXP_s111
                                 + 0.0404252542154122 w_k1_AXP_s112
                                 + 0.0378134290761378 w_k1_AXP_s113
                                 + 0.0570588248038154 w_k1_AXP_s114
                                 + 0.0636858221990346 w_k1_AXP_s115
                                 + 0.0144309853145281 w_k1_AXP_s116
                                 + 0.0341572701360301 w_k1_AXP_s117
                                 + 0.0211981405051717 w_k1_DIS_s000
                                 + 0.0229802865742185 w_k1_DIS_s001
                                 + 0.0138540177696839 w_k1_DIS_s002
                                 + 0.0191076332278797 w_k1_DIS_s003
                                 + 0.0183910599363408 w_k1_DIS_s004
                                 + 0.0328489569598671 w_k1_DIS_s005
                                 + 0.0364774437287007 w_k1_DIS_s006
                                 + 0.0240191405568363 w_k1_DIS_s007
                                 + 0.0591402440957811 w_k1_DIS_s010
                                 + 0.0359920632061417 w_k1_DIS_s011
                                 + 0.0699264595569419 w_k1_DIS_s012
                                 + 0.0662961333922928 w_k1_DIS_s013
                                 + 0.0699146389242226 w_k1_DIS_s014
                                 + 0.0422491193308458 w_k1_DIS_s015
                                 + 0.0195187343136168 w_k1_DIS_s016
                                 + 0.0412414299373065 w_k1_DIS_s017
                                 + 0.0130387920642403 w_k1_DIS_s100
                                 + 0.0209062281209793 w_k1_DIS_s101
                                 + 0.0143852765066424 w_k1_DIS_s102
                                 + 0.00954907806622861 w_k1_DIS_s103
                                 + 0.0108450041822036 w_k1_DIS_s104
                                 + 0.0204624062230631 w_k1_DIS_s105
                                 + 0.00847360685137795 w_k1_DIS_s106
                                 + 0.0259320195098738 w_k1_DIS_s107
                                 + 0.0476648937777667 w_k1_DIS_s110
                                 + 0.0247428669812228 w_k1_DIS_s111
                                 + 0.0374707943939685 w_k1_DIS_s112
                                 + 0.0378954766665288 w_k1_DIS_s113
                                 + 0.0558319572622994 w_k1_DIS_s114
                                 + 0.0633282197359602 w_k1_DIS_s115
                                 + 0.0148723375956657 w_k1_DIS_s116
                                 + 0.036851457256887 w_k1_DIS_s117
                                 + 0.0205748376518793 w_k1_FDX_s000
                                 + 0.0224996127555149 w_k1_FDX_s001
                                 + 0.0128493004394475 w_k1_FDX_s002
                                 + 0.0199945799243322 w_k1_FDX_s003
                                 + 0.0179326969519539 w_k1_FDX_s004
                                 + 0.0311515036372301 w_k1_FDX_s005
                                 + 0.0352915244535913 w_k1_FDX_s006
                                 + 0.0232079662939118 w_k1_FDX_s007
                                 + 0.0610283914807497 w_k1_FDX_s010
                                 + 0.0360458213272256 w_k1_FDX_s011
                                 + 0.0684954987570652 w_k1_FDX_s012
                                 + 0.0663832878720103 w_k1_FDX_s013
                                 + 0.0665731686435738 w_k1_FDX_s014
                                 + 0.0400901345913384 w_k1_FDX_s015
                                 + 0.0193645862300599 w_k1_FDX_s016
                                 + 0.0390828810313798 w_k1_FDX_s017
                                 + 0.012246278781739 w_k1_FDX_s100
                                 + 0.0192160436967352 w_k1_FDX_s101
                                 + 0.013834509101574 w_k1_FDX_s102
                                 + 0.00908415672324382 w_k1_FDX_s103
                                 + 0.0108587917897721 w_k1_FDX_s104
                                 + 0.021445222003118 w_k1_FDX_s105
                                 + 0.00841692070876606 w_k1_FDX_s106
                                 + 0.0265844327431638 w_k1_FDX_s107
                                 + 0.0457591007017163 w_k1_FDX_s110
                                 + 0.0227260921325225 w_k1_FDX_s111
                                 + 0.0365037368999769 w_k1_FDX_s112
                                 + 0.0374117382564883 w_k1_FDX_s113
                                 + 0.052604764195528 w_k1_FDX_s114
                                 + 0.062682293459784 w_k1_FDX_s115
                                 + 0.0133280588926212 w_k1_FDX_s116
                                 + 0.0359458369947939 w_k1_FDX_s117
                                 + 0.0210813233578375 w_k1_IBM_s000
                                 + 0.0235445922029586 w_k1_IBM_s001
                                 + 0.0134552767744402 w_k1_IBM_s002
                                 + 0.0186829186956039 w_k1_IBM_s003
                                 + 0.0175603997491821 w_k1_IBM_s004
                                 + 0.0327029006455228 w_k1_IBM_s005
                                 + 0.0354851386108037 w_k1_IBM_s006
                                 + 0.0221452362975585 w_k1_IBM_s007
                                 + 0.0624341066593063 w_k1_IBM_s010
                                 + 0.0359405651804031 w_k1_IBM_s011
                                 + 0.069888839612097 w_k1_IBM_s012
                                 + 0.0675732087112233 w_k1_IBM_s013
                                 + 0.0661503266953959 w_k1_IBM_s014
                                 + 0.0414553472420611 w_k1_IBM_s015
                                 + 0.020582663523499 w_k1_IBM_s016
                                 + 0.0412060602901152 w_k1_IBM_s017
                                 + 0.0131894551206183 w_k1_IBM_s100
                                 + 0.0215477775961715 w_k1_IBM_s101
                                 + 0.0140976425714146 w_k1_IBM_s102
                                 + 0.00908296012948118 w_k1_IBM_s103
                                 + 0.0106641708036791 w_k1_IBM_s104
                                 + 0.0199035933831901 w_k1_IBM_s105
                                 + 0.00888395636653442 w_k1_IBM_s106
                                 + 0.0265138900025275 w_k1_IBM_s107
                                 + 0.0457689238426169 w_k1_IBM_s110
                                 + 0.0232646472070736 w_k1_IBM_s111
                                 + 0.0377783740686999 w_k1_IBM_s112
                                 + 0.0381717616898237 w_k1_IBM_s113
                                 + 0.0521372964944429 w_k1_IBM_s114
                                 + 0.0663985040111266 w_k1_IBM_s115
                                 + 0.0134144357077259 w_k1_IBM_s116
                                 + 0.0371650223394505 w_k1_IBM_s117
                                 + 0.0198410805904768 w_k1_INTC_s000
                                 + 0.0213181237394566 w_k1_INTC_s001
                                 + 0.0130761864224112 w_k1_INTC_s002
                                 + 0.0196787315833994 w_k1_INTC_s003
                                 + 0.0183825880687879 w_k1_INTC_s004
                                 + 0.0297567527382046 w_k1_INTC_s005
                                 + 0.0371111584970167 w_k1_INTC_s006
                                 + 0.0208912935283407 w_k1_INTC_s007
                                 + 0.0593481820134253 w_k1_INTC_s010
                                 + 0.0367495105377359 w_k1_INTC_s011
                                 + 0.0800717543855525 w_k1_INTC_s012
                                 + 0.0715061479658931 w_k1_INTC_s013
                                 + 0.0640362563018052 w_k1_INTC_s014
                                 + 0.0450603892525201 w_k1_INTC_s015
                                 + 0.0164178110752084 w_k1_INTC_s016
                                 + 0.0390510974201176 w_k1_INTC_s017
                                 + 0.0123769179506937 w_k1_INTC_s100
                                 + 0.0223460747227999 w_k1_INTC_s101
                                 + 0.0140777990625347 w_k1_INTC_s102
                                 + 0.00930279955406448 w_k1_INTC_s103
                                 + 0.0115100110342165 w_k1_INTC_s104
                                 + 0.0194554363831264 w_k1_INTC_s105
                                 + 0.00818752609360332 w_k1_INTC_s106
                                 + 0.0278853132014027 w_k1_INTC_s107
                                 + 0.0480682723883299 w_k1_INTC_s110
                                 + 0.0276465668415442 w_k1_INTC_s111
                                 + 0.0409068609846671 w_k1_INTC_s112
                                 + 0.0324250546249298 w_k1_INTC_s113
                                 + 0.0472634909387405 w_k1_INTC_s114
                                 + 0.0640116003456736 w_k1_INTC_s115
                                 + 0.0115665492049461 w_k1_INTC_s116
                                 + 0.0338594220440559 w_k1_INTC_s117
                                 + 0.0208339219522737 w_k1_KO_s000
                                 + 0.0253885393368222 w_k1_KO_s001
                                 + 0.0128222081663444 w_k1_KO_s002
                                 + 0.019376541684797 w_k1_KO_s003
                                 + 0.0163685443483845 w_k1_KO_s004
                                 + 0.035258098760872 w_k1_KO_s005
                                 + 0.0345939366137241 w_k1_KO_s006
                                 + 0.0207119706990591 w_k1_KO_s007
                                 + 0.0604326432422678 w_k1_KO_s010
                                 + 0.0342646597948672 w_k1_KO_s011
                                 + 0.069100611171036 w_k1_KO_s012
                                 + 0.0686362786553821 w_k1_KO_s013
                                 + 0.0637644647133469 w_k1_KO_s014
                                 + 0.0408043210692894 w_k1_KO_s015
                                 + 0.0181299349714965 w_k1_KO_s016
                                 + 0.0398485341176364 w_k1_KO_s017
                                 + 0.0130143545222519 w_k1_KO_s100
                                 + 0.0236171555945074 w_k1_KO_s101
                                 + 0.0148438887023817 w_k1_KO_s102
                                 + 0.00791316071050768 w_k1_KO_s103
                                 + 0.0111829705814365 w_k1_KO_s104
                                 + 0.0187383922117793 w_k1_KO_s105
                                 + 0.00879855525925281 w_k1_KO_s106
                                 + 0.0253401713650975 w_k1_KO_s107
                                 + 0.0461478130076718 w_k1_KO_s110
                                 + 0.0232120368791866 w_k1_KO_s111
                                 + 0.0394524874136247 w_k1_KO_s112
                                 + 0.038307966131838 w_k1_KO_s113
                                 + 0.049216864590814 w_k1_KO_s114
                                 + 0.0699377977066931 w_k1_KO_s115
                                 + 0.0129281745946826 w_k1_KO_s116
                                 + 0.0365602350649415 w_k1_KO_s117
                                 + 0.0236408986326395 w_k1_MAT_s000
                                 + 0.0263076174474613 w_k1_MAT_s001
                                 + 0.0129693359351334 w_k1_MAT_s002
                                 + 0.0224437588618674 w_k1_MAT_s003
                                 + 0.017726466091601 w_k1_MAT_s004
                                 + 0.0350757096184196 w_k1_MAT_s005
                                 + 0.0327122056315807 w_k1_MAT_s006
                                 + 0.0231251724773014 w_k1_MAT_s007
                                 + 0.0707910066876599 w_k1_MAT_s010
                                 + 0.0399931049021681 w_k1_MAT_s011
                                 + 0.0745614513622555 w_k1_MAT_s012
                                 + 0.0730045091612971 w_k1_MAT_s013
                                 + 0.0624203800524799 w_k1_MAT_s014
                                 + 0.040426252729576 w_k1_MAT_s015
                                 + 0.0222073118813604 w_k1_MAT_s016
                                 + 0.040965647837293 w_k1_MAT_s017
                                 + 0.0148674624719069 w_k1_MAT_s100
                                 + 0.0212732847387943 w_k1_MAT_s101
                                 + 0.0119877136078527 w_k1_MAT_s102
                                 + 0.00696500810034513 w_k1_MAT_s103
                                 + 0.0101513556823041 w_k1_MAT_s104
                                 + 0.0211993130683262 w_k1_MAT_s105
                                 + 0.0097377561599808 w_k1_MAT_s106
                                 + 0.0279320982853563 w_k1_MAT_s107
                                 + 0.0486275827676802 w_k1_MAT_s110
                                 + 0.0217656171622309 w_k1_MAT_s111
                                 + 0.034182352605562 w_k1_MAT_s112
                                 + 0.0396096638321176 w_k1_MAT_s113
                                 + 0.0637102715392785 w_k1_MAT_s114
                                 + 0.0623418273348929 w_k1_MAT_s115
                                 + 0.0134441257309857 w_k1_MAT_s116
                                 + 0.0339710397007891 w_k1_MAT_s117
                                 + 0.0210266573274846 w_k1_MSFT_s000
                                 + 0.0230507582728086 w_k1_MSFT_s001
                                 + 0.0129302409587186 w_k1_MSFT_s002
                                 + 0.0203771668420071 w_k1_MSFT_s003
                                 + 0.016891045932802 w_k1_MSFT_s004
                                 + 0.0325792535976677 w_k1_MSFT_s005
                                 + 0.0339018539111369 w_k1_MSFT_s006
                                 + 0.0215217308842901 w_k1_MSFT_s007
                                 + 0.0627204602321455 w_k1_MSFT_s010
                                 + 0.0345488522776437 w_k1_MSFT_s011
                                 + 0.0678582546603759 w_k1_MSFT_s012
                                 + 0.0707272824540315 w_k1_MSFT_s013
                                 + 0.0655044290568145 w_k1_MSFT_s014
                                 + 0.0406190174755539 w_k1_MSFT_s015
                                 + 0.0180036666581357 w_k1_MSFT_s016
                                 + 0.0373383043486206 w_k1_MSFT_s017
                                 + 0.0121907938092708 w_k1_MSFT_s100
                                 + 0.0216314653099116 w_k1_MSFT_s101
                                 + 0.0135679120399274 w_k1_MSFT_s102
                                 + 0.00945106405105015 w_k1_MSFT_s103
                                 + 0.0113077324486108 w_k1_MSFT_s104
                                 + 0.0206581037293749 w_k1_MSFT_s105
                                 + 0.00828655667072819 w_k1_MSFT_s106
                                 + 0.0270433976861848 w_k1_MSFT_s107
                                 + 0.0462553551279932 w_k1_MSFT_s110
                                 + 0.0235717938949312 w_k1_MSFT_s111
                                 + 0.0388333049744185 w_k1_MSFT_s112
                                 + 0.0356475058244511 w_k1_MSFT_s113
                                 + 0.0515847262385757 w_k1_MSFT_s114
                                 + 0.0642551568757087 w_k1_MSFT_s115
                                 + 0.0122864865691312 w_k1_MSFT_s116
                                 + 0.0328751322499426 w_k1_MSFT_s117
                                 + 0.021229436011875 w_k1_PG_s000
                                 + 0.0230199325527951 w_k1_PG_s001
                                 + 0.0126295197846018 w_k1_PG_s002
                                 + 0.0187480173015584 w_k1_PG_s003
                                 + 0.0164106291622538 w_k1_PG_s004
                                 + 0.0322742322938435 w_k1_PG_s005
                                 + 0.0336091060754991 w_k1_PG_s006
                                 + 0.021281393504413 w_k1_PG_s007
                                 + 0.0575954074617081 w_k1_PG_s010
                                 + 0.0347451980183755 w_k1_PG_s011
                                 + 0.063744741712069 w_k1_PG_s012
                                 + 0.0677972694798669 w_k1_PG_s013
                                 + 0.0663444635534492 w_k1_PG_s014
                                 + 0.0447856773168372 w_k1_PG_s015
                                 + 0.0195951584930732 w_k1_PG_s016
                                 + 0.0396762164409871 w_k1_PG_s017
                                 + 0.011873357583538 w_k1_PG_s100
                                 + 0.0196762495841182 w_k1_PG_s101
                                 + 0.0140537147230708 w_k1_PG_s102
                                 + 0.00805780622842746 w_k1_PG_s103
                                 + 0.0110483067175005 w_k1_PG_s104
                                 + 0.0198413102579263 w_k1_PG_s105
                                 + 0.00811233097553531 w_k1_PG_s106
                                 + 0.0289999254877532 w_k1_PG_s107
                                 + 0.0448323539212925 w_k1_PG_s110
                                 + 0.0244299744996759 w_k1_PG_s111
                                 + 0.0377899486624384 w_k1_PG_s112
                                 + 0.0372019439070587 w_k1_PG_s113
                                 + 0.0504429168404237 w_k1_PG_s114
                                 + 0.062020655109499 w_k1_PG_s115
                                 + 0.0124389602328269 w_k1_PG_s116
                                 + 0.036033514857611 w_k1_PG_s117 >= 1.07
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
