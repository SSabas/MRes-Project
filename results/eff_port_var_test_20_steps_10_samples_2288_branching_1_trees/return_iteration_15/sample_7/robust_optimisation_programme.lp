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
 w_k1_ADM_s0_return_constraint:  1.01714 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.956784 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.2474 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.984442 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.11737 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.929524 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.21374 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.818077 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.883449 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.06501 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.24165 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.39607 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.904427 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.06179 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.89094 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.17871 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.855191 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.69368 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.08799 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.908928 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.923277 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.40814 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.861485 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.835514 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.02088 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.04979 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.245 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.29353 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.854135 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.29446 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.05414 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.954971 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.24653 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.931583 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.967106 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.655587 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.980865 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.754188 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.07755 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.01708 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.31592 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.8704 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.25591 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.830949 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.43763 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.928538 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.811776 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.41997 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.11725 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.69338 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.820787 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.03058 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.03156 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.14302 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.833159 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.59879 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.22374 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.911347 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.895881 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.43813 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.795114 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.602067 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.969313 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.09878 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.77167 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.20038 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.80306 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.28638 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.43681 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.01305 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.41542 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.653698 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.26771 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.629757 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.994374 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.687987 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.0486 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.945997 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.54284 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.859906 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.38314 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.880385 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.14031 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.06332 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.707152 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.31738 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.66659 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 2.00326 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.651303 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.884232 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.08798 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.22219 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.782383 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.618909 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.33067 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.904567 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.905702 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.75755 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.96887 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.580104 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.780082 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.12395 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.82387 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.03231 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.881539 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.29293 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.42608 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.8735 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.30955 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.671206 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.07383 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.542224 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.94177 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.582021 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.08575 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.901479 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.23306 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.89694 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.24709 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.854106 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.32014 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.986466 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.861915 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.17041 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.33253 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.37303 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.728118 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.04196 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.00888 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.18413 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.804545 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.69157 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.07726 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.924392 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.04878 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.42381 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.951224 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.72424 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.988455 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.06232 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.32428 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.986708 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.90402 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.3369 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.24819 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.19369 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.2784 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.960866 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.990055 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.780452 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.05046 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.75419 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.941989 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.1998 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.38692 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.848199 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.08583 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.994357 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.862548 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.982456 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.656449 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.40762 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.95037 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.84894 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.1076 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.902468 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.03584 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.879613 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.1793 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.662399 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.57781 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.848745 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.605791 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.2332 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.15177 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.764717 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.72376 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.23506 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.8509 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.50753 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.824768 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.03382 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.50997 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.66017 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.05985 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.503722 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.85576 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.547618 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.97074 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.793033 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.999119 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.00109 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.12102 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.891741 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.05393 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.903119 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.36137 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.14568 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.04598 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.35548 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.1936 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.40521 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.730527 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.156 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.14526 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.09745 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.915262 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.806812 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.06653 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.03376 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.04722 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.53148 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.14794 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.786483 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.819607 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.938805 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.31381 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.929572 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.03665 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.37686 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.13106 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.911777 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.08598 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.826956 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.11211 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.797726 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.0261 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.84456 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.976032 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.03686 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.33221 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.926987 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.22158 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.923538 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.13553 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.973262 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.781877 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.3654 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.24187 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.7041 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.888947 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.03501 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.05657 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.03099 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.924771 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.656068 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.27558 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.823924 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.77381 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.33024 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.06311 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.694713 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.796625 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.995402 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.54299 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.04623 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.848007 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.22262 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.35503 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.847885 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.11513 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.723104 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.30413 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.592979 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.896141 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.756879 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.962356 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.19719 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.57401 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.909706 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.29305 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.947011 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.902856 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.823684 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.544297 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.54569 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.12602 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.65005 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.08746 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.878047 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.971452 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.883066 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.9095 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.509615 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.27199 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.759644 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.53436 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.00607 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.908947 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.613456 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.766904 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.19914 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.65799 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.4446 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.740075 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.12576 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.57117 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.776188 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.06085 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.50363 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.81112 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.583258 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.985096 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.591521 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.0561 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.952475 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.17437 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.908655 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.18969 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.896805 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.31897 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.02243 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.905923 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.22789 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.29083 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.44572 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.753165 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.06013 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.04661 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.1391 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.76445 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.732288 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.04457 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.900985 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.996509 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.40268 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.967469 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.695061 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.854573 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.985698 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.23043 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.944919 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.967845 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.36141 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.24788 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.03589 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.13321 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.867211 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.999826 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.77929 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.959121 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.691944 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.00987 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.980862 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.24572 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.915543 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.07857 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.915727 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.04772 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.907033 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.817438 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.09043 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.07933 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.26127 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.810857 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.0173 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.07132 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.09297 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.937952 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.785223 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.10102 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.995835 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.984592 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.2858 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.13397 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.842888 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.945719 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.0437 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.29251 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.23365 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.97958 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.1895 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.27695 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.02341 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.24947 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.815795 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.1097 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.744145 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.986811 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.812205 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.279999648141838 z_k1_s0000
                                 - 0.0580000420722174 z_k1_s0001
                                 - 0.0710000301573734 z_k1_s0002
                                 - 0.25000043748365 z_k1_s0003
                                 - 0.100999983615429 z_k1_s0004
                                 - 0.201999472030946 z_k1_s0005
                                 - 0.169999653728997 z_k1_s0006
                                 - 0.106000017125098 z_k1_s0007
                                 - 0.219000548492792 z_k1_s0010
                                 - 0.404999961956765 z_k1_s0011
                                 - 0.273000640053924 z_k1_s0012
                                 - 0.583001885065842 z_k1_s0013
                                 - 0.481000327035959 z_k1_s0014
                                 - 0.171000531527494 z_k1_s0015
                                 - 0.404999961956765 z_k1_s0016
                                 - 0.362001891509153 z_k1_s0017
                                 - 0.315999353516322 z_k1_s0020
                                 - 0.718001924879556 z_k1_s0021
                                 - 0.204000205393524 z_k1_s0022
                                 - 0.158000197358753 z_k1_s0023
                                 - 0.191000288003359 z_k1_s0024
                                 - 0.471000370862872 z_k1_s0025
                                 - 0.180000257788491 z_k1_s0026
                                 - 0.365000884265962 z_k1_s0027
                                 - 0.33099887224666 z_k1_s0030
                                 - 0.580999148296078 z_k1_s0031
                                 - 0.304000158844711 z_k1_s0032
                                 - 0.27799857794901 z_k1_s0033
                                 - 0.392997919014496 z_k1_s0034
                                 - 0.304000158844711 z_k1_s0035
                                 - 0.135999541085017 z_k1_s0036
                                 - 0.12899947441184 z_k1_s0037
                                 - 0.254000602889717 z_k1_s0040
                                 - 0.356999136294489 z_k1_s0041
                                 - 0.247999180255082 z_k1_s0042
                                 - 0.169000061391323 z_k1_s0043
                                 - 0.274000638007022 z_k1_s0044
                                 - 0.081999987081014 z_k1_s0045
                                 - 0.0940000081513973 z_k1_s0046
                                 - 0.0909999244338324 z_k1_s0047
                                 - 0.0950000065061491 z_k1_s0050
                                 - 0.0670000224474923 z_k1_s0051
                                 - 0.10199957837087 z_k1_s0052
                                 - 0.160000336620839 z_k1_s0053
                                 - 0.140000419293218 z_k1_s0054
                                 - 0.075999985244922 z_k1_s0055
                                 - 0.20700014234075 z_k1_s0056
                                 - 0.15100037382341 z_k1_s0057
                                 - 0.409999186030475 z_k1_s0060
                                 - 0.493999001378215 z_k1_s0061
                                 - 0.330999316036847 z_k1_s0062
                                 - 0.55099874511442 z_k1_s0063
                                 - 0.772998519032968 z_k1_s0064
                                 - 0.469998530279819 z_k1_s0065
                                 - 0.336999250461781 z_k1_s0066
                                 - 0.300999277212846 z_k1_s0067
                                 - 0.370000335756524 z_k1_s0070
                                 - 0.449000909285615 z_k1_s0071
                                 - 0.453998404620187 z_k1_s0072
                                 - 0.626000066223205 z_k1_s0073
                                 - 0.465001258365173 z_k1_s0074
                                 - 0.30500027677227 z_k1_s0075
                                 - 0.32400037765391 z_k1_s0076
                                 - 1.18900241719956 z_k1_s0077
                                 - 0.277999900162393 z_k1_s0100
                                 - 0.345999837583407 z_k1_s0101
                                 - 0.32900018472818 z_k1_s0102
                                 - 0.329999988807885 z_k1_s0103
                                 - 0.802999144015788 z_k1_s0104
                                 - 0.740998694073059 z_k1_s0105
                                 - 0.172000114711998 z_k1_s0106
                                 - 0.3160000796918 z_k1_s0107
                                 - 1.19300084631044 z_k1_s0110
                                 - 0.358000683571287 z_k1_s0111
                                 - 0.250000587985732 z_k1_s0112
                                 - 0.778999536897468 z_k1_s0113
                                 - 0.407000936494734 z_k1_s0114
                                 - 0.692002917570794 z_k1_s0115
                                 - 0.459000877739445 z_k1_s0116
                                 - 0.576999148561151 z_k1_s0117
                                 - 1.34199915662798 z_k1_s0120
                                 - 1.72999796380665 z_k1_s0121
                                 - 0.752999068960424 z_k1_s0122
                                 - 1.10400246093165 z_k1_s0123
                                 - 0.694998883552875 z_k1_s0124
                                 - 1.63400074285111 z_k1_s0125
                                 - 0.78999878666073 z_k1_s0126
                                 - 0.556999006185706 z_k1_s0127
                                 - 0.924996741837465 z_k1_s0130
                                 - 1.52000104483234 z_k1_s0131
                                 - 0.536999977528468 z_k1_s0132
                                 - 1.38099772408682 z_k1_s0133
                                 - 0.348999725099268 z_k1_s0134
                                 - 0.374999980571278 z_k1_s0135
                                 - 1.26100168959799 z_k1_s0136
                                 - 0.339999688112777 z_k1_s0137
                                 - 0.505999939560292 z_k1_s0140
                                 - 1.0890036783156 z_k1_s0141
                                 - 1.23099514763304 z_k1_s0142
                                 - 2.00699672683252 z_k1_s0143
                                 - 0.86799967029282 z_k1_s0144
                                 - 1.5290020471511 z_k1_s0145
                                 - 1.04800288135166 z_k1_s0146
                                 - 0.79899996734815 z_k1_s0147
                                 - 0.615000543999873 z_k1_s0150
                                 - 0.468000156507277 z_k1_s0151
                                 - 0.401999999102435 z_k1_s0152
                                 - 0.579000531960852 z_k1_s0153
                                 - 0.408001489775276 z_k1_s0154
                                 - 0.219999983349595 z_k1_s0155
                                 - 0.629998180683322 z_k1_s0156
                                 - 0.737998216800386 z_k1_s0157
                                 - 0.319000083026275 z_k1_s0160
                                 - 0.379998304658378 z_k1_s0161
                                 - 0.807998133115593 z_k1_s0162
                                 - 0.640999940252498 z_k1_s0163
                                 - 0.354999941725488 z_k1_s0164
                                 - 0.310999829059802 z_k1_s0165
                                 - 0.239000111661375 z_k1_s0166
                                 - 0.615999375919755 z_k1_s0167
                                 - 0.424000579326732 z_k1_s0170
                                 - 0.879002194798699 z_k1_s0171
                                 - 0.860999960551333 z_k1_s0172
                                 - 0.429000480672454 z_k1_s0173
                                 - 1.49799892660856 z_k1_s0174
                                 - 0.958002519263109 z_k1_s0175
                                 - 0.515000431620622 z_k1_s0176
                                 - 0.321000610702734 z_k1_s0177
                                 - 0.156000290215595 z_k1_s1000
                                 - 0.216999585421529 z_k1_s1001
                                 - 0.206000123220541 z_k1_s1002
                                 - 0.25299956062511 z_k1_s1003
                                 - 0.0950000406094698 z_k1_s1004
                                 - 0.131999511413049 z_k1_s1005
                                 - 0.0789999986078678 z_k1_s1006
                                 - 0.0550000549054888 z_k1_s1007
                                 - 0.244000186264796 z_k1_s1010
                                 - 0.26700019439631 z_k1_s1011
                                 - 0.27399909687073 z_k1_s1012
                                 - 0.316999912073409 z_k1_s1013
                                 - 0.139000089142644 z_k1_s1014
                                 - 0.285000000760033 z_k1_s1015
                                 - 0.20600003283007 z_k1_s1016
                                 - 0.568001300813438 z_k1_s1017
                                 - 0.277000039186696 z_k1_s1020
                                 - 0.277000039186696 z_k1_s1021
                                 - 0.201999913949522 z_k1_s1022
                                 - 0.523999096118671 z_k1_s1023
                                 - 0.747000862846314 z_k1_s1024
                                 - 0.155999864325385 z_k1_s1025
                                 - 0.342999869747454 z_k1_s1026
                                 - 0.387000727787805 z_k1_s1027
                                 - 0.141000505030715 z_k1_s1030
                                 - 0.188999768998657 z_k1_s1031
                                 - 0.169999732803027 z_k1_s1032
                                 - 0.148000163734383 z_k1_s1033
                                 - 0.131999660411768 z_k1_s1034
                                 - 0.0639999918905475 z_k1_s1035
                                 - 0.161000011184032 z_k1_s1036
                                 - 0.118000343004432 z_k1_s1037
                                 - 0.298999350520999 z_k1_s1040
                                 - 0.111999518382419 z_k1_s1041
                                 - 0.16099918474398 z_k1_s1042
                                 - 0.189998737177355 z_k1_s1043
                                 - 0.0849996898730378 z_k1_s1044
                                 - 0.179999361060478 z_k1_s1045
                                 - 0.104999501202461 z_k1_s1046
                                 - 0.380999076259592 z_k1_s1047
                                 - 0.261999740976648 z_k1_s1050
                                 - 0.0810000675844629 z_k1_s1051
                                 - 0.128000060874188 z_k1_s1052
                                 - 0.168999690935242 z_k1_s1053
                                 - 0.204000445768425 z_k1_s1054
                                 - 0.181000417392585 z_k1_s1055
                                 - 0.191999719311082 z_k1_s1056
                                 - 0.271000657819855 z_k1_s1057
                                 - 0.3490004708353 z_k1_s1060
                                 - 0.307001176397801 z_k1_s1061
                                 - 0.404999530085298 z_k1_s1062
                                 - 0.136000099782113 z_k1_s1063
                                 - 0.525001612484245 z_k1_s1064
                                 - 0.42700065079287 z_k1_s1065
                                 - 0.127000139202482 z_k1_s1066
                                 - 0.332001501564086 z_k1_s1067
                                 - 0.409998468308225 z_k1_s1070
                                 - 0.242999050507205 z_k1_s1071
                                 - 0.127999591288684 z_k1_s1072
                                 - 0.445997694644677 z_k1_s1073
                                 - 0.167999427960263 z_k1_s1074
                                 - 0.269000074245069 z_k1_s1075
                                 - 0.101999706947127 z_k1_s1076
                                 - 0.132999642084901 z_k1_s1077
                                 - 0.354998100763836 z_k1_s1100
                                 - 0.41499858398889 z_k1_s1101
                                 - 0.423997514476393 z_k1_s1102
                                 - 0.237999156968429 z_k1_s1103
                                 - 0.134999659761051 z_k1_s1104
                                 - 0.521997923078562 z_k1_s1105
                                 - 0.477999662373323 z_k1_s1106
                                 - 0.288000035487746 z_k1_s1107
                                 - 0.180999394055626 z_k1_s1110
                                 - 0.364999827681084 z_k1_s1111
                                 - 0.250000764165 z_k1_s1112
                                 - 0.156999932495698 z_k1_s1113
                                 - 0.25400000359175 z_k1_s1114
                                 - 0.127000001795875 z_k1_s1115
                                 - 0.0789999516761815 z_k1_s1116
                                 - 0.196999571762167 z_k1_s1117
                                 - 0.335998251099989 z_k1_s1120
                                 - 0.240999936622036 z_k1_s1121
                                 - 0.0709998615383544 z_k1_s1122
                                 - 0.315998710735923 z_k1_s1123
                                 - 0.262998435525228 z_k1_s1124
                                 - 0.303998986517483 z_k1_s1125
                                 - 0.132999631263695 z_k1_s1126
                                 - 0.326998955684799 z_k1_s1127
                                 - 0.237999683853079 z_k1_s1130
                                 - 0.0769999431347929 z_k1_s1131
                                 - 0.179000248842994 z_k1_s1132
                                 - 0.100999885301505 z_k1_s1133
                                 - 0.142999453193448 z_k1_s1134
                                 - 0.132000343716526 z_k1_s1135
                                 - 0.116999552612821 z_k1_s1136
                                 - 0.116000235205472 z_k1_s1137
                                 - 0.230999478673524 z_k1_s1140
                                 - 0.237000227132381 z_k1_s1141
                                 - 0.0909999724289184 z_k1_s1142
                                 - 0.255000694509408 z_k1_s1143
                                 - 0.134999949328618 z_k1_s1144
                                 - 0.418999816390309 z_k1_s1145
                                 - 0.134999949328618 z_k1_s1146
                                 - 0.274999633373178 z_k1_s1147
                                 - 0.361999067445683 z_k1_s1150
                                 - 0.241999533827904 z_k1_s1151
                                 - 0.231999233860672 z_k1_s1152
                                 - 0.178999473729395 z_k1_s1153
                                 - 0.228000333670482 z_k1_s1154
                                 - 0.0869997635226147 z_k1_s1155
                                 - 0.275999333919791 z_k1_s1156
                                 - 0.426997967740936 z_k1_s1157
                                 - 0.360000186774684 z_k1_s1160
                                 - 1.22000060263084 z_k1_s1161
                                 - 0.638998543700444 z_k1_s1162
                                 - 0.559002262843863 z_k1_s1163
                                 - 0.735999629720685 z_k1_s1164
                                 - 0.72900118342219 z_k1_s1165
                                 - 0.457000726894277 z_k1_s1166
                                 - 0.75899842399179 z_k1_s1167
                                 - 0.481000760526502 z_k1_s1170
                                 - 0.223000756690239 z_k1_s1171
                                 - 0.1260002993349 z_k1_s1172
                                 - 0.375002324024922 z_k1_s1173
                                 - 0.528000824411596 z_k1_s1174
                                 - 0.221000620567061 z_k1_s1175
                                 - 0.370001356715056 z_k1_s1176
                                 - 0.184000484895555 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.22542 w_k1_ADM_s000
                                 + 1.03431 w_k1_AXP_s000
                                 + 1.17876 w_k1_DIS_s000
                                 + 1.17997 w_k1_FDX_s000
                                 + 0.914569 w_k1_IBM_s000
                                 + 1.09195 w_k1_INTC_s000
                                 + 0.960475 w_k1_KO_s000
                                 + 0.942814 w_k1_MAT_s000
                                 + 0.988244 w_k1_MSFT_s000
                                 + 1.05647 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.04769 w_k1_ADM_s000
                                 + 0.93642 w_k1_AXP_s000
                                 + 0.751601 w_k1_DIS_s000
                                 + 0.926993 w_k1_FDX_s000
                                 + 0.61877 w_k1_IBM_s000
                                 + 0.802279 w_k1_INTC_s000
                                 + 0.667659 w_k1_KO_s000
                                 + 0.79343 w_k1_MAT_s000
                                 + 0.815513 w_k1_MSFT_s000
                                 + 0.838949 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.29549 w_k1_ADM_s000
                                 + 1.79184 w_k1_AXP_s000
                                 + 1.35374 w_k1_DIS_s000
                                 + 1.35179 w_k1_FDX_s000
                                 + 1.06304 w_k1_IBM_s000
                                 + 1.17875 w_k1_INTC_s000
                                 + 1.06686 w_k1_KO_s000 + 1.5275 w_k1_MAT_s000
                                 + 0.953225 w_k1_MSFT_s000
                                 + 1.17888 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.04074 w_k1_ADM_s000
                                 + 1.1334 w_k1_AXP_s000
                                 + 0.940647 w_k1_DIS_s000
                                 + 0.937246 w_k1_FDX_s000
                                 + 1.28288 w_k1_IBM_s000
                                 + 0.816811 w_k1_INTC_s000
                                 + 1.01019 w_k1_KO_s000 + 1.37089 w_k1_MAT_s000
                                 + 0.923626 w_k1_MSFT_s000
                                 + 1.02947 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.941764 w_k1_ADM_s000
                                 + 0.628471 w_k1_AXP_s000
                                 + 0.719792 w_k1_DIS_s000
                                 + 0.744929 w_k1_FDX_s000
                                 + 0.773417 w_k1_IBM_s000
                                 + 0.96853 w_k1_INTC_s000
                                 + 0.752688 w_k1_KO_s000
                                 + 0.677267 w_k1_MAT_s000
                                 + 0.76705 w_k1_MSFT_s000
                                 + 0.885507 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.788759 w_k1_ADM_s000
                                 + 0.809707 w_k1_AXP_s000
                                 + 0.814299 w_k1_DIS_s000
                                 + 0.794236 w_k1_FDX_s000
                                 + 1.29935 w_k1_IBM_s000
                                 + 0.791908 w_k1_INTC_s000
                                 + 0.865073 w_k1_KO_s000
                                 + 1.09775 w_k1_MAT_s000
                                 + 0.74015 w_k1_MSFT_s000
                                 + 0.859748 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.752994 w_k1_ADM_s000
                                 + 1.0174 w_k1_AXP_s000 + 0.97608 w_k1_DIS_s000
                                 + 0.912113 w_k1_FDX_s000
                                 + 0.898853 w_k1_IBM_s000
                                 + 0.766153 w_k1_INTC_s000
                                 + 0.825776 w_k1_KO_s000
                                 + 1.49555 w_k1_MAT_s000
                                 + 0.736316 w_k1_MSFT_s000
                                 + 0.976094 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.45637 w_k1_ADM_s000
                                 + 1.44393 w_k1_AXP_s000
                                 + 1.67873 w_k1_DIS_s000
                                 + 1.31626 w_k1_FDX_s000
                                 + 1.43005 w_k1_IBM_s000
                                 + 0.978113 w_k1_INTC_s000
                                 + 1.2544 w_k1_KO_s000 + 1.64434 w_k1_MAT_s000
                                 + 1.15154 w_k1_MSFT_s000
                                 + 1.12362 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.6074 w_k1_ADM_s001
                                 + 1.01231 w_k1_AXP_s001
                                 + 1.60649 w_k1_DIS_s001
                                 + 1.24578 w_k1_FDX_s001
                                 + 0.985127 w_k1_IBM_s001
                                 + 1.23719 w_k1_INTC_s001
                                 + 1.29299 w_k1_KO_s001
                                 + 0.782088 w_k1_MAT_s001
                                 + 1.39026 w_k1_MSFT_s001
                                 + 1.29023 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.20301 w_k1_ADM_s001
                                 + 0.809685 w_k1_AXP_s001
                                 + 1.13881 w_k1_DIS_s001
                                 + 1.05449 w_k1_FDX_s001
                                 + 0.707761 w_k1_IBM_s001
                                 + 1.11505 w_k1_INTC_s001 + 1.018 w_k1_KO_s001
                                 + 0.675491 w_k1_MAT_s001
                                 + 1.14143 w_k1_MSFT_s001
                                 + 1.07799 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.48744 w_k1_ADM_s001
                                 + 1.37063 w_k1_AXP_s001
                                 + 1.51277 w_k1_DIS_s001
                                 + 1.33286 w_k1_FDX_s001
                                 + 1.09141 w_k1_IBM_s001
                                 + 1.16572 w_k1_INTC_s001
                                 + 1.18914 w_k1_KO_s001 + 1.47721 w_k1_MAT_s001
                                 + 1.2719 w_k1_MSFT_s001 + 1.14467 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.17979 w_k1_ADM_s001
                                 + 1.07686 w_k1_AXP_s001
                                 + 1.02522 w_k1_DIS_s001
                                 + 0.969468 w_k1_FDX_s001
                                 + 1.05733 w_k1_IBM_s001
                                 + 0.946006 w_k1_INTC_s001
                                 + 1.04884 w_k1_KO_s001 + 1.16873 w_k1_MAT_s001
                                 + 1.04921 w_k1_MSFT_s001
                                 + 1.11771 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.926925 w_k1_ADM_s001
                                 + 0.783307 w_k1_AXP_s001
                                 + 0.763812 w_k1_DIS_s001
                                 + 0.871736 w_k1_FDX_s001
                                 + 0.683938 w_k1_IBM_s001
                                 + 0.972502 w_k1_INTC_s001
                                 + 0.762893 w_k1_KO_s001
                                 + 0.546569 w_k1_MAT_s001
                                 + 0.924046 w_k1_MSFT_s001
                                 + 0.870126 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.798069 w_k1_ADM_s001
                                 + 0.625849 w_k1_AXP_s001
                                 + 0.603274 w_k1_DIS_s001
                                 + 0.666841 w_k1_FDX_s001
                                 + 0.819944 w_k1_IBM_s001
                                 + 0.731928 w_k1_INTC_s001
                                 + 0.822971 w_k1_KO_s001
                                 + 0.880888 w_k1_MAT_s001
                                 + 0.752321 w_k1_MSFT_s001
                                 + 0.864102 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.27608 w_k1_ADM_s001
                                 + 0.95236 w_k1_AXP_s001
                                 + 1.17074 w_k1_DIS_s001
                                 + 0.927822 w_k1_FDX_s001
                                 + 1.62423 w_k1_IBM_s001
                                 + 0.879649 w_k1_INTC_s001
                                 + 1.45973 w_k1_KO_s001 + 1.73319 w_k1_MAT_s001
                                 + 1.15812 w_k1_MSFT_s001
                                 + 1.18749 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.930696 w_k1_ADM_s001
                                 + 0.703291 w_k1_AXP_s001
                                 + 0.801044 w_k1_DIS_s001
                                 + 0.809987 w_k1_FDX_s001
                                 + 1.09409 w_k1_IBM_s001
                                 + 0.847671 w_k1_INTC_s001
                                 + 0.929551 w_k1_KO_s001
                                 + 0.812717 w_k1_MAT_s001
                                 + 0.875967 w_k1_MSFT_s001
                                 + 0.885366 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.878807 w_k1_ADM_s002
                                 + 0.761796 w_k1_AXP_s002
                                 + 0.71246 w_k1_DIS_s002
                                 + 0.865414 w_k1_FDX_s002
                                 + 0.777338 w_k1_IBM_s002
                                 + 0.712568 w_k1_INTC_s002
                                 + 0.746093 w_k1_KO_s002
                                 + 0.981186 w_k1_MAT_s002
                                 + 0.83954 w_k1_MSFT_s002
                                 + 0.961124 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.07214 w_k1_ADM_s002
                                 + 0.954089 w_k1_AXP_s002
                                 + 0.987888 w_k1_DIS_s002
                                 + 1.03198 w_k1_FDX_s002
                                 + 0.852323 w_k1_IBM_s002
                                 + 0.998762 w_k1_INTC_s002
                                 + 0.899608 w_k1_KO_s002
                                 + 1.07414 w_k1_MAT_s002
                                 + 0.997621 w_k1_MSFT_s002
                                 + 0.990048 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.00535 w_k1_ADM_s002
                                 + 1.1081 w_k1_AXP_s002 + 1.0767 w_k1_DIS_s002
                                 + 0.93412 w_k1_FDX_s002
                                 + 1.45226 w_k1_IBM_s002
                                 + 1.01392 w_k1_INTC_s002
                                 + 1.16084 w_k1_KO_s002 + 1.29248 w_k1_MAT_s002
                                 + 1.00066 w_k1_MSFT_s002
                                 + 1.02435 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.659126 w_k1_ADM_s002
                                 + 0.738351 w_k1_AXP_s002
                                 + 0.839153 w_k1_DIS_s002
                                 + 0.78862 w_k1_FDX_s002
                                 + 1.07428 w_k1_IBM_s002
                                 + 0.862487 w_k1_INTC_s002
                                 + 0.978368 w_k1_KO_s002
                                 + 0.91145 w_k1_MAT_s002
                                 + 0.772414 w_k1_MSFT_s002
                                 + 0.858592 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.771114 w_k1_ADM_s002
                                 + 1.06681 w_k1_AXP_s002
                                 + 0.796387 w_k1_DIS_s002
                                 + 0.915423 w_k1_FDX_s002
                                 + 1.23925 w_k1_IBM_s002
                                 + 0.936623 w_k1_INTC_s002
                                 + 1.03643 w_k1_KO_s002 + 1.40066 w_k1_MAT_s002
                                 + 0.805118 w_k1_MSFT_s002
                                 + 1.0022 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.34343 w_k1_ADM_s002
                                 + 1.39653 w_k1_AXP_s002
                                 + 1.37168 w_k1_DIS_s002
                                 + 1.29646 w_k1_FDX_s002
                                 + 0.977466 w_k1_IBM_s002
                                 + 0.960254 w_k1_INTC_s002
                                 + 1.10502 w_k1_KO_s002 + 1.19067 w_k1_MAT_s002
                                 + 1.2176 w_k1_MSFT_s002 + 1.22708 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.588557 w_k1_ADM_s002
                                 + 0.667536 w_k1_AXP_s002
                                 + 0.555501 w_k1_DIS_s002
                                 + 0.711871 w_k1_FDX_s002
                                 + 0.660658 w_k1_IBM_s002
                                 + 0.676214 w_k1_INTC_s002
                                 + 0.697373 w_k1_KO_s002
                                 + 1.09238 w_k1_MAT_s002
                                 + 0.730518 w_k1_MSFT_s002
                                 + 0.729527 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.10014 w_k1_ADM_s002
                                 + 1.20337 w_k1_AXP_s002
                                 + 1.54575 w_k1_DIS_s002
                                 + 1.28581 w_k1_FDX_s002
                                 + 1.41492 w_k1_IBM_s002
                                 + 1.08909 w_k1_INTC_s002 + 1.3366 w_k1_KO_s002
                                 + 1.53122 w_k1_MAT_s002
                                 + 1.30497 w_k1_MSFT_s002
                                 + 1.20268 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.10441 w_k1_ADM_s003
                                 + 1.00594 w_k1_AXP_s003
                                 + 1.16939 w_k1_DIS_s003
                                 + 1.10553 w_k1_FDX_s003
                                 + 0.802438 w_k1_IBM_s003
                                 + 1.00786 w_k1_INTC_s003
                                 + 0.949126 w_k1_KO_s003
                                 + 0.697725 w_k1_MAT_s003
                                 + 1.04023 w_k1_MSFT_s003
                                 + 1.03188 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.819213 w_k1_ADM_s003
                                 + 0.816621 w_k1_AXP_s003
                                 + 0.740457 w_k1_DIS_s003
                                 + 0.82343 w_k1_FDX_s003
                                 + 0.817388 w_k1_IBM_s003
                                 + 0.870091 w_k1_INTC_s003
                                 + 0.803211 w_k1_KO_s003
                                 + 0.882233 w_k1_MAT_s003
                                 + 0.82615 w_k1_MSFT_s003
                                 + 0.838193 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.50635 w_k1_ADM_s003
                                 + 1.15979 w_k1_AXP_s003
                                 + 1.54993 w_k1_DIS_s003
                                 + 1.38843 w_k1_FDX_s003
                                 + 1.10243 w_k1_IBM_s003
                                 + 1.36324 w_k1_INTC_s003
                                 + 1.31218 w_k1_KO_s003
                                 + 0.914783 w_k1_MAT_s003
                                 + 1.51123 w_k1_MSFT_s003 + 1.3249 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.37673 w_k1_ADM_s003
                                 + 1.59592 w_k1_AXP_s003 + 1.2605 w_k1_DIS_s003
                                 + 1.42589 w_k1_FDX_s003
                                 + 0.852084 w_k1_IBM_s003
                                 + 1.33977 w_k1_INTC_s003
                                 + 0.93017 w_k1_KO_s003
                                 + 0.964242 w_k1_MAT_s003
                                 + 1.1986 w_k1_MSFT_s003 + 1.17339 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.905707 w_k1_ADM_s003
                                 + 1.13406 w_k1_AXP_s003
                                 + 1.07058 w_k1_DIS_s003
                                 + 1.00664 w_k1_FDX_s003
                                 + 1.22688 w_k1_IBM_s003
                                 + 1.27841 w_k1_INTC_s003
                                 + 1.06504 w_k1_KO_s003 + 1.06089 w_k1_MAT_s003
                                 + 1.04012 w_k1_MSFT_s003
                                 + 1.07196 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.07771 w_k1_ADM_s003
                                 + 0.979996 w_k1_AXP_s003
                                 + 0.774578 w_k1_DIS_s003
                                 + 0.852865 w_k1_FDX_s003
                                 + 1.20248 w_k1_IBM_s003
                                 + 0.894369 w_k1_INTC_s003
                                 + 0.989935 w_k1_KO_s003 + 1.0834 w_k1_MAT_s003
                                 + 0.713055 w_k1_MSFT_s003
                                 + 0.994688 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.00182 w_k1_ADM_s003
                                 + 1.22276 w_k1_AXP_s003
                                 + 1.06614 w_k1_DIS_s003
                                 + 1.04236 w_k1_FDX_s003
                                 + 1.80603 w_k1_IBM_s003
                                 + 1.12127 w_k1_INTC_s003
                                 + 1.14222 w_k1_KO_s003 + 1.99106 w_k1_MAT_s003
                                 + 1.07782 w_k1_MSFT_s003
                                 + 1.11529 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.894255 w_k1_ADM_s003
                                 + 0.895007 w_k1_AXP_s003
                                 + 0.843355 w_k1_DIS_s003
                                 + 1.03309 w_k1_FDX_s003
                                 + 0.608462 w_k1_IBM_s003
                                 + 1.0137 w_k1_INTC_s003
                                 + 0.678052 w_k1_KO_s003
                                 + 0.53326 w_k1_MAT_s003
                                 + 0.905698 w_k1_MSFT_s003
                                 + 0.889925 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.915248 w_k1_ADM_s004
                                 + 0.927858 w_k1_AXP_s004
                                 + 0.905477 w_k1_DIS_s004
                                 + 0.913572 w_k1_FDX_s004
                                 + 1.24544 w_k1_IBM_s004
                                 + 1.11115 w_k1_INTC_s004
                                 + 1.13396 w_k1_KO_s004
                                 + 0.907519 w_k1_MAT_s004
                                 + 1.05479 w_k1_MSFT_s004
                                 + 1.08183 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.895545 w_k1_ADM_s004
                                 + 0.836918 w_k1_AXP_s004
                                 + 0.720059 w_k1_DIS_s004
                                 + 0.865715 w_k1_FDX_s004
                                 + 0.716014 w_k1_IBM_s004
                                 + 0.963381 w_k1_INTC_s004
                                 + 0.716395 w_k1_KO_s004
                                 + 0.434193 w_k1_MAT_s004
                                 + 0.881339 w_k1_MSFT_s004
                                 + 0.842456 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.818713 w_k1_ADM_s004
                                 + 0.792006 w_k1_AXP_s004
                                 + 0.74686 w_k1_DIS_s004
                                 + 0.799613 w_k1_FDX_s004
                                 + 1.11276 w_k1_IBM_s004
                                 + 0.963643 w_k1_INTC_s004
                                 + 0.913908 w_k1_KO_s004
                                 + 0.646506 w_k1_MAT_s004
                                 + 0.879347 w_k1_MSFT_s004
                                 + 0.920029 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.10207 w_k1_ADM_s004
                                 + 1.19313 w_k1_AXP_s004
                                 + 1.33027 w_k1_DIS_s004
                                 + 1.18739 w_k1_FDX_s004
                                 + 0.86117 w_k1_IBM_s004
                                 + 1.24429 w_k1_INTC_s004
                                 + 1.03669 w_k1_KO_s004
                                 + 0.624167 w_k1_MAT_s004
                                 + 1.27212 w_k1_MSFT_s004
                                 + 0.984233 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.19649 w_k1_ADM_s004
                                 + 1.29453 w_k1_AXP_s004
                                 + 1.03637 w_k1_DIS_s004
                                 + 1.08466 w_k1_FDX_s004
                                 + 0.927366 w_k1_IBM_s004
                                 + 1.16041 w_k1_INTC_s004
                                 + 0.983978 w_k1_KO_s004
                                 + 0.690448 w_k1_MAT_s004
                                 + 1.12883 w_k1_MSFT_s004
                                 + 1.10467 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.998361 w_k1_ADM_s004
                                 + 1.42527 w_k1_AXP_s004
                                 + 1.61069 w_k1_DIS_s004
                                 + 1.33875 w_k1_FDX_s004
                                 + 1.61149 w_k1_IBM_s004
                                 + 1.2104 w_k1_INTC_s004 + 1.48533 w_k1_KO_s004
                                 + 1.28952 w_k1_MAT_s004
                                 + 1.44067 w_k1_MSFT_s004
                                 + 1.45014 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.9844 w_k1_ADM_s004
                                 + 1.18611 w_k1_AXP_s004
                                 + 1.26628 w_k1_DIS_s004
                                 + 1.04189 w_k1_FDX_s004
                                 + 1.78091 w_k1_IBM_s004
                                 + 1.19483 w_k1_INTC_s004
                                 + 1.46728 w_k1_KO_s004 + 1.38232 w_k1_MAT_s004
                                 + 1.16188 w_k1_MSFT_s004
                                 + 1.16443 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.37863 w_k1_ADM_s004
                                 + 1.73542 w_k1_AXP_s004
                                 + 1.65507 w_k1_DIS_s004
                                 + 1.54648 w_k1_FDX_s004
                                 + 0.944797 w_k1_IBM_s004
                                 + 1.5138 w_k1_INTC_s004 + 1.30461 w_k1_KO_s004
                                 + 0.750872 w_k1_MAT_s004
                                 + 1.35673 w_k1_MSFT_s004
                                 + 1.31345 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.45637 w_k1_ADM_s005
                                 + 1.55332 w_k1_AXP_s005
                                 + 1.12409 w_k1_DIS_s005 + 1.4303 w_k1_FDX_s005
                                 + 1.03509 w_k1_IBM_s005
                                 + 1.12235 w_k1_INTC_s005
                                 + 1.05767 w_k1_KO_s005 + 1.34501 w_k1_MAT_s005
                                 + 1.05622 w_k1_MSFT_s005
                                 + 1.21896 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.2954 w_k1_ADM_s005
                                 + 1.34919 w_k1_AXP_s005
                                 + 1.82716 w_k1_DIS_s005
                                 + 1.46143 w_k1_FDX_s005 + 1.4007 w_k1_IBM_s005
                                 + 1.3657 w_k1_INTC_s005 + 1.56496 w_k1_KO_s005
                                 + 1.96721 w_k1_MAT_s005
                                 + 1.51919 w_k1_MSFT_s005
                                 + 1.24051 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.09341 w_k1_ADM_s005
                                 + 1.31735 w_k1_AXP_s005
                                 + 0.855555 w_k1_DIS_s005
                                 + 1.00214 w_k1_FDX_s005
                                 + 1.42191 w_k1_IBM_s005
                                 + 0.817701 w_k1_INTC_s005
                                 + 1.00343 w_k1_KO_s005 + 1.88019 w_k1_MAT_s005
                                 + 0.890713 w_k1_MSFT_s005
                                 + 1.14326 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.708737 w_k1_ADM_s005
                                 + 0.61978 w_k1_AXP_s005
                                 + 0.51893 w_k1_DIS_s005
                                 + 0.673503 w_k1_FDX_s005
                                 + 1.00995 w_k1_IBM_s005
                                 + 0.669336 w_k1_INTC_s005
                                 + 0.766871 w_k1_KO_s005
                                 + 1.15674 w_k1_MAT_s005
                                 + 0.716766 w_k1_MSFT_s005
                                 + 0.801136 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.18322 w_k1_ADM_s005
                                 + 1.05806 w_k1_AXP_s005
                                 + 1.20225 w_k1_DIS_s005
                                 + 1.34257 w_k1_FDX_s005
                                 + 0.659401 w_k1_IBM_s005
                                 + 1.39946 w_k1_INTC_s005
                                 + 0.866652 w_k1_KO_s005 + 0.8445 w_k1_MAT_s005
                                 + 1.10756 w_k1_MSFT_s005
                                 + 0.973568 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.12923 w_k1_ADM_s005
                                 + 1.01481 w_k1_AXP_s005
                                 + 1.22267 w_k1_DIS_s005
                                 + 1.01316 w_k1_FDX_s005
                                 + 1.17593 w_k1_IBM_s005
                                 + 1.08316 w_k1_INTC_s005
                                 + 1.20856 w_k1_KO_s005 + 1.60567 w_k1_MAT_s005
                                 + 1.19731 w_k1_MSFT_s005
                                 + 1.20949 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.745123 w_k1_ADM_s005
                                 + 0.903041 w_k1_AXP_s005
                                 + 0.910296 w_k1_DIS_s005
                                 + 0.937397 w_k1_FDX_s005
                                 + 0.815316 w_k1_IBM_s005
                                 + 0.938345 w_k1_INTC_s005
                                 + 0.923161 w_k1_KO_s005
                                 + 0.820116 w_k1_MAT_s005
                                 + 0.933502 w_k1_MSFT_s005
                                 + 0.948709 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.975984 w_k1_ADM_s005
                                 + 0.852452 w_k1_AXP_s005
                                 + 0.753552 w_k1_DIS_s005
                                 + 0.786713 w_k1_FDX_s005
                                 + 1.22433 w_k1_IBM_s005
                                 + 0.838562 w_k1_INTC_s005
                                 + 1.13453 w_k1_KO_s005 + 1.38895 w_k1_MAT_s005
                                 + 0.797237 w_k1_MSFT_s005
                                 + 0.967724 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.09429 w_k1_ADM_s006
                                 + 1.47191 w_k1_AXP_s006
                                 + 1.48329 w_k1_DIS_s006
                                 + 1.25784 w_k1_FDX_s006
                                 + 1.67483 w_k1_IBM_s006
                                 + 1.50326 w_k1_INTC_s006
                                 + 1.50199 w_k1_KO_s006 + 1.45055 w_k1_MAT_s006
                                 + 1.32074 w_k1_MSFT_s006
                                 + 1.36881 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.820952 w_k1_ADM_s006
                                 + 0.696759 w_k1_AXP_s006
                                 + 0.852246 w_k1_DIS_s006
                                 + 0.910157 w_k1_FDX_s006
                                 + 0.993987 w_k1_IBM_s006
                                 + 1.00296 w_k1_INTC_s006
                                 + 0.89074 w_k1_KO_s006
                                 + 0.683222 w_k1_MAT_s006
                                 + 0.993274 w_k1_MSFT_s006
                                 + 0.99269 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.886835 w_k1_ADM_s006
                                 + 0.77824 w_k1_AXP_s006
                                 + 0.787989 w_k1_DIS_s006
                                 + 0.814771 w_k1_FDX_s006
                                 + 0.873286 w_k1_IBM_s006
                                 + 0.99864 w_k1_INTC_s006
                                 + 0.966224 w_k1_KO_s006
                                 + 0.971234 w_k1_MAT_s006
                                 + 0.954884 w_k1_MSFT_s006
                                 + 0.909793 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.982618 w_k1_ADM_s006
                                 + 1.10098 w_k1_AXP_s006
                                 + 1.39382 w_k1_DIS_s006
                                 + 1.23942 w_k1_FDX_s006
                                 + 0.976477 w_k1_IBM_s006
                                 + 1.51763 w_k1_INTC_s006
                                 + 1.22422 w_k1_KO_s006
                                 + 0.861539 w_k1_MAT_s006
                                 + 1.30906 w_k1_MSFT_s006
                                 + 1.28704 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.980593 w_k1_ADM_s006
                                 + 1.07825 w_k1_AXP_s006
                                 + 1.08601 w_k1_DIS_s006
                                 + 0.976242 w_k1_FDX_s006
                                 + 1.30742 w_k1_IBM_s006
                                 + 1.26312 w_k1_INTC_s006
                                 + 1.12181 w_k1_KO_s006 + 1.00538 w_k1_MAT_s006
                                 + 0.967235 w_k1_MSFT_s006
                                 + 1.13955 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.654354 w_k1_ADM_s006
                                 + 0.600605 w_k1_AXP_s006
                                 + 0.587883 w_k1_DIS_s006
                                 + 0.788486 w_k1_FDX_s006
                                 + 0.619775 w_k1_IBM_s006
                                 + 1.06308 w_k1_INTC_s006
                                 + 0.674042 w_k1_KO_s006
                                 + 0.613207 w_k1_MAT_s006
                                 + 0.827095 w_k1_MSFT_s006
                                 + 0.840855 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.996583 w_k1_ADM_s006
                                 + 0.814431 w_k1_AXP_s006
                                 + 1.05109 w_k1_DIS_s006 + 1.077 w_k1_FDX_s006
                                 + 0.776761 w_k1_IBM_s006
                                 + 1.31241 w_k1_INTC_s006
                                 + 1.01464 w_k1_KO_s006 + 0.51321 w_k1_MAT_s006
                                 + 1.12592 w_k1_MSFT_s006
                                 + 0.97548 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.824907 w_k1_ADM_s006
                                 + 0.885907 w_k1_AXP_s006
                                 + 0.830641 w_k1_DIS_s006
                                 + 1.04615 w_k1_FDX_s006 + 0.5486 w_k1_IBM_s006
                                 + 1.07551 w_k1_INTC_s006
                                 + 0.756405 w_k1_KO_s006
                                 + 0.670489 w_k1_MAT_s006
                                 + 1.00709 w_k1_MSFT_s006
                                 + 0.930197 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.81076 w_k1_ADM_s007
                                 + 0.932226 w_k1_AXP_s007
                                 + 0.826403 w_k1_DIS_s007
                                 + 0.735378 w_k1_FDX_s007
                                 + 1.36406 w_k1_IBM_s007
                                 + 0.826749 w_k1_INTC_s007
                                 + 1.00036 w_k1_KO_s007 + 1.36322 w_k1_MAT_s007
                                 + 0.907659 w_k1_MSFT_s007
                                 + 0.934486 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.30745 w_k1_ADM_s007
                                 + 1.22239 w_k1_AXP_s007
                                 + 1.18856 w_k1_DIS_s007
                                 + 1.09958 w_k1_FDX_s007
                                 + 1.17139 w_k1_IBM_s007
                                 + 0.913492 w_k1_INTC_s007
                                 + 1.08938 w_k1_KO_s007 + 1.40361 w_k1_MAT_s007
                                 + 1.08809 w_k1_MSFT_s007
                                 + 1.08654 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.11202 w_k1_ADM_s007
                                 + 1.15947 w_k1_AXP_s007
                                 + 0.778126 w_k1_DIS_s007
                                 + 0.921271 w_k1_FDX_s007
                                 + 1.1496 w_k1_IBM_s007
                                 + 0.888569 w_k1_INTC_s007
                                 + 0.889275 w_k1_KO_s007
                                 + 1.23623 w_k1_MAT_s007
                                 + 0.873873 w_k1_MSFT_s007
                                 + 0.907208 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.876505 w_k1_ADM_s007
                                 + 0.738487 w_k1_AXP_s007
                                 + 0.708921 w_k1_DIS_s007
                                 + 0.812819 w_k1_FDX_s007
                                 + 0.743963 w_k1_IBM_s007
                                 + 0.900938 w_k1_INTC_s007
                                 + 0.716264 w_k1_KO_s007
                                 + 0.623678 w_k1_MAT_s007
                                 + 0.791439 w_k1_MSFT_s007
                                 + 0.754821 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.1457 w_k1_ADM_s007
                                 + 1.18724 w_k1_AXP_s007 + 1.6083 w_k1_DIS_s007
                                 + 1.34442 w_k1_FDX_s007
                                 + 0.841042 w_k1_IBM_s007
                                 + 1.45407 w_k1_INTC_s007
                                 + 0.961963 w_k1_KO_s007
                                 + 0.858471 w_k1_MAT_s007
                                 + 1.33583 w_k1_MSFT_s007
                                 + 0.992567 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.23474 w_k1_ADM_s007
                                 + 1.36416 w_k1_AXP_s007
                                 + 1.25782 w_k1_DIS_s007
                                 + 1.06382 w_k1_FDX_s007
                                 + 1.65126 w_k1_IBM_s007
                                 + 0.946829 w_k1_INTC_s007
                                 + 1.23848 w_k1_KO_s007 + 1.77094 w_k1_MAT_s007
                                 + 1.02717 w_k1_MSFT_s007 + 1.0811 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.37345 w_k1_ADM_s007
                                 + 1.46966 w_k1_AXP_s007
                                 + 1.50378 w_k1_DIS_s007 + 1.2174 w_k1_FDX_s007
                                 + 1.11072 w_k1_IBM_s007
                                 + 1.04713 w_k1_INTC_s007 + 1.1957 w_k1_KO_s007
                                 + 1.39317 w_k1_MAT_s007
                                 + 1.13316 w_k1_MSFT_s007
                                 + 1.06767 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.878442 w_k1_ADM_s007
                                 + 0.914631 w_k1_AXP_s007
                                 + 1.06209 w_k1_DIS_s007
                                 + 0.972949 w_k1_FDX_s007
                                 + 0.846135 w_k1_IBM_s007
                                 + 0.851213 w_k1_INTC_s007
                                 + 0.842326 w_k1_KO_s007
                                 + 1.01617 w_k1_MAT_s007
                                 + 0.962306 w_k1_MSFT_s007
                                 + 0.898327 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.880027 w_k1_ADM_s010
                                 + 0.781192 w_k1_AXP_s010
                                 + 0.699012 w_k1_DIS_s010
                                 + 0.80907 w_k1_FDX_s010 + 1.1118 w_k1_IBM_s010
                                 + 0.904132 w_k1_INTC_s010
                                 + 0.788378 w_k1_KO_s010
                                 + 0.98567 w_k1_MAT_s010
                                 + 0.793071 w_k1_MSFT_s010
                                 + 0.756651 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.03453 w_k1_ADM_s010
                                 + 0.707166 w_k1_AXP_s010
                                 + 0.916625 w_k1_DIS_s010
                                 + 0.971202 w_k1_FDX_s010
                                 + 0.806216 w_k1_IBM_s010
                                 + 0.971348 w_k1_INTC_s010
                                 + 0.923498 w_k1_KO_s010
                                 + 0.719026 w_k1_MAT_s010
                                 + 1.11395 w_k1_MSFT_s010
                                 + 0.930934 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.768271 w_k1_ADM_s010
                                 + 0.727417 w_k1_AXP_s010
                                 + 0.733296 w_k1_DIS_s010
                                 + 0.780805 w_k1_FDX_s010
                                 + 0.848524 w_k1_IBM_s010
                                 + 0.905995 w_k1_INTC_s010
                                 + 0.767072 w_k1_KO_s010
                                 + 0.696094 w_k1_MAT_s010
                                 + 0.85612 w_k1_MSFT_s010
                                 + 0.831684 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.2442 w_k1_ADM_s010
                                 + 1.22579 w_k1_AXP_s010
                                 + 1.05818 w_k1_DIS_s010
                                 + 1.12812 w_k1_FDX_s010
                                 + 0.953076 w_k1_IBM_s010
                                 + 0.90747 w_k1_INTC_s010
                                 + 0.891169 w_k1_KO_s010
                                 + 1.02413 w_k1_MAT_s010
                                 + 0.979402 w_k1_MSFT_s010
                                 + 1.03739 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.29941 w_k1_ADM_s010
                                 + 1.22473 w_k1_AXP_s010
                                 + 1.40307 w_k1_DIS_s010
                                 + 1.24778 w_k1_FDX_s010
                                 + 1.13969 w_k1_IBM_s010
                                 + 1.10757 w_k1_INTC_s010
                                 + 1.16713 w_k1_KO_s010 + 1.31505 w_k1_MAT_s010
                                 + 1.23889 w_k1_MSFT_s010
                                 + 1.11621 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.16411 w_k1_ADM_s010
                                 + 1.21826 w_k1_AXP_s010
                                 + 0.936395 w_k1_DIS_s010
                                 + 1.01866 w_k1_FDX_s010
                                 + 1.06042 w_k1_IBM_s010
                                 + 0.967352 w_k1_INTC_s010
                                 + 0.909982 w_k1_KO_s010
                                 + 1.02149 w_k1_MAT_s010
                                 + 1.03616 w_k1_MSFT_s010
                                 + 0.995997 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.880021 w_k1_ADM_s010
                                 + 0.757687 w_k1_AXP_s010
                                 + 0.636716 w_k1_DIS_s010
                                 + 0.903868 w_k1_FDX_s010
                                 + 0.611604 w_k1_IBM_s010
                                 + 0.801167 w_k1_INTC_s010
                                 + 0.690849 w_k1_KO_s010
                                 + 0.722242 w_k1_MAT_s010
                                 + 0.905845 w_k1_MSFT_s010
                                 + 0.823128 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.25105 w_k1_ADM_s010
                                 + 1.35391 w_k1_AXP_s010
                                 + 1.21175 w_k1_DIS_s010
                                 + 1.13241 w_k1_FDX_s010
                                 + 1.60193 w_k1_IBM_s010
                                 + 1.22338 w_k1_INTC_s010
                                 + 1.44453 w_k1_KO_s010 + 1.61849 w_k1_MAT_s010
                                 + 1.24792 w_k1_MSFT_s010
                                 + 1.09462 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.903934 w_k1_ADM_s011
                                 + 0.811981 w_k1_AXP_s011
                                 + 0.662762 w_k1_DIS_s011
                                 + 0.801501 w_k1_FDX_s011
                                 + 0.829799 w_k1_IBM_s011
                                 + 0.820024 w_k1_INTC_s011
                                 + 0.76184 w_k1_KO_s011
                                 + 0.869486 w_k1_MAT_s011
                                 + 0.891577 w_k1_MSFT_s011
                                 + 0.835369 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.979442 w_k1_ADM_s011
                                 + 0.892744 w_k1_AXP_s011
                                 + 1.21818 w_k1_DIS_s011 + 1.0149 w_k1_FDX_s011
                                 + 1.12488 w_k1_IBM_s011
                                 + 0.957312 w_k1_INTC_s011
                                 + 1.06565 w_k1_KO_s011 + 1.10768 w_k1_MAT_s011
                                 + 1.03383 w_k1_MSFT_s011
                                 + 1.04578 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.0104 w_k1_ADM_s011
                                 + 1.45026 w_k1_AXP_s011
                                 + 1.15674 w_k1_DIS_s011
                                 + 1.20112 w_k1_FDX_s011
                                 + 0.865385 w_k1_IBM_s011
                                 + 1.38301 w_k1_INTC_s011
                                 + 0.879589 w_k1_KO_s011
                                 + 0.948075 w_k1_MAT_s011
                                 + 1.1284 w_k1_MSFT_s011 + 1.12842 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.85568 w_k1_ADM_s011
                                 + 1.09247 w_k1_AXP_s011
                                 + 0.879439 w_k1_DIS_s011
                                 + 0.969742 w_k1_FDX_s011
                                 + 1.24643 w_k1_IBM_s011
                                 + 1.0891 w_k1_INTC_s011 + 0.93117 w_k1_KO_s011
                                 + 1.24176 w_k1_MAT_s011
                                 + 0.933958 w_k1_MSFT_s011
                                 + 0.90748 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.1177 w_k1_ADM_s011
                                 + 1.48698 w_k1_AXP_s011
                                 + 1.37591 w_k1_DIS_s011
                                 + 1.29482 w_k1_FDX_s011
                                 + 1.42398 w_k1_IBM_s011
                                 + 1.41988 w_k1_INTC_s011
                                 + 1.36631 w_k1_KO_s011 + 1.84226 w_k1_MAT_s011
                                 + 1.36753 w_k1_MSFT_s011
                                 + 1.24405 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.974037 w_k1_ADM_s011
                                 + 1.01983 w_k1_AXP_s011
                                 + 0.993902 w_k1_DIS_s011
                                 + 1.12399 w_k1_FDX_s011
                                 + 0.80177 w_k1_IBM_s011
                                 + 1.07201 w_k1_INTC_s011
                                 + 0.837294 w_k1_KO_s011
                                 + 0.836039 w_k1_MAT_s011
                                 + 1.09451 w_k1_MSFT_s011
                                 + 0.95829 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.28564 w_k1_ADM_s011
                                 + 1.28782 w_k1_AXP_s011
                                 + 1.47205 w_k1_DIS_s011 + 1.3849 w_k1_FDX_s011
                                 + 0.954937 w_k1_IBM_s011
                                 + 1.38801 w_k1_INTC_s011
                                 + 1.28225 w_k1_KO_s011 + 1.01763 w_k1_MAT_s011
                                 + 1.23623 w_k1_MSFT_s011
                                 + 1.16684 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.971634 w_k1_ADM_s011
                                 + 1.17891 w_k1_AXP_s011
                                 + 1.06759 w_k1_DIS_s011
                                 + 1.08029 w_k1_FDX_s011
                                 + 1.24697 w_k1_IBM_s011
                                 + 1.14893 w_k1_INTC_s011
                                 + 0.978199 w_k1_KO_s011
                                 + 1.20131 w_k1_MAT_s011
                                 + 1.02247 w_k1_MSFT_s011
                                 + 1.00935 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.43853 w_k1_ADM_s012
                                 + 1.33933 w_k1_AXP_s012
                                 + 1.45656 w_k1_DIS_s012
                                 + 1.27665 w_k1_FDX_s012
                                 + 1.10347 w_k1_IBM_s012
                                 + 1.01085 w_k1_INTC_s012
                                 + 1.10758 w_k1_KO_s012 + 1.40695 w_k1_MAT_s012
                                 + 1.34619 w_k1_MSFT_s012
                                 + 1.11808 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.05738 w_k1_ADM_s012
                                 + 0.849783 w_k1_AXP_s012
                                 + 0.925146 w_k1_DIS_s012
                                 + 0.994184 w_k1_FDX_s012
                                 + 0.8358 w_k1_IBM_s012
                                 + 0.882765 w_k1_INTC_s012
                                 + 0.778305 w_k1_KO_s012
                                 + 0.813951 w_k1_MAT_s012
                                 + 1.00761 w_k1_MSFT_s012
                                 + 0.942331 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.34678 w_k1_ADM_s012
                                 + 1.33099 w_k1_AXP_s012
                                 + 1.15091 w_k1_DIS_s012
                                 + 1.28402 w_k1_FDX_s012
                                 + 0.791523 w_k1_IBM_s012
                                 + 1.28528 w_k1_INTC_s012
                                 + 1.04484 w_k1_KO_s012 + 1.12997 w_k1_MAT_s012
                                 + 1.22795 w_k1_MSFT_s012
                                 + 1.06864 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.802128 w_k1_ADM_s012
                                 + 0.935433 w_k1_AXP_s012
                                 + 0.907588 w_k1_DIS_s012
                                 + 0.871437 w_k1_FDX_s012
                                 + 1.1839 w_k1_IBM_s012 + 1.0985 w_k1_INTC_s012
                                 + 1.07795 w_k1_KO_s012 + 1.2365 w_k1_MAT_s012
                                 + 1.12287 w_k1_MSFT_s012
                                 + 0.946307 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.889968 w_k1_ADM_s012
                                 + 1.1929 w_k1_AXP_s012 + 0.91978 w_k1_DIS_s012
                                 + 0.904113 w_k1_FDX_s012
                                 + 1.50414 w_k1_IBM_s012
                                 + 1.15169 w_k1_INTC_s012
                                 + 1.13294 w_k1_KO_s012 + 1.78483 w_k1_MAT_s012
                                 + 1.15346 w_k1_MSFT_s012
                                 + 1.12131 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.776269 w_k1_ADM_s012
                                 + 0.722029 w_k1_AXP_s012
                                 + 0.669435 w_k1_DIS_s012
                                 + 0.750373 w_k1_FDX_s012
                                 + 0.892041 w_k1_IBM_s012
                                 + 0.839579 w_k1_INTC_s012
                                 + 0.785615 w_k1_KO_s012
                                 + 0.950591 w_k1_MAT_s012
                                 + 0.862111 w_k1_MSFT_s012
                                 + 0.771983 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.17509 w_k1_ADM_s012
                                 + 0.97237 w_k1_AXP_s012
                                 + 0.910978 w_k1_DIS_s012
                                 + 1.11773 w_k1_FDX_s012
                                 + 0.640446 w_k1_IBM_s012
                                 + 1.17867 w_k1_INTC_s012
                                 + 0.771792 w_k1_KO_s012
                                 + 0.791886 w_k1_MAT_s012
                                 + 1.09372 w_k1_MSFT_s012
                                 + 0.940852 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.993462 w_k1_ADM_s012
                                 + 1.47232 w_k1_AXP_s012 + 1.5401 w_k1_DIS_s012
                                 + 1.26678 w_k1_FDX_s012
                                 + 1.62869 w_k1_IBM_s012
                                 + 1.26792 w_k1_INTC_s012
                                 + 1.33759 w_k1_KO_s012 + 1.72266 w_k1_MAT_s012
                                 + 1.34563 w_k1_MSFT_s012
                                 + 1.21695 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.94201 w_k1_ADM_s013
                                 + 0.917315 w_k1_AXP_s013
                                 + 0.790324 w_k1_DIS_s013
                                 + 1.01896 w_k1_FDX_s013
                                 + 0.71647 w_k1_IBM_s013
                                 + 1.00893 w_k1_INTC_s013
                                 + 0.767517 w_k1_KO_s013
                                 + 0.840465 w_k1_MAT_s013
                                 + 1.05323 w_k1_MSFT_s013
                                 + 0.936747 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.32163 w_k1_ADM_s013
                                 + 1.2264 w_k1_AXP_s013 + 1.15488 w_k1_DIS_s013
                                 + 1.21069 w_k1_FDX_s013
                                 + 1.03703 w_k1_IBM_s013
                                 + 1.04025 w_k1_INTC_s013
                                 + 1.06778 w_k1_KO_s013 + 1.26306 w_k1_MAT_s013
                                 + 1.17255 w_k1_MSFT_s013
                                 + 1.05988 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.76677 w_k1_ADM_s013
                                 + 0.612735 w_k1_AXP_s013
                                 + 0.48693 w_k1_DIS_s013
                                 + 0.722602 w_k1_FDX_s013
                                 + 0.721474 w_k1_IBM_s013
                                 + 0.741157 w_k1_INTC_s013
                                 + 0.654853 w_k1_KO_s013
                                 + 0.501313 w_k1_MAT_s013
                                 + 0.678248 w_k1_MSFT_s013
                                 + 0.761904 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.885827 w_k1_ADM_s013
                                 + 0.822865 w_k1_AXP_s013
                                 + 0.857361 w_k1_DIS_s013
                                 + 0.899304 w_k1_FDX_s013
                                 + 0.980997 w_k1_IBM_s013
                                 + 0.986678 w_k1_INTC_s013
                                 + 0.93408 w_k1_KO_s013 + 1.15204 w_k1_MAT_s013
                                 + 0.930343 w_k1_MSFT_s013
                                 + 0.884731 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.958868 w_k1_ADM_s013
                                 + 1.23946 w_k1_AXP_s013
                                 + 0.802619 w_k1_DIS_s013
                                 + 0.924667 w_k1_FDX_s013
                                 + 1.14235 w_k1_IBM_s013
                                 + 0.949083 w_k1_INTC_s013
                                 + 1.06118 w_k1_KO_s013 + 1.38836 w_k1_MAT_s013
                                 + 0.937038 w_k1_MSFT_s013
                                 + 0.986976 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.25244 w_k1_ADM_s013
                                 + 1.04325 w_k1_AXP_s013
                                 + 1.67103 w_k1_DIS_s013 + 1.4208 w_k1_FDX_s013
                                 + 1.05182 w_k1_IBM_s013
                                 + 1.13421 w_k1_INTC_s013
                                 + 1.20011 w_k1_KO_s013 + 1.05366 w_k1_MAT_s013
                                 + 1.37831 w_k1_MSFT_s013
                                 + 1.11713 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.996757 w_k1_ADM_s013
                                 + 0.951629 w_k1_AXP_s013
                                 + 0.825694 w_k1_DIS_s013
                                 + 0.867505 w_k1_FDX_s013
                                 + 1.56741 w_k1_IBM_s013
                                 + 1.10866 w_k1_INTC_s013
                                 + 1.23313 w_k1_KO_s013 + 1.46468 w_k1_MAT_s013
                                 + 0.985824 w_k1_MSFT_s013
                                 + 1.01661 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.62822 w_k1_ADM_s013
                                 + 1.48831 w_k1_AXP_s013
                                 + 1.62262 w_k1_DIS_s013
                                 + 1.52464 w_k1_FDX_s013
                                 + 1.49486 w_k1_IBM_s013
                                 + 1.50529 w_k1_INTC_s013
                                 + 1.41178 w_k1_KO_s013 + 1.43783 w_k1_MAT_s013
                                 + 1.3892 w_k1_MSFT_s013 + 1.4174 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.4434 w_k1_ADM_s014
                                 + 1.56127 w_k1_AXP_s014
                                 + 1.64351 w_k1_DIS_s014
                                 + 1.46143 w_k1_FDX_s014
                                 + 1.13375 w_k1_IBM_s014
                                 + 1.64298 w_k1_INTC_s014
                                 + 1.25407 w_k1_KO_s014 + 1.32066 w_k1_MAT_s014
                                 + 1.54462 w_k1_MSFT_s014
                                 + 1.38744 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.858963 w_k1_ADM_s014
                                 + 1.15512 w_k1_AXP_s014
                                 + 0.962889 w_k1_DIS_s014
                                 + 1.00174 w_k1_FDX_s014
                                 + 1.23996 w_k1_IBM_s014
                                 + 1.12073 w_k1_INTC_s014 + 1.089 w_k1_KO_s014
                                 + 1.302 w_k1_MAT_s014 + 1.06936 w_k1_MSFT_s014
                                 + 0.991465 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.06544 w_k1_ADM_s014
                                 + 1.18118 w_k1_AXP_s014
                                 + 0.983675 w_k1_DIS_s014
                                 + 1.18521 w_k1_FDX_s014
                                 + 0.992614 w_k1_IBM_s014
                                 + 1.28124 w_k1_INTC_s014
                                 + 1.04249 w_k1_KO_s014 + 1.22282 w_k1_MAT_s014
                                 + 1.11864 w_k1_MSFT_s014
                                 + 1.03289 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.827455 w_k1_ADM_s014
                                 + 0.781095 w_k1_AXP_s014
                                 + 0.685739 w_k1_DIS_s014
                                 + 0.730008 w_k1_FDX_s014
                                 + 0.989671 w_k1_IBM_s014
                                 + 0.85758 w_k1_INTC_s014
                                 + 0.925537 w_k1_KO_s014
                                 + 1.04637 w_k1_MAT_s014
                                 + 0.911222 w_k1_MSFT_s014
                                 + 0.969667 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.02598 w_k1_ADM_s014
                                 + 0.775415 w_k1_AXP_s014
                                 + 0.960431 w_k1_DIS_s014
                                 + 0.919109 w_k1_FDX_s014
                                 + 0.851338 w_k1_IBM_s014
                                 + 0.995031 w_k1_INTC_s014
                                 + 0.949049 w_k1_KO_s014
                                 + 0.928347 w_k1_MAT_s014
                                 + 1.09652 w_k1_MSFT_s014
                                 + 1.03966 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.898761 w_k1_ADM_s014
                                 + 0.815474 w_k1_AXP_s014
                                 + 0.761815 w_k1_DIS_s014
                                 + 0.942901 w_k1_FDX_s014
                                 + 0.647859 w_k1_IBM_s014
                                 + 1.18551 w_k1_INTC_s014
                                 + 0.794414 w_k1_KO_s014
                                 + 0.540042 w_k1_MAT_s014
                                 + 0.976416 w_k1_MSFT_s014
                                 + 0.908516 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.1226 w_k1_ADM_s014
                                 + 1.32234 w_k1_AXP_s014
                                 + 1.28474 w_k1_DIS_s014
                                 + 1.10573 w_k1_FDX_s014
                                 + 1.60623 w_k1_IBM_s014
                                 + 1.27618 w_k1_INTC_s014
                                 + 1.47146 w_k1_KO_s014 + 1.69846 w_k1_MAT_s014
                                 + 1.07413 w_k1_MSFT_s014
                                 + 1.29546 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.992878 w_k1_ADM_s014
                                 + 1.0373 w_k1_AXP_s014 + 1.30287 w_k1_DIS_s014
                                 + 1.15932 w_k1_FDX_s014
                                 + 0.849554 w_k1_IBM_s014
                                 + 1.27134 w_k1_INTC_s014
                                 + 1.20166 w_k1_KO_s014
                                 + 0.910333 w_k1_MAT_s014
                                 + 1.25134 w_k1_MSFT_s014
                                 + 1.20632 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.854128 w_k1_ADM_s015
                                 + 0.700711 w_k1_AXP_s015
                                 + 0.658071 w_k1_DIS_s015
                                 + 0.844567 w_k1_FDX_s015
                                 + 0.66418 w_k1_IBM_s015
                                 + 0.765661 w_k1_INTC_s015
                                 + 0.720454 w_k1_KO_s015
                                 + 0.841984 w_k1_MAT_s015
                                 + 0.818727 w_k1_MSFT_s015
                                 + 0.792377 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.977415 w_k1_ADM_s015
                                 + 0.91266 w_k1_AXP_s015
                                 + 0.989396 w_k1_DIS_s015
                                 + 1.04136 w_k1_FDX_s015
                                 + 0.808249 w_k1_IBM_s015
                                 + 1.01641 w_k1_INTC_s015
                                 + 0.903579 w_k1_KO_s015
                                 + 1.05709 w_k1_MAT_s015
                                 + 0.998901 w_k1_MSFT_s015
                                 + 0.932251 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.1165 w_k1_ADM_s015
                                 + 1.54288 w_k1_AXP_s015
                                 + 1.35598 w_k1_DIS_s015
                                 + 1.26899 w_k1_FDX_s015
                                 + 0.995096 w_k1_IBM_s015
                                 + 1.12391 w_k1_INTC_s015
                                 + 1.14743 w_k1_KO_s015 + 1.16481 w_k1_MAT_s015
                                 + 1.12031 w_k1_MSFT_s015
                                 + 1.04595 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.863694 w_k1_ADM_s015
                                 + 0.606859 w_k1_AXP_s015
                                 + 0.870889 w_k1_DIS_s015
                                 + 0.826294 w_k1_FDX_s015
                                 + 0.978951 w_k1_IBM_s015
                                 + 0.901127 w_k1_INTC_s015
                                 + 0.964272 w_k1_KO_s015
                                 + 0.798325 w_k1_MAT_s015
                                 + 1.01971 w_k1_MSFT_s015
                                 + 0.875073 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.09858 w_k1_ADM_s015
                                 + 0.985225 w_k1_AXP_s015
                                 + 0.836713 w_k1_DIS_s015
                                 + 0.894281 w_k1_FDX_s015
                                 + 1.03222 w_k1_IBM_s015
                                 + 0.914452 w_k1_INTC_s015
                                 + 0.928383 w_k1_KO_s015
                                 + 1.05447 w_k1_MAT_s015
                                 + 0.973081 w_k1_MSFT_s015
                                 + 0.961729 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.756046 w_k1_ADM_s015
                                 + 0.981672 w_k1_AXP_s015
                                 + 0.975853 w_k1_DIS_s015
                                 + 0.858497 w_k1_FDX_s015
                                 + 1.51077 w_k1_IBM_s015
                                 + 0.804369 w_k1_INTC_s015
                                 + 1.20137 w_k1_KO_s015 + 2.07385 w_k1_MAT_s015
                                 + 0.942949 w_k1_MSFT_s015
                                 + 0.920557 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.22221 w_k1_ADM_s015
                                 + 1.11401 w_k1_AXP_s015
                                 + 1.39911 w_k1_DIS_s015
                                 + 1.05396 w_k1_FDX_s015 + 1.4104 w_k1_IBM_s015
                                 + 1.15818 w_k1_INTC_s015
                                 + 1.37249 w_k1_KO_s015 + 1.01885 w_k1_MAT_s015
                                 + 1.19225 w_k1_MSFT_s015
                                 + 1.15924 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.18131 w_k1_ADM_s015
                                 + 0.951112 w_k1_AXP_s015
                                 + 1.38327 w_k1_DIS_s015
                                 + 1.12659 w_k1_FDX_s015
                                 + 1.06033 w_k1_IBM_s015
                                 + 1.30249 w_k1_INTC_s015 + 1.3502 w_k1_KO_s015
                                 + 1.05173 w_k1_MAT_s015
                                 + 1.30739 w_k1_MSFT_s015
                                 + 1.02175 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.02917 w_k1_ADM_s016
                                 + 0.662272 w_k1_AXP_s016
                                 + 0.54758 w_k1_DIS_s016
                                 + 0.715845 w_k1_FDX_s016
                                 + 0.855017 w_k1_IBM_s016
                                 + 0.785722 w_k1_INTC_s016
                                 + 0.704283 w_k1_KO_s016
                                 + 0.650874 w_k1_MAT_s016
                                 + 0.742806 w_k1_MSFT_s016
                                 + 0.666867 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.02376 w_k1_ADM_s016
                                 + 0.83834 w_k1_AXP_s016
                                 + 0.764851 w_k1_DIS_s016
                                 + 1.00589 w_k1_FDX_s016
                                 + 0.692932 w_k1_IBM_s016
                                 + 0.904179 w_k1_INTC_s016
                                 + 0.705458 w_k1_KO_s016
                                 + 0.715137 w_k1_MAT_s016
                                 + 0.930452 w_k1_MSFT_s016
                                 + 0.824421 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.955395 w_k1_ADM_s016
                                 + 0.979235 w_k1_AXP_s016
                                 + 1.08637 w_k1_DIS_s016
                                 + 1.04707 w_k1_FDX_s016
                                 + 1.23568 w_k1_IBM_s016
                                 + 1.17429 w_k1_INTC_s016
                                 + 1.13666 w_k1_KO_s016 + 1.15528 w_k1_MAT_s016
                                 + 1.10659 w_k1_MSFT_s016
                                 + 0.952361 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.964286 w_k1_ADM_s016
                                 + 0.866745 w_k1_AXP_s016 + 0.958 w_k1_DIS_s016
                                 + 1.02591 w_k1_FDX_s016
                                 + 0.872014 w_k1_IBM_s016
                                 + 1.10457 w_k1_INTC_s016
                                 + 0.873277 w_k1_KO_s016
                                 + 0.841231 w_k1_MAT_s016
                                 + 1.06789 w_k1_MSFT_s016
                                 + 0.88324 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.24111 w_k1_ADM_s016
                                 + 1.35081 w_k1_AXP_s016
                                 + 1.01966 w_k1_DIS_s016
                                 + 1.18574 w_k1_FDX_s016
                                 + 1.24804 w_k1_IBM_s016
                                 + 1.14602 w_k1_INTC_s016
                                 + 1.15121 w_k1_KO_s016 + 1.65896 w_k1_MAT_s016
                                 + 1.03081 w_k1_MSFT_s016 + 1.1151 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.859892 w_k1_ADM_s016
                                 + 0.895436 w_k1_AXP_s016
                                 + 0.842037 w_k1_DIS_s016
                                 + 0.87134 w_k1_FDX_s016
                                 + 1.24821 w_k1_IBM_s016
                                 + 0.883825 w_k1_INTC_s016
                                 + 0.979614 w_k1_KO_s016
                                 + 1.55073 w_k1_MAT_s016
                                 + 0.766411 w_k1_MSFT_s016
                                 + 0.865951 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.974948 w_k1_ADM_s016
                                 + 0.663818 w_k1_AXP_s016
                                 + 0.783141 w_k1_DIS_s016
                                 + 0.859322 w_k1_FDX_s016
                                 + 0.89617 w_k1_IBM_s016
                                 + 0.839765 w_k1_INTC_s016
                                 + 0.960608 w_k1_KO_s016
                                 + 1.14546 w_k1_MAT_s016
                                 + 0.960857 w_k1_MSFT_s016
                                 + 0.933929 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.62778 w_k1_ADM_s016
                                 + 1.2973 w_k1_AXP_s016 + 1.68525 w_k1_DIS_s016
                                 + 1.37098 w_k1_FDX_s016
                                 + 1.18358 w_k1_IBM_s016
                                 + 1.2895 w_k1_INTC_s016 + 1.23356 w_k1_KO_s016
                                 + 1.33027 w_k1_MAT_s016
                                 + 1.42949 w_k1_MSFT_s016
                                 + 1.12577 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.07649 w_k1_ADM_s017
                                 + 0.987583 w_k1_AXP_s017
                                 + 1.28414 w_k1_DIS_s017
                                 + 0.979871 w_k1_FDX_s017
                                 + 1.49285 w_k1_IBM_s017
                                 + 0.934155 w_k1_INTC_s017
                                 + 1.29473 w_k1_KO_s017 + 1.29406 w_k1_MAT_s017
                                 + 0.963583 w_k1_MSFT_s017
                                 + 1.08514 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.716333 w_k1_ADM_s017
                                 + 0.793999 w_k1_AXP_s017
                                 + 0.843905 w_k1_DIS_s017
                                 + 0.798695 w_k1_FDX_s017
                                 + 1.0119 w_k1_IBM_s017
                                 + 0.745802 w_k1_INTC_s017
                                 + 0.963623 w_k1_KO_s017
                                 + 1.51042 w_k1_MAT_s017
                                 + 0.766036 w_k1_MSFT_s017
                                 + 0.945113 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.935722 w_k1_ADM_s017
                                 + 1.17716 w_k1_AXP_s017
                                 + 1.45039 w_k1_DIS_s017
                                 + 1.19303 w_k1_FDX_s017
                                 + 1.03697 w_k1_IBM_s017
                                 + 1.07833 w_k1_INTC_s017
                                 + 1.07032 w_k1_KO_s017 + 1.18012 w_k1_MAT_s017
                                 + 1.18951 w_k1_MSFT_s017
                                 + 1.04022 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.03982 w_k1_ADM_s017
                                 + 1.24587 w_k1_AXP_s017 + 1.0298 w_k1_DIS_s017
                                 + 1.26103 w_k1_FDX_s017
                                 + 0.640648 w_k1_IBM_s017
                                 + 1.00715 w_k1_INTC_s017
                                 + 0.861886 w_k1_KO_s017
                                 + 0.701618 w_k1_MAT_s017
                                 + 1.02251 w_k1_MSFT_s017
                                 + 1.02429 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.848365 w_k1_ADM_s017
                                 + 0.964631 w_k1_AXP_s017
                                 + 0.924765 w_k1_DIS_s017
                                 + 0.994024 w_k1_FDX_s017
                                 + 1.03891 w_k1_IBM_s017
                                 + 0.941312 w_k1_INTC_s017
                                 + 0.919984 w_k1_KO_s017
                                 + 1.42494 w_k1_MAT_s017
                                 + 1.06757 w_k1_MSFT_s017
                                 + 0.912913 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.722938 w_k1_ADM_s017
                                 + 0.651451 w_k1_AXP_s017
                                 + 0.661172 w_k1_DIS_s017
                                 + 0.831653 w_k1_FDX_s017
                                 + 0.607724 w_k1_IBM_s017
                                 + 0.856284 w_k1_INTC_s017
                                 + 0.752145 w_k1_KO_s017
                                 + 0.729062 w_k1_MAT_s017
                                 + 0.87142 w_k1_MSFT_s017
                                 + 0.837033 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.27252 w_k1_ADM_s017
                                 + 1.70994 w_k1_AXP_s017
                                 + 1.32112 w_k1_DIS_s017 + 1.2494 w_k1_FDX_s017
                                 + 1.49915 w_k1_IBM_s017
                                 + 0.917214 w_k1_INTC_s017
                                 + 1.35996 w_k1_KO_s017 + 2.12768 w_k1_MAT_s017
                                 + 1.02685 w_k1_MSFT_s017
                                 + 1.31835 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.27337 w_k1_ADM_s017
                                 + 1.57778 w_k1_AXP_s017
                                 + 1.70239 w_k1_DIS_s017
                                 + 1.59989 w_k1_FDX_s017
                                 + 0.947792 w_k1_IBM_s017
                                 + 1.21153 w_k1_INTC_s017
                                 + 1.31274 w_k1_KO_s017 + 1.12147 w_k1_MAT_s017
                                 + 1.54325 w_k1_MSFT_s017
                                 + 1.21861 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.24279 w_k1_ADM_s100
                                 + 1.20991 w_k1_AXP_s100
                                 + 1.37845 w_k1_DIS_s100
                                 + 1.11761 w_k1_FDX_s100
                                 + 1.14646 w_k1_IBM_s100
                                 + 0.926363 w_k1_INTC_s100
                                 + 1.14926 w_k1_KO_s100 + 1.33749 w_k1_MAT_s100
                                 + 1.23323 w_k1_MSFT_s100
                                 + 1.05791 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.23214 w_k1_ADM_s100
                                 + 1.34369 w_k1_AXP_s100
                                 + 1.11811 w_k1_DIS_s100
                                 + 1.01626 w_k1_FDX_s100
                                 + 1.34503 w_k1_IBM_s100
                                 + 0.79083 w_k1_INTC_s100
                                 + 1.04632 w_k1_KO_s100 + 1.30383 w_k1_MAT_s100
                                 + 0.881197 w_k1_MSFT_s100
                                 + 0.891397 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.871691 w_k1_ADM_s100
                                 + 0.701966 w_k1_AXP_s100
                                 + 0.717504 w_k1_DIS_s100
                                 + 0.737816 w_k1_FDX_s100
                                 + 1.02218 w_k1_IBM_s100
                                 + 0.663216 w_k1_INTC_s100
                                 + 0.763126 w_k1_KO_s100
                                 + 1.08518 w_k1_MAT_s100
                                 + 0.814407 w_k1_MSFT_s100
                                 + 0.788831 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.19558 w_k1_ADM_s100
                                 + 0.952755 w_k1_AXP_s100
                                 + 0.836472 w_k1_DIS_s100
                                 + 0.980449 w_k1_FDX_s100
                                 + 0.869858 w_k1_IBM_s100
                                 + 0.814337 w_k1_INTC_s100
                                 + 0.769736 w_k1_KO_s100
                                 + 0.856407 w_k1_MAT_s100
                                 + 0.9617 w_k1_MSFT_s100 + 0.93893 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.965241 w_k1_ADM_s100
                                 + 0.881789 w_k1_AXP_s100
                                 + 0.740553 w_k1_DIS_s100
                                 + 0.884892 w_k1_FDX_s100
                                 + 0.63402 w_k1_IBM_s100
                                 + 0.73448 w_k1_INTC_s100
                                 + 0.745231 w_k1_KO_s100
                                 + 0.826204 w_k1_MAT_s100
                                 + 0.835794 w_k1_MSFT_s100
                                 + 0.795314 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.1973 w_k1_ADM_s100
                                 + 0.900386 w_k1_AXP_s100
                                 + 1.17571 w_k1_DIS_s100
                                 + 1.08843 w_k1_FDX_s100
                                 + 0.884905 w_k1_IBM_s100
                                 + 0.975688 w_k1_INTC_s100
                                 + 0.947502 w_k1_KO_s100
                                 + 0.829708 w_k1_MAT_s100
                                 + 1.09594 w_k1_MSFT_s100
                                 + 0.880643 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.46929 w_k1_ADM_s100
                                 + 1.62901 w_k1_AXP_s100 + 1.6975 w_k1_DIS_s100
                                 + 1.40248 w_k1_FDX_s100
                                 + 1.06599 w_k1_IBM_s100
                                 + 1.28294 w_k1_INTC_s100
                                 + 1.18473 w_k1_KO_s100
                                 + 0.831366 w_k1_MAT_s100
                                 + 1.41525 w_k1_MSFT_s100
                                 + 0.996185 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.6416 w_k1_ADM_s100
                                 + 1.70159 w_k1_AXP_s100
                                 + 1.77942 w_k1_DIS_s100
                                 + 1.29511 w_k1_FDX_s100
                                 + 1.40396 w_k1_IBM_s100
                                 + 1.3835 w_k1_INTC_s100 + 1.43435 w_k1_KO_s100
                                 + 1.44079 w_k1_MAT_s100
                                 + 1.30866 w_k1_MSFT_s100 + 1.189 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.39306 w_k1_ADM_s101
                                 + 1.39342 w_k1_AXP_s101
                                 + 1.36834 w_k1_DIS_s101
                                 + 1.18171 w_k1_FDX_s101
                                 + 1.32307 w_k1_IBM_s101
                                 + 1.28331 w_k1_INTC_s101
                                 + 1.35585 w_k1_KO_s101 + 1.38993 w_k1_MAT_s101
                                 + 1.21435 w_k1_MSFT_s101
                                 + 1.23564 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.772138 w_k1_ADM_s101
                                 + 0.829701 w_k1_AXP_s101
                                 + 0.788589 w_k1_DIS_s101
                                 + 0.7676 w_k1_FDX_s101 + 0.95602 w_k1_IBM_s101
                                 + 0.844913 w_k1_INTC_s101
                                 + 0.907106 w_k1_KO_s101
                                 + 0.648366 w_k1_MAT_s101
                                 + 0.840637 w_k1_MSFT_s101
                                 + 0.883576 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.991864 w_k1_ADM_s101
                                 + 1.34053 w_k1_AXP_s101
                                 + 1.36472 w_k1_DIS_s101 + 1.1147 w_k1_FDX_s101
                                 + 1.6548 w_k1_IBM_s101
                                 + 1.21442 w_k1_INTC_s101
                                 + 1.51834 w_k1_KO_s101 + 2.06826 w_k1_MAT_s101
                                 + 1.37212 w_k1_MSFT_s101
                                 + 1.27226 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.12348 w_k1_ADM_s101
                                 + 1.2744 w_k1_AXP_s101 + 1.19757 w_k1_DIS_s101
                                 + 1.16133 w_k1_FDX_s101
                                 + 0.879688 w_k1_IBM_s101
                                 + 1.22316 w_k1_INTC_s101
                                 + 0.935693 w_k1_KO_s101
                                 + 0.984748 w_k1_MAT_s101
                                 + 1.29596 w_k1_MSFT_s101
                                 + 0.955472 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.07676 w_k1_ADM_s101
                                 + 1.0945 w_k1_AXP_s101
                                 + 0.859481 w_k1_DIS_s101
                                 + 1.17739 w_k1_FDX_s101
                                 + 0.583667 w_k1_IBM_s101
                                 + 1.08593 w_k1_INTC_s101
                                 + 0.775666 w_k1_KO_s101
                                 + 0.648837 w_k1_MAT_s101
                                 + 1.12942 w_k1_MSFT_s101
                                 + 0.924303 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.827464 w_k1_ADM_s101
                                 + 0.702516 w_k1_AXP_s101
                                 + 0.647182 w_k1_DIS_s101
                                 + 0.747594 w_k1_FDX_s101
                                 + 0.643822 w_k1_IBM_s101
                                 + 0.716035 w_k1_INTC_s101
                                 + 0.711849 w_k1_KO_s101
                                 + 0.578037 w_k1_MAT_s101
                                 + 0.858138 w_k1_MSFT_s101
                                 + 0.782232 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.755571 w_k1_ADM_s101
                                 + 0.915483 w_k1_AXP_s101
                                 + 0.779883 w_k1_DIS_s101
                                 + 0.932661 w_k1_FDX_s101
                                 + 0.629508 w_k1_IBM_s101
                                 + 0.919101 w_k1_INTC_s101
                                 + 0.684266 w_k1_KO_s101
                                 + 0.554563 w_k1_MAT_s101
                                 + 0.86459 w_k1_MSFT_s101
                                 + 0.848589 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.04352 w_k1_ADM_s101
                                 + 1.0275 w_k1_AXP_s101 + 1.04073 w_k1_DIS_s101
                                 + 1.0102 w_k1_FDX_s101 + 1.10428 w_k1_IBM_s101
                                 + 1.20429 w_k1_INTC_s101
                                 + 1.08797 w_k1_KO_s101 + 1.19103 w_k1_MAT_s101
                                 + 1.19981 w_k1_MSFT_s101 + 1.0956 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.38443 w_k1_ADM_s102
                                 + 1.69685 w_k1_AXP_s102
                                 + 1.81411 w_k1_DIS_s102
                                 + 1.37889 w_k1_FDX_s102 + 1.4211 w_k1_IBM_s102
                                 + 1.49424 w_k1_INTC_s102
                                 + 1.34577 w_k1_KO_s102 + 1.34898 w_k1_MAT_s102
                                 + 1.29612 w_k1_MSFT_s102
                                 + 1.26939 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.777127 w_k1_ADM_s102
                                 + 0.552678 w_k1_AXP_s102
                                 + 0.614853 w_k1_DIS_s102
                                 + 0.739632 w_k1_FDX_s102
                                 + 0.64784 w_k1_IBM_s102
                                 + 0.863568 w_k1_INTC_s102
                                 + 0.628956 w_k1_KO_s102
                                 + 0.618414 w_k1_MAT_s102
                                 + 0.784228 w_k1_MSFT_s102
                                 + 0.757913 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.36558 w_k1_ADM_s102
                                 + 1.15531 w_k1_AXP_s102
                                 + 1.01104 w_k1_DIS_s102 + 1.2797 w_k1_FDX_s102
                                 + 0.562598 w_k1_IBM_s102
                                 + 1.41826 w_k1_INTC_s102
                                 + 0.759225 w_k1_KO_s102
                                 + 0.609019 w_k1_MAT_s102
                                 + 1.20589 w_k1_MSFT_s102
                                 + 1.06553 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.830358 w_k1_ADM_s102
                                 + 0.946788 w_k1_AXP_s102
                                 + 1.00358 w_k1_DIS_s102
                                 + 1.00119 w_k1_FDX_s102
                                 + 1.16149 w_k1_IBM_s102
                                 + 1.38667 w_k1_INTC_s102
                                 + 1.01882 w_k1_KO_s102
                                 + 0.725733 w_k1_MAT_s102
                                 + 1.10876 w_k1_MSFT_s102
                                 + 1.03037 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.07972 w_k1_ADM_s102
                                 + 0.886855 w_k1_AXP_s102
                                 + 1.01909 w_k1_DIS_s102
                                 + 1.04955 w_k1_FDX_s102
                                 + 0.783877 w_k1_IBM_s102
                                 + 1.20297 w_k1_INTC_s102
                                 + 0.842826 w_k1_KO_s102
                                 + 0.647717 w_k1_MAT_s102
                                 + 1.10513 w_k1_MSFT_s102
                                 + 1.06863 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04743 w_k1_ADM_s102
                                 + 1.32975 w_k1_AXP_s102
                                 + 1.24353 w_k1_DIS_s102
                                 + 1.09585 w_k1_FDX_s102 + 1.6133 w_k1_IBM_s102
                                 + 1.09607 w_k1_INTC_s102
                                 + 1.15519 w_k1_KO_s102 + 1.74437 w_k1_MAT_s102
                                 + 1.00916 w_k1_MSFT_s102
                                 + 1.19125 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.989575 w_k1_ADM_s102
                                 + 1.08065 w_k1_AXP_s102
                                 + 0.886413 w_k1_DIS_s102
                                 + 0.880896 w_k1_FDX_s102
                                 + 1.24001 w_k1_IBM_s102
                                 + 1.16269 w_k1_INTC_s102
                                 + 0.955039 w_k1_KO_s102 + 1.2808 w_k1_MAT_s102
                                 + 0.961948 w_k1_MSFT_s102
                                 + 1.01504 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.095 w_k1_ADM_s102
                                 + 0.876938 w_k1_AXP_s102
                                 + 1.04307 w_k1_DIS_s102
                                 + 0.949322 w_k1_FDX_s102
                                 + 0.92991 w_k1_IBM_s102
                                 + 1.1365 w_k1_INTC_s102 + 1.01642 w_k1_KO_s102
                                 + 0.596551 w_k1_MAT_s102
                                 + 1.01298 w_k1_MSFT_s102
                                 + 1.06082 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.855886 w_k1_ADM_s103
                                 + 0.760302 w_k1_AXP_s103
                                 + 0.753196 w_k1_DIS_s103
                                 + 0.828308 w_k1_FDX_s103
                                 + 0.68628 w_k1_IBM_s103
                                 + 0.92214 w_k1_INTC_s103
                                 + 0.847179 w_k1_KO_s103
                                 + 0.480958 w_k1_MAT_s103
                                 + 0.830267 w_k1_MSFT_s103
                                 + 0.934373 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.00701 w_k1_ADM_s103
                                 + 1.00601 w_k1_AXP_s103
                                 + 0.850696 w_k1_DIS_s103
                                 + 0.885062 w_k1_FDX_s103
                                 + 1.04912 w_k1_IBM_s103
                                 + 0.935187 w_k1_INTC_s103
                                 + 1.14923 w_k1_KO_s103
                                 + 0.854453 w_k1_MAT_s103
                                 + 0.909471 w_k1_MSFT_s103
                                 + 1.02819 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.13064 w_k1_ADM_s103
                                 + 1.34937 w_k1_AXP_s103
                                 + 1.56576 w_k1_DIS_s103
                                 + 1.24085 w_k1_FDX_s103
                                 + 1.20723 w_k1_IBM_s103
                                 + 1.54689 w_k1_INTC_s103
                                 + 1.61324 w_k1_KO_s103 + 1.14716 w_k1_MAT_s103
                                 + 1.24512 w_k1_MSFT_s103
                                 + 1.33338 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.765945 w_k1_ADM_s103
                                 + 0.839653 w_k1_AXP_s103
                                 + 0.640771 w_k1_DIS_s103
                                 + 0.753625 w_k1_FDX_s103
                                 + 0.899179 w_k1_IBM_s103
                                 + 0.990717 w_k1_INTC_s103
                                 + 0.831327 w_k1_KO_s103
                                 + 0.895272 w_k1_MAT_s103
                                 + 0.845937 w_k1_MSFT_s103
                                 + 0.882596 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.10361 w_k1_ADM_s103
                                 + 1.26512 w_k1_AXP_s103
                                 + 0.966669 w_k1_DIS_s103
                                 + 1.0291 w_k1_FDX_s103
                                 + 0.998765 w_k1_IBM_s103
                                 + 1.4096 w_k1_INTC_s103 + 1.13155 w_k1_KO_s103
                                 + 1.05333 w_k1_MAT_s103
                                 + 1.15348 w_k1_MSFT_s103
                                 + 1.06263 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.931981 w_k1_ADM_s103
                                 + 1.39649 w_k1_AXP_s103 + 1.211 w_k1_DIS_s103
                                 + 1.1782 w_k1_FDX_s103 + 1.23018 w_k1_IBM_s103
                                 + 1.35066 w_k1_INTC_s103
                                 + 1.39597 w_k1_KO_s103 + 1.55713 w_k1_MAT_s103
                                 + 1.20592 w_k1_MSFT_s103
                                 + 1.30685 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.08173 w_k1_ADM_s103
                                 + 1.02927 w_k1_AXP_s103
                                 + 1.13674 w_k1_DIS_s103
                                 + 1.15654 w_k1_FDX_s103
                                 + 0.871568 w_k1_IBM_s103
                                 + 1.46705 w_k1_INTC_s103
                                 + 1.06483 w_k1_KO_s103
                                 + 0.967063 w_k1_MAT_s103
                                 + 1.23798 w_k1_MSFT_s103
                                 + 1.02003 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.03224 w_k1_ADM_s103
                                 + 1.39125 w_k1_AXP_s103
                                 + 1.04241 w_k1_DIS_s103
                                 + 0.978607 w_k1_FDX_s103
                                 + 1.52342 w_k1_IBM_s103
                                 + 1.26578 w_k1_INTC_s103
                                 + 1.29442 w_k1_KO_s103 + 1.35629 w_k1_MAT_s103
                                 + 0.958699 w_k1_MSFT_s103
                                 + 1.11289 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.30634 w_k1_ADM_s104
                                 + 1.3807 w_k1_AXP_s104 + 1.41115 w_k1_DIS_s104
                                 + 1.33837 w_k1_FDX_s104
                                 + 0.965252 w_k1_IBM_s104
                                 + 1.33126 w_k1_INTC_s104
                                 + 1.15915 w_k1_KO_s104 + 1.19281 w_k1_MAT_s104
                                 + 1.31 w_k1_MSFT_s104 + 1.11488 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.967089 w_k1_ADM_s104
                                 + 0.617326 w_k1_AXP_s104
                                 + 0.607554 w_k1_DIS_s104
                                 + 0.675535 w_k1_FDX_s104
                                 + 1.09426 w_k1_IBM_s104
                                 + 0.922672 w_k1_INTC_s104
                                 + 0.895777 w_k1_KO_s104
                                 + 1.10383 w_k1_MAT_s104
                                 + 0.841741 w_k1_MSFT_s104
                                 + 0.950422 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.21134 w_k1_ADM_s104
                                 + 0.767284 w_k1_AXP_s104
                                 + 0.725729 w_k1_DIS_s104
                                 + 0.88798 w_k1_FDX_s104
                                 + 0.789817 w_k1_IBM_s104
                                 + 0.811844 w_k1_INTC_s104
                                 + 0.918181 w_k1_KO_s104
                                 + 0.956738 w_k1_MAT_s104
                                 + 0.883137 w_k1_MSFT_s104
                                 + 1.01542 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.06238 w_k1_ADM_s104
                                 + 0.726877 w_k1_AXP_s104
                                 + 0.962902 w_k1_DIS_s104
                                 + 1.02979 w_k1_FDX_s104
                                 + 0.682719 w_k1_IBM_s104
                                 + 1.00448 w_k1_INTC_s104
                                 + 0.862479 w_k1_KO_s104
                                 + 0.57341 w_k1_MAT_s104
                                 + 1.08191 w_k1_MSFT_s104
                                 + 0.938905 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.882157 w_k1_ADM_s104
                                 + 0.590022 w_k1_AXP_s104
                                 + 0.555624 w_k1_DIS_s104
                                 + 0.687833 w_k1_FDX_s104
                                 + 0.762485 w_k1_IBM_s104
                                 + 0.811938 w_k1_INTC_s104
                                 + 0.778827 w_k1_KO_s104
                                 + 0.733198 w_k1_MAT_s104
                                 + 0.76373 w_k1_MSFT_s104
                                 + 0.842314 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.819256 w_k1_ADM_s104
                                 + 1.35293 w_k1_AXP_s104
                                 + 1.41467 w_k1_DIS_s104
                                 + 1.10015 w_k1_FDX_s104
                                 + 1.76596 w_k1_IBM_s104
                                 + 1.40803 w_k1_INTC_s104
                                 + 1.53214 w_k1_KO_s104 + 1.75329 w_k1_MAT_s104
                                 + 1.41738 w_k1_MSFT_s104
                                 + 1.19692 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.998797 w_k1_ADM_s104
                                 + 0.993614 w_k1_AXP_s104
                                 + 0.85975 w_k1_DIS_s104
                                 + 1.07919 w_k1_FDX_s104
                                 + 0.910523 w_k1_IBM_s104
                                 + 1.0342 w_k1_INTC_s104
                                 + 0.965867 w_k1_KO_s104
                                 + 1.06308 w_k1_MAT_s104
                                 + 1.04489 w_k1_MSFT_s104
                                 + 1.22852 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.807714 w_k1_ADM_s104
                                 + 0.843639 w_k1_AXP_s104
                                 + 0.845059 w_k1_DIS_s104
                                 + 0.982876 w_k1_FDX_s104
                                 + 1.27361 w_k1_IBM_s104
                                 + 1.11347 w_k1_INTC_s104
                                 + 0.956778 w_k1_KO_s104
                                 + 1.40421 w_k1_MAT_s104
                                 + 1.04598 w_k1_MSFT_s104
                                 + 0.95484 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.654914 w_k1_ADM_s105
                                 + 0.705144 w_k1_AXP_s105
                                 + 0.761657 w_k1_DIS_s105
                                 + 0.82315 w_k1_FDX_s105
                                 + 0.973287 w_k1_IBM_s105
                                 + 1.03745 w_k1_INTC_s105
                                 + 1.01181 w_k1_KO_s105
                                 + 0.762012 w_k1_MAT_s105
                                 + 0.904491 w_k1_MSFT_s105
                                 + 0.824325 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.907208 w_k1_ADM_s105
                                 + 0.703446 w_k1_AXP_s105
                                 + 0.652068 w_k1_DIS_s105
                                 + 0.746499 w_k1_FDX_s105
                                 + 0.639881 w_k1_IBM_s105
                                 + 0.938636 w_k1_INTC_s105
                                 + 0.80436 w_k1_KO_s105
                                 + 0.542145 w_k1_MAT_s105
                                 + 0.778242 w_k1_MSFT_s105
                                 + 0.771369 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.547681 w_k1_ADM_s105
                                 + 0.714879 w_k1_AXP_s105
                                 + 0.785905 w_k1_DIS_s105
                                 + 0.83071 w_k1_FDX_s105
                                 + 0.784333 w_k1_IBM_s105
                                 + 1.13713 w_k1_INTC_s105
                                 + 0.989876 w_k1_KO_s105
                                 + 0.655043 w_k1_MAT_s105
                                 + 0.923917 w_k1_MSFT_s105
                                 + 0.719105 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.773644 w_k1_ADM_s105
                                 + 1.16641 w_k1_AXP_s105
                                 + 1.13702 w_k1_DIS_s105
                                 + 0.921816 w_k1_FDX_s105
                                 + 1.48847 w_k1_IBM_s105
                                 + 1.15555 w_k1_INTC_s105
                                 + 1.67404 w_k1_KO_s105 + 1.57041 w_k1_MAT_s105
                                 + 1.05995 w_k1_MSFT_s105
                                 + 1.01975 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.926959 w_k1_ADM_s105
                                 + 1.10418 w_k1_AXP_s105
                                 + 1.39767 w_k1_DIS_s105
                                 + 1.34602 w_k1_FDX_s105
                                 + 0.995085 w_k1_IBM_s105
                                 + 1.50798 w_k1_INTC_s105
                                 + 1.30403 w_k1_KO_s105 + 1.1011 w_k1_MAT_s105
                                 + 1.31521 w_k1_MSFT_s105 + 1.0258 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.05242 w_k1_ADM_s105
                                 + 0.976213 w_k1_AXP_s105 + 1.449 w_k1_DIS_s105
                                 + 1.27437 w_k1_FDX_s105
                                 + 1.30491 w_k1_IBM_s105
                                 + 1.15395 w_k1_INTC_s105
                                 + 1.49154 w_k1_KO_s105 + 1.0346 w_k1_MAT_s105
                                 + 1.19185 w_k1_MSFT_s105
                                 + 1.03378 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.890186 w_k1_ADM_s105
                                 + 1.24179 w_k1_AXP_s105
                                 + 1.16302 w_k1_DIS_s105
                                 + 1.15718 w_k1_FDX_s105
                                 + 0.997585 w_k1_IBM_s105
                                 + 1.15343 w_k1_INTC_s105
                                 + 1.19644 w_k1_KO_s105 + 1.18823 w_k1_MAT_s105
                                 + 1.097 w_k1_MSFT_s105 + 1.0122 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.835423 w_k1_ADM_s105
                                 + 0.912641 w_k1_AXP_s105
                                 + 0.798824 w_k1_DIS_s105
                                 + 0.948995 w_k1_FDX_s105
                                 + 0.965093 w_k1_IBM_s105
                                 + 1.1824 w_k1_INTC_s105 + 1.05898 w_k1_KO_s105
                                 + 0.972768 w_k1_MAT_s105
                                 + 0.977953 w_k1_MSFT_s105
                                 + 0.880146 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.13459 w_k1_ADM_s106
                                 + 0.831578 w_k1_AXP_s106
                                 + 0.67876 w_k1_DIS_s106
                                 + 0.839002 w_k1_FDX_s106
                                 + 0.902123 w_k1_IBM_s106
                                 + 0.740061 w_k1_INTC_s106
                                 + 0.914307 w_k1_KO_s106
                                 + 0.877528 w_k1_MAT_s106
                                 + 0.782254 w_k1_MSFT_s106
                                 + 0.925306 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.892464 w_k1_ADM_s106
                                 + 0.712065 w_k1_AXP_s106
                                 + 0.696686 w_k1_DIS_s106
                                 + 0.83262 w_k1_FDX_s106
                                 + 0.725313 w_k1_IBM_s106
                                 + 0.695974 w_k1_INTC_s106
                                 + 0.816179 w_k1_KO_s106
                                 + 0.506941 w_k1_MAT_s106
                                 + 0.833126 w_k1_MSFT_s106
                                 + 0.843796 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.3177 w_k1_ADM_s106
                                 + 1.44466 w_k1_AXP_s106
                                 + 1.47179 w_k1_DIS_s106
                                 + 1.31346 w_k1_FDX_s106
                                 + 1.17025 w_k1_IBM_s106
                                 + 1.03343 w_k1_INTC_s106
                                 + 1.42764 w_k1_KO_s106 + 1.26147 w_k1_MAT_s106
                                 + 1.23381 w_k1_MSFT_s106
                                 + 1.27907 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.17604 w_k1_ADM_s106
                                 + 0.637578 w_k1_AXP_s106
                                 + 0.773008 w_k1_DIS_s106
                                 + 0.845067 w_k1_FDX_s106
                                 + 0.519581 w_k1_IBM_s106
                                 + 0.839845 w_k1_INTC_s106
                                 + 0.768186 w_k1_KO_s106
                                 + 0.66349 w_k1_MAT_s106
                                 + 0.907001 w_k1_MSFT_s106
                                 + 0.833396 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.03717 w_k1_ADM_s106
                                 + 0.905652 w_k1_AXP_s106
                                 + 1.00967 w_k1_DIS_s106
                                 + 0.938521 w_k1_FDX_s106
                                 + 1.11942 w_k1_IBM_s106
                                 + 0.962822 w_k1_INTC_s106
                                 + 1.05223 w_k1_KO_s106 + 1.09262 w_k1_MAT_s106
                                 + 1.00368 w_k1_MSFT_s106
                                 + 0.965565 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.29273 w_k1_ADM_s106
                                 + 0.966973 w_k1_AXP_s106
                                 + 1.1679 w_k1_DIS_s106 + 1.14517 w_k1_FDX_s106
                                 + 0.820439 w_k1_IBM_s106
                                 + 1.08643 w_k1_INTC_s106
                                 + 0.989555 w_k1_KO_s106
                                 + 0.836987 w_k1_MAT_s106
                                 + 1.00623 w_k1_MSFT_s106
                                 + 1.00588 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.08436 w_k1_ADM_s106
                                 + 1.02634 w_k1_AXP_s106
                                 + 1.42975 w_k1_DIS_s106
                                 + 1.15581 w_k1_FDX_s106
                                 + 1.05139 w_k1_IBM_s106
                                 + 1.26101 w_k1_INTC_s106
                                 + 1.28873 w_k1_KO_s106 + 1.07813 w_k1_MAT_s106
                                 + 1.10836 w_k1_MSFT_s106
                                 + 1.10888 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.6084 w_k1_ADM_s106
                                 + 1.32981 w_k1_AXP_s106
                                 + 1.18903 w_k1_DIS_s106
                                 + 1.10444 w_k1_FDX_s106
                                 + 1.60992 w_k1_IBM_s106
                                 + 0.983725 w_k1_INTC_s106
                                 + 1.35891 w_k1_KO_s106 + 1.23115 w_k1_MAT_s106
                                 + 1.07394 w_k1_MSFT_s106
                                 + 1.18123 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.9156 w_k1_ADM_s107
                                 + 0.840043 w_k1_AXP_s107
                                 + 1.03178 w_k1_DIS_s107
                                 + 0.978416 w_k1_FDX_s107
                                 + 0.879827 w_k1_IBM_s107
                                 + 0.907351 w_k1_INTC_s107
                                 + 0.902229 w_k1_KO_s107
                                 + 0.539511 w_k1_MAT_s107
                                 + 0.988341 w_k1_MSFT_s107
                                 + 1.05037 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.00746 w_k1_ADM_s107
                                 + 0.936827 w_k1_AXP_s107
                                 + 0.768042 w_k1_DIS_s107
                                 + 0.827824 w_k1_FDX_s107
                                 + 1.0411 w_k1_IBM_s107
                                 + 0.683327 w_k1_INTC_s107
                                 + 0.807115 w_k1_KO_s107
                                 + 0.920245 w_k1_MAT_s107
                                 + 0.639364 w_k1_MSFT_s107
                                 + 0.978369 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.787945 w_k1_ADM_s107
                                 + 0.849277 w_k1_AXP_s107
                                 + 1.17243 w_k1_DIS_s107
                                 + 0.867309 w_k1_FDX_s107
                                 + 1.23778 w_k1_IBM_s107
                                 + 0.720844 w_k1_INTC_s107
                                 + 1.04481 w_k1_KO_s107 + 1.16773 w_k1_MAT_s107
                                 + 0.916835 w_k1_MSFT_s107
                                 + 1.16196 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.21748 w_k1_ADM_s107
                                 + 1.32547 w_k1_AXP_s107
                                 + 1.62775 w_k1_DIS_s107
                                 + 1.24198 w_k1_FDX_s107
                                 + 1.07155 w_k1_IBM_s107
                                 + 0.994266 w_k1_INTC_s107
                                 + 1.07718 w_k1_KO_s107 + 1.04836 w_k1_MAT_s107
                                 + 1.1673 w_k1_MSFT_s107 + 1.09202 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.83941 w_k1_ADM_s107
                                 + 0.830067 w_k1_AXP_s107
                                 + 0.655524 w_k1_DIS_s107
                                 + 0.752234 w_k1_FDX_s107
                                 + 0.668979 w_k1_IBM_s107
                                 + 0.738949 w_k1_INTC_s107
                                 + 0.729255 w_k1_KO_s107
                                 + 0.699784 w_k1_MAT_s107
                                 + 0.685753 w_k1_MSFT_s107
                                 + 0.836866 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.00542 w_k1_ADM_s107
                                 + 1.28455 w_k1_AXP_s107
                                 + 1.12139 w_k1_DIS_s107
                                 + 1.08004 w_k1_FDX_s107 + 1.1359 w_k1_IBM_s107
                                 + 1.05803 w_k1_INTC_s107
                                 + 1.08627 w_k1_KO_s107 + 1.3012 w_k1_MAT_s107
                                 + 1.03351 w_k1_MSFT_s107
                                 + 1.11627 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.23387 w_k1_ADM_s107
                                 + 1.10707 w_k1_AXP_s107
                                 + 0.974555 w_k1_DIS_s107
                                 + 0.941248 w_k1_FDX_s107
                                 + 1.49694 w_k1_IBM_s107
                                 + 0.890745 w_k1_INTC_s107
                                 + 1.08651 w_k1_KO_s107 + 1.31013 w_k1_MAT_s107
                                 + 0.875541 w_k1_MSFT_s107
                                 + 1.05522 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.870283 w_k1_ADM_s107
                                 + 0.63207 w_k1_AXP_s107
                                 + 0.734826 w_k1_DIS_s107
                                 + 0.805292 w_k1_FDX_s107
                                 + 0.673153 w_k1_IBM_s107
                                 + 0.778655 w_k1_INTC_s107
                                 + 0.785584 w_k1_KO_s107
                                 + 0.517399 w_k1_MAT_s107
                                 + 0.740786 w_k1_MSFT_s107
                                 + 0.799437 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.39881 w_k1_ADM_s110
                                 + 0.948921 w_k1_AXP_s110
                                 + 1.1964 w_k1_DIS_s110 + 1.08371 w_k1_FDX_s110
                                 + 1.16228 w_k1_IBM_s110
                                 + 0.884481 w_k1_INTC_s110
                                 + 1.26492 w_k1_KO_s110 + 1.09341 w_k1_MAT_s110
                                 + 0.949308 w_k1_MSFT_s110
                                 + 1.06778 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.23604 w_k1_ADM_s110
                                 + 1.18911 w_k1_AXP_s110
                                 + 1.20306 w_k1_DIS_s110
                                 + 1.19274 w_k1_FDX_s110
                                 + 0.881861 w_k1_IBM_s110
                                 + 1.37284 w_k1_INTC_s110
                                 + 1.16268 w_k1_KO_s110
                                 + 0.940151 w_k1_MAT_s110
                                 + 1.35266 w_k1_MSFT_s110
                                 + 1.00111 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.842099 w_k1_ADM_s110
                                 + 0.765029 w_k1_AXP_s110
                                 + 0.779985 w_k1_DIS_s110
                                 + 0.797149 w_k1_FDX_s110
                                 + 0.989427 w_k1_IBM_s110
                                 + 1.02707 w_k1_INTC_s110
                                 + 0.901324 w_k1_KO_s110
                                 + 0.73089 w_k1_MAT_s110
                                 + 0.85156 w_k1_MSFT_s110
                                 + 0.824705 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.816946 w_k1_ADM_s110
                                 + 1.01604 w_k1_AXP_s110
                                 + 1.14733 w_k1_DIS_s110
                                 + 0.960857 w_k1_FDX_s110
                                 + 1.4332 w_k1_IBM_s110
                                 + 1.00814 w_k1_INTC_s110
                                 + 1.21562 w_k1_KO_s110 + 1.15774 w_k1_MAT_s110
                                 + 0.948416 w_k1_MSFT_s110
                                 + 1.00396 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.960923 w_k1_ADM_s110
                                 + 1.06226 w_k1_AXP_s110
                                 + 0.948495 w_k1_DIS_s110
                                 + 0.982662 w_k1_FDX_s110
                                 + 1.02664 w_k1_IBM_s110
                                 + 0.863976 w_k1_INTC_s110
                                 + 0.949742 w_k1_KO_s110
                                 + 1.20963 w_k1_MAT_s110
                                 + 0.973591 w_k1_MSFT_s110
                                 + 0.968228 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.871565 w_k1_ADM_s110
                                 + 0.831031 w_k1_AXP_s110
                                 + 0.922505 w_k1_DIS_s110
                                 + 0.9987 w_k1_FDX_s110
                                 + 0.773108 w_k1_IBM_s110
                                 + 1.00773 w_k1_INTC_s110
                                 + 0.837475 w_k1_KO_s110
                                 + 0.631244 w_k1_MAT_s110
                                 + 0.875364 w_k1_MSFT_s110
                                 + 0.988642 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.63388 w_k1_ADM_s110
                                 + 1.61948 w_k1_AXP_s110
                                 + 1.86352 w_k1_DIS_s110
                                 + 1.41848 w_k1_FDX_s110
                                 + 1.37621 w_k1_IBM_s110
                                 + 1.20883 w_k1_INTC_s110 + 1.5235 w_k1_KO_s110
                                 + 1.65444 w_k1_MAT_s110
                                 + 1.3518 w_k1_MSFT_s110 + 1.33047 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.936115 w_k1_ADM_s110
                                 + 0.683034 w_k1_AXP_s110
                                 + 0.620703 w_k1_DIS_s110
                                 + 0.774347 w_k1_FDX_s110
                                 + 0.664806 w_k1_IBM_s110
                                 + 0.92927 w_k1_INTC_s110
                                 + 0.758116 w_k1_KO_s110
                                 + 0.710226 w_k1_MAT_s110
                                 + 0.786793 w_k1_MSFT_s110
                                 + 0.742914 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.01651 w_k1_ADM_s111
                                 + 0.984158 w_k1_AXP_s111
                                 + 1.10083 w_k1_DIS_s111
                                 + 1.10639 w_k1_FDX_s111
                                 + 0.69327 w_k1_IBM_s111
                                 + 1.11834 w_k1_INTC_s111
                                 + 0.940452 w_k1_KO_s111
                                 + 0.547646 w_k1_MAT_s111
                                 + 1.14128 w_k1_MSFT_s111
                                 + 1.07271 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.06874 w_k1_ADM_s111
                                 + 0.987567 w_k1_AXP_s111
                                 + 1.12387 w_k1_DIS_s111
                                 + 0.985487 w_k1_FDX_s111
                                 + 1.13747 w_k1_IBM_s111
                                 + 1.06347 w_k1_INTC_s111
                                 + 1.00056 w_k1_KO_s111
                                 + 0.666977 w_k1_MAT_s111
                                 + 1.0865 w_k1_MSFT_s111
                                 + 0.955562 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.905166 w_k1_ADM_s111
                                 + 0.792546 w_k1_AXP_s111
                                 + 0.778048 w_k1_DIS_s111
                                 + 0.790034 w_k1_FDX_s111
                                 + 0.755311 w_k1_IBM_s111
                                 + 1.01693 w_k1_INTC_s111
                                 + 0.791354 w_k1_KO_s111
                                 + 0.528675 w_k1_MAT_s111
                                 + 0.789703 w_k1_MSFT_s111
                                 + 0.791312 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.05325 w_k1_ADM_s111
                                 + 1.0308 w_k1_AXP_s111
                                 + 0.892854 w_k1_DIS_s111
                                 + 0.862608 w_k1_FDX_s111
                                 + 1.44903 w_k1_IBM_s111
                                 + 1.24849 w_k1_INTC_s111
                                 + 1.07471 w_k1_KO_s111 + 1.00052 w_k1_MAT_s111
                                 + 0.840588 w_k1_MSFT_s111
                                 + 1.07968 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.37174 w_k1_ADM_s111
                                 + 1.29199 w_k1_AXP_s111
                                 + 1.58002 w_k1_DIS_s111
                                 + 1.31723 w_k1_FDX_s111
                                 + 1.10793 w_k1_IBM_s111
                                 + 1.3656 w_k1_INTC_s111 + 1.2599 w_k1_KO_s111
                                 + 1.11001 w_k1_MAT_s111
                                 + 1.22139 w_k1_MSFT_s111
                                 + 1.16483 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.739696 w_k1_ADM_s111
                                 + 0.795029 w_k1_AXP_s111
                                 + 0.644897 w_k1_DIS_s111
                                 + 0.670426 w_k1_FDX_s111
                                 + 0.91756 w_k1_IBM_s111
                                 + 0.870201 w_k1_INTC_s111
                                 + 0.838081 w_k1_KO_s111
                                 + 0.926207 w_k1_MAT_s111
                                 + 0.737659 w_k1_MSFT_s111
                                 + 0.707281 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.09061 w_k1_ADM_s111
                                 + 1.05171 w_k1_AXP_s111
                                 + 0.755114 w_k1_DIS_s111
                                 + 0.805133 w_k1_FDX_s111
                                 + 1.1748 w_k1_IBM_s111 + 1.0526 w_k1_INTC_s111
                                 + 0.885081 w_k1_KO_s111
                                 + 0.845596 w_k1_MAT_s111
                                 + 0.791151 w_k1_MSFT_s111
                                 + 0.83882 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.05705 w_k1_ADM_s111
                                 + 1.35524 w_k1_AXP_s111
                                 + 0.845942 w_k1_DIS_s111
                                 + 1.01008 w_k1_FDX_s111
                                 + 0.937814 w_k1_IBM_s111
                                 + 1.22979 w_k1_INTC_s111
                                 + 0.946145 w_k1_KO_s111
                                 + 0.850263 w_k1_MAT_s111
                                 + 1.02332 w_k1_MSFT_s111
                                 + 0.990243 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.23387 w_k1_ADM_s112
                                 + 1.0606 w_k1_AXP_s112 + 1.61886 w_k1_DIS_s112
                                 + 1.24319 w_k1_FDX_s112 + 1.3114 w_k1_IBM_s112
                                 + 1.07165 w_k1_INTC_s112
                                 + 1.44396 w_k1_KO_s112 + 1.43621 w_k1_MAT_s112
                                 + 1.29593 w_k1_MSFT_s112
                                 + 1.12982 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.664859 w_k1_ADM_s112
                                 + 0.636571 w_k1_AXP_s112
                                 + 0.661577 w_k1_DIS_s112
                                 + 0.76523 w_k1_FDX_s112
                                 + 0.686499 w_k1_IBM_s112
                                 + 1.00597 w_k1_INTC_s112
                                 + 0.785091 w_k1_KO_s112
                                 + 0.620171 w_k1_MAT_s112
                                 + 0.912888 w_k1_MSFT_s112
                                 + 0.720375 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.75503 w_k1_ADM_s112
                                 + 0.671608 w_k1_AXP_s112
                                 + 0.745091 w_k1_DIS_s112
                                 + 0.699452 w_k1_FDX_s112
                                 + 1.13001 w_k1_IBM_s112
                                 + 0.752149 w_k1_INTC_s112
                                 + 1.08395 w_k1_KO_s112 + 1.26082 w_k1_MAT_s112
                                 + 0.863776 w_k1_MSFT_s112
                                 + 0.884239 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.845464 w_k1_ADM_s112
                                 + 0.837795 w_k1_AXP_s112
                                 + 0.979174 w_k1_DIS_s112
                                 + 0.958675 w_k1_FDX_s112
                                 + 1.13674 w_k1_IBM_s112
                                 + 0.918051 w_k1_INTC_s112
                                 + 0.948457 w_k1_KO_s112
                                 + 1.32741 w_k1_MAT_s112
                                 + 0.98615 w_k1_MSFT_s112
                                 + 0.91259 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.07564 w_k1_ADM_s112
                                 + 0.968542 w_k1_AXP_s112
                                 + 1.14293 w_k1_DIS_s112 + 1.2543 w_k1_FDX_s112
                                 + 0.746545 w_k1_IBM_s112
                                 + 0.999407 w_k1_INTC_s112
                                 + 0.979737 w_k1_KO_s112
                                 + 1.10716 w_k1_MAT_s112
                                 + 1.35037 w_k1_MSFT_s112 + 1.0569 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.879335 w_k1_ADM_s112
                                 + 0.857637 w_k1_AXP_s112
                                 + 0.786856 w_k1_DIS_s112
                                 + 0.896844 w_k1_FDX_s112
                                 + 0.918112 w_k1_IBM_s112
                                 + 0.875608 w_k1_INTC_s112
                                 + 0.879393 w_k1_KO_s112
                                 + 0.986986 w_k1_MAT_s112
                                 + 0.874237 w_k1_MSFT_s112
                                 + 0.846666 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.833558 w_k1_ADM_s112
                                 + 1.06678 w_k1_AXP_s112
                                 + 1.02654 w_k1_DIS_s112
                                 + 0.999172 w_k1_FDX_s112
                                 + 1.67513 w_k1_IBM_s112
                                 + 0.987187 w_k1_INTC_s112
                                 + 1.21583 w_k1_KO_s112 + 1.88188 w_k1_MAT_s112
                                 + 0.880523 w_k1_MSFT_s112
                                 + 0.893428 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.12618 w_k1_ADM_s112
                                 + 1.08121 w_k1_AXP_s112
                                 + 0.918483 w_k1_DIS_s112
                                 + 1.10097 w_k1_FDX_s112
                                 + 0.965276 w_k1_IBM_s112
                                 + 1.11846 w_k1_INTC_s112
                                 + 0.97329 w_k1_KO_s112 + 1.30769 w_k1_MAT_s112
                                 + 1.08944 w_k1_MSFT_s112
                                 + 0.917742 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.880872 w_k1_ADM_s113
                                 + 0.825158 w_k1_AXP_s113
                                 + 0.695154 w_k1_DIS_s113
                                 + 0.875484 w_k1_FDX_s113
                                 + 0.801936 w_k1_IBM_s113
                                 + 0.825432 w_k1_INTC_s113
                                 + 0.705625 w_k1_KO_s113
                                 + 0.813006 w_k1_MAT_s113
                                 + 0.826214 w_k1_MSFT_s113
                                 + 0.937083 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.13756 w_k1_ADM_s113
                                 + 1.51598 w_k1_AXP_s113
                                 + 1.66256 w_k1_DIS_s113
                                 + 1.19771 w_k1_FDX_s113
                                 + 1.67695 w_k1_IBM_s113
                                 + 1.3733 w_k1_INTC_s113 + 1.28965 w_k1_KO_s113
                                 + 1.33493 w_k1_MAT_s113
                                 + 1.23833 w_k1_MSFT_s113
                                 + 1.30104 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.01387 w_k1_ADM_s113
                                 + 1.2144 w_k1_AXP_s113 + 1.37557 w_k1_DIS_s113
                                 + 1.16799 w_k1_FDX_s113
                                 + 0.951417 w_k1_IBM_s113
                                 + 1.02322 w_k1_INTC_s113
                                 + 0.965448 w_k1_KO_s113
                                 + 0.843558 w_k1_MAT_s113
                                 + 1.07742 w_k1_MSFT_s113
                                 + 1.06864 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.00235 w_k1_ADM_s113
                                 + 1.3589 w_k1_AXP_s113 + 1.91162 w_k1_DIS_s113
                                 + 1.34961 w_k1_FDX_s113
                                 + 1.18778 w_k1_IBM_s113
                                 + 1.20839 w_k1_INTC_s113
                                 + 1.31277 w_k1_KO_s113 + 1.44389 w_k1_MAT_s113
                                 + 1.28155 w_k1_MSFT_s113
                                 + 1.25631 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.844558 w_k1_ADM_s113
                                 + 1.07902 w_k1_AXP_s113
                                 + 1.20795 w_k1_DIS_s113
                                 + 1.02778 w_k1_FDX_s113
                                 + 1.09578 w_k1_IBM_s113
                                 + 1.11077 w_k1_INTC_s113
                                 + 1.11913 w_k1_KO_s113 + 1.25803 w_k1_MAT_s113
                                 + 1.19464 w_k1_MSFT_s113
                                 + 1.06624 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.936369 w_k1_ADM_s113
                                 + 1.1573 w_k1_AXP_s113
                                 + 0.897354 w_k1_DIS_s113
                                 + 1.10374 w_k1_FDX_s113
                                 + 0.742105 w_k1_IBM_s113
                                 + 0.934055 w_k1_INTC_s113
                                 + 0.803439 w_k1_KO_s113
                                 + 0.81128 w_k1_MAT_s113
                                 + 1.01877 w_k1_MSFT_s113
                                 + 1.13147 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.974773 w_k1_ADM_s113
                                 + 1.13438 w_k1_AXP_s113
                                 + 0.928802 w_k1_DIS_s113
                                 + 0.918699 w_k1_FDX_s113
                                 + 1.32028 w_k1_IBM_s113
                                 + 0.910358 w_k1_INTC_s113
                                 + 1.08158 w_k1_KO_s113 + 1.22193 w_k1_MAT_s113
                                 + 1.00564 w_k1_MSFT_s113
                                 + 1.06138 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.770994 w_k1_ADM_s113
                                 + 0.864068 w_k1_AXP_s113
                                 + 0.87025 w_k1_DIS_s113
                                 + 0.859452 w_k1_FDX_s113
                                 + 0.871666 w_k1_IBM_s113
                                 + 0.968655 w_k1_INTC_s113
                                 + 0.851354 w_k1_KO_s113
                                 + 0.839976 w_k1_MAT_s113
                                 + 0.981991 w_k1_MSFT_s113
                                 + 0.942031 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.866877 w_k1_ADM_s114
                                 + 1.02279 w_k1_AXP_s114
                                 + 1.16062 w_k1_DIS_s114
                                 + 1.20622 w_k1_FDX_s114
                                 + 0.851984 w_k1_IBM_s114
                                 + 1.01782 w_k1_INTC_s114
                                 + 0.921505 w_k1_KO_s114
                                 + 0.966326 w_k1_MAT_s114
                                 + 0.960168 w_k1_MSFT_s114
                                 + 1.12727 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.693995 w_k1_ADM_s114
                                 + 0.844364 w_k1_AXP_s114
                                 + 0.768214 w_k1_DIS_s114
                                 + 0.844919 w_k1_FDX_s114
                                 + 0.909271 w_k1_IBM_s114
                                 + 0.674159 w_k1_INTC_s114
                                 + 0.891415 w_k1_KO_s114
                                 + 0.974248 w_k1_MAT_s114
                                 + 0.788875 w_k1_MSFT_s114
                                 + 0.979632 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.14657 w_k1_ADM_s114
                                 + 1.76844 w_k1_AXP_s114
                                 + 1.76666 w_k1_DIS_s114
                                 + 1.34674 w_k1_FDX_s114 + 2.1354 w_k1_IBM_s114
                                 + 1.37392 w_k1_INTC_s114
                                 + 1.72804 w_k1_KO_s114 + 2.23463 w_k1_MAT_s114
                                 + 1.37384 w_k1_MSFT_s114
                                 + 1.24069 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.03323 w_k1_ADM_s114
                                 + 0.863374 w_k1_AXP_s114
                                 + 0.760488 w_k1_DIS_s114
                                 + 0.922437 w_k1_FDX_s114
                                 + 0.582601 w_k1_IBM_s114
                                 + 0.786345 w_k1_INTC_s114
                                 + 0.765238 w_k1_KO_s114
                                 + 0.687634 w_k1_MAT_s114
                                 + 0.845951 w_k1_MSFT_s114
                                 + 0.93065 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.73075 w_k1_ADM_s114
                                 + 0.599649 w_k1_AXP_s114
                                 + 0.551678 w_k1_DIS_s114
                                 + 0.709176 w_k1_FDX_s114
                                 + 0.707388 w_k1_IBM_s114
                                 + 0.668393 w_k1_INTC_s114
                                 + 0.774066 w_k1_KO_s114
                                 + 0.661183 w_k1_MAT_s114
                                 + 0.737093 w_k1_MSFT_s114
                                 + 0.821327 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.06414 w_k1_ADM_s114
                                 + 1.42333 w_k1_AXP_s114
                                 + 1.45551 w_k1_DIS_s114
                                 + 1.27886 w_k1_FDX_s114
                                 + 1.17921 w_k1_IBM_s114
                                 + 0.919235 w_k1_INTC_s114
                                 + 1.29429 w_k1_KO_s114 + 1.37838 w_k1_MAT_s114
                                 + 1.1113 w_k1_MSFT_s114 + 1.17888 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.915897 w_k1_ADM_s114
                                 + 0.892345 w_k1_AXP_s114
                                 + 0.863939 w_k1_DIS_s114
                                 + 0.950809 w_k1_FDX_s114
                                 + 0.719726 w_k1_IBM_s114
                                 + 0.823109 w_k1_INTC_s114
                                 + 0.823191 w_k1_KO_s114
                                 + 0.784257 w_k1_MAT_s114
                                 + 0.923679 w_k1_MSFT_s114
                                 + 0.994523 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.27485 w_k1_ADM_s114
                                 + 1.14302 w_k1_AXP_s114
                                 + 1.42169 w_k1_DIS_s114
                                 + 1.23198 w_k1_FDX_s114
                                 + 1.38651 w_k1_IBM_s114
                                 + 1.02112 w_k1_INTC_s114
                                 + 1.21782 w_k1_KO_s114
                                 + 0.987519 w_k1_MAT_s114
                                 + 1.10463 w_k1_MSFT_s114
                                 + 1.01781 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.04537 w_k1_ADM_s115
                                 + 0.875906 w_k1_AXP_s115
                                 + 0.980687 w_k1_DIS_s115
                                 + 1.00987 w_k1_FDX_s115
                                 + 0.869787 w_k1_IBM_s115
                                 + 0.822813 w_k1_INTC_s115
                                 + 1.00267 w_k1_KO_s115 + 1.01129 w_k1_MAT_s115
                                 + 0.95655 w_k1_MSFT_s115
                                 + 1.08321 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.75904 w_k1_ADM_s115
                                 + 1.59542 w_k1_AXP_s115
                                 + 1.70947 w_k1_DIS_s115
                                 + 1.36545 w_k1_FDX_s115
                                 + 1.37956 w_k1_IBM_s115 + 1.228 w_k1_INTC_s115
                                 + 1.6787 w_k1_KO_s115 + 1.55033 w_k1_MAT_s115
                                 + 1.33326 w_k1_MSFT_s115
                                 + 1.52891 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.15913 w_k1_ADM_s115
                                 + 1.12033 w_k1_AXP_s115
                                 + 0.810438 w_k1_DIS_s115
                                 + 0.954822 w_k1_FDX_s115
                                 + 1.18002 w_k1_IBM_s115
                                 + 1.04346 w_k1_INTC_s115
                                 + 1.00867 w_k1_KO_s115 + 1.33862 w_k1_MAT_s115
                                 + 0.961379 w_k1_MSFT_s115
                                 + 1.10913 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.82511 w_k1_ADM_s115
                                 + 0.597897 w_k1_AXP_s115
                                 + 0.788495 w_k1_DIS_s115
                                 + 0.697632 w_k1_FDX_s115
                                 + 1.08043 w_k1_IBM_s115
                                 + 0.664249 w_k1_INTC_s115
                                 + 0.968443 w_k1_KO_s115
                                 + 0.786414 w_k1_MAT_s115
                                 + 0.847452 w_k1_MSFT_s115
                                 + 0.965541 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.51231 w_k1_ADM_s115
                                 + 1.03669 w_k1_AXP_s115
                                 + 1.25665 w_k1_DIS_s115
                                 + 1.23386 w_k1_FDX_s115
                                 + 0.892256 w_k1_IBM_s115
                                 + 1.06382 w_k1_INTC_s115
                                 + 1.00714 w_k1_KO_s115
                                 + 0.703847 w_k1_MAT_s115
                                 + 1.20728 w_k1_MSFT_s115
                                 + 1.13351 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.06042 w_k1_ADM_s115
                                 + 1.0225 w_k1_AXP_s115
                                 + 0.674371 w_k1_DIS_s115
                                 + 0.80641 w_k1_FDX_s115
                                 + 0.845552 w_k1_IBM_s115
                                 + 0.75823 w_k1_INTC_s115
                                 + 0.815227 w_k1_KO_s115
                                 + 0.836684 w_k1_MAT_s115
                                 + 0.697033 w_k1_MSFT_s115
                                 + 0.943293 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.31432 w_k1_ADM_s115
                                 + 0.971611 w_k1_AXP_s115
                                 + 1.19381 w_k1_DIS_s115
                                 + 1.03535 w_k1_FDX_s115
                                 + 1.43398 w_k1_IBM_s115
                                 + 1.05887 w_k1_INTC_s115 + 1.3618 w_k1_KO_s115
                                 + 0.948525 w_k1_MAT_s115
                                 + 1.05994 w_k1_MSFT_s115
                                 + 1.14924 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.811398 w_k1_ADM_s115
                                 + 0.61874 w_k1_AXP_s115
                                 + 0.740922 w_k1_DIS_s115
                                 + 0.780777 w_k1_FDX_s115
                                 + 0.668872 w_k1_IBM_s115
                                 + 0.774161 w_k1_INTC_s115
                                 + 0.872849 w_k1_KO_s115
                                 + 0.420519 w_k1_MAT_s115
                                 + 0.804025 w_k1_MSFT_s115
                                 + 0.811932 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.23116 w_k1_ADM_s116
                                 + 0.99889 w_k1_AXP_s116
                                 + 1.48588 w_k1_DIS_s116
                                 + 1.20091 w_k1_FDX_s116
                                 + 0.729043 w_k1_IBM_s116
                                 + 0.93822 w_k1_INTC_s116
                                 + 1.10928 w_k1_KO_s116
                                 + 0.847967 w_k1_MAT_s116
                                 + 1.34871 w_k1_MSFT_s116
                                 + 1.06232 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.16106 w_k1_ADM_s116
                                 + 1.25948 w_k1_AXP_s116
                                 + 1.12477 w_k1_DIS_s116 + 1.0749 w_k1_FDX_s116
                                 + 0.970569 w_k1_IBM_s116
                                 + 0.906379 w_k1_INTC_s116
                                 + 1.11896 w_k1_KO_s116
                                 + 0.997061 w_k1_MAT_s116
                                 + 1.06148 w_k1_MSFT_s116
                                 + 0.990546 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.885826 w_k1_ADM_s116
                                 + 0.781449 w_k1_AXP_s116
                                 + 0.664985 w_k1_DIS_s116
                                 + 0.815165 w_k1_FDX_s116
                                 + 0.742766 w_k1_IBM_s116
                                 + 0.694174 w_k1_INTC_s116
                                 + 0.768001 w_k1_KO_s116
                                 + 0.803057 w_k1_MAT_s116
                                 + 0.851047 w_k1_MSFT_s116
                                 + 0.813717 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.979128 w_k1_ADM_s116
                                 + 1.2731 w_k1_AXP_s116 + 1.19134 w_k1_DIS_s116
                                 + 0.981798 w_k1_FDX_s116
                                 + 1.54428 w_k1_IBM_s116
                                 + 1.0746 w_k1_INTC_s116 + 1.3519 w_k1_KO_s116
                                 + 1.45977 w_k1_MAT_s116
                                 + 0.947082 w_k1_MSFT_s116
                                 + 1.21226 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.834452 w_k1_ADM_s116
                                 + 1.0065 w_k1_AXP_s116
                                 + 0.947025 w_k1_DIS_s116
                                 + 0.92973 w_k1_FDX_s116
                                 + 1.18325 w_k1_IBM_s116
                                 + 0.802451 w_k1_INTC_s116
                                 + 1.18215 w_k1_KO_s116 + 1.21181 w_k1_MAT_s116
                                 + 0.93025 w_k1_MSFT_s116
                                 + 1.10809 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.934797 w_k1_ADM_s116
                                 + 0.734892 w_k1_AXP_s116
                                 + 0.911736 w_k1_DIS_s116
                                 + 0.931858 w_k1_FDX_s116
                                 + 0.725067 w_k1_IBM_s116
                                 + 0.97119 w_k1_INTC_s116
                                 + 1.11128 w_k1_KO_s116
                                 + 0.576256 w_k1_MAT_s116
                                 + 0.978838 w_k1_MSFT_s116
                                 + 0.922386 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.17492 w_k1_ADM_s116
                                 + 1.28992 w_k1_AXP_s116
                                 + 1.76631 w_k1_DIS_s116
                                 + 1.29304 w_k1_FDX_s116
                                 + 1.32634 w_k1_IBM_s116
                                 + 1.43832 w_k1_INTC_s116
                                 + 1.49709 w_k1_KO_s116 + 1.24834 w_k1_MAT_s116
                                 + 1.29427 w_k1_MSFT_s116
                                 + 1.26788 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.814904 w_k1_ADM_s116
                                 + 0.72928 w_k1_AXP_s116
                                 + 0.747795 w_k1_DIS_s116
                                 + 0.72944 w_k1_FDX_s116
                                 + 1.07227 w_k1_IBM_s116
                                 + 0.9015 w_k1_INTC_s116
                                 + 0.899982 w_k1_KO_s116
                                 + 0.85863 w_k1_MAT_s116
                                 + 0.977164 w_k1_MSFT_s116
                                 + 0.918981 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.84334 w_k1_ADM_s117
                                 + 0.699675 w_k1_AXP_s117
                                 + 0.639429 w_k1_DIS_s117
                                 + 0.748295 w_k1_FDX_s117
                                 + 0.770914 w_k1_IBM_s117
                                 + 0.780351 w_k1_INTC_s117
                                 + 0.859471 w_k1_KO_s117
                                 + 0.758089 w_k1_MAT_s117
                                 + 0.894143 w_k1_MSFT_s117
                                 + 0.782094 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.872001 w_k1_ADM_s117
                                 + 0.761423 w_k1_AXP_s117
                                 + 0.832396 w_k1_DIS_s117
                                 + 0.80319 w_k1_FDX_s117
                                 + 1.20757 w_k1_IBM_s117
                                 + 0.819637 w_k1_INTC_s117
                                 + 0.942836 w_k1_KO_s117
                                 + 1.14939 w_k1_MAT_s117
                                 + 0.915559 w_k1_MSFT_s117
                                 + 0.877806 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.12607 w_k1_ADM_s117
                                 + 1.215 w_k1_AXP_s117 + 1.18039 w_k1_DIS_s117
                                 + 1.20584 w_k1_FDX_s117
                                 + 1.22219 w_k1_IBM_s117
                                 + 0.935344 w_k1_INTC_s117
                                 + 1.04173 w_k1_KO_s117 + 1.84651 w_k1_MAT_s117
                                 + 1.38144 w_k1_MSFT_s117
                                 + 1.14449 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.11412 w_k1_ADM_s117
                                 + 1.37847 w_k1_AXP_s117
                                 + 1.36129 w_k1_DIS_s117
                                 + 1.38051 w_k1_FDX_s117
                                 + 1.06614 w_k1_IBM_s117
                                 + 1.28407 w_k1_INTC_s117
                                 + 1.18839 w_k1_KO_s117 + 1.33363 w_k1_MAT_s117
                                 + 1.23305 w_k1_MSFT_s117
                                 + 1.16586 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.936756 w_k1_ADM_s117
                                 + 0.945356 w_k1_AXP_s117
                                 + 1.08062 w_k1_DIS_s117
                                 + 1.04385 w_k1_FDX_s117
                                 + 0.839668 w_k1_IBM_s117
                                 + 0.771884 w_k1_INTC_s117
                                 + 1.01547 w_k1_KO_s117 + 1.23461 w_k1_MAT_s117
                                 + 1.08931 w_k1_MSFT_s117
                                 + 0.986031 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.871576 w_k1_ADM_s117
                                 + 1.09957 w_k1_AXP_s117 + 1.3572 w_k1_DIS_s117
                                 + 1.15451 w_k1_FDX_s117
                                 + 1.49359 w_k1_IBM_s117
                                 + 1.00116 w_k1_INTC_s117
                                 + 1.26955 w_k1_KO_s117 + 1.70506 w_k1_MAT_s117
                                 + 1.27154 w_k1_MSFT_s117
                                 + 1.15215 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.06673 w_k1_ADM_s117
                                 + 1.02109 w_k1_AXP_s117
                                 + 1.16408 w_k1_DIS_s117
                                 + 0.988228 w_k1_FDX_s117
                                 + 1.07353 w_k1_IBM_s117
                                 + 0.964193 w_k1_INTC_s117
                                 + 1.08081 w_k1_KO_s117 + 1.13851 w_k1_MAT_s117
                                 + 1.12071 w_k1_MSFT_s117
                                 + 0.954175 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.826604 w_k1_ADM_s117
                                 + 1.07666 w_k1_AXP_s117
                                 + 1.01792 w_k1_DIS_s117
                                 + 0.839271 w_k1_FDX_s117
                                 + 1.66799 w_k1_IBM_s117
                                 + 0.876369 w_k1_INTC_s117
                                 + 1.19841 w_k1_KO_s117 + 2.34615 w_k1_MAT_s117
                                 + 0.959643 w_k1_MSFT_s117
                                 + 1.13425 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0129287953160967 w_k1_ADM_s000
                                 + 0.0336917986711178 w_k1_ADM_s001
                                 + 0.0264426618103396 w_k1_ADM_s002
                                 + 0.0261735060393183 w_k1_ADM_s003
                                 + 0.0156916583214583 w_k1_ADM_s004
                                 + 0.0100315978147726 w_k1_ADM_s005
                                 + 0.033388570306727 w_k1_ADM_s006
                                 + 0.043393911685233 w_k1_ADM_s007
                                 + 0.0371866663724179 w_k1_ADM_s010
                                 + 0.0462789312207249 w_k1_ADM_s011
                                 + 0.0902807457578113 w_k1_ADM_s012
                                 + 0.0713013190273502 w_k1_ADM_s013
                                 + 0.0887258036080634 w_k1_ADM_s014
                                 + 0.0418797897601763 w_k1_ADM_s015
                                 + 0.0405114141228301 w_k1_ADM_s016
                                 + 0.0536535772928574 w_k1_ADM_s017
                                 + 0.013994020852609 w_k1_ADM_s100
                                 + 0.0230784614185806 w_k1_ADM_s101
                                 + 0.0304284586020612 w_k1_ADM_s102
                                 + 0.0111785963676618 w_k1_ADM_s103
                                 + 0.0153084402729153 w_k1_ADM_s104
                                 + 0.0122282415908254 w_k1_ADM_s105
                                 + 0.031317869400199 w_k1_ADM_s106
                                 + 0.0191713649263599 w_k1_ADM_s107
                                 + 0.0319628472477855 w_k1_ADM_s110
                                 + 0.0170248987336875 w_k1_ADM_s111
                                 + 0.0192491287492599 w_k1_ADM_s112
                                 + 0.0102781538243057 w_k1_ADM_s113
                                 + 0.0175129232246009 w_k1_ADM_s114
                                 + 0.0236700156826849 w_k1_ADM_s115
                                 + 0.0542416264925229 w_k1_ADM_s116
                                 + 0.0239380067174089 w_k1_ADM_s117
                                 + 0.0130754022498187 w_k1_AXP_s000
                                 + 0.0267570298754683 w_k1_AXP_s001
                                 + 0.0268939194468255 w_k1_AXP_s002
                                 + 0.0262901333693218 w_k1_AXP_s003
                                 + 0.016735029825859 w_k1_AXP_s004
                                 + 0.0101239989845962 w_k1_AXP_s005
                                 + 0.0346880478187664 w_k1_AXP_s006
                                 + 0.0441427473202417 w_k1_AXP_s007
                                 + 0.0355002576231062 w_k1_AXP_s010
                                 + 0.0508416462382453 w_k1_AXP_s011
                                 + 0.0889956914247928 w_k1_AXP_s012
                                 + 0.0670787668277072 w_k1_AXP_s013
                                 + 0.0920415458735504 w_k1_AXP_s014
                                 + 0.0385135643612411 w_k1_AXP_s015
                                 + 0.0359244262752861 w_k1_AXP_s016
                                 + 0.0612087225356895 w_k1_AXP_s017
                                 + 0.0129087799700078 w_k1_AXP_s100
                                 + 0.0245737728785212 w_k1_AXP_s101
                                 + 0.0293256864422622 w_k1_AXP_s102
                                 + 0.0123725116229036 w_k1_AXP_s103
                                 + 0.0146303930516182 w_k1_AXP_s104
                                 + 0.0141805201938404 w_k1_AXP_s105
                                 + 0.0264083429927261 w_k1_AXP_s106
                                 + 0.0195390881389687 w_k1_AXP_s107
                                 + 0.0294455708098187 w_k1_AXP_s110
                                 + 0.016777673104116 w_k1_AXP_s111
                                 + 0.0183308122769695 w_k1_AXP_s112
                                 + 0.0120776167474786 w_k1_AXP_s113
                                 + 0.0192959158143809 w_k1_AXP_s114
                                 + 0.019277946369492 w_k1_AXP_s115
                                 + 0.0552671501265711 w_k1_AXP_s116
                                 + 0.0249442697668847 w_k1_AXP_s117
                                 + 0.0128598863625018 w_k1_DIS_s000
                                 + 0.0305841566377978 w_k1_DIS_s001
                                 + 0.0274903812085624 w_k1_DIS_s002
                                 + 0.0254886228092044 w_k1_DIS_s003
                                 + 0.015827727000737 w_k1_DIS_s004
                                 + 0.00962959983606673 w_k1_DIS_s005
                                 + 0.0377800040695838 w_k1_DIS_s006
                                 + 0.045180363009015 w_k1_DIS_s007
                                 + 0.0341489107769065 w_k1_DIS_s010
                                 + 0.0474050365889213 w_k1_DIS_s011
                                 + 0.0873442557627722 w_k1_DIS_s012
                                 + 0.0643160277134062 w_k1_DIS_s013
                                 + 0.0885325549892173 w_k1_DIS_s014
                                 + 0.043754560010953 w_k1_DIS_s015
                                 + 0.0380631936177168 w_k1_DIS_s016
                                 + 0.0622238789782703 w_k1_DIS_s017
                                 + 0.0127461737336347 w_k1_DIS_s100
                                 + 0.0235369698686301 w_k1_DIS_s101
                                 + 0.0306588786534401 w_k1_DIS_s102
                                 + 0.0113911921859973 w_k1_DIS_s103
                                 + 0.0150387821359075 w_k1_DIS_s104
                                 + 0.0153229965608571 w_k1_DIS_s105
                                 + 0.0275708503724812 w_k1_DIS_s106
                                 + 0.0209462257432362 w_k1_DIS_s107
                                 + 0.0320687981146892 w_k1_DIS_s110
                                 + 0.0165368371865859 w_k1_DIS_s111
                                 + 0.0204235787500766 w_k1_DIS_s112
                                 + 0.0123357007980229 w_k1_DIS_s113
                                 + 0.0199679136167499 w_k1_DIS_s114
                                 + 0.0208890959759358 w_k1_DIS_s115
                                 + 0.0573447021043932 w_k1_DIS_s116
                                 + 0.0264092642619146 w_k1_DIS_s117
                                 + 0.0124470140478024 w_k1_FDX_s000
                                 + 0.0283128838513225 w_k1_FDX_s001
                                 + 0.0271253462642839 w_k1_FDX_s002
                                 + 0.0259272671410812 w_k1_FDX_s003
                                 + 0.015857219438338 w_k1_FDX_s004
                                 + 0.0102156869379768 w_k1_FDX_s005
                                 + 0.0372099700161789 w_k1_FDX_s006
                                 + 0.0419378199756551 w_k1_FDX_s007
                                 + 0.0346022074164849 w_k1_FDX_s010
                                 + 0.049390295779196 w_k1_FDX_s011
                                 + 0.0880520208592301 w_k1_FDX_s012
                                 + 0.0688056967647798 w_k1_FDX_s013
                                 + 0.0907908219824218 w_k1_FDX_s014
                                 + 0.0404447605140336 w_k1_FDX_s015
                                 + 0.0385605506335864 w_k1_FDX_s016
                                 + 0.0612848154921903 w_k1_FDX_s017
                                 + 0.0120468287945626 w_k1_FDX_s100
                                 + 0.0230949982672451 w_k1_FDX_s101
                                 + 0.0299445534702949 w_k1_FDX_s102
                                 + 0.0111947370728995 w_k1_FDX_s103
                                 + 0.0155873535956962 w_k1_FDX_s104
                                 + 0.0152285579229058 w_k1_FDX_s105
                                 + 0.0269048443453036 w_k1_FDX_s106
                                 + 0.0188726121148944 w_k1_FDX_s107
                                 + 0.0300139603948206 w_k1_FDX_s110
                                 + 0.0157520832973133 w_k1_FDX_s111
                                 + 0.0205015525900664 w_k1_FDX_s112
                                 + 0.011458195164688 w_k1_FDX_s113
                                 + 0.0193539160898241 w_k1_FDX_s114
                                 + 0.0201302806867444 w_k1_FDX_s115
                                 + 0.0532159357442022 w_k1_FDX_s116
                                 + 0.0253504615986936 w_k1_FDX_s117
                                 + 0.013331372590863 w_k1_IBM_s000
                                 + 0.0293982854229285 w_k1_IBM_s001
                                 + 0.0265605526727579 w_k1_IBM_s002
                                 + 0.0248435076755702 w_k1_IBM_s003
                                 + 0.0163308328609616 w_k1_IBM_s004
                                 + 0.0103413894349553 w_k1_IBM_s005
                                 + 0.0373362533358517 w_k1_IBM_s006
                                 + 0.0427894589144494 w_k1_IBM_s007
                                 + 0.0349405641156836 w_k1_IBM_s010
                                 + 0.0487217979205005 w_k1_IBM_s011
                                 + 0.0874593425473881 w_k1_IBM_s012
                                 + 0.0725907615286683 w_k1_IBM_s013
                                 + 0.0922383403450136 w_k1_IBM_s014
                                 + 0.0417816121655775 w_k1_IBM_s015
                                 + 0.0386796802958053 w_k1_IBM_s016
                                 + 0.0590489291054226 w_k1_IBM_s017
                                 + 0.0123983198185519 w_k1_IBM_s100
                                 + 0.0233189175708984 w_k1_IBM_s101
                                 + 0.029177914105582 w_k1_IBM_s102
                                 + 0.011640113720452 w_k1_IBM_s103
                                 + 0.0163157330955772 w_k1_IBM_s104
                                 + 0.0155104090393748 w_k1_IBM_s105
                                 + 0.026881748896723 w_k1_IBM_s106
                                 + 0.019102200206993 w_k1_IBM_s107
                                 + 0.0293064212731357 w_k1_IBM_s110
                                 + 0.0163248618852479 w_k1_IBM_s111
                                 + 0.020593944507876 w_k1_IBM_s112
                                 + 0.0112049477886551 w_k1_IBM_s113
                                 + 0.0182322912694968 w_k1_IBM_s114
                                 + 0.0207425709640441 w_k1_IBM_s115
                                 + 0.0560386863496632 w_k1_IBM_s116
                                 + 0.0267144876601706 w_k1_IBM_s117
                                 + 0.0113188447591345 w_k1_INTC_s000
                                 + 0.0284835625757269 w_k1_INTC_s001
                                 + 0.0243580983054347 w_k1_INTC_s002
                                 + 0.026835617450016 w_k1_INTC_s003
                                 + 0.0174270229533065 w_k1_INTC_s004
                                 + 0.00987730185345415 w_k1_INTC_s005
                                 + 0.0452060284319403 w_k1_INTC_s006
                                 + 0.0399974461908095 w_k1_INTC_s007
                                 + 0.0331554579222024 w_k1_INTC_s010
                                 + 0.0513492536265375 w_k1_INTC_s011
                                 + 0.0887397687036164 w_k1_INTC_s012
                                 + 0.0694141811214965 w_k1_INTC_s013
                                 + 0.103797620815088 w_k1_INTC_s014
                                 + 0.0416107233718961 w_k1_INTC_s015
                                 + 0.0392782688323716 w_k1_INTC_s016
                                 + 0.0550382784821842 w_k1_INTC_s017
                                 + 0.0103478296320064 w_k1_INTC_s100
                                 + 0.0248759887503456 w_k1_INTC_s101
                                 + 0.035744503779245 w_k1_INTC_s102
                                 + 0.0137443598850868 w_k1_INTC_s103
                                 + 0.016782213527952 w_k1_INTC_s104
                                 + 0.0174706495070674 w_k1_INTC_s105
                                 + 0.0246084018460474 w_k1_INTC_s106
                                 + 0.0167693813676791 w_k1_INTC_s107
                                 + 0.0304724688641454 w_k1_INTC_s110
                                 + 0.0182363273787545 w_k1_INTC_s111
                                 + 0.0197207382028411 w_k1_INTC_s112
                                 + 0.0110841123319413 w_k1_INTC_s113
                                 + 0.0158775654502346 w_k1_INTC_s114
                                 + 0.0188734484228541 w_k1_INTC_s115
                                 + 0.0512797497110638 w_k1_INTC_s116
                                 + 0.0230432936260035 w_k1_INTC_s117
                                 + 0.0116006576672943 w_k1_KO_s000
                                 + 0.030669372343604 w_k1_KO_s001
                                 + 0.0260488987680944 w_k1_KO_s002
                                 + 0.0240062205656087 w_k1_KO_s003
                                 + 0.0159367941234277 w_k1_KO_s004
                                 + 0.0100597116514644 w_k1_KO_s005
                                 + 0.0380376901054788 w_k1_KO_s006
                                 + 0.0392536253904535 w_k1_KO_s007
                                 + 0.0327194931997558 w_k1_KO_s010
                                 + 0.045241311651709 w_k1_KO_s011
                                 + 0.0823549906405821 w_k1_KO_s012
                                 + 0.0682997236849748 w_k1_KO_s013
                                 + 0.0950198968284293 w_k1_KO_s014
                                 + 0.0438972999767732 w_k1_KO_s015
                                 + 0.0367372572117351 w_k1_KO_s016
                                 + 0.0590775016612127 w_k1_KO_s017
                                 + 0.0112663259301227 w_k1_KO_s100
                                 + 0.0235528532075204 w_k1_KO_s101
                                 + 0.0276495808800011 w_k1_KO_s102
                                 + 0.0129682970276099 w_k1_KO_s103
                                 + 0.0156653936495723 w_k1_KO_s104
                                 + 0.0179255608779471 w_k1_KO_s105
                                 + 0.0284212051400589 w_k1_KO_s106
                                 + 0.0181022296231517 w_k1_KO_s107
                                 + 0.0311497540020286 w_k1_KO_s110
                                 + 0.0158475638098523 w_k1_KO_s111
                                 + 0.0205602003923838 w_k1_KO_s112
                                 + 0.0106403739782058 w_k1_KO_s113
                                 + 0.0186935748582043 w_k1_KO_s114
                                 + 0.0222568513129465 w_k1_KO_s115
                                 + 0.0605838209643918 w_k1_KO_s116
                                 + 0.0263771617222043 w_k1_KO_s117
                                 + 0.0147987413407104 w_k1_MAT_s000
                                 + 0.0293918175818823 w_k1_MAT_s001
                                 + 0.0307282200519785 w_k1_MAT_s002
                                 + 0.0237552973223739 w_k1_MAT_s003
                                 + 0.0114452532084612 w_k1_MAT_s004
                                 + 0.0125619379790897 w_k1_MAT_s005
                                 + 0.0316855173212631 w_k1_MAT_s006
                                 + 0.0468522641887683 w_k1_MAT_s007
                                 + 0.0353835244011417 w_k1_MAT_s010
                                 + 0.0512678838626744 w_k1_MAT_s011
                                 + 0.0989105483160007 w_k1_MAT_s012
                                 + 0.077729350382532 w_k1_MAT_s013
                                 + 0.0983036054543237 w_k1_MAT_s014
                                 + 0.0424754392945119 w_k1_MAT_s015
                                 + 0.0411649083755369 w_k1_MAT_s016
                                 + 0.0748217291386166 w_k1_MAT_s017
                                 + 0.0126472911217046 w_k1_MAT_s100
                                 + 0.0243679896023585 w_k1_MAT_s101
                                 + 0.0247441998226137 w_k1_MAT_s102
                                 + 0.011112593852911 w_k1_MAT_s103
                                 + 0.0176779732255152 w_k1_MAT_s104
                                 + 0.014964535938316 w_k1_MAT_s105
                                 + 0.0253970619622328 w_k1_MAT_s106
                                 + 0.0173188541694382 w_k1_MAT_s107
                                 + 0.0285193106445001 w_k1_MAT_s110
                                 + 0.0126569513535278 w_k1_MAT_s111
                                 + 0.0241012981597493 w_k1_MAT_s112
                                 + 0.0112050403920351 w_k1_MAT_s113
                                 + 0.0187705887209174 w_k1_MAT_s114
                                 + 0.0186721336835941 w_k1_MAT_s115
                                 + 0.0538502440742775 w_k1_MAT_s116
                                 + 0.0323536878559545 w_k1_MAT_s117
                                 + 0.0109681233981803 w_k1_MSFT_s000
                                 + 0.0308492171695495 w_k1_MSFT_s001
                                 + 0.0264284075922212 w_k1_MSFT_s002
                                 + 0.0250588207896101 w_k1_MSFT_s003
                                 + 0.0167573436949912 w_k1_MSFT_s004
                                 + 0.00967334361328979 w_k1_MSFT_s005
                                 + 0.038885072723537 w_k1_MSFT_s006
                                 + 0.0416234354167137 w_k1_MSFT_s007
                                 + 0.0352353554482638 w_k1_MSFT_s010
                                 + 0.0491480255372442 w_k1_MSFT_s011
                                 + 0.0953794101529178 w_k1_MSFT_s012
                                 + 0.0696487779390376 w_k1_MSFT_s013
                                 + 0.0972220759888145 w_k1_MSFT_s014
                                 + 0.0433221470771679 w_k1_MSFT_s015
                                 + 0.0388366649665279 w_k1_MSFT_s016
                                 + 0.0600298607518441 w_k1_MSFT_s017
                                 + 0.0120252636974368 w_k1_MSFT_s100
                                 + 0.0256868873590153 w_k1_MSFT_s101
                                 + 0.0310576931414536 w_k1_MSFT_s102
                                 + 0.0116770620023402 w_k1_MSFT_s103
                                 + 0.0166198425666612 w_k1_MSFT_s104
                                 + 0.0155708508911189 w_k1_MSFT_s105
                                 + 0.0260572842325911 w_k1_MSFT_s106
                                 + 0.0177960033206936 w_k1_MSFT_s107
                                 + 0.0294626505274315 w_k1_MSFT_s110
                                 + 0.0160055216519929 w_k1_MSFT_s111
                                 + 0.0212265360464015 w_k1_MSFT_s112
                                 + 0.0115116633548398 w_k1_MSFT_s113
                                 + 0.0174311083307468 w_k1_MSFT_s114
                                 + 0.0201541288390623 w_k1_MSFT_s115
                                 + 0.0558516684901004 w_k1_MSFT_s116
                                 + 0.0271811898689253 w_k1_MSFT_s117
                                 + 0.0123368298731713 w_k1_PG_s000
                                 + 0.0305100002963749 w_k1_PG_s001
                                 + 0.0269886618348571 w_k1_PG_s002
                                 + 0.0254765324194494 w_k1_PG_s003
                                 + 0.0162061543811679 w_k1_PG_s004
                                 + 0.010144402075866 w_k1_PG_s005
                                 + 0.0394669287020416 w_k1_PG_s006
                                 + 0.0392332572325592 w_k1_PG_s007
                                 + 0.0327024106178098 w_k1_PG_s010
                                 + 0.0464746170779214 w_k1_PG_s011
                                 + 0.0844194518271534 w_k1_PG_s012
                                 + 0.0663570112519926 w_k1_PG_s013
                                 + 0.0961239747559431 w_k1_PG_s014
                                 + 0.039300236899464 w_k1_PG_s015
                                 + 0.034435257086782 w_k1_PG_s016
                                 + 0.0586545446132496 w_k1_PG_s017
                                 + 0.010944081940483 w_k1_PG_s100
                                 + 0.0233741989191328 w_k1_PG_s101
                                 + 0.0305951199770312 w_k1_PG_s102
                                 + 0.0120282476528978 w_k1_PG_s103
                                 + 0.0156149616788032 w_k1_PG_s104
                                 + 0.0137207732584483 w_k1_PG_s105
                                 + 0.0268277592703998 w_k1_PG_s106
                                 + 0.0195898832459339 w_k1_PG_s107
                                 + 0.0287983659865854 w_k1_PG_s110
                                 + 0.0155731381600536 w_k1_PG_s111
                                 + 0.0185866090597336 w_k1_PG_s112
                                 + 0.0117666242418786 w_k1_PG_s113
                                 + 0.018617784793466 w_k1_PG_s114
                                 + 0.0219665487283788 w_k1_PG_s115
                                 + 0.0555342370404982 w_k1_PG_s116
                                 + 0.0249034514352204 w_k1_PG_s117 >= 1.18
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
