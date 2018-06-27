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
 w_k1_ADM_s0_return_constraint:  1.18185 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.918435 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.05884 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.972708 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.928297 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.06601 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.816297 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.901689 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.641064 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.18543 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.98497 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.08922 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.915769 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.720153 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.18267 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.14247 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.808251 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.62479 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.04036 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.13351 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.20125 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.06603 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.884148 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.93865 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.963122 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.736255 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.941633 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.22197 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.10693 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.45499 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.22288 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.06498 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.22376 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.827685 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.1393 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.973831 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.06933 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.900194 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.2582 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.922222 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.05623 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.973747 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.770748 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.17522 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.771925 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.07537 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.63352 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.57668 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.01613 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.28667 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.817183 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.59871 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.19581 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.04256 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.659658 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.73347 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.834447 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.00618 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.36744 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.33596 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.956991 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.22132 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.12813 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.770863 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.819193 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.21055 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.61956 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.59863 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.37292 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.870159 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.32401 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.70984 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.960695 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.665095 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.13019 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.725075 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.32685 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.8961 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.12215 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.929869 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.734343 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.26823 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.750933 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.00053 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.504345 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.78725 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.16788 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.07808 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.762275 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.518484 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.984041 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.18828 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.594153 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.92189 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.795484 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.1509 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.60559 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.27297 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.939865 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.15179 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.09062 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.643739 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.827538 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.20396 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.19962 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.73275 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.56362 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.80232 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.42048 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.565909 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.952106 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.653235 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.18523 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.954497 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18349 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.924563 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.03296 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.06365 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.854434 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.14491 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.770628 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.987056 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.647871 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.39262 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.20207 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.10697 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.882237 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.634704 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.95358 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.02969 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.693683 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.55538 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.884068 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.00569 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.24985 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.26206 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.84812 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.03393 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.05415 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.695193 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.911251 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.35422 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.16893 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.41667 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.35897 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.861915 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.03929 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.685029 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.0355 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.844382 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.992182 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.86108 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.22238 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.915738 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.22033 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.729654 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.824111 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.21209 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.09756 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.01044 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.637415 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.41515 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.794903 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.36731 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.709893 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.95292 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.53714 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.829131 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.892266 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.2488 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.842581 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.2128 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.3518 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.96643 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.34534 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.39201 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.857196 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.940294 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.630524 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.678677 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.3582 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.2908 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.91514 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.03194 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 2.00129 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.804402 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.720765 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.556372 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.44699 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.08776 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.17537 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.94536 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.07571 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.996272 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.865664 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.16682 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.780817 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.969541 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.640726 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.29871 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.07505 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.12408 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.859843 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.638265 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.04817 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.997221 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.737821 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.40703 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.841869 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.967162 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.12374 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.18818 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.00356 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.03306 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.95466 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.857224 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.9694 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.23017 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.19517 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.28547 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.16839 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.840459 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.1941 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.734051 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.01348 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.866323 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.963453 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.964308 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.26182 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.91728 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.16116 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.832333 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.86641 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.29635 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.861495 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.959356 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.602643 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.36879 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.970298 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.11084 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.75528 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.74972 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.13492 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.999434 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.762705 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.4643 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.78839 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.18371 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.35225 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.996529 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.970681 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.24605 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.928256 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.770662 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.658242 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.856934 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.05846 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.45758 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.00246 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.84315 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.42421 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.666331 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.743988 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.611987 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.20756 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.94427 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.20179 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.810754 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.4703 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.785351 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.971326 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.27457 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.00984 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.02086 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.540737 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.31057 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.72092 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.18565 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.541426 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.874428 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.94922 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.823885 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.760523 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 2.01863 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.738686 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.06823 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.96399 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.889435 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.11164 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.46472 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.02905 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.609288 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.450152 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.738852 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.61868 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.30635 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.26436 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.20102 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 2.23764 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.876584 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.749786 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.621746 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.72354 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.878936 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18773 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.914433 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.09317 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.05128 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.875219 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.10178 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.780367 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.98734 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.56703 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.30651 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.0879 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.968834 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.805527 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.606311 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.875823 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01225 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.707019 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.37025 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.818912 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.955914 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.1694 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.06524 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.925173 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.08099 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.985986 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.738343 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.908839 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.26756 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.985998 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.4143 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.33764 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.974723 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.23158 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.82819 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.06337 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.89747 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.14677 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.993576 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13154 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.94317 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.051 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.96641 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.889221 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10648 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.879764 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.99828 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.75094 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.39627 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.09468 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.09837 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.884909 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.829549 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.968522 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.969065 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.84584 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.28042 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.889081 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.05429 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.25748 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.08128 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.947748 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.07757 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.985677 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.857915 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.868048 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.03088 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.11662 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.18729 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.13796 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.900299 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.19846 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.779137 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.891964 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.88009 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.06623 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.855004 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.27000107407691 z_k1_s0000
                                 - 0.250000807400222 z_k1_s0001
                                 - 0.315002139426004 z_k1_s0002
                                 - 0.768001518835891 z_k1_s0003
                                 - 0.194000752047789 z_k1_s0004
                                 - 0.222000646773564 z_k1_s0005
                                 - 0.249000754181255 z_k1_s0006
                                 - 0.39100192965336 z_k1_s0007
                                 - 0.238000498447282 z_k1_s0010
                                 - 0.60400200770111 z_k1_s0011
                                 - 0.394000087127416 z_k1_s0012
                                 - 0.279999763760084 z_k1_s0013
                                 - 0.365000740921211 z_k1_s0014
                                 - 0.153000360387607 z_k1_s0015
                                 - 0.426000619529536 z_k1_s0016
                                 - 0.33699992544375 z_k1_s0017
                                 - 0.463000169357829 z_k1_s0020
                                 - 0.157999894306819 z_k1_s0021
                                 - 0.260999669301179 z_k1_s0022
                                 - 0.24199950437509 z_k1_s0023
                                 - 0.103999962727265 z_k1_s0024
                                 - 0.16999993580226 z_k1_s0025
                                 - 0.109999898425008 z_k1_s0026
                                 - 0.193000511460057 z_k1_s0027
                                 - 0.199999606144818 z_k1_s0030
                                 - 0.178999759350467 z_k1_s0031
                                 - 0.650999180519782 z_k1_s0032
                                 - 0.180999773549891 z_k1_s0033
                                 - 0.551999082247478 z_k1_s0034
                                 - 0.673998134614737 z_k1_s0035
                                 - 0.368998992198555 z_k1_s0036
                                 - 0.216999726839927 z_k1_s0037
                                 - 0.717001010236876 z_k1_s0040
                                 - 0.415001915161087 z_k1_s0041
                                 - 0.46500011023605 z_k1_s0042
                                 - 0.285000623952408 z_k1_s0043
                                 - 0.408000619047027 z_k1_s0044
                                 - 0.424002575876849 z_k1_s0045
                                 - 0.581001977223901 z_k1_s0046
                                 - 0.225000677857591 z_k1_s0047
                                 - 0.52300275768357 z_k1_s0050
                                 - 0.658002166030892 z_k1_s0051
                                 - 0.199000472641124 z_k1_s0052
                                 - 0.419999805973389 z_k1_s0053
                                 - 0.552001862141525 z_k1_s0054
                                 - 0.158000473938918 z_k1_s0055
                                 - 0.495000300612761 z_k1_s0056
                                 - 0.70000214662823 z_k1_s0057
                                 - 0.369999310839036 z_k1_s0060
                                 - 0.189000562165905 z_k1_s0061
                                 - 0.661001813187271 z_k1_s0062
                                 - 0.242000507826738 z_k1_s0063
                                 - 0.145000439979154 z_k1_s0064
                                 - 0.647002087783144 z_k1_s0065
                                 - 0.322000909112167 z_k1_s0066
                                 - 0.708002533333617 z_k1_s0067
                                 - 0.197999967377564 z_k1_s0070
                                 - 0.0639999712250829 z_k1_s0071
                                 - 0.240999299444838 z_k1_s0072
                                 - 0.233999687991844 z_k1_s0073
                                 - 0.136999872603736 z_k1_s0074
                                 - 0.205999738180846 z_k1_s0075
                                 - 0.166999539515701 z_k1_s0076
                                 - 0.257000345050424 z_k1_s0077
                                 - 0.0939999871228542 z_k1_s0100
                                 - 0.108000031162442 z_k1_s0101
                                 - 0.225000514921883 z_k1_s0102
                                 - 0.0989999835655584 z_k1_s0103
                                 - 0.287999633099714 z_k1_s0104
                                 - 0.260999625309104 z_k1_s0105
                                 - 0.139000090107231 z_k1_s0106
                                 - 0.136000389241694 z_k1_s0107
                                 - 0.1080000316885 z_k1_s0110
                                 - 0.101000091517051 z_k1_s0111
                                 - 0.131000303480612 z_k1_s0112
                                 - 0.162999453521272 z_k1_s0113
                                 - 0.240999903746487 z_k1_s0114
                                 - 0.179000289542155 z_k1_s0115
                                 - 0.121000064692684 z_k1_s0116
                                 - 0.217000541215992 z_k1_s0117
                                 - 0.159000437772738 z_k1_s0120
                                 - 0.512002508975696 z_k1_s0121
                                 - 0.351000338236874 z_k1_s0122
                                 - 0.340000517339742 z_k1_s0123
                                 - 0.48500061556972 z_k1_s0124
                                 - 0.528999205557085 z_k1_s0125
                                 - 0.431999082443141 z_k1_s0126
                                 - 0.65999965784547 z_k1_s0127
                                 - 0.223000507210445 z_k1_s0130
                                 - 0.0739999755764677 z_k1_s0131
                                 - 0.0970000266340467 z_k1_s0132
                                 - 0.0850000608649043 z_k1_s0133
                                 - 0.308999748555719 z_k1_s0134
                                 - 0.231000111056361 z_k1_s0135
                                 - 0.0920000642302486 z_k1_s0136
                                 - 0.28900073894093 z_k1_s0137
                                 - 0.540001444278258 z_k1_s0140
                                 - 0.409999410358356 z_k1_s0141
                                 - 0.403999466577531 z_k1_s0142
                                 - 0.574001125702927 z_k1_s0143
                                 - 0.552999696469019 z_k1_s0144
                                 - 0.399001139427869 z_k1_s0145
                                 - 0.242000008894669 z_k1_s0146
                                 - 0.130000082718673 z_k1_s0147
                                 - 0.369000073958727 z_k1_s0150
                                 - 0.167000362598763 z_k1_s0151
                                 - 0.0849999022752004 z_k1_s0152
                                 - 0.349999879603386 z_k1_s0153
                                 - 0.103999936830757 z_k1_s0154
                                 - 0.0749999513721849 z_k1_s0155
                                 - 0.113999887733773 z_k1_s0156
                                 - 0.333999127199682 z_k1_s0157
                                 - 0.130999691988203 z_k1_s0160
                                 - 0.139000076720344 z_k1_s0161
                                 - 0.109000201974959 z_k1_s0162
                                 - 0.112999946340989 z_k1_s0163
                                 - 0.172999695831759 z_k1_s0164
                                 - 0.0509999662671709 z_k1_s0165
                                 - 0.0670000189313872 z_k1_s0166
                                 - 0.112999946340989 z_k1_s0167
                                 - 0.106000066772667 z_k1_s0170
                                 - 0.222000514682428 z_k1_s0171
                                 - 0.0830000004974586 z_k1_s0172
                                 - 0.06499999409078 z_k1_s0173
                                 - 0.226999790350852 z_k1_s0174
                                 - 0.250000379426131 z_k1_s0175
                                 - 0.154999764724138 z_k1_s0176
                                 - 0.19899992560715 z_k1_s0177
                                 - 0.169000136078025 z_k1_s1000
                                 - 0.171000164746408 z_k1_s1001
                                 - 0.383000631394621 z_k1_s1002
                                 - 0.240000010605455 z_k1_s1003
                                 - 0.576001397293276 z_k1_s1004
                                 - 0.287000398512408 z_k1_s1005
                                 - 0.711999345542757 z_k1_s1006
                                 - 0.319998870940453 z_k1_s1007
                                 - 0.597000965362209 z_k1_s1010
                                 - 0.809999487962142 z_k1_s1011
                                 - 0.495998119910575 z_k1_s1012
                                 - 0.476999123000991 z_k1_s1013
                                 - 0.512998812510951 z_k1_s1014
                                 - 0.202999663867439 z_k1_s1015
                                 - 0.403999297028858 z_k1_s1016
                                 - 0.815998716881797 z_k1_s1017
                                 - 0.370999702453344 z_k1_s1020
                                 - 1.50400089541114 z_k1_s1021
                                 - 1.45100119536036 z_k1_s1022
                                 - 0.394999441953838 z_k1_s1023
                                 - 0.59499927345565 z_k1_s1024
                                 - 0.673996638909318 z_k1_s1025
                                 - 0.460999476454332 z_k1_s1026
                                 - 0.555999471505106 z_k1_s1027
                                 - 0.415001201645775 z_k1_s1030
                                 - 0.602000954712778 z_k1_s1031
                                 - 0.824002721803674 z_k1_s1032
                                 - 0.898001697495937 z_k1_s1033
                                 - 1.06900423593425 z_k1_s1034
                                 - 0.299001067528508 z_k1_s1035
                                 - 0.258000824020971 z_k1_s1036
                                 - 0.475001404534406 z_k1_s1037
                                 - 0.270999479136513 z_k1_s1040
                                 - 0.435998898475383 z_k1_s1041
                                 - 0.172999726726909 z_k1_s1042
                                 - 0.403998979321226 z_k1_s1043
                                 - 0.839997877796609 z_k1_s1044
                                 - 0.200999655986796 z_k1_s1045
                                 - 0.358000733534626 z_k1_s1046
                                 - 0.592997682826713 z_k1_s1047
                                 - 0.697999436516443 z_k1_s1050
                                 - 0.563000332204898 z_k1_s1051
                                 - 0.148999981381352 z_k1_s1052
                                 - 0.377000627597434 z_k1_s1053
                                 - 0.174000159691061 z_k1_s1054
                                 - 0.467000030471797 z_k1_s1055
                                 - 0.173000197614694 z_k1_s1056
                                 - 0.215000012822775 z_k1_s1057
                                 - 0.0860000130983826 z_k1_s1060
                                 - 0.129000019647574 z_k1_s1061
                                 - 0.217000141297057 z_k1_s1062
                                 - 0.289999311913315 z_k1_s1063
                                 - 0.138000645027497 z_k1_s1064
                                 - 0.1099999467119 z_k1_s1065
                                 - 0.238999966359474 z_k1_s1066
                                 - 0.160000470089914 z_k1_s1067
                                 - 0.197000068722064 z_k1_s1070
                                 - 0.371001479120019 z_k1_s1071
                                 - 0.353000496347005 z_k1_s1072
                                 - 0.264000256957139 z_k1_s1073
                                 - 0.19600003395136 z_k1_s1074
                                 - 0.33000041042151 z_k1_s1075
                                 - 0.907001557399235 z_k1_s1076
                                 - 0.95100023210743 z_k1_s1077
                                 - 0.757000899424809 z_k1_s1100
                                 - 0.575000902971695 z_k1_s1101
                                 - 0.443000548950095 z_k1_s1102
                                 - 0.728002345816821 z_k1_s1103
                                 - 1.20600352947061 z_k1_s1104
                                 - 0.468000665166398 z_k1_s1105
                                 - 1.08000289654036 z_k1_s1106
                                 - 0.643000888679428 z_k1_s1107
                                 - 0.456999022140834 z_k1_s1110
                                 - 2.14299806476605 z_k1_s1111
                                 - 0.480998618214667 z_k1_s1112
                                 - 0.44599915085712 z_k1_s1113
                                 - 0.96599558944519 z_k1_s1114
                                 - 0.495998873509673 z_k1_s1115
                                 - 1.0149967393415 z_k1_s1116
                                 - 0.765996698834979 z_k1_s1117
                                 - 0.2860003678091 z_k1_s1120
                                 - 0.105999939132783 z_k1_s1121
                                 - 0.368999855921657 z_k1_s1122
                                 - 0.131000044746151 z_k1_s1123
                                 - 0.10199989274665 z_k1_s1124
                                 - 0.343999566008326 z_k1_s1125
                                 - 0.227999326610248 z_k1_s1126
                                 - 0.276999157640526 z_k1_s1127
                                 - 0.306999842445715 z_k1_s1130
                                 - 0.288999896930317 z_k1_s1131
                                 - 0.210999922563657 z_k1_s1132
                                 - 0.589001303986223 z_k1_s1133
                                 - 0.214000018716191 z_k1_s1134
                                 - 0.639999465880383 z_k1_s1135
                                 - 0.560998722829778 z_k1_s1136
                                 - 0.346000776728745 z_k1_s1137
                                 - 0.965002936173303 z_k1_s1140
                                 - 0.617000191100779 z_k1_s1141
                                 - 1.44900231738267 z_k1_s1142
                                 - 0.453000463322424 z_k1_s1143
                                 - 0.448000271643733 z_k1_s1144
                                 - 0.48400001190963 z_k1_s1145
                                 - 1.24500231757531 z_k1_s1146
                                 - 0.646001050557085 z_k1_s1147
                                 - 0.171000556480685 z_k1_s1150
                                 - 0.0970000866186851 z_k1_s1151
                                 - 0.275999668283632 z_k1_s1152
                                 - 0.0950001398224674 z_k1_s1153
                                 - 0.105000026203677 z_k1_s1154
                                 - 0.331000872181746 z_k1_s1155
                                 - 0.10300007940746 z_k1_s1156
                                 - 0.346000092153075 z_k1_s1157
                                 - 0.773001971070936 z_k1_s1160
                                 - 1.05799938270803 z_k1_s1161
                                 - 0.496999726538646 z_k1_s1162
                                 - 0.689002427430561 z_k1_s1163
                                 - 1.11099777743363 z_k1_s1164
                                 - 0.418000086400681 z_k1_s1165
                                 - 0.264000146526737 z_k1_s1166
                                 - 1.01399906708693 z_k1_s1167
                                 - 0.379000505565656 z_k1_s1170
                                 - 0.265999882117116 z_k1_s1171
                                 - 0.495998267478064 z_k1_s1172
                                 - 0.325000869529617 z_k1_s1173
                                 - 0.423998752763346 z_k1_s1174
                                 - 0.388999149721359 z_k1_s1175
                                 - 0.299999910643334 z_k1_s1176
                                 - 0.320000097952758 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.876697 w_k1_ADM_s000
                                 + 0.887577 w_k1_AXP_s000
                                 + 0.727776 w_k1_DIS_s000
                                 + 0.820384 w_k1_FDX_s000
                                 + 1.30412 w_k1_IBM_s000
                                 + 0.802375 w_k1_INTC_s000
                                 + 1.03295 w_k1_KO_s000 + 1.07325 w_k1_MAT_s000
                                 + 0.792755 w_k1_MSFT_s000
                                 + 0.863661 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.00939 w_k1_ADM_s000
                                 + 0.865512 w_k1_AXP_s000
                                 + 0.864251 w_k1_DIS_s000
                                 + 0.990392 w_k1_FDX_s000
                                 + 0.625681 w_k1_IBM_s000
                                 + 0.85518 w_k1_INTC_s000
                                 + 0.785891 w_k1_KO_s000
                                 + 0.469068 w_k1_MAT_s000
                                 + 0.866678 w_k1_MSFT_s000
                                 + 0.991077 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.85163 w_k1_ADM_s000
                                 + 0.925347 w_k1_AXP_s000
                                 + 1.14413 w_k1_DIS_s000
                                 + 0.922304 w_k1_FDX_s000
                                 + 1.47163 w_k1_IBM_s000
                                 + 1.0575 w_k1_INTC_s000 + 1.234 w_k1_KO_s000
                                 + 1.19276 w_k1_MAT_s000
                                 + 1.06674 w_k1_MSFT_s000
                                 + 1.16783 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.03933 w_k1_ADM_s000
                                 + 0.923738 w_k1_AXP_s000
                                 + 1.04895 w_k1_DIS_s000
                                 + 1.07029 w_k1_FDX_s000
                                 + 0.834252 w_k1_IBM_s000
                                 + 0.887791 w_k1_INTC_s000
                                 + 0.989643 w_k1_KO_s000
                                 + 0.751091 w_k1_MAT_s000
                                 + 0.931925 w_k1_MSFT_s000
                                 + 0.97745 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.05194 w_k1_ADM_s000
                                 + 1.71871 w_k1_AXP_s000
                                 + 1.53481 w_k1_DIS_s000
                                 + 1.34079 w_k1_FDX_s000
                                 + 1.53471 w_k1_IBM_s000
                                 + 1.52835 w_k1_INTC_s000
                                 + 1.33269 w_k1_KO_s000 + 1.37723 w_k1_MAT_s000
                                 + 1.39115 w_k1_MSFT_s000
                                 + 1.16828 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.28722 w_k1_ADM_s000
                                 + 1.34414 w_k1_AXP_s000
                                 + 1.64104 w_k1_DIS_s000
                                 + 1.32346 w_k1_FDX_s000
                                 + 1.11891 w_k1_IBM_s000
                                 + 1.16574 w_k1_INTC_s000
                                 + 1.19671 w_k1_KO_s000
                                 + 0.846652 w_k1_MAT_s000
                                 + 1.10985 w_k1_MSFT_s000
                                 + 1.09533 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.575918 w_k1_ADM_s000
                                 + 0.753536 w_k1_AXP_s000
                                 + 0.71855 w_k1_DIS_s000
                                 + 0.731436 w_k1_FDX_s000
                                 + 0.833442 w_k1_IBM_s000
                                 + 0.790487 w_k1_INTC_s000
                                 + 0.835149 w_k1_KO_s000
                                 + 0.726032 w_k1_MAT_s000
                                 + 0.830477 w_k1_MSFT_s000
                                 + 0.858257 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.10792 w_k1_ADM_s000
                                 + 1.1085 w_k1_AXP_s000 + 1.07064 w_k1_DIS_s000
                                 + 1.02947 w_k1_FDX_s000
                                 + 1.14282 w_k1_IBM_s000
                                 + 1.1969 w_k1_INTC_s000 + 1.1779 w_k1_KO_s000
                                 + 0.781014 w_k1_MAT_s000
                                 + 1.02292 w_k1_MSFT_s000
                                 + 1.06819 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.16276 w_k1_ADM_s001
                                 + 1.34204 w_k1_AXP_s001
                                 + 1.19625 w_k1_DIS_s001
                                 + 1.10853 w_k1_FDX_s001
                                 + 1.17341 w_k1_IBM_s001
                                 + 1.04643 w_k1_INTC_s001
                                 + 1.23988 w_k1_KO_s001 + 1.53249 w_k1_MAT_s001
                                 + 1.1412 w_k1_MSFT_s001 + 1.05567 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.86776 w_k1_ADM_s001
                                 + 0.742085 w_k1_AXP_s001
                                 + 0.773395 w_k1_DIS_s001
                                 + 0.826289 w_k1_FDX_s001
                                 + 0.799842 w_k1_IBM_s001
                                 + 1.01349 w_k1_INTC_s001
                                 + 0.84462 w_k1_KO_s001
                                 + 0.667573 w_k1_MAT_s001
                                 + 0.938549 w_k1_MSFT_s001
                                 + 0.920425 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.13893 w_k1_ADM_s001
                                 + 1.18399 w_k1_AXP_s001
                                 + 1.50066 w_k1_DIS_s001
                                 + 1.27853 w_k1_FDX_s001
                                 + 0.968513 w_k1_IBM_s001
                                 + 1.4012 w_k1_INTC_s001 + 1.37991 w_k1_KO_s001
                                 + 1.22962 w_k1_MAT_s001
                                 + 1.25868 w_k1_MSFT_s001
                                 + 1.26399 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.29636 w_k1_ADM_s001
                                 + 1.12997 w_k1_AXP_s001 + 1.1516 w_k1_DIS_s001
                                 + 0.990954 w_k1_FDX_s001
                                 + 1.56494 w_k1_IBM_s001
                                 + 1.00135 w_k1_INTC_s001
                                 + 1.11657 w_k1_KO_s001 + 1.20421 w_k1_MAT_s001
                                 + 0.981938 w_k1_MSFT_s001
                                 + 1.1685 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.971682 w_k1_ADM_s001
                                 + 0.794923 w_k1_AXP_s001
                                 + 0.994214 w_k1_DIS_s001
                                 + 0.972779 w_k1_FDX_s001
                                 + 1.12554 w_k1_IBM_s001
                                 + 0.853863 w_k1_INTC_s001
                                 + 1.06316 w_k1_KO_s001
                                 + 0.938329 w_k1_MAT_s001
                                 + 0.979924 w_k1_MSFT_s001
                                 + 0.965293 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.767928 w_k1_ADM_s001
                                 + 0.56526 w_k1_AXP_s001
                                 + 0.567129 w_k1_DIS_s001
                                 + 0.708692 w_k1_FDX_s001
                                 + 0.680846 w_k1_IBM_s001
                                 + 0.647978 w_k1_INTC_s001
                                 + 0.600623 w_k1_KO_s001
                                 + 0.580897 w_k1_MAT_s001
                                 + 0.673627 w_k1_MSFT_s001
                                 + 0.886556 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.10899 w_k1_ADM_s001
                                 + 0.884132 w_k1_AXP_s001
                                 + 1.16359 w_k1_DIS_s001 + 1.0724 w_k1_FDX_s001
                                 + 0.724109 w_k1_IBM_s001
                                 + 1.14116 w_k1_INTC_s001
                                 + 1.06273 w_k1_KO_s001 + 0.70294 w_k1_MAT_s001
                                 + 1.12608 w_k1_MSFT_s001
                                 + 1.05487 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.0238 w_k1_ADM_s001
                                 + 1.10323 w_k1_AXP_s001
                                 + 0.805186 w_k1_DIS_s001
                                 + 0.940787 w_k1_FDX_s001
                                 + 1.02993 w_k1_IBM_s001
                                 + 0.912602 w_k1_INTC_s001
                                 + 0.861056 w_k1_KO_s001
                                 + 1.18897 w_k1_MAT_s001
                                 + 0.843291 w_k1_MSFT_s001
                                 + 0.979047 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.05941 w_k1_ADM_s002
                                 + 0.751703 w_k1_AXP_s002
                                 + 0.872264 w_k1_DIS_s002
                                 + 0.883743 w_k1_FDX_s002
                                 + 0.921907 w_k1_IBM_s002
                                 + 1.06309 w_k1_INTC_s002
                                 + 0.955694 w_k1_KO_s002
                                 + 0.747605 w_k1_MAT_s002
                                 + 1.01376 w_k1_MSFT_s002
                                 + 0.918943 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.910553 w_k1_ADM_s002
                                 + 0.899102 w_k1_AXP_s002
                                 + 0.848067 w_k1_DIS_s002
                                 + 0.863845 w_k1_FDX_s002
                                 + 1.17667 w_k1_IBM_s002
                                 + 1.01848 w_k1_INTC_s002
                                 + 1.01954 w_k1_KO_s002 + 1.015 w_k1_MAT_s002
                                 + 1.00052 w_k1_MSFT_s002
                                 + 0.95462 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.3754 w_k1_ADM_s002
                                 + 1.57408 w_k1_AXP_s002
                                 + 1.37509 w_k1_DIS_s002
                                 + 1.25369 w_k1_FDX_s002
                                 + 1.43872 w_k1_IBM_s002
                                 + 1.34663 w_k1_INTC_s002
                                 + 1.28602 w_k1_KO_s002 + 1.51392 w_k1_MAT_s002
                                 + 1.2559 w_k1_MSFT_s002 + 1.15445 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.0425 w_k1_ADM_s002
                                 + 1.01522 w_k1_AXP_s002
                                 + 1.44882 w_k1_DIS_s002
                                 + 1.25977 w_k1_FDX_s002
                                 + 0.97767 w_k1_IBM_s002
                                 + 0.95903 w_k1_INTC_s002
                                 + 1.14824 w_k1_KO_s002 + 1.27195 w_k1_MAT_s002
                                 + 1.41935 w_k1_MSFT_s002
                                 + 1.02686 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.925504 w_k1_ADM_s002
                                 + 1.07967 w_k1_AXP_s002
                                 + 0.968074 w_k1_DIS_s002
                                 + 0.923519 w_k1_FDX_s002
                                 + 1.49885 w_k1_IBM_s002
                                 + 0.967816 w_k1_INTC_s002
                                 + 1.13352 w_k1_KO_s002 + 1.35233 w_k1_MAT_s002
                                 + 1.00154 w_k1_MSFT_s002
                                 + 1.12935 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.1353 w_k1_ADM_s002
                                 + 0.957739 w_k1_AXP_s002
                                 + 0.878938 w_k1_DIS_s002
                                 + 1.02306 w_k1_FDX_s002
                                 + 0.787737 w_k1_IBM_s002
                                 + 1.0663 w_k1_INTC_s002
                                 + 0.823764 w_k1_KO_s002
                                 + 0.646141 w_k1_MAT_s002
                                 + 1.07542 w_k1_MSFT_s002
                                 + 0.861645 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.2535 w_k1_ADM_s002
                                 + 1.4941 w_k1_AXP_s002 + 1.11931 w_k1_DIS_s002
                                 + 1.29492 w_k1_FDX_s002 + 1.0384 w_k1_IBM_s002
                                 + 1.35902 w_k1_INTC_s002
                                 + 1.07623 w_k1_KO_s002 + 0.98305 w_k1_MAT_s002
                                 + 1.22593 w_k1_MSFT_s002
                                 + 1.15242 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.928713 w_k1_ADM_s002
                                 + 0.724531 w_k1_AXP_s002
                                 + 0.812879 w_k1_DIS_s002
                                 + 0.912541 w_k1_FDX_s002
                                 + 0.623786 w_k1_IBM_s002
                                 + 1.03563 w_k1_INTC_s002
                                 + 0.874147 w_k1_KO_s002
                                 + 0.576004 w_k1_MAT_s002
                                 + 1.20371 w_k1_MSFT_s002
                                 + 0.836079 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.879498 w_k1_ADM_s003
                                 + 1.24762 w_k1_AXP_s003
                                 + 1.43082 w_k1_DIS_s003
                                 + 1.10929 w_k1_FDX_s003 + 2.091 w_k1_IBM_s003
                                 + 1.02719 w_k1_INTC_s003
                                 + 1.58733 w_k1_KO_s003 + 2.39826 w_k1_MAT_s003
                                 + 1.13271 w_k1_MSFT_s003
                                 + 1.23144 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.02239 w_k1_ADM_s003
                                 + 1.15676 w_k1_AXP_s003 + 1.4118 w_k1_DIS_s003
                                 + 1.18159 w_k1_FDX_s003
                                 + 0.883968 w_k1_IBM_s003
                                 + 1.00037 w_k1_INTC_s003 + 1.0279 w_k1_KO_s003
                                 + 0.897967 w_k1_MAT_s003
                                 + 1.05105 w_k1_MSFT_s003
                                 + 0.926183 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.13226 w_k1_ADM_s003
                                 + 0.771088 w_k1_AXP_s003
                                 + 0.978935 w_k1_DIS_s003
                                 + 1.02182 w_k1_FDX_s003
                                 + 0.740197 w_k1_IBM_s003
                                 + 1.14093 w_k1_INTC_s003
                                 + 0.874429 w_k1_KO_s003
                                 + 0.74131 w_k1_MAT_s003
                                 + 1.19988 w_k1_MSFT_s003
                                 + 0.814349 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.34373 w_k1_ADM_s003
                                 + 1.04467 w_k1_AXP_s003
                                 + 1.15208 w_k1_DIS_s003
                                 + 1.20135 w_k1_FDX_s003
                                 + 1.01937 w_k1_IBM_s003
                                 + 0.964557 w_k1_INTC_s003
                                 + 1.00312 w_k1_KO_s003 + 1.10519 w_k1_MAT_s003
                                 + 0.958709 w_k1_MSFT_s003
                                 + 0.935994 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.842966 w_k1_ADM_s003
                                 + 0.737447 w_k1_AXP_s003
                                 + 0.710776 w_k1_DIS_s003
                                 + 0.754434 w_k1_FDX_s003
                                 + 0.738531 w_k1_IBM_s003
                                 + 0.815598 w_k1_INTC_s003
                                 + 0.79733 w_k1_KO_s003
                                 + 0.675335 w_k1_MAT_s003
                                 + 0.744194 w_k1_MSFT_s003
                                 + 0.727452 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.982464 w_k1_ADM_s003
                                 + 0.724548 w_k1_AXP_s003
                                 + 0.866828 w_k1_DIS_s003
                                 + 0.803995 w_k1_FDX_s003
                                 + 1.32343 w_k1_IBM_s003
                                 + 0.97516 w_k1_INTC_s003
                                 + 1.05228 w_k1_KO_s003 + 1.45046 w_k1_MAT_s003
                                 + 0.854516 w_k1_MSFT_s003
                                 + 0.843519 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.34125 w_k1_ADM_s003
                                 + 1.8362 w_k1_AXP_s003 + 1.63431 w_k1_DIS_s003
                                 + 1.53336 w_k1_FDX_s003
                                 + 0.933354 w_k1_IBM_s003
                                 + 1.46189 w_k1_INTC_s003
                                 + 1.25506 w_k1_KO_s003 + 1.34069 w_k1_MAT_s003
                                 + 1.42875 w_k1_MSFT_s003
                                 + 1.08943 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.19015 w_k1_ADM_s003
                                 + 1.48797 w_k1_AXP_s003
                                 + 1.26536 w_k1_DIS_s003 + 1.2088 w_k1_FDX_s003
                                 + 1.56521 w_k1_IBM_s003
                                 + 0.897021 w_k1_INTC_s003
                                 + 1.21378 w_k1_KO_s003 + 1.61193 w_k1_MAT_s003
                                 + 0.933543 w_k1_MSFT_s003
                                 + 0.92375 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.15383 w_k1_ADM_s004
                                 + 1.03626 w_k1_AXP_s004
                                 + 0.992827 w_k1_DIS_s004
                                 + 0.95148 w_k1_FDX_s004
                                 + 1.22267 w_k1_IBM_s004
                                 + 0.758722 w_k1_INTC_s004
                                 + 0.959033 w_k1_KO_s004
                                 + 1.27737 w_k1_MAT_s004
                                 + 0.911789 w_k1_MSFT_s004
                                 + 0.925123 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.39144 w_k1_ADM_s004
                                 + 1.4693 w_k1_AXP_s004 + 1.40884 w_k1_DIS_s004
                                 + 1.27185 w_k1_FDX_s004
                                 + 1.04827 w_k1_IBM_s004
                                 + 1.15666 w_k1_INTC_s004
                                 + 1.11284 w_k1_KO_s004
                                 + 0.962754 w_k1_MAT_s004
                                 + 1.23169 w_k1_MSFT_s004
                                 + 1.10765 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.88512 w_k1_ADM_s004
                                 + 0.701331 w_k1_AXP_s004
                                 + 0.841249 w_k1_DIS_s004
                                 + 0.843281 w_k1_FDX_s004
                                 + 0.852077 w_k1_IBM_s004
                                 + 0.745208 w_k1_INTC_s004
                                 + 0.906882 w_k1_KO_s004
                                 + 0.59595 w_k1_MAT_s004
                                 + 0.836828 w_k1_MSFT_s004
                                 + 0.851128 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.00047 w_k1_ADM_s004
                                 + 1.33721 w_k1_AXP_s004
                                 + 1.37888 w_k1_DIS_s004
                                 + 1.09533 w_k1_FDX_s004
                                 + 1.67058 w_k1_IBM_s004
                                 + 1.19182 w_k1_INTC_s004
                                 + 1.17725 w_k1_KO_s004 + 1.48537 w_k1_MAT_s004
                                 + 1.06638 w_k1_MSFT_s004
                                 + 0.967501 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.00454 w_k1_ADM_s004
                                 + 0.833974 w_k1_AXP_s004
                                 + 0.717962 w_k1_DIS_s004
                                 + 0.699775 w_k1_FDX_s004
                                 + 1.06861 w_k1_IBM_s004
                                 + 0.702533 w_k1_INTC_s004
                                 + 0.743415 w_k1_KO_s004
                                 + 0.853534 w_k1_MAT_s004
                                 + 0.737269 w_k1_MSFT_s004
                                 + 0.861806 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.895552 w_k1_ADM_s004
                                 + 0.905541 w_k1_AXP_s004
                                 + 1.0114 w_k1_DIS_s004
                                 + 0.971015 w_k1_FDX_s004
                                 + 0.67456 w_k1_IBM_s004
                                 + 0.8099 w_k1_INTC_s004
                                 + 0.712564 w_k1_KO_s004
                                 + 0.668039 w_k1_MAT_s004
                                 + 0.960908 w_k1_MSFT_s004
                                 + 0.782933 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.24164 w_k1_ADM_s004
                                 + 1.10385 w_k1_AXP_s004
                                 + 0.919371 w_k1_DIS_s004
                                 + 1.04103 w_k1_FDX_s004
                                 + 0.898578 w_k1_IBM_s004
                                 + 0.867759 w_k1_INTC_s004
                                 + 0.785441 w_k1_KO_s004
                                 + 0.894379 w_k1_MAT_s004
                                 + 0.858174 w_k1_MSFT_s004
                                 + 0.902318 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.32387 w_k1_ADM_s004
                                 + 1.816 w_k1_AXP_s004 + 1.93171 w_k1_DIS_s004
                                 + 1.53007 w_k1_FDX_s004
                                 + 1.38284 w_k1_IBM_s004
                                 + 1.40716 w_k1_INTC_s004
                                 + 1.33755 w_k1_KO_s004 + 1.2963 w_k1_MAT_s004
                                 + 1.37336 w_k1_MSFT_s004
                                 + 1.09813 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.801053 w_k1_ADM_s005
                                 + 1.00138 w_k1_AXP_s005
                                 + 0.716451 w_k1_DIS_s005
                                 + 0.851251 w_k1_FDX_s005
                                 + 0.75743 w_k1_IBM_s005
                                 + 0.847463 w_k1_INTC_s005
                                 + 0.726149 w_k1_KO_s005
                                 + 0.907603 w_k1_MAT_s005
                                 + 0.96478 w_k1_MSFT_s005
                                 + 0.912675 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.833904 w_k1_ADM_s005
                                 + 0.945783 w_k1_AXP_s005
                                 + 1.0108 w_k1_DIS_s005
                                 + 0.836282 w_k1_FDX_s005
                                 + 1.38155 w_k1_IBM_s005
                                 + 0.942078 w_k1_INTC_s005
                                 + 1.22285 w_k1_KO_s005 + 1.43071 w_k1_MAT_s005
                                 + 0.981124 w_k1_MSFT_s005
                                 + 1.0143 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.821565 w_k1_ADM_s005
                                 + 1.05068 w_k1_AXP_s005
                                 + 0.734003 w_k1_DIS_s005
                                 + 0.783044 w_k1_FDX_s005
                                 + 1.04398 w_k1_IBM_s005
                                 + 0.831658 w_k1_INTC_s005
                                 + 0.891766 w_k1_KO_s005
                                 + 1.38103 w_k1_MAT_s005
                                 + 0.813037 w_k1_MSFT_s005
                                 + 0.878017 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.01899 w_k1_ADM_s005
                                 + 0.920137 w_k1_AXP_s005
                                 + 0.860448 w_k1_DIS_s005
                                 + 0.977415 w_k1_FDX_s005
                                 + 0.638521 w_k1_IBM_s005
                                 + 0.918688 w_k1_INTC_s005
                                 + 0.800467 w_k1_KO_s005
                                 + 0.639775 w_k1_MAT_s005
                                 + 1.0246 w_k1_MSFT_s005
                                 + 0.889865 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.41555 w_k1_ADM_s005
                                 + 1.36368 w_k1_AXP_s005
                                 + 1.54479 w_k1_DIS_s005
                                 + 1.39317 w_k1_FDX_s005
                                 + 0.926912 w_k1_IBM_s005
                                 + 1.31731 w_k1_INTC_s005
                                 + 1.10453 w_k1_KO_s005
                                 + 0.843465 w_k1_MAT_s005
                                 + 1.2713 w_k1_MSFT_s005 + 1.28405 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.3406 w_k1_ADM_s005
                                 + 1.98238 w_k1_AXP_s005
                                 + 1.71824 w_k1_DIS_s005
                                 + 1.44177 w_k1_FDX_s005
                                 + 1.61026 w_k1_IBM_s005
                                 + 1.25685 w_k1_INTC_s005 + 1.2484 w_k1_KO_s005
                                 + 1.92684 w_k1_MAT_s005
                                 + 1.29784 w_k1_MSFT_s005
                                 + 1.22457 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.05376 w_k1_ADM_s005
                                 + 0.757782 w_k1_AXP_s005
                                 + 0.956905 w_k1_DIS_s005 + 0.876 w_k1_FDX_s005
                                 + 0.946507 w_k1_IBM_s005
                                 + 0.886054 w_k1_INTC_s005
                                 + 1.09912 w_k1_KO_s005
                                 + 0.748521 w_k1_MAT_s005
                                 + 1.01707 w_k1_MSFT_s005
                                 + 0.93979 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.03965 w_k1_ADM_s005
                                 + 1.43881 w_k1_AXP_s005
                                 + 1.26172 w_k1_DIS_s005
                                 + 1.05726 w_k1_FDX_s005
                                 + 1.11588 w_k1_IBM_s005
                                 + 0.947611 w_k1_INTC_s005
                                 + 1.09033 w_k1_KO_s005 + 1.0799 w_k1_MAT_s005
                                 + 1.09159 w_k1_MSFT_s005
                                 + 1.09105 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.04225 w_k1_ADM_s006
                                 + 0.837688 w_k1_AXP_s006
                                 + 0.839021 w_k1_DIS_s006
                                 + 0.966965 w_k1_FDX_s006
                                 + 0.729571 w_k1_IBM_s006
                                 + 0.967144 w_k1_INTC_s006
                                 + 0.843627 w_k1_KO_s006
                                 + 0.754289 w_k1_MAT_s006
                                 + 1.0901 w_k1_MSFT_s006 + 1.00437 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.803345 w_k1_ADM_s006
                                 + 0.901379 w_k1_AXP_s006
                                 + 1.18506 w_k1_DIS_s006
                                 + 1.10651 w_k1_FDX_s006
                                 + 0.889638 w_k1_IBM_s006
                                 + 1.12051 w_k1_INTC_s006
                                 + 1.01474 w_k1_KO_s006
                                 + 0.811951 w_k1_MAT_s006
                                 + 1.19185 w_k1_MSFT_s006
                                 + 1.03627 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.757683 w_k1_ADM_s006
                                 + 0.789189 w_k1_AXP_s006
                                 + 0.702368 w_k1_DIS_s006
                                 + 0.788003 w_k1_FDX_s006
                                 + 0.828479 w_k1_IBM_s006
                                 + 0.881018 w_k1_INTC_s006
                                 + 0.794131 w_k1_KO_s006
                                 + 0.680525 w_k1_MAT_s006
                                 + 0.761775 w_k1_MSFT_s006
                                 + 0.850958 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.20656 w_k1_ADM_s006
                                 + 1.19156 w_k1_AXP_s006
                                 + 1.25008 w_k1_DIS_s006
                                 + 1.21732 w_k1_FDX_s006
                                 + 0.788192 w_k1_IBM_s006
                                 + 1.083 w_k1_INTC_s006 + 1.0477 w_k1_KO_s006
                                 + 0.9774 w_k1_MAT_s006
                                 + 1.25378 w_k1_MSFT_s006
                                 + 1.23139 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.45602 w_k1_ADM_s006
                                 + 1.43823 w_k1_AXP_s006 + 1.8683 w_k1_DIS_s006
                                 + 1.43217 w_k1_FDX_s006
                                 + 0.961967 w_k1_IBM_s006
                                 + 1.05165 w_k1_INTC_s006
                                 + 1.28961 w_k1_KO_s006 + 1.26054 w_k1_MAT_s006
                                 + 1.47868 w_k1_MSFT_s006
                                 + 1.27833 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.942535 w_k1_ADM_s006
                                 + 0.944087 w_k1_AXP_s006
                                 + 0.942213 w_k1_DIS_s006
                                 + 0.860803 w_k1_FDX_s006
                                 + 1.24085 w_k1_IBM_s006
                                 + 0.948585 w_k1_INTC_s006
                                 + 1.05781 w_k1_KO_s006 + 1.12357 w_k1_MAT_s006
                                 + 1.09159 w_k1_MSFT_s006
                                 + 0.907378 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.911685 w_k1_ADM_s006
                                 + 1.03702 w_k1_AXP_s006
                                 + 1.22742 w_k1_DIS_s006
                                 + 1.17734 w_k1_FDX_s006
                                 + 1.27497 w_k1_IBM_s006
                                 + 1.06307 w_k1_INTC_s006
                                 + 1.17408 w_k1_KO_s006 + 1.80717 w_k1_MAT_s006
                                 + 1.24383 w_k1_MSFT_s006
                                 + 1.09584 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.2856 w_k1_ADM_s006
                                 + 1.21546 w_k1_AXP_s006
                                 + 1.19516 w_k1_DIS_s006
                                 + 1.14011 w_k1_FDX_s006 + 1.0241 w_k1_IBM_s006
                                 + 0.888756 w_k1_INTC_s006
                                 + 1.09861 w_k1_KO_s006 + 1.48335 w_k1_MAT_s006
                                 + 1.11432 w_k1_MSFT_s006
                                 + 1.05734 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.05682 w_k1_ADM_s007
                                 + 1.26733 w_k1_AXP_s007
                                 + 1.37454 w_k1_DIS_s007
                                 + 1.22868 w_k1_FDX_s007
                                 + 1.29108 w_k1_IBM_s007
                                 + 1.35838 w_k1_INTC_s007 + 1.2672 w_k1_KO_s007
                                 + 1.34734 w_k1_MAT_s007
                                 + 1.36272 w_k1_MSFT_s007
                                 + 1.08857 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.13728 w_k1_ADM_s007
                                 + 1.60364 w_k1_AXP_s007
                                 + 1.07434 w_k1_DIS_s007
                                 + 1.02729 w_k1_FDX_s007 + 1.2053 w_k1_IBM_s007
                                 + 1.17384 w_k1_INTC_s007 + 1.0463 w_k1_KO_s007
                                 + 1.6342 w_k1_MAT_s007
                                 + 1.15091 w_k1_MSFT_s007
                                 + 1.07522 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.793656 w_k1_ADM_s007
                                 + 0.855346 w_k1_AXP_s007
                                 + 1.07457 w_k1_DIS_s007
                                 + 0.967979 w_k1_FDX_s007
                                 + 0.929013 w_k1_IBM_s007
                                 + 1.12468 w_k1_INTC_s007
                                 + 0.98135 w_k1_KO_s007
                                 + 0.808782 w_k1_MAT_s007
                                 + 1.0037 w_k1_MSFT_s007
                                 + 0.885308 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.77447 w_k1_ADM_s007
                                 + 0.944417 w_k1_AXP_s007
                                 + 0.829457 w_k1_DIS_s007
                                 + 0.781877 w_k1_FDX_s007
                                 + 1.13336 w_k1_IBM_s007
                                 + 0.90318 w_k1_INTC_s007
                                 + 0.888129 w_k1_KO_s007
                                 + 0.823116 w_k1_MAT_s007
                                 + 0.929653 w_k1_MSFT_s007
                                 + 0.958425 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.0441 w_k1_ADM_s007
                                 + 0.955952 w_k1_AXP_s007
                                 + 1.28794 w_k1_DIS_s007
                                 + 1.18716 w_k1_FDX_s007
                                 + 0.738878 w_k1_IBM_s007
                                 + 1.16632 w_k1_INTC_s007
                                 + 1.03066 w_k1_KO_s007
                                 + 0.636266 w_k1_MAT_s007
                                 + 1.25329 w_k1_MSFT_s007
                                 + 1.03721 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.22667 w_k1_ADM_s007
                                 + 1.35627 w_k1_AXP_s007
                                 + 1.06226 w_k1_DIS_s007
                                 + 1.19695 w_k1_FDX_s007
                                 + 0.902371 w_k1_IBM_s007
                                 + 1.18921 w_k1_INTC_s007
                                 + 0.887644 w_k1_KO_s007
                                 + 0.762203 w_k1_MAT_s007
                                 + 0.945572 w_k1_MSFT_s007
                                 + 1.0226 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.18106 w_k1_ADM_s007
                                 + 1.40858 w_k1_AXP_s007
                                 + 1.16153 w_k1_DIS_s007
                                 + 1.08974 w_k1_FDX_s007
                                 + 1.60376 w_k1_IBM_s007
                                 + 1.22863 w_k1_INTC_s007
                                 + 1.12711 w_k1_KO_s007 + 1.37529 w_k1_MAT_s007
                                 + 1.11321 w_k1_MSFT_s007
                                 + 1.09799 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.748696 w_k1_ADM_s007
                                 + 0.775648 w_k1_AXP_s007
                                 + 0.753461 w_k1_DIS_s007
                                 + 0.777883 w_k1_FDX_s007
                                 + 0.773567 w_k1_IBM_s007
                                 + 0.858959 w_k1_INTC_s007
                                 + 0.709322 w_k1_KO_s007
                                 + 0.519038 w_k1_MAT_s007
                                 + 0.817419 w_k1_MSFT_s007
                                 + 0.771128 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.15957 w_k1_ADM_s010
                                 + 1.52082 w_k1_AXP_s010
                                 + 1.49397 w_k1_DIS_s010
                                 + 1.45486 w_k1_FDX_s010
                                 + 0.968159 w_k1_IBM_s010
                                 + 0.949454 w_k1_INTC_s010
                                 + 1.00856 w_k1_KO_s010 + 1.22895 w_k1_MAT_s010
                                 + 1.12485 w_k1_MSFT_s010
                                 + 1.19747 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.841406 w_k1_ADM_s010
                                 + 0.998883 w_k1_AXP_s010
                                 + 1.17268 w_k1_DIS_s010
                                 + 1.09669 w_k1_FDX_s010
                                 + 0.768903 w_k1_IBM_s010
                                 + 0.956667 w_k1_INTC_s010
                                 + 0.94439 w_k1_KO_s010
                                 + 0.583204 w_k1_MAT_s010
                                 + 1.22986 w_k1_MSFT_s010
                                 + 1.00228 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.690597 w_k1_ADM_s010
                                 + 0.788365 w_k1_AXP_s010
                                 + 0.932304 w_k1_DIS_s010
                                 + 0.811583 w_k1_FDX_s010
                                 + 1.22692 w_k1_IBM_s010
                                 + 0.85298 w_k1_INTC_s010
                                 + 1.12445 w_k1_KO_s010 + 1.00362 w_k1_MAT_s010
                                 + 1.02129 w_k1_MSFT_s010
                                 + 1.06392 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.590135 w_k1_ADM_s010
                                 + 0.717596 w_k1_AXP_s010
                                 + 0.622806 w_k1_DIS_s010
                                 + 0.67325 w_k1_FDX_s010
                                 + 0.965965 w_k1_IBM_s010
                                 + 0.78466 w_k1_INTC_s010
                                 + 0.857999 w_k1_KO_s010
                                 + 0.803873 w_k1_MAT_s010
                                 + 0.805939 w_k1_MSFT_s010
                                 + 0.907789 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.07324 w_k1_ADM_s010
                                 + 0.967245 w_k1_AXP_s010
                                 + 1.04503 w_k1_DIS_s010
                                 + 1.02358 w_k1_FDX_s010
                                 + 1.00407 w_k1_IBM_s010
                                 + 0.88457 w_k1_INTC_s010
                                 + 1.02844 w_k1_KO_s010
                                 + 0.697366 w_k1_MAT_s010
                                 + 1.02473 w_k1_MSFT_s010
                                 + 1.19805 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.812358 w_k1_ADM_s010
                                 + 0.758082 w_k1_AXP_s010
                                 + 0.862744 w_k1_DIS_s010
                                 + 0.826791 w_k1_FDX_s010
                                 + 0.764645 w_k1_IBM_s010
                                 + 0.816091 w_k1_INTC_s010
                                 + 0.835677 w_k1_KO_s010
                                 + 0.736841 w_k1_MAT_s010
                                 + 0.865497 w_k1_MSFT_s010
                                 + 1.04412 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.29148 w_k1_ADM_s010
                                 + 1.37952 w_k1_AXP_s010
                                 + 1.42935 w_k1_DIS_s010
                                 + 1.22859 w_k1_FDX_s010
                                 + 1.12635 w_k1_IBM_s010
                                 + 0.997175 w_k1_INTC_s010
                                 + 1.06992 w_k1_KO_s010 + 1.02241 w_k1_MAT_s010
                                 + 1.2384 w_k1_MSFT_s010 + 1.19499 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.03255 w_k1_ADM_s010
                                 + 1.25432 w_k1_AXP_s010
                                 + 1.60375 w_k1_DIS_s010
                                 + 1.20651 w_k1_FDX_s010
                                 + 1.42996 w_k1_IBM_s010
                                 + 0.923653 w_k1_INTC_s010
                                 + 1.4169 w_k1_KO_s010 + 1.53822 w_k1_MAT_s010
                                 + 1.25986 w_k1_MSFT_s010
                                 + 1.23844 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.54072 w_k1_ADM_s011
                                 + 1.35045 w_k1_AXP_s011
                                 + 1.14256 w_k1_DIS_s011 + 1.3108 w_k1_FDX_s011
                                 + 1.05533 w_k1_IBM_s011
                                 + 1.30988 w_k1_INTC_s011
                                 + 1.11452 w_k1_KO_s011 + 1.21862 w_k1_MAT_s011
                                 + 1.32455 w_k1_MSFT_s011
                                 + 1.11772 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.19501 w_k1_ADM_s011
                                 + 0.997911 w_k1_AXP_s011
                                 + 1.22735 w_k1_DIS_s011
                                 + 1.16263 w_k1_FDX_s011
                                 + 1.00058 w_k1_IBM_s011
                                 + 1.16691 w_k1_INTC_s011
                                 + 1.07504 w_k1_KO_s011 + 1.09974 w_k1_MAT_s011
                                 + 1.12746 w_k1_MSFT_s011
                                 + 1.16094 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.94046 w_k1_ADM_s011
                                 + 0.953042 w_k1_AXP_s011
                                 + 0.984372 w_k1_DIS_s011
                                 + 1.18216 w_k1_FDX_s011
                                 + 0.626635 w_k1_IBM_s011
                                 + 1.09486 w_k1_INTC_s011
                                 + 0.890087 w_k1_KO_s011
                                 + 0.680028 w_k1_MAT_s011
                                 + 1.1172 w_k1_MSFT_s011
                                 + 0.968465 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.08693 w_k1_ADM_s011
                                 + 1.14821 w_k1_AXP_s011 + 1.0173 w_k1_DIS_s011
                                 + 1.00365 w_k1_FDX_s011
                                 + 1.50639 w_k1_IBM_s011
                                 + 1.18376 w_k1_INTC_s011
                                 + 1.23822 w_k1_KO_s011 + 1.51396 w_k1_MAT_s011
                                 + 1.09112 w_k1_MSFT_s011 + 1.2076 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.693922 w_k1_ADM_s011
                                 + 0.663714 w_k1_AXP_s011
                                 + 0.688119 w_k1_DIS_s011
                                 + 0.775038 w_k1_FDX_s011
                                 + 0.975234 w_k1_IBM_s011
                                 + 0.884738 w_k1_INTC_s011
                                 + 0.787177 w_k1_KO_s011
                                 + 0.914081 w_k1_MAT_s011
                                 + 0.819088 w_k1_MSFT_s011
                                 + 0.80248 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.74115 w_k1_ADM_s011
                                 + 0.972529 w_k1_AXP_s011
                                 + 0.780046 w_k1_DIS_s011
                                 + 0.940246 w_k1_FDX_s011
                                 + 0.883354 w_k1_IBM_s011
                                 + 0.856423 w_k1_INTC_s011
                                 + 0.859866 w_k1_KO_s011
                                 + 1.04472 w_k1_MAT_s011
                                 + 0.898379 w_k1_MSFT_s011
                                 + 0.997212 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.17826 w_k1_ADM_s011
                                 + 1.19918 w_k1_AXP_s011
                                 + 1.61189 w_k1_DIS_s011
                                 + 1.25844 w_k1_FDX_s011 + 1.4922 w_k1_IBM_s011
                                 + 1.28231 w_k1_INTC_s011
                                 + 1.48582 w_k1_KO_s011 + 1.42476 w_k1_MAT_s011
                                 + 1.22418 w_k1_MSFT_s011 + 1.4307 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.799224 w_k1_ADM_s011
                                 + 0.863272 w_k1_AXP_s011
                                 + 0.958515 w_k1_DIS_s011
                                 + 0.920374 w_k1_FDX_s011
                                 + 1.04911 w_k1_IBM_s011
                                 + 1.09941 w_k1_INTC_s011
                                 + 1.06091 w_k1_KO_s011
                                 + 0.882817 w_k1_MAT_s011
                                 + 1.01275 w_k1_MSFT_s011
                                 + 0.986517 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.09427 w_k1_ADM_s012
                                 + 1.69991 w_k1_AXP_s012
                                 + 1.49544 w_k1_DIS_s012
                                 + 1.15659 w_k1_FDX_s012
                                 + 1.78774 w_k1_IBM_s012
                                 + 1.09377 w_k1_INTC_s012
                                 + 1.47791 w_k1_KO_s012 + 2.40692 w_k1_MAT_s012
                                 + 1.05859 w_k1_MSFT_s012
                                 + 1.24925 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.00591 w_k1_ADM_s012
                                 + 1.16244 w_k1_AXP_s012
                                 + 1.26173 w_k1_DIS_s012
                                 + 1.20335 w_k1_FDX_s012
                                 + 0.727246 w_k1_IBM_s012
                                 + 1.08529 w_k1_INTC_s012
                                 + 1.04221 w_k1_KO_s012
                                 + 0.920616 w_k1_MAT_s012
                                 + 1.12379 w_k1_MSFT_s012
                                 + 0.974303 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.12724 w_k1_ADM_s012
                                 + 0.821292 w_k1_AXP_s012
                                 + 1.14644 w_k1_DIS_s012
                                 + 0.990535 w_k1_FDX_s012
                                 + 0.911273 w_k1_IBM_s012
                                 + 0.966713 w_k1_INTC_s012
                                 + 1.06894 w_k1_KO_s012
                                 + 0.827656 w_k1_MAT_s012
                                 + 1.08006 w_k1_MSFT_s012
                                 + 1.04932 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.747048 w_k1_ADM_s012
                                 + 0.593716 w_k1_AXP_s012
                                 + 0.67047 w_k1_DIS_s012
                                 + 0.804789 w_k1_FDX_s012
                                 + 0.594928 w_k1_IBM_s012
                                 + 1.03715 w_k1_INTC_s012
                                 + 0.733088 w_k1_KO_s012
                                 + 0.447069 w_k1_MAT_s012
                                 + 0.879029 w_k1_MSFT_s012
                                 + 0.763332 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.821733 w_k1_ADM_s012
                                 + 0.751473 w_k1_AXP_s012
                                 + 0.750597 w_k1_DIS_s012
                                 + 0.781315 w_k1_FDX_s012
                                 + 0.733246 w_k1_IBM_s012
                                 + 0.87342 w_k1_INTC_s012
                                 + 0.734704 w_k1_KO_s012
                                 + 0.634673 w_k1_MAT_s012
                                 + 0.85775 w_k1_MSFT_s012
                                 + 0.755799 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.3027 w_k1_ADM_s012
                                 + 1.32927 w_k1_AXP_s012
                                 + 1.59368 w_k1_DIS_s012
                                 + 1.21902 w_k1_FDX_s012
                                 + 1.20146 w_k1_IBM_s012
                                 + 1.33648 w_k1_INTC_s012
                                 + 1.37987 w_k1_KO_s012 + 1.38478 w_k1_MAT_s012
                                 + 1.14161 w_k1_MSFT_s012
                                 + 1.10843 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.870488 w_k1_ADM_s012
                                 + 0.89242 w_k1_AXP_s012
                                 + 0.788349 w_k1_DIS_s012
                                 + 0.771653 w_k1_FDX_s012
                                 + 1.13324 w_k1_IBM_s012
                                 + 0.931834 w_k1_INTC_s012
                                 + 0.939595 w_k1_KO_s012
                                 + 0.886802 w_k1_MAT_s012
                                 + 0.850672 w_k1_MSFT_s012
                                 + 0.854225 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.863935 w_k1_ADM_s012
                                 + 1.04778 w_k1_AXP_s012
                                 + 1.11244 w_k1_DIS_s012
                                 + 0.930928 w_k1_FDX_s012
                                 + 1.19942 w_k1_IBM_s012
                                 + 1.12141 w_k1_INTC_s012
                                 + 1.22183 w_k1_KO_s012 + 1.22705 w_k1_MAT_s012
                                 + 0.913389 w_k1_MSFT_s012
                                 + 0.933149 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.803971 w_k1_ADM_s013
                                 + 0.816246 w_k1_AXP_s013
                                 + 1.03413 w_k1_DIS_s013
                                 + 0.894748 w_k1_FDX_s013
                                 + 1.27055 w_k1_IBM_s013
                                 + 0.852074 w_k1_INTC_s013
                                 + 0.890351 w_k1_KO_s013
                                 + 0.706722 w_k1_MAT_s013
                                 + 0.908593 w_k1_MSFT_s013
                                 + 0.957188 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.954721 w_k1_ADM_s013
                                 + 0.922543 w_k1_AXP_s013
                                 + 0.760623 w_k1_DIS_s013
                                 + 0.943826 w_k1_FDX_s013
                                 + 0.88358 w_k1_IBM_s013
                                 + 0.971862 w_k1_INTC_s013
                                 + 0.966634 w_k1_KO_s013
                                 + 0.650434 w_k1_MAT_s013
                                 + 0.906728 w_k1_MSFT_s013
                                 + 1.07601 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.713658 w_k1_ADM_s013
                                 + 0.591277 w_k1_AXP_s013
                                 + 0.532042 w_k1_DIS_s013
                                 + 0.686894 w_k1_FDX_s013
                                 + 0.833576 w_k1_IBM_s013
                                 + 0.817425 w_k1_INTC_s013
                                 + 0.732928 w_k1_KO_s013
                                 + 0.49381 w_k1_MAT_s013
                                 + 0.669178 w_k1_MSFT_s013
                                 + 0.75077 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.756245 w_k1_ADM_s013
                                 + 1.15451 w_k1_AXP_s013
                                 + 0.813584 w_k1_DIS_s013
                                 + 1.01456 w_k1_FDX_s013
                                 + 0.778646 w_k1_IBM_s013
                                 + 0.92589 w_k1_INTC_s013
                                 + 0.666412 w_k1_KO_s013
                                 + 0.551709 w_k1_MAT_s013
                                 + 0.792527 w_k1_MSFT_s013
                                 + 0.773144 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.98213 w_k1_ADM_s013
                                 + 1.40212 w_k1_AXP_s013
                                 + 1.34731 w_k1_DIS_s013
                                 + 1.20805 w_k1_FDX_s013
                                 + 1.43325 w_k1_IBM_s013
                                 + 1.15381 w_k1_INTC_s013
                                 + 1.27069 w_k1_KO_s013 + 1.07701 w_k1_MAT_s013
                                 + 1.20879 w_k1_MSFT_s013
                                 + 1.21402 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.775174 w_k1_ADM_s013
                                 + 0.745822 w_k1_AXP_s013
                                 + 0.802176 w_k1_DIS_s013
                                 + 0.856085 w_k1_FDX_s013
                                 + 0.816392 w_k1_IBM_s013
                                 + 0.969589 w_k1_INTC_s013
                                 + 0.768967 w_k1_KO_s013
                                 + 0.507597 w_k1_MAT_s013
                                 + 0.910665 w_k1_MSFT_s013
                                 + 0.894483 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.769537 w_k1_ADM_s013
                                 + 0.89325 w_k1_AXP_s013
                                 + 0.950487 w_k1_DIS_s013
                                 + 0.914793 w_k1_FDX_s013
                                 + 0.912415 w_k1_IBM_s013
                                 + 0.939794 w_k1_INTC_s013
                                 + 1.07141 w_k1_KO_s013
                                 + 0.645312 w_k1_MAT_s013
                                 + 0.942385 w_k1_MSFT_s013
                                 + 1.00424 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.02896 w_k1_ADM_s013
                                 + 1.05247 w_k1_AXP_s013
                                 + 1.32038 w_k1_DIS_s013
                                 + 1.26665 w_k1_FDX_s013
                                 + 0.787956 w_k1_IBM_s013
                                 + 0.997556 w_k1_INTC_s013
                                 + 0.869922 w_k1_KO_s013
                                 + 0.759207 w_k1_MAT_s013
                                 + 1.14421 w_k1_MSFT_s013
                                 + 1.21779 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.776331 w_k1_ADM_s014
                                 + 1.06423 w_k1_AXP_s014
                                 + 0.775843 w_k1_DIS_s014
                                 + 0.836996 w_k1_FDX_s014
                                 + 1.27398 w_k1_IBM_s014
                                 + 0.891478 w_k1_INTC_s014
                                 + 1.08178 w_k1_KO_s014 + 1.60834 w_k1_MAT_s014
                                 + 0.801879 w_k1_MSFT_s014
                                 + 0.960604 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.04382 w_k1_ADM_s014
                                 + 1.11082 w_k1_AXP_s014
                                 + 1.36836 w_k1_DIS_s014
                                 + 1.25867 w_k1_FDX_s014
                                 + 0.982476 w_k1_IBM_s014
                                 + 1.22983 w_k1_INTC_s014
                                 + 1.15929 w_k1_KO_s014 + 1.58117 w_k1_MAT_s014
                                 + 1.21397 w_k1_MSFT_s014
                                 + 1.20996 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.907248 w_k1_ADM_s014
                                 + 0.800102 w_k1_AXP_s014
                                 + 0.931016 w_k1_DIS_s014
                                 + 0.880831 w_k1_FDX_s014
                                 + 0.750831 w_k1_IBM_s014
                                 + 0.998323 w_k1_INTC_s014
                                 + 0.918687 w_k1_KO_s014
                                 + 0.797672 w_k1_MAT_s014
                                 + 0.97057 w_k1_MSFT_s014
                                 + 0.922047 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.00168 w_k1_ADM_s014
                                 + 1.3911 w_k1_AXP_s014 + 1.20737 w_k1_DIS_s014
                                 + 1.0879 w_k1_FDX_s014 + 1.2195 w_k1_IBM_s014
                                 + 0.909012 w_k1_INTC_s014
                                 + 1.22564 w_k1_KO_s014 + 1.75309 w_k1_MAT_s014
                                 + 1.04493 w_k1_MSFT_s014
                                 + 1.05903 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.880802 w_k1_ADM_s014
                                 + 0.87402 w_k1_AXP_s014
                                 + 0.82321 w_k1_DIS_s014
                                 + 0.921524 w_k1_FDX_s014
                                 + 0.804489 w_k1_IBM_s014
                                 + 0.974053 w_k1_INTC_s014
                                 + 0.848435 w_k1_KO_s014
                                 + 0.876104 w_k1_MAT_s014
                                 + 0.865909 w_k1_MSFT_s014
                                 + 0.835526 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.15047 w_k1_ADM_s014
                                 + 1.13686 w_k1_AXP_s014
                                 + 1.01478 w_k1_DIS_s014
                                 + 1.17609 w_k1_FDX_s014
                                 + 0.761999 w_k1_IBM_s014
                                 + 1.10752 w_k1_INTC_s014
                                 + 0.803046 w_k1_KO_s014
                                 + 0.858505 w_k1_MAT_s014
                                 + 1.0163 w_k1_MSFT_s014
                                 + 0.939865 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14334 w_k1_ADM_s014
                                 + 1.44138 w_k1_AXP_s014
                                 + 1.53137 w_k1_DIS_s014 + 1.1916 w_k1_FDX_s014
                                 + 1.79883 w_k1_IBM_s014
                                 + 1.33339 w_k1_INTC_s014
                                 + 1.53076 w_k1_KO_s014 + 1.81101 w_k1_MAT_s014
                                 + 1.24053 w_k1_MSFT_s014
                                 + 1.25372 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.15819 w_k1_ADM_s014
                                 + 1.55759 w_k1_AXP_s014
                                 + 1.84102 w_k1_DIS_s014
                                 + 1.59903 w_k1_FDX_s014
                                 + 1.16013 w_k1_IBM_s014
                                 + 1.53809 w_k1_INTC_s014
                                 + 1.57781 w_k1_KO_s014 + 1.46553 w_k1_MAT_s014
                                 + 1.54053 w_k1_MSFT_s014
                                 + 1.40533 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.983293 w_k1_ADM_s015
                                 + 0.923095 w_k1_AXP_s015
                                 + 1.10293 w_k1_DIS_s015
                                 + 0.990792 w_k1_FDX_s015
                                 + 1.0303 w_k1_IBM_s015
                                 + 1.01347 w_k1_INTC_s015
                                 + 1.22678 w_k1_KO_s015 + 1.34599 w_k1_MAT_s015
                                 + 1.21383 w_k1_MSFT_s015
                                 + 1.10202 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.788922 w_k1_ADM_s015
                                 + 1.23707 w_k1_AXP_s015
                                 + 1.14071 w_k1_DIS_s015
                                 + 1.02093 w_k1_FDX_s015 + 1.3403 w_k1_IBM_s015
                                 + 1.12512 w_k1_INTC_s015
                                 + 0.983694 w_k1_KO_s015 + 1.5134 w_k1_MAT_s015
                                 + 1.00589 w_k1_MSFT_s015
                                 + 1.09647 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.749164 w_k1_ADM_s015
                                 + 0.731411 w_k1_AXP_s015
                                 + 0.571357 w_k1_DIS_s015
                                 + 0.737358 w_k1_FDX_s015
                                 + 0.726825 w_k1_IBM_s015
                                 + 0.835204 w_k1_INTC_s015
                                 + 0.724868 w_k1_KO_s015
                                 + 0.766917 w_k1_MAT_s015
                                 + 0.807044 w_k1_MSFT_s015
                                 + 0.810512 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.802456 w_k1_ADM_s015
                                 + 0.829277 w_k1_AXP_s015
                                 + 0.795349 w_k1_DIS_s015
                                 + 0.888713 w_k1_FDX_s015
                                 + 0.988405 w_k1_IBM_s015
                                 + 1.05506 w_k1_INTC_s015
                                 + 0.881987 w_k1_KO_s015
                                 + 0.817105 w_k1_MAT_s015
                                 + 0.808823 w_k1_MSFT_s015
                                 + 0.849215 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.08401 w_k1_ADM_s015
                                 + 1.45814 w_k1_AXP_s015
                                 + 1.43585 w_k1_DIS_s015
                                 + 1.22075 w_k1_FDX_s015
                                 + 1.23615 w_k1_IBM_s015
                                 + 1.42253 w_k1_INTC_s015
                                 + 1.15898 w_k1_KO_s015 + 1.19875 w_k1_MAT_s015
                                 + 1.15852 w_k1_MSFT_s015
                                 + 1.05676 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.0411 w_k1_ADM_s015
                                 + 1.72056 w_k1_AXP_s015 + 1.8131 w_k1_DIS_s015
                                 + 1.47446 w_k1_FDX_s015
                                 + 1.24918 w_k1_IBM_s015
                                 + 1.51548 w_k1_INTC_s015
                                 + 1.20563 w_k1_KO_s015 + 2.09612 w_k1_MAT_s015
                                 + 1.50595 w_k1_MSFT_s015
                                 + 1.24769 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.830767 w_k1_ADM_s015
                                 + 0.652533 w_k1_AXP_s015
                                 + 0.660718 w_k1_DIS_s015
                                 + 0.855945 w_k1_FDX_s015
                                 + 0.66898 w_k1_IBM_s015
                                 + 0.823055 w_k1_INTC_s015
                                 + 0.725373 w_k1_KO_s015
                                 + 0.820284 w_k1_MAT_s015
                                 + 0.932886 w_k1_MSFT_s015
                                 + 0.773247 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.09418 w_k1_ADM_s015
                                 + 1.13802 w_k1_AXP_s015
                                 + 1.17668 w_k1_DIS_s015
                                 + 1.25873 w_k1_FDX_s015
                                 + 0.860144 w_k1_IBM_s015
                                 + 1.10124 w_k1_INTC_s015 + 1.0448 w_k1_KO_s015
                                 + 1.15062 w_k1_MAT_s015
                                 + 1.12473 w_k1_MSFT_s015 + 1.1104 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.29647 w_k1_ADM_s016
                                 + 1.18077 w_k1_AXP_s016 + 1.2162 w_k1_DIS_s016
                                 + 1.04335 w_k1_FDX_s016
                                 + 1.16667 w_k1_IBM_s016
                                 + 1.15463 w_k1_INTC_s016
                                 + 1.18779 w_k1_KO_s016
                                 + 0.723782 w_k1_MAT_s016
                                 + 1.12085 w_k1_MSFT_s016
                                 + 1.10141 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.02423 w_k1_ADM_s016
                                 + 1.16328 w_k1_AXP_s016
                                 + 0.912802 w_k1_DIS_s016
                                 + 0.946036 w_k1_FDX_s016
                                 + 1.10875 w_k1_IBM_s016
                                 + 1.10036 w_k1_INTC_s016
                                 + 0.94292 w_k1_KO_s016
                                 + 0.922175 w_k1_MAT_s016
                                 + 0.876116 w_k1_MSFT_s016
                                 + 0.970435 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.12919 w_k1_ADM_s016
                                 + 1.36193 w_k1_AXP_s016
                                 + 1.30977 w_k1_DIS_s016
                                 + 1.08204 w_k1_FDX_s016
                                 + 1.51914 w_k1_IBM_s016
                                 + 1.12911 w_k1_INTC_s016
                                 + 1.36539 w_k1_KO_s016 + 1.06016 w_k1_MAT_s016
                                 + 1.01073 w_k1_MSFT_s016
                                 + 1.05603 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.05239 w_k1_ADM_s016
                                 + 1.10833 w_k1_AXP_s016
                                 + 0.876765 w_k1_DIS_s016
                                 + 1.07041 w_k1_FDX_s016
                                 + 0.899353 w_k1_IBM_s016
                                 + 1.05478 w_k1_INTC_s016
                                 + 0.990638 w_k1_KO_s016
                                 + 0.738514 w_k1_MAT_s016
                                 + 1.14526 w_k1_MSFT_s016
                                 + 1.06091 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.884361 w_k1_ADM_s016
                                 + 0.713072 w_k1_AXP_s016
                                 + 0.777165 w_k1_DIS_s016
                                 + 0.840665 w_k1_FDX_s016
                                 + 0.836677 w_k1_IBM_s016
                                 + 0.911927 w_k1_INTC_s016
                                 + 0.875659 w_k1_KO_s016
                                 + 0.761966 w_k1_MAT_s016
                                 + 0.941006 w_k1_MSFT_s016
                                 + 0.886117 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.867488 w_k1_ADM_s016
                                 + 0.710987 w_k1_AXP_s016
                                 + 0.726088 w_k1_DIS_s016
                                 + 0.822301 w_k1_FDX_s016
                                 + 0.603959 w_k1_IBM_s016
                                 + 0.871074 w_k1_INTC_s016
                                 + 0.680296 w_k1_KO_s016
                                 + 0.339728 w_k1_MAT_s016
                                 + 0.871728 w_k1_MSFT_s016
                                 + 0.872876 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.10176 w_k1_ADM_s016
                                 + 0.928128 w_k1_AXP_s016
                                 + 1.06841 w_k1_DIS_s016
                                 + 0.956734 w_k1_FDX_s016
                                 + 0.759094 w_k1_IBM_s016
                                 + 1.13866 w_k1_INTC_s016
                                 + 0.933708 w_k1_KO_s016
                                 + 0.528448 w_k1_MAT_s016
                                 + 1.10598 w_k1_MSFT_s016
                                 + 1.01835 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.1261 w_k1_ADM_s016
                                 + 1.13037 w_k1_AXP_s016
                                 + 1.29526 w_k1_DIS_s016
                                 + 1.24837 w_k1_FDX_s016
                                 + 0.810831 w_k1_IBM_s016
                                 + 1.1261 w_k1_INTC_s016 + 1.13867 w_k1_KO_s016
                                 + 0.654623 w_k1_MAT_s016
                                 + 1.16904 w_k1_MSFT_s016
                                 + 1.16398 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.35872 w_k1_ADM_s017
                                 + 1.41388 w_k1_AXP_s017
                                 + 1.27174 w_k1_DIS_s017
                                 + 1.08496 w_k1_FDX_s017 + 1.7522 w_k1_IBM_s017
                                 + 0.981789 w_k1_INTC_s017
                                 + 1.52866 w_k1_KO_s017 + 2.61246 w_k1_MAT_s017
                                 + 1.06829 w_k1_MSFT_s017
                                 + 1.22746 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.883473 w_k1_ADM_s017
                                 + 0.846032 w_k1_AXP_s017
                                 + 0.800154 w_k1_DIS_s017
                                 + 0.787332 w_k1_FDX_s017
                                 + 1.17216 w_k1_IBM_s017
                                 + 0.996785 w_k1_INTC_s017
                                 + 1.08164 w_k1_KO_s017 + 1.07133 w_k1_MAT_s017
                                 + 0.837757 w_k1_MSFT_s017
                                 + 0.8936 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.769501 w_k1_ADM_s017
                                 + 0.539677 w_k1_AXP_s017
                                 + 0.662652 w_k1_DIS_s017
                                 + 0.752267 w_k1_FDX_s017
                                 + 0.677594 w_k1_IBM_s017
                                 + 0.772181 w_k1_INTC_s017
                                 + 0.763333 w_k1_KO_s017
                                 + 0.676079 w_k1_MAT_s017
                                 + 0.869603 w_k1_MSFT_s017
                                 + 0.756856 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.53686 w_k1_ADM_s017
                                 + 1.62061 w_k1_AXP_s017 + 1.7131 w_k1_DIS_s017
                                 + 1.4076 w_k1_FDX_s017 + 1.15515 w_k1_IBM_s017
                                 + 1.2173 w_k1_INTC_s017 + 1.60949 w_k1_KO_s017
                                 + 1.77852 w_k1_MAT_s017
                                 + 1.52221 w_k1_MSFT_s017
                                 + 1.40108 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.30399 w_k1_ADM_s017
                                 + 1.02574 w_k1_AXP_s017
                                 + 1.02642 w_k1_DIS_s017
                                 + 1.05306 w_k1_FDX_s017
                                 + 0.832607 w_k1_IBM_s017
                                 + 0.941233 w_k1_INTC_s017
                                 + 1.14428 w_k1_KO_s017 + 1.00709 w_k1_MAT_s017
                                 + 1.13946 w_k1_MSFT_s017
                                 + 1.00514 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.01393 w_k1_ADM_s017
                                 + 1.03558 w_k1_AXP_s017
                                 + 1.11468 w_k1_DIS_s017
                                 + 0.907498 w_k1_FDX_s017
                                 + 1.28913 w_k1_IBM_s017
                                 + 0.941217 w_k1_INTC_s017
                                 + 1.26142 w_k1_KO_s017 + 1.79097 w_k1_MAT_s017
                                 + 0.945244 w_k1_MSFT_s017
                                 + 1.10306 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.64773 w_k1_ADM_s017
                                 + 1.21665 w_k1_AXP_s017
                                 + 1.28997 w_k1_DIS_s017
                                 + 1.26553 w_k1_FDX_s017
                                 + 0.976757 w_k1_IBM_s017
                                 + 1.05837 w_k1_INTC_s017
                                 + 1.06808 w_k1_KO_s017 + 1.15325 w_k1_MAT_s017
                                 + 1.18076 w_k1_MSFT_s017
                                 + 1.04193 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.0181 w_k1_ADM_s017
                                 + 0.890804 w_k1_AXP_s017
                                 + 0.757638 w_k1_DIS_s017
                                 + 0.871529 w_k1_FDX_s017
                                 + 0.782051 w_k1_IBM_s017
                                 + 0.850409 w_k1_INTC_s017
                                 + 0.956442 w_k1_KO_s017
                                 + 1.18041 w_k1_MAT_s017
                                 + 0.851309 w_k1_MSFT_s017
                                 + 0.888152 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.884546 w_k1_ADM_s100
                                 + 1.55997 w_k1_AXP_s100
                                 + 1.27144 w_k1_DIS_s100
                                 + 1.07639 w_k1_FDX_s100
                                 + 1.58743 w_k1_IBM_s100
                                 + 0.911669 w_k1_INTC_s100
                                 + 1.29691 w_k1_KO_s100 + 1.63972 w_k1_MAT_s100
                                 + 0.934674 w_k1_MSFT_s100
                                 + 1.29767 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.9409 w_k1_ADM_s100
                                 + 0.926801 w_k1_AXP_s100
                                 + 0.931981 w_k1_DIS_s100
                                 + 0.891716 w_k1_FDX_s100
                                 + 1.84877 w_k1_IBM_s100
                                 + 1.01507 w_k1_INTC_s100
                                 + 1.17505 w_k1_KO_s100 + 1.32371 w_k1_MAT_s100
                                 + 0.891804 w_k1_MSFT_s100
                                 + 1.12685 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.036 w_k1_ADM_s100
                                 + 0.958624 w_k1_AXP_s100
                                 + 1.01559 w_k1_DIS_s100
                                 + 0.984199 w_k1_FDX_s100
                                 + 1.27066 w_k1_IBM_s100
                                 + 0.785814 w_k1_INTC_s100
                                 + 1.12986 w_k1_KO_s100 + 1.17551 w_k1_MAT_s100
                                 + 0.890542 w_k1_MSFT_s100
                                 + 1.09612 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.842417 w_k1_ADM_s100
                                 + 0.729515 w_k1_AXP_s100
                                 + 0.571817 w_k1_DIS_s100
                                 + 0.736379 w_k1_FDX_s100
                                 + 0.979184 w_k1_IBM_s100
                                 + 0.722657 w_k1_INTC_s100
                                 + 0.894539 w_k1_KO_s100
                                 + 1.04241 w_k1_MAT_s100
                                 + 0.770407 w_k1_MSFT_s100
                                 + 0.856469 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.96246 w_k1_ADM_s100
                                 + 0.738924 w_k1_AXP_s100
                                 + 0.707687 w_k1_DIS_s100
                                 + 0.87382 w_k1_FDX_s100
                                 + 0.649185 w_k1_IBM_s100
                                 + 0.84874 w_k1_INTC_s100
                                 + 0.792853 w_k1_KO_s100
                                 + 0.40033 w_k1_MAT_s100
                                 + 0.798406 w_k1_MSFT_s100
                                 + 0.884712 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.819979 w_k1_ADM_s100
                                 + 0.992222 w_k1_AXP_s100
                                 + 0.751103 w_k1_DIS_s100
                                 + 0.909074 w_k1_FDX_s100
                                 + 1.05227 w_k1_IBM_s100
                                 + 0.850969 w_k1_INTC_s100
                                 + 0.95594 w_k1_KO_s100 + 1.12542 w_k1_MAT_s100
                                 + 0.958535 w_k1_MSFT_s100
                                 + 0.924455 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.21971 w_k1_ADM_s100
                                 + 1.10263 w_k1_AXP_s100
                                 + 1.16078 w_k1_DIS_s100
                                 + 1.09439 w_k1_FDX_s100
                                 + 0.859298 w_k1_IBM_s100
                                 + 0.886099 w_k1_INTC_s100
                                 + 1.06606 w_k1_KO_s100
                                 + 0.816418 w_k1_MAT_s100
                                 + 1.0472 w_k1_MSFT_s100 + 1.09423 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.19273 w_k1_ADM_s100
                                 + 1.21114 w_k1_AXP_s100
                                 + 1.56725 w_k1_DIS_s100
                                 + 1.37908 w_k1_FDX_s100
                                 + 1.12626 w_k1_IBM_s100
                                 + 1.09416 w_k1_INTC_s100
                                 + 1.39296 w_k1_KO_s100 + 1.68366 w_k1_MAT_s100
                                 + 1.29174 w_k1_MSFT_s100
                                 + 1.19442 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.22856 w_k1_ADM_s101
                                 + 1.31935 w_k1_AXP_s101
                                 + 1.59689 w_k1_DIS_s101
                                 + 1.33117 w_k1_FDX_s101
                                 + 1.06006 w_k1_IBM_s101
                                 + 1.16289 w_k1_INTC_s101
                                 + 1.17996 w_k1_KO_s101 + 1.12809 w_k1_MAT_s101
                                 + 1.16057 w_k1_MSFT_s101
                                 + 1.27016 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.20526 w_k1_ADM_s101
                                 + 0.959631 w_k1_AXP_s101
                                 + 1.09274 w_k1_DIS_s101
                                 + 1.01568 w_k1_FDX_s101
                                 + 1.10967 w_k1_IBM_s101
                                 + 1.10536 w_k1_INTC_s101
                                 + 1.00616 w_k1_KO_s101
                                 + 0.891354 w_k1_MAT_s101
                                 + 1.12173 w_k1_MSFT_s101
                                 + 1.06892 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.0853 w_k1_ADM_s101
                                 + 0.964358 w_k1_AXP_s101
                                 + 0.847352 w_k1_DIS_s101
                                 + 0.8556 w_k1_FDX_s101
                                 + 0.894122 w_k1_IBM_s101
                                 + 0.960815 w_k1_INTC_s101
                                 + 0.819337 w_k1_KO_s101
                                 + 0.939689 w_k1_MAT_s101
                                 + 0.799883 w_k1_MSFT_s101
                                 + 0.844232 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.18751 w_k1_ADM_s101
                                 + 1.19224 w_k1_AXP_s101
                                 + 1.09145 w_k1_DIS_s101
                                 + 1.16423 w_k1_FDX_s101
                                 + 0.941618 w_k1_IBM_s101
                                 + 1.0032 w_k1_INTC_s101
                                 + 0.974992 w_k1_KO_s101
                                 + 0.942201 w_k1_MAT_s101
                                 + 1.02854 w_k1_MSFT_s101
                                 + 1.07999 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.850253 w_k1_ADM_s101
                                 + 0.687994 w_k1_AXP_s101
                                 + 0.667897 w_k1_DIS_s101
                                 + 0.766985 w_k1_FDX_s101
                                 + 0.719467 w_k1_IBM_s101
                                 + 0.962229 w_k1_INTC_s101
                                 + 0.696134 w_k1_KO_s101
                                 + 0.481756 w_k1_MAT_s101
                                 + 0.805938 w_k1_MSFT_s101
                                 + 0.778043 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.39996 w_k1_ADM_s101
                                 + 1.0204 w_k1_AXP_s101 + 1.12595 w_k1_DIS_s101
                                 + 1.2572 w_k1_FDX_s101 + 0.68487 w_k1_IBM_s101
                                 + 1.16018 w_k1_INTC_s101
                                 + 0.852783 w_k1_KO_s101
                                 + 0.677987 w_k1_MAT_s101
                                 + 1.21774 w_k1_MSFT_s101
                                 + 1.09298 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.973896 w_k1_ADM_s101
                                 + 0.815076 w_k1_AXP_s101
                                 + 0.923732 w_k1_DIS_s101
                                 + 0.957779 w_k1_FDX_s101
                                 + 0.654936 w_k1_IBM_s101
                                 + 0.970474 w_k1_INTC_s101
                                 + 0.709498 w_k1_KO_s101
                                 + 0.636179 w_k1_MAT_s101
                                 + 1.02235 w_k1_MSFT_s101
                                 + 0.935794 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.970521 w_k1_ADM_s101
                                 + 1.138 w_k1_AXP_s101 + 1.09524 w_k1_DIS_s101
                                 + 0.909197 w_k1_FDX_s101
                                 + 1.61693 w_k1_IBM_s101
                                 + 1.08776 w_k1_INTC_s101
                                 + 1.22294 w_k1_KO_s101 + 1.51005 w_k1_MAT_s101
                                 + 0.927036 w_k1_MSFT_s101
                                 + 1.03419 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.964928 w_k1_ADM_s102
                                 + 0.993231 w_k1_AXP_s102
                                 + 0.849756 w_k1_DIS_s102
                                 + 0.913403 w_k1_FDX_s102
                                 + 1.2303 w_k1_IBM_s102
                                 + 1.16282 w_k1_INTC_s102
                                 + 1.02941 w_k1_KO_s102 + 1.15277 w_k1_MAT_s102
                                 + 0.955918 w_k1_MSFT_s102
                                 + 0.884526 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.28905 w_k1_ADM_s102
                                 + 1.28372 w_k1_AXP_s102 + 1.3724 w_k1_DIS_s102
                                 + 1.24131 w_k1_FDX_s102
                                 + 1.19879 w_k1_IBM_s102
                                 + 1.26567 w_k1_INTC_s102
                                 + 1.06282 w_k1_KO_s102
                                 + 0.883812 w_k1_MAT_s102
                                 + 1.1416 w_k1_MSFT_s102 + 1.16206 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.939872 w_k1_ADM_s102
                                 + 1.00872 w_k1_AXP_s102
                                 + 0.904212 w_k1_DIS_s102
                                 + 0.981969 w_k1_FDX_s102
                                 + 0.762122 w_k1_IBM_s102
                                 + 0.955028 w_k1_INTC_s102
                                 + 0.757154 w_k1_KO_s102
                                 + 0.739364 w_k1_MAT_s102
                                 + 1.05813 w_k1_MSFT_s102
                                 + 0.906334 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.970166 w_k1_ADM_s102
                                 + 0.876824 w_k1_AXP_s102
                                 + 1.09168 w_k1_DIS_s102
                                 + 0.923124 w_k1_FDX_s102
                                 + 1.19189 w_k1_IBM_s102
                                 + 1.05988 w_k1_INTC_s102
                                 + 0.965452 w_k1_KO_s102
                                 + 0.825281 w_k1_MAT_s102
                                 + 1.03268 w_k1_MSFT_s102 + 1.0583 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.08575 w_k1_ADM_s102
                                 + 1.0757 w_k1_AXP_s102 + 1.20679 w_k1_DIS_s102
                                 + 1.01077 w_k1_FDX_s102
                                 + 1.50243 w_k1_IBM_s102
                                 + 1.06261 w_k1_INTC_s102
                                 + 1.21421 w_k1_KO_s102 + 1.41264 w_k1_MAT_s102
                                 + 1.18968 w_k1_MSFT_s102
                                 + 1.07867 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.810733 w_k1_ADM_s102
                                 + 0.594252 w_k1_AXP_s102
                                 + 0.718788 w_k1_DIS_s102
                                 + 0.730807 w_k1_FDX_s102
                                 + 0.97212 w_k1_IBM_s102
                                 + 0.949032 w_k1_INTC_s102
                                 + 0.909124 w_k1_KO_s102
                                 + 0.742059 w_k1_MAT_s102
                                 + 0.921424 w_k1_MSFT_s102
                                 + 0.895152 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.11883 w_k1_ADM_s102
                                 + 0.771254 w_k1_AXP_s102
                                 + 1.06552 w_k1_DIS_s102
                                 + 0.944103 w_k1_FDX_s102
                                 + 1.07347 w_k1_IBM_s102
                                 + 1.08308 w_k1_INTC_s102
                                 + 1.07774 w_k1_KO_s102
                                 + 0.693556 w_k1_MAT_s102
                                 + 1.11543 w_k1_MSFT_s102
                                 + 1.01021 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.851598 w_k1_ADM_s102
                                 + 0.911163 w_k1_AXP_s102
                                 + 0.605564 w_k1_DIS_s102
                                 + 0.788199 w_k1_FDX_s102
                                 + 0.978489 w_k1_IBM_s102
                                 + 0.714852 w_k1_INTC_s102
                                 + 0.738551 w_k1_KO_s102
                                 + 1.05243 w_k1_MAT_s102
                                 + 0.7951 w_k1_MSFT_s102
                                 + 0.872415 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.686794 w_k1_ADM_s103
                                 + 0.5799 w_k1_AXP_s103
                                 + 0.718366 w_k1_DIS_s103
                                 + 0.752151 w_k1_FDX_s103
                                 + 0.809266 w_k1_IBM_s103
                                 + 0.802705 w_k1_INTC_s103
                                 + 0.760194 w_k1_KO_s103
                                 + 0.760015 w_k1_MAT_s103
                                 + 0.787625 w_k1_MSFT_s103
                                 + 0.785679 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.991574 w_k1_ADM_s103
                                 + 1.38522 w_k1_AXP_s103
                                 + 1.37563 w_k1_DIS_s103
                                 + 1.11187 w_k1_FDX_s103
                                 + 1.71215 w_k1_IBM_s103
                                 + 0.98782 w_k1_INTC_s103
                                 + 1.49899 w_k1_KO_s103 + 2.55645 w_k1_MAT_s103
                                 + 1.11462 w_k1_MSFT_s103
                                 + 1.08308 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.14798 w_k1_ADM_s103
                                 + 1.14765 w_k1_AXP_s103
                                 + 1.50375 w_k1_DIS_s103
                                 + 1.35623 w_k1_FDX_s103
                                 + 1.12521 w_k1_IBM_s103
                                 + 1.12493 w_k1_INTC_s103
                                 + 1.32328 w_k1_KO_s103 + 1.86272 w_k1_MAT_s103
                                 + 1.33507 w_k1_MSFT_s103
                                 + 1.09635 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.92306 w_k1_ADM_s103
                                 + 0.723338 w_k1_AXP_s103
                                 + 1.03033 w_k1_DIS_s103
                                 + 0.992926 w_k1_FDX_s103
                                 + 0.765771 w_k1_IBM_s103
                                 + 0.876999 w_k1_INTC_s103
                                 + 0.864775 w_k1_KO_s103
                                 + 0.904082 w_k1_MAT_s103
                                 + 0.994267 w_k1_MSFT_s103
                                 + 0.878449 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.15181 w_k1_ADM_s103
                                 + 0.983083 w_k1_AXP_s103
                                 + 1.03196 w_k1_DIS_s103
                                 + 1.02332 w_k1_FDX_s103 + 1.0453 w_k1_IBM_s103
                                 + 0.912184 w_k1_INTC_s103
                                 + 1.15741 w_k1_KO_s103 + 1.52178 w_k1_MAT_s103
                                 + 1.11747 w_k1_MSFT_s103
                                 + 0.98702 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.774073 w_k1_ADM_s103
                                 + 0.51724 w_k1_AXP_s103
                                 + 0.652518 w_k1_DIS_s103
                                 + 0.811542 w_k1_FDX_s103
                                 + 0.521842 w_k1_IBM_s103
                                 + 0.793181 w_k1_INTC_s103
                                 + 0.670307 w_k1_KO_s103
                                 + 0.595278 w_k1_MAT_s103
                                 + 0.849736 w_k1_MSFT_s103
                                 + 0.755887 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.995427 w_k1_ADM_s103
                                 + 1.09266 w_k1_AXP_s103
                                 + 1.18143 w_k1_DIS_s103
                                 + 1.23427 w_k1_FDX_s103
                                 + 0.723554 w_k1_IBM_s103
                                 + 0.983896 w_k1_INTC_s103
                                 + 0.894896 w_k1_KO_s103
                                 + 1.19409 w_k1_MAT_s103
                                 + 1.20045 w_k1_MSFT_s103
                                 + 0.988207 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.885745 w_k1_ADM_s103
                                 + 0.970741 w_k1_AXP_s103
                                 + 0.940319 w_k1_DIS_s103
                                 + 0.83244 w_k1_FDX_s103
                                 + 1.26018 w_k1_IBM_s103
                                 + 0.795258 w_k1_INTC_s103
                                 + 1.0515 w_k1_KO_s103 + 1.47217 w_k1_MAT_s103
                                 + 0.987623 w_k1_MSFT_s103
                                 + 0.913119 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.01488 w_k1_ADM_s104
                                 + 0.885682 w_k1_AXP_s104
                                 + 0.874299 w_k1_DIS_s104
                                 + 0.974124 w_k1_FDX_s104
                                 + 0.741151 w_k1_IBM_s104
                                 + 0.818792 w_k1_INTC_s104
                                 + 0.853748 w_k1_KO_s104
                                 + 0.865767 w_k1_MAT_s104
                                 + 0.955169 w_k1_MSFT_s104
                                 + 0.860693 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.857636 w_k1_ADM_s104
                                 + 0.630032 w_k1_AXP_s104
                                 + 0.696736 w_k1_DIS_s104
                                 + 0.749442 w_k1_FDX_s104
                                 + 0.880214 w_k1_IBM_s104
                                 + 0.735053 w_k1_INTC_s104
                                 + 0.907365 w_k1_KO_s104
                                 + 0.823135 w_k1_MAT_s104
                                 + 0.871459 w_k1_MSFT_s104
                                 + 0.877404 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.727182 w_k1_ADM_s104
                                 + 0.674792 w_k1_AXP_s104
                                 + 0.594628 w_k1_DIS_s104
                                 + 0.78238 w_k1_FDX_s104
                                 + 0.497427 w_k1_IBM_s104
                                 + 0.759946 w_k1_INTC_s104
                                 + 0.637106 w_k1_KO_s104
                                 + 0.440517 w_k1_MAT_s104
                                 + 0.746572 w_k1_MSFT_s104
                                 + 0.757982 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.02161 w_k1_ADM_s104
                                 + 0.775313 w_k1_AXP_s104
                                 + 0.727185 w_k1_DIS_s104
                                 + 0.8794 w_k1_FDX_s104 + 1.01151 w_k1_IBM_s104
                                 + 0.879525 w_k1_INTC_s104
                                 + 0.987249 w_k1_KO_s104
                                 + 1.10479 w_k1_MAT_s104
                                 + 0.895791 w_k1_MSFT_s104
                                 + 0.938136 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.37814 w_k1_ADM_s104
                                 + 1.2739 w_k1_AXP_s104 + 1.4118 w_k1_DIS_s104
                                 + 1.20424 w_k1_FDX_s104
                                 + 1.38061 w_k1_IBM_s104
                                 + 0.964489 w_k1_INTC_s104
                                 + 1.20078 w_k1_KO_s104 + 1.50214 w_k1_MAT_s104
                                 + 1.12541 w_k1_MSFT_s104
                                 + 1.19359 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.929206 w_k1_ADM_s104
                                 + 0.858991 w_k1_AXP_s104
                                 + 1.04242 w_k1_DIS_s104 + 1.1061 w_k1_FDX_s104
                                 + 0.648562 w_k1_IBM_s104
                                 + 0.852866 w_k1_INTC_s104
                                 + 0.950193 w_k1_KO_s104
                                 + 0.958145 w_k1_MAT_s104
                                 + 1.096 w_k1_MSFT_s104 + 1.00485 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.935877 w_k1_ADM_s104
                                 + 1.20851 w_k1_AXP_s104
                                 + 0.960936 w_k1_DIS_s104
                                 + 1.1393 w_k1_FDX_s104
                                 + 0.842238 w_k1_IBM_s104
                                 + 0.935748 w_k1_INTC_s104
                                 + 1.04147 w_k1_KO_s104 + 1.26594 w_k1_MAT_s104
                                 + 1.09562 w_k1_MSFT_s104
                                 + 1.05259 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.1772 w_k1_ADM_s104
                                 + 0.986643 w_k1_AXP_s104
                                 + 1.4002 w_k1_DIS_s104 + 1.18242 w_k1_FDX_s104
                                 + 1.09099 w_k1_IBM_s104
                                 + 1.03721 w_k1_INTC_s104
                                 + 1.46582 w_k1_KO_s104 + 1.11421 w_k1_MAT_s104
                                 + 1.26771 w_k1_MSFT_s104
                                 + 1.11554 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.25474 w_k1_ADM_s105
                                 + 1.44113 w_k1_AXP_s105
                                 + 1.50127 w_k1_DIS_s105
                                 + 1.29057 w_k1_FDX_s105 + 1.1891 w_k1_IBM_s105
                                 + 1.05405 w_k1_INTC_s105
                                 + 1.21165 w_k1_KO_s105 + 1.16437 w_k1_MAT_s105
                                 + 1.19345 w_k1_MSFT_s105
                                 + 1.02314 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.866873 w_k1_ADM_s105
                                 + 0.72529 w_k1_AXP_s105
                                 + 0.92252 w_k1_DIS_s105
                                 + 0.888037 w_k1_FDX_s105
                                 + 0.913825 w_k1_IBM_s105
                                 + 0.825247 w_k1_INTC_s105
                                 + 0.919703 w_k1_KO_s105
                                 + 0.720327 w_k1_MAT_s105
                                 + 0.986949 w_k1_MSFT_s105
                                 + 0.995485 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.825496 w_k1_ADM_s105
                                 + 0.929334 w_k1_AXP_s105
                                 + 0.707406 w_k1_DIS_s105
                                 + 0.779179 w_k1_FDX_s105
                                 + 1.12327 w_k1_IBM_s105
                                 + 0.806433 w_k1_INTC_s105
                                 + 0.831901 w_k1_KO_s105
                                 + 1.08465 w_k1_MAT_s105
                                 + 0.758145 w_k1_MSFT_s105
                                 + 0.830204 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.91683 w_k1_ADM_s105
                                 + 1.23777 w_k1_AXP_s105
                                 + 0.883081 w_k1_DIS_s105
                                 + 0.941272 w_k1_FDX_s105
                                 + 1.03094 w_k1_IBM_s105
                                 + 0.903715 w_k1_INTC_s105
                                 + 0.94948 w_k1_KO_s105 + 1.14848 w_k1_MAT_s105
                                 + 0.861784 w_k1_MSFT_s105
                                 + 1.04109 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.916394 w_k1_ADM_s105
                                 + 0.716431 w_k1_AXP_s105
                                 + 0.619491 w_k1_DIS_s105
                                 + 0.735239 w_k1_FDX_s105
                                 + 0.591654 w_k1_IBM_s105
                                 + 0.810071 w_k1_INTC_s105
                                 + 0.679796 w_k1_KO_s105
                                 + 0.473524 w_k1_MAT_s105
                                 + 0.751956 w_k1_MSFT_s105
                                 + 0.808038 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.955522 w_k1_ADM_s105
                                 + 0.838355 w_k1_AXP_s105
                                 + 1.42231 w_k1_DIS_s105
                                 + 1.01419 w_k1_FDX_s105
                                 + 1.03646 w_k1_IBM_s105
                                 + 0.934838 w_k1_INTC_s105
                                 + 1.26745 w_k1_KO_s105
                                 + 0.754946 w_k1_MAT_s105
                                 + 0.998347 w_k1_MSFT_s105
                                 + 1.07098 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.01108 w_k1_ADM_s105
                                 + 1.08239 w_k1_AXP_s105
                                 + 0.943918 w_k1_DIS_s105
                                 + 1.13979 w_k1_FDX_s105
                                 + 1.00091 w_k1_IBM_s105
                                 + 1.02757 w_k1_INTC_s105
                                 + 1.06328 w_k1_KO_s105 + 1.40812 w_k1_MAT_s105
                                 + 1.09442 w_k1_MSFT_s105
                                 + 1.05573 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.713988 w_k1_ADM_s105
                                 + 0.662107 w_k1_AXP_s105
                                 + 0.768278 w_k1_DIS_s105
                                 + 0.717812 w_k1_FDX_s105
                                 + 1.01213 w_k1_IBM_s105
                                 + 0.766985 w_k1_INTC_s105
                                 + 0.863273 w_k1_KO_s105
                                 + 0.818402 w_k1_MAT_s105
                                 + 0.670863 w_k1_MSFT_s105
                                 + 0.810867 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.40816 w_k1_ADM_s106
                                 + 1.6051 w_k1_AXP_s106 + 1.59489 w_k1_DIS_s106
                                 + 1.38457 w_k1_FDX_s106
                                 + 1.05537 w_k1_IBM_s106
                                 + 1.31922 w_k1_INTC_s106
                                 + 1.18182 w_k1_KO_s106 + 1.03559 w_k1_MAT_s106
                                 + 1.29048 w_k1_MSFT_s106
                                 + 1.29366 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.828144 w_k1_ADM_s106
                                 + 0.821698 w_k1_AXP_s106
                                 + 0.703802 w_k1_DIS_s106
                                 + 0.776044 w_k1_FDX_s106
                                 + 1.10756 w_k1_IBM_s106
                                 + 0.653246 w_k1_INTC_s106
                                 + 0.980234 w_k1_KO_s106
                                 + 1.32422 w_k1_MAT_s106
                                 + 0.823814 w_k1_MSFT_s106
                                 + 0.842348 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.941505 w_k1_ADM_s106
                                 + 1.07097 w_k1_AXP_s106
                                 + 1.03798 w_k1_DIS_s106
                                 + 0.991873 w_k1_FDX_s106
                                 + 1.43795 w_k1_IBM_s106
                                 + 1.03268 w_k1_INTC_s106
                                 + 1.25319 w_k1_KO_s106 + 1.55198 w_k1_MAT_s106
                                 + 1.28503 w_k1_MSFT_s106
                                 + 1.11105 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.838869 w_k1_ADM_s106
                                 + 0.901819 w_k1_AXP_s106
                                 + 0.694155 w_k1_DIS_s106
                                 + 0.807822 w_k1_FDX_s106
                                 + 0.719745 w_k1_IBM_s106
                                 + 0.787009 w_k1_INTC_s106
                                 + 0.720994 w_k1_KO_s106
                                 + 0.656043 w_k1_MAT_s106
                                 + 0.809644 w_k1_MSFT_s106
                                 + 0.900105 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.12267 w_k1_ADM_s106
                                 + 1.13135 w_k1_AXP_s106
                                 + 1.05392 w_k1_DIS_s106 + 1.1551 w_k1_FDX_s106
                                 + 0.669899 w_k1_IBM_s106
                                 + 1.22192 w_k1_INTC_s106
                                 + 0.948822 w_k1_KO_s106
                                 + 0.616607 w_k1_MAT_s106
                                 + 1.21338 w_k1_MSFT_s106
                                 + 0.975714 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.968068 w_k1_ADM_s106
                                 + 0.931344 w_k1_AXP_s106
                                 + 0.917321 w_k1_DIS_s106
                                 + 0.909069 w_k1_FDX_s106
                                 + 1.11309 w_k1_IBM_s106
                                 + 0.8226 w_k1_INTC_s106 + 1.02628 w_k1_KO_s106
                                 + 1.29389 w_k1_MAT_s106
                                 + 1.02012 w_k1_MSFT_s106
                                 + 1.06952 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.06577 w_k1_ADM_s106
                                 + 1.23681 w_k1_AXP_s106
                                 + 1.02552 w_k1_DIS_s106
                                 + 1.06264 w_k1_FDX_s106
                                 + 0.867734 w_k1_IBM_s106
                                 + 0.830965 w_k1_INTC_s106
                                 + 0.940811 w_k1_KO_s106
                                 + 0.995679 w_k1_MAT_s106
                                 + 1.03362 w_k1_MSFT_s106
                                 + 0.980298 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.43515 w_k1_ADM_s106
                                 + 1.20279 w_k1_AXP_s106
                                 + 1.38509 w_k1_DIS_s106
                                 + 1.13542 w_k1_FDX_s106
                                 + 1.22121 w_k1_IBM_s106
                                 + 1.01421 w_k1_INTC_s106
                                 + 1.27421 w_k1_KO_s106 + 1.29207 w_k1_MAT_s106
                                 + 1.30742 w_k1_MSFT_s106
                                 + 1.13303 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.894721 w_k1_ADM_s107
                                 + 1.30402 w_k1_AXP_s107
                                 + 1.07038 w_k1_DIS_s107
                                 + 1.14467 w_k1_FDX_s107
                                 + 1.05758 w_k1_IBM_s107
                                 + 1.14053 w_k1_INTC_s107
                                 + 0.949884 w_k1_KO_s107
                                 + 1.22581 w_k1_MAT_s107
                                 + 0.98129 w_k1_MSFT_s107
                                 + 1.23072 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.05869 w_k1_ADM_s107
                                 + 1.41171 w_k1_AXP_s107 + 1.3234 w_k1_DIS_s107
                                 + 1.358 w_k1_FDX_s107 + 0.940642 w_k1_IBM_s107
                                 + 1.26075 w_k1_INTC_s107
                                 + 0.931465 w_k1_KO_s107
                                 + 0.776703 w_k1_MAT_s107
                                 + 1.26679 w_k1_MSFT_s107
                                 + 1.15711 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.618811 w_k1_ADM_s107
                                 + 0.919994 w_k1_AXP_s107
                                 + 0.988539 w_k1_DIS_s107
                                 + 0.865787 w_k1_FDX_s107
                                 + 1.80059 w_k1_IBM_s107
                                 + 0.940101 w_k1_INTC_s107
                                 + 1.15312 w_k1_KO_s107 + 1.81226 w_k1_MAT_s107
                                 + 0.887626 w_k1_MSFT_s107
                                 + 1.21879 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.1142 w_k1_ADM_s107
                                 + 1.1013 w_k1_AXP_s107 + 1.5644 w_k1_DIS_s107
                                 + 1.21203 w_k1_FDX_s107
                                 + 0.962329 w_k1_IBM_s107
                                 + 1.1703 w_k1_INTC_s107
                                 + 0.945653 w_k1_KO_s107
                                 + 0.707695 w_k1_MAT_s107
                                 + 1.13796 w_k1_MSFT_s107
                                 + 1.15275 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.946478 w_k1_ADM_s107
                                 + 1.11892 w_k1_AXP_s107
                                 + 0.825311 w_k1_DIS_s107
                                 + 1.04814 w_k1_FDX_s107
                                 + 0.89581 w_k1_IBM_s107
                                 + 0.934459 w_k1_INTC_s107
                                 + 0.71712 w_k1_KO_s107 + 1.00774 w_k1_MAT_s107
                                 + 0.885383 w_k1_MSFT_s107
                                 + 0.930289 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.988091 w_k1_ADM_s107
                                 + 1.36462 w_k1_AXP_s107
                                 + 1.48169 w_k1_DIS_s107
                                 + 1.18185 w_k1_FDX_s107
                                 + 1.41451 w_k1_IBM_s107
                                 + 1.12235 w_k1_INTC_s107
                                 + 1.33002 w_k1_KO_s107 + 1.76898 w_k1_MAT_s107
                                 + 1.08384 w_k1_MSFT_s107
                                 + 1.28954 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.921525 w_k1_ADM_s107
                                 + 0.94801 w_k1_AXP_s107
                                 + 0.875774 w_k1_DIS_s107
                                 + 0.877965 w_k1_FDX_s107
                                 + 1.13689 w_k1_IBM_s107
                                 + 0.831206 w_k1_INTC_s107
                                 + 0.901252 w_k1_KO_s107
                                 + 1.13535 w_k1_MAT_s107
                                 + 0.845074 w_k1_MSFT_s107
                                 + 1.03228 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.664647 w_k1_ADM_s107
                                 + 0.669816 w_k1_AXP_s107
                                 + 0.778809 w_k1_DIS_s107
                                 + 0.873615 w_k1_FDX_s107
                                 + 0.72043 w_k1_IBM_s107
                                 + 0.893294 w_k1_INTC_s107
                                 + 0.70789 w_k1_KO_s107
                                 + 0.549146 w_k1_MAT_s107
                                 + 0.934901 w_k1_MSFT_s107
                                 + 0.997017 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.12895 w_k1_ADM_s110
                                 + 0.81804 w_k1_AXP_s110
                                 + 1.06854 w_k1_DIS_s110
                                 + 1.06962 w_k1_FDX_s110
                                 + 0.759286 w_k1_IBM_s110
                                 + 1.14941 w_k1_INTC_s110
                                 + 0.948348 w_k1_KO_s110
                                 + 0.73408 w_k1_MAT_s110
                                 + 1.15353 w_k1_MSFT_s110
                                 + 0.961101 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.869712 w_k1_ADM_s110
                                 + 0.80409 w_k1_AXP_s110
                                 + 0.762481 w_k1_DIS_s110
                                 + 0.857132 w_k1_FDX_s110
                                 + 1.18668 w_k1_IBM_s110
                                 + 0.90436 w_k1_INTC_s110
                                 + 0.990308 w_k1_KO_s110
                                 + 0.877196 w_k1_MAT_s110
                                 + 0.878201 w_k1_MSFT_s110
                                 + 0.989204 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.986497 w_k1_ADM_s110
                                 + 0.732294 w_k1_AXP_s110
                                 + 1.09645 w_k1_DIS_s110
                                 + 0.88687 w_k1_FDX_s110
                                 + 0.968492 w_k1_IBM_s110
                                 + 0.938298 w_k1_INTC_s110
                                 + 1.04146 w_k1_KO_s110
                                 + 0.807033 w_k1_MAT_s110
                                 + 1.03832 w_k1_MSFT_s110
                                 + 1.00386 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.21251 w_k1_ADM_s110
                                 + 1.13305 w_k1_AXP_s110
                                 + 1.19166 w_k1_DIS_s110
                                 + 1.01225 w_k1_FDX_s110
                                 + 1.64331 w_k1_IBM_s110
                                 + 1.13707 w_k1_INTC_s110
                                 + 1.51532 w_k1_KO_s110 + 1.53503 w_k1_MAT_s110
                                 + 1.23507 w_k1_MSFT_s110
                                 + 1.17919 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.887638 w_k1_ADM_s110
                                 + 0.643614 w_k1_AXP_s110
                                 + 0.670074 w_k1_DIS_s110
                                 + 0.821632 w_k1_FDX_s110
                                 + 0.79224 w_k1_IBM_s110
                                 + 0.787705 w_k1_INTC_s110
                                 + 0.780679 w_k1_KO_s110
                                 + 0.676455 w_k1_MAT_s110
                                 + 0.888794 w_k1_MSFT_s110
                                 + 0.865268 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.06756 w_k1_ADM_s110
                                 + 1.00258 w_k1_AXP_s110
                                 + 0.69078 w_k1_DIS_s110
                                 + 0.977406 w_k1_FDX_s110
                                 + 0.687851 w_k1_IBM_s110
                                 + 0.946014 w_k1_INTC_s110
                                 + 0.750636 w_k1_KO_s110
                                 + 0.740687 w_k1_MAT_s110
                                 + 0.969004 w_k1_MSFT_s110
                                 + 0.910092 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.36067 w_k1_ADM_s110
                                 + 1.22087 w_k1_AXP_s110
                                 + 1.14531 w_k1_DIS_s110 + 1.1454 w_k1_FDX_s110
                                 + 1.18975 w_k1_IBM_s110
                                 + 1.04226 w_k1_INTC_s110
                                 + 1.18921 w_k1_KO_s110
                                 + 0.959997 w_k1_MAT_s110
                                 + 1.05528 w_k1_MSFT_s110
                                 + 1.05943 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.50258 w_k1_ADM_s110
                                 + 1.466 w_k1_AXP_s110 + 1.55396 w_k1_DIS_s110
                                 + 1.46702 w_k1_FDX_s110
                                 + 1.22793 w_k1_IBM_s110
                                 + 1.30039 w_k1_INTC_s110
                                 + 1.40888 w_k1_KO_s110 + 1.26015 w_k1_MAT_s110
                                 + 1.35425 w_k1_MSFT_s110
                                 + 1.45439 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.743773 w_k1_ADM_s111
                                 + 0.810304 w_k1_AXP_s111
                                 + 0.826358 w_k1_DIS_s111
                                 + 0.865539 w_k1_FDX_s111
                                 + 0.821493 w_k1_IBM_s111
                                 + 0.824807 w_k1_INTC_s111
                                 + 0.820732 w_k1_KO_s111
                                 + 0.570186 w_k1_MAT_s111
                                 + 0.880936 w_k1_MSFT_s111
                                 + 0.922356 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.974408 w_k1_ADM_s111
                                 + 1.14528 w_k1_AXP_s111
                                 + 1.06087 w_k1_DIS_s111
                                 + 1.05633 w_k1_FDX_s111 + 1.033 w_k1_IBM_s111
                                 + 0.995228 w_k1_INTC_s111
                                 + 1.04969 w_k1_KO_s111
                                 + 0.895318 w_k1_MAT_s111
                                 + 0.900679 w_k1_MSFT_s111
                                 + 1.19127 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.47176 w_k1_ADM_s111
                                 + 1.88425 w_k1_AXP_s111
                                 + 1.77664 w_k1_DIS_s111
                                 + 1.38419 w_k1_FDX_s111
                                 + 1.42644 w_k1_IBM_s111
                                 + 1.39178 w_k1_INTC_s111
                                 + 1.40675 w_k1_KO_s111 + 1.49095 w_k1_MAT_s111
                                 + 1.26374 w_k1_MSFT_s111
                                 + 1.27447 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.06421 w_k1_ADM_s111
                                 + 1.16238 w_k1_AXP_s111
                                 + 1.06545 w_k1_DIS_s111
                                 + 1.16622 w_k1_FDX_s111
                                 + 0.655529 w_k1_IBM_s111
                                 + 1.17355 w_k1_INTC_s111
                                 + 0.923095 w_k1_KO_s111
                                 + 0.75436 w_k1_MAT_s111
                                 + 1.11051 w_k1_MSFT_s111
                                 + 1.07637 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.18161 w_k1_ADM_s111
                                 + 1.28993 w_k1_AXP_s111
                                 + 1.37511 w_k1_DIS_s111
                                 + 1.34061 w_k1_FDX_s111
                                 + 1.08107 w_k1_IBM_s111
                                 + 1.0304 w_k1_INTC_s111 + 1.24138 w_k1_KO_s111
                                 + 1.12507 w_k1_MAT_s111
                                 + 1.10168 w_k1_MSFT_s111
                                 + 1.18101 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.767102 w_k1_ADM_s111
                                 + 0.762149 w_k1_AXP_s111
                                 + 0.657801 w_k1_DIS_s111
                                 + 0.791515 w_k1_FDX_s111
                                 + 0.644765 w_k1_IBM_s111
                                 + 0.734039 w_k1_INTC_s111
                                 + 0.699223 w_k1_KO_s111
                                 + 0.634691 w_k1_MAT_s111
                                 + 0.756305 w_k1_MSFT_s111
                                 + 0.825649 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.755471 w_k1_ADM_s111
                                 + 0.780676 w_k1_AXP_s111
                                 + 0.715698 w_k1_DIS_s111
                                 + 0.79482 w_k1_FDX_s111
                                 + 1.05917 w_k1_IBM_s111
                                 + 0.969775 w_k1_INTC_s111
                                 + 0.936885 w_k1_KO_s111
                                 + 0.979553 w_k1_MAT_s111
                                 + 0.837485 w_k1_MSFT_s111
                                 + 0.926812 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.901621 w_k1_ADM_s111
                                 + 1.12161 w_k1_AXP_s111
                                 + 1.02345 w_k1_DIS_s111
                                 + 0.864059 w_k1_FDX_s111
                                 + 1.56389 w_k1_IBM_s111
                                 + 0.967305 w_k1_INTC_s111
                                 + 1.17082 w_k1_KO_s111 + 1.2695 w_k1_MAT_s111
                                 + 0.904803 w_k1_MSFT_s111
                                 + 0.951148 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.788475 w_k1_ADM_s112
                                 + 0.81985 w_k1_AXP_s112
                                 + 0.667562 w_k1_DIS_s112
                                 + 0.785288 w_k1_FDX_s112
                                 + 0.914151 w_k1_IBM_s112
                                 + 0.692535 w_k1_INTC_s112
                                 + 0.881232 w_k1_KO_s112
                                 + 0.895746 w_k1_MAT_s112
                                 + 0.772166 w_k1_MSFT_s112
                                 + 0.926681 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.41263 w_k1_ADM_s112
                                 + 1.04771 w_k1_AXP_s112
                                 + 1.14482 w_k1_DIS_s112
                                 + 1.32475 w_k1_FDX_s112
                                 + 0.74285 w_k1_IBM_s112
                                 + 0.948416 w_k1_INTC_s112
                                 + 0.962425 w_k1_KO_s112
                                 + 0.604789 w_k1_MAT_s112
                                 + 1.27648 w_k1_MSFT_s112
                                 + 1.11553 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.5452 w_k1_ADM_s112
                                 + 1.20621 w_k1_AXP_s112
                                 + 1.63015 w_k1_DIS_s112
                                 + 1.44382 w_k1_FDX_s112
                                 + 1.25615 w_k1_IBM_s112
                                 + 1.05783 w_k1_INTC_s112 + 1.2456 w_k1_KO_s112
                                 + 1.07916 w_k1_MAT_s112
                                 + 1.11444 w_k1_MSFT_s112
                                 + 1.18534 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.705201 w_k1_ADM_s112
                                 + 0.909901 w_k1_AXP_s112
                                 + 0.905345 w_k1_DIS_s112
                                 + 0.989186 w_k1_FDX_s112
                                 + 0.72174 w_k1_IBM_s112
                                 + 0.768319 w_k1_INTC_s112
                                 + 0.864631 w_k1_KO_s112
                                 + 0.683939 w_k1_MAT_s112
                                 + 0.887366 w_k1_MSFT_s112
                                 + 0.978984 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.6921 w_k1_ADM_s112
                                 + 0.604635 w_k1_AXP_s112
                                 + 0.383365 w_k1_DIS_s112
                                 + 0.615458 w_k1_FDX_s112
                                 + 0.739941 w_k1_IBM_s112
                                 + 0.580057 w_k1_INTC_s112
                                 + 0.53619 w_k1_KO_s112
                                 + 0.625925 w_k1_MAT_s112
                                 + 0.533536 w_k1_MSFT_s112
                                 + 0.680531 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.74446 w_k1_ADM_s112
                                 + 0.765265 w_k1_AXP_s112
                                 + 0.754949 w_k1_DIS_s112
                                 + 0.731099 w_k1_FDX_s112
                                 + 1.33526 w_k1_IBM_s112
                                 + 0.628344 w_k1_INTC_s112
                                 + 0.97927 w_k1_KO_s112 + 1.41878 w_k1_MAT_s112
                                 + 0.742547 w_k1_MSFT_s112
                                 + 0.908778 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.41714 w_k1_ADM_s112
                                 + 1.61723 w_k1_AXP_s112
                                 + 1.27997 w_k1_DIS_s112
                                 + 1.37934 w_k1_FDX_s112
                                 + 1.04287 w_k1_IBM_s112
                                 + 0.839968 w_k1_INTC_s112
                                 + 0.982093 w_k1_KO_s112
                                 + 1.19156 w_k1_MAT_s112
                                 + 0.976172 w_k1_MSFT_s112
                                 + 1.10904 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.04273 w_k1_ADM_s112
                                 + 1.13339 w_k1_AXP_s112
                                 + 0.814956 w_k1_DIS_s112
                                 + 1.05668 w_k1_FDX_s112
                                 + 1.11868 w_k1_IBM_s112
                                 + 1.05844 w_k1_INTC_s112
                                 + 0.933051 w_k1_KO_s112
                                 + 1.37858 w_k1_MAT_s112
                                 + 0.939846 w_k1_MSFT_s112
                                 + 0.988254 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.08406 w_k1_ADM_s113
                                 + 1.11287 w_k1_AXP_s113
                                 + 0.994165 w_k1_DIS_s113
                                 + 0.998617 w_k1_FDX_s113
                                 + 1.22621 w_k1_IBM_s113
                                 + 0.795012 w_k1_INTC_s113
                                 + 1.02415 w_k1_KO_s113 + 1.2106 w_k1_MAT_s113
                                 + 0.883837 w_k1_MSFT_s113
                                 + 1.05027 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.640171 w_k1_ADM_s113
                                 + 0.599178 w_k1_AXP_s113
                                 + 0.752305 w_k1_DIS_s113
                                 + 0.789872 w_k1_FDX_s113
                                 + 0.957033 w_k1_IBM_s113
                                 + 0.955653 w_k1_INTC_s113
                                 + 1.03536 w_k1_KO_s113
                                 + 0.798682 w_k1_MAT_s113
                                 + 0.877479 w_k1_MSFT_s113
                                 + 0.914545 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.07824 w_k1_ADM_s113
                                 + 0.793784 w_k1_AXP_s113
                                 + 0.688894 w_k1_DIS_s113
                                 + 0.765819 w_k1_FDX_s113
                                 + 1.48241 w_k1_IBM_s113
                                 + 0.747962 w_k1_INTC_s113
                                 + 0.983827 w_k1_KO_s113
                                 + 1.44763 w_k1_MAT_s113
                                 + 0.667822 w_k1_MSFT_s113
                                 + 0.881667 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.16017 w_k1_ADM_s113
                                 + 0.937036 w_k1_AXP_s113
                                 + 1.32559 w_k1_DIS_s113
                                 + 1.23478 w_k1_FDX_s113
                                 + 0.849811 w_k1_IBM_s113
                                 + 1.15892 w_k1_INTC_s113
                                 + 1.21528 w_k1_KO_s113
                                 + 0.773452 w_k1_MAT_s113
                                 + 1.129 w_k1_MSFT_s113 + 1.08512 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.922909 w_k1_ADM_s113
                                 + 0.94985 w_k1_AXP_s113
                                 + 0.806367 w_k1_DIS_s113
                                 + 0.925389 w_k1_FDX_s113
                                 + 0.984939 w_k1_IBM_s113
                                 + 0.939005 w_k1_INTC_s113
                                 + 0.817257 w_k1_KO_s113
                                 + 0.903391 w_k1_MAT_s113
                                 + 0.736231 w_k1_MSFT_s113
                                 + 0.899384 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.98388 w_k1_ADM_s113
                                 + 1.16565 w_k1_AXP_s113
                                 + 1.36652 w_k1_DIS_s113
                                 + 1.20606 w_k1_FDX_s113 + 1.2957 w_k1_IBM_s113
                                 + 1.11245 w_k1_INTC_s113
                                 + 1.37125 w_k1_KO_s113 + 1.38818 w_k1_MAT_s113
                                 + 1.10169 w_k1_MSFT_s113
                                 + 1.29335 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.925285 w_k1_ADM_s113
                                 + 0.693293 w_k1_AXP_s113
                                 + 0.784278 w_k1_DIS_s113
                                 + 0.870499 w_k1_FDX_s113
                                 + 0.722178 w_k1_IBM_s113
                                 + 0.953666 w_k1_INTC_s113
                                 + 0.81905 w_k1_KO_s113
                                 + 0.666839 w_k1_MAT_s113
                                 + 0.913757 w_k1_MSFT_s113
                                 + 0.885008 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.27072 w_k1_ADM_s113
                                 + 1.22637 w_k1_AXP_s113
                                 + 1.01605 w_k1_DIS_s113
                                 + 1.16732 w_k1_FDX_s113
                                 + 0.969559 w_k1_IBM_s113
                                 + 1.03915 w_k1_INTC_s113
                                 + 0.917499 w_k1_KO_s113
                                 + 0.750331 w_k1_MAT_s113
                                 + 1.0686 w_k1_MSFT_s113 + 1.11126 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.796877 w_k1_ADM_s114
                                 + 0.81602 w_k1_AXP_s114
                                 + 0.894556 w_k1_DIS_s114
                                 + 0.856933 w_k1_FDX_s114
                                 + 1.10568 w_k1_IBM_s114
                                 + 0.979735 w_k1_INTC_s114
                                 + 1.05855 w_k1_KO_s114 + 1.15462 w_k1_MAT_s114
                                 + 0.893886 w_k1_MSFT_s114
                                 + 0.998953 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.930315 w_k1_ADM_s114
                                 + 0.925723 w_k1_AXP_s114
                                 + 0.923991 w_k1_DIS_s114
                                 + 1.04474 w_k1_FDX_s114
                                 + 0.792992 w_k1_IBM_s114
                                 + 0.852619 w_k1_INTC_s114
                                 + 0.847362 w_k1_KO_s114
                                 + 0.83904 w_k1_MAT_s114
                                 + 0.916942 w_k1_MSFT_s114
                                 + 1.04336 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.804882 w_k1_ADM_s114
                                 + 0.70426 w_k1_AXP_s114
                                 + 0.723223 w_k1_DIS_s114
                                 + 0.810364 w_k1_FDX_s114
                                 + 0.751117 w_k1_IBM_s114
                                 + 0.765653 w_k1_INTC_s114
                                 + 0.833165 w_k1_KO_s114
                                 + 0.886769 w_k1_MAT_s114
                                 + 0.902231 w_k1_MSFT_s114
                                 + 0.922467 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.29752 w_k1_ADM_s114
                                 + 1.50508 w_k1_AXP_s114
                                 + 1.28314 w_k1_DIS_s114
                                 + 1.39943 w_k1_FDX_s114
                                 + 0.850973 w_k1_IBM_s114
                                 + 1.20557 w_k1_INTC_s114
                                 + 1.16567 w_k1_KO_s114 + 1.31126 w_k1_MAT_s114
                                 + 1.28798 w_k1_MSFT_s114
                                 + 1.19776 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.0977 w_k1_ADM_s114
                                 + 1.42903 w_k1_AXP_s114
                                 + 1.67306 w_k1_DIS_s114
                                 + 1.31261 w_k1_FDX_s114
                                 + 1.42368 w_k1_IBM_s114
                                 + 1.18504 w_k1_INTC_s114
                                 + 1.62599 w_k1_KO_s114 + 1.84961 w_k1_MAT_s114
                                 + 1.42916 w_k1_MSFT_s114
                                 + 1.43201 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.23336 w_k1_ADM_s114
                                 + 1.12722 w_k1_AXP_s114
                                 + 1.13198 w_k1_DIS_s114
                                 + 1.23157 w_k1_FDX_s114
                                 + 0.746445 w_k1_IBM_s114
                                 + 1.12955 w_k1_INTC_s114
                                 + 0.921227 w_k1_KO_s114
                                 + 0.856125 w_k1_MAT_s114
                                 + 1.08638 w_k1_MSFT_s114
                                 + 0.968556 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.981203 w_k1_ADM_s114
                                 + 1.25338 w_k1_AXP_s114
                                 + 1.12841 w_k1_DIS_s114
                                 + 1.08688 w_k1_FDX_s114
                                 + 1.15404 w_k1_IBM_s114
                                 + 1.04143 w_k1_INTC_s114 + 1.1743 w_k1_KO_s114
                                 + 1.11813 w_k1_MAT_s114
                                 + 1.14251 w_k1_MSFT_s114
                                 + 1.16121 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.974752 w_k1_ADM_s114
                                 + 1.08135 w_k1_AXP_s114
                                 + 1.03307 w_k1_DIS_s114
                                 + 0.974148 w_k1_FDX_s114
                                 + 1.41369 w_k1_IBM_s114
                                 + 0.973555 w_k1_INTC_s114
                                 + 1.23214 w_k1_KO_s114 + 1.19982 w_k1_MAT_s114
                                 + 0.987359 w_k1_MSFT_s114
                                 + 1.04818 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.0556 w_k1_ADM_s115
                                 + 0.957608 w_k1_AXP_s115
                                 + 1.20804 w_k1_DIS_s115
                                 + 1.10685 w_k1_FDX_s115
                                 + 0.88788 w_k1_IBM_s115
                                 + 0.943053 w_k1_INTC_s115
                                 + 1.0714 w_k1_KO_s115 + 1.03022 w_k1_MAT_s115
                                 + 1.20093 w_k1_MSFT_s115
                                 + 1.01293 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.24236 w_k1_ADM_s115
                                 + 1.31871 w_k1_AXP_s115 + 1.445 w_k1_DIS_s115
                                 + 1.09657 w_k1_FDX_s115
                                 + 1.68562 w_k1_IBM_s115
                                 + 1.01942 w_k1_INTC_s115
                                 + 1.25082 w_k1_KO_s115 + 1.21815 w_k1_MAT_s115
                                 + 1.10951 w_k1_MSFT_s115
                                 + 1.16186 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.934694 w_k1_ADM_s115
                                 + 0.748956 w_k1_AXP_s115
                                 + 0.901863 w_k1_DIS_s115
                                 + 0.911699 w_k1_FDX_s115
                                 + 0.677445 w_k1_IBM_s115
                                 + 0.889387 w_k1_INTC_s115
                                 + 0.892105 w_k1_KO_s115
                                 + 0.480556 w_k1_MAT_s115
                                 + 0.896147 w_k1_MSFT_s115
                                 + 0.840693 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.12082 w_k1_ADM_s115
                                 + 1.22086 w_k1_AXP_s115
                                 + 0.997085 w_k1_DIS_s115
                                 + 1.07889 w_k1_FDX_s115
                                 + 1.01527 w_k1_IBM_s115
                                 + 0.986486 w_k1_INTC_s115
                                 + 1.11321 w_k1_KO_s115 + 0.93821 w_k1_MAT_s115
                                 + 1.0215 w_k1_MSFT_s115 + 0.9953 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.757425 w_k1_ADM_s115
                                 + 0.569971 w_k1_AXP_s115
                                 + 0.628414 w_k1_DIS_s115
                                 + 0.706505 w_k1_FDX_s115
                                 + 0.710565 w_k1_IBM_s115
                                 + 0.678321 w_k1_INTC_s115
                                 + 0.628305 w_k1_KO_s115
                                 + 0.312588 w_k1_MAT_s115
                                 + 0.680965 w_k1_MSFT_s115
                                 + 0.634569 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.13987 w_k1_ADM_s115
                                 + 1.15524 w_k1_AXP_s115
                                 + 1.41127 w_k1_DIS_s115
                                 + 1.29979 w_k1_FDX_s115
                                 + 1.01945 w_k1_IBM_s115
                                 + 1.14099 w_k1_INTC_s115
                                 + 1.07995 w_k1_KO_s115
                                 + 0.649529 w_k1_MAT_s115
                                 + 1.15882 w_k1_MSFT_s115
                                 + 0.961615 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.675297 w_k1_ADM_s115
                                 + 0.972115 w_k1_AXP_s115
                                 + 1.10736 w_k1_DIS_s115
                                 + 0.896368 w_k1_FDX_s115
                                 + 1.33115 w_k1_IBM_s115
                                 + 0.918841 w_k1_INTC_s115
                                 + 1.12668 w_k1_KO_s115 + 1.38558 w_k1_MAT_s115
                                 + 1.02542 w_k1_MSFT_s115
                                 + 0.969238 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.681291 w_k1_ADM_s115
                                 + 0.851428 w_k1_AXP_s115
                                 + 0.937164 w_k1_DIS_s115
                                 + 0.822229 w_k1_FDX_s115
                                 + 0.973092 w_k1_IBM_s115
                                 + 0.898834 w_k1_INTC_s115
                                 + 0.976997 w_k1_KO_s115
                                 + 0.867954 w_k1_MAT_s115
                                 + 0.914042 w_k1_MSFT_s115
                                 + 0.939562 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.14111 w_k1_ADM_s116
                                 + 1.19412 w_k1_AXP_s116
                                 + 0.88771 w_k1_DIS_s116
                                 + 1.12002 w_k1_FDX_s116
                                 + 0.841637 w_k1_IBM_s116
                                 + 1.14217 w_k1_INTC_s116
                                 + 0.876919 w_k1_KO_s116
                                 + 0.97152 w_k1_MAT_s116
                                 + 0.976974 w_k1_MSFT_s116
                                 + 1.03292 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.858588 w_k1_ADM_s116
                                 + 0.783561 w_k1_AXP_s116
                                 + 0.769119 w_k1_DIS_s116
                                 + 0.819706 w_k1_FDX_s116
                                 + 1.03646 w_k1_IBM_s116
                                 + 0.708481 w_k1_INTC_s116
                                 + 0.773433 w_k1_KO_s116
                                 + 0.803549 w_k1_MAT_s116
                                 + 0.792837 w_k1_MSFT_s116
                                 + 0.970921 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.93435 w_k1_ADM_s116
                                 + 0.921561 w_k1_AXP_s116
                                 + 0.559572 w_k1_DIS_s116
                                 + 0.852072 w_k1_FDX_s116
                                 + 0.79753 w_k1_IBM_s116
                                 + 0.812424 w_k1_INTC_s116
                                 + 0.652947 w_k1_KO_s116
                                 + 0.819259 w_k1_MAT_s116
                                 + 0.770648 w_k1_MSFT_s116
                                 + 0.912862 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.07207 w_k1_ADM_s116
                                 + 0.879119 w_k1_AXP_s116
                                 + 0.938721 w_k1_DIS_s116
                                 + 0.993521 w_k1_FDX_s116
                                 + 0.808405 w_k1_IBM_s116
                                 + 0.834713 w_k1_INTC_s116
                                 + 0.986662 w_k1_KO_s116
                                 + 0.671252 w_k1_MAT_s116
                                 + 0.917346 w_k1_MSFT_s116
                                 + 0.970751 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.33508 w_k1_ADM_s116
                                 + 1.53691 w_k1_AXP_s116
                                 + 1.37249 w_k1_DIS_s116
                                 + 1.41245 w_k1_FDX_s116
                                 + 0.984163 w_k1_IBM_s116
                                 + 1.40506 w_k1_INTC_s116
                                 + 1.12398 w_k1_KO_s116 + 1.14937 w_k1_MAT_s116
                                 + 1.23145 w_k1_MSFT_s116
                                 + 1.22527 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.1924 w_k1_ADM_s116
                                 + 1.34249 w_k1_AXP_s116
                                 + 1.37474 w_k1_DIS_s116
                                 + 1.13553 w_k1_FDX_s116
                                 + 1.63911 w_k1_IBM_s116
                                 + 1.19686 w_k1_INTC_s116
                                 + 1.30007 w_k1_KO_s116 + 1.30037 w_k1_MAT_s116
                                 + 0.865149 w_k1_MSFT_s116
                                 + 1.08579 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.969202 w_k1_ADM_s116
                                 + 0.721932 w_k1_AXP_s116
                                 + 0.545904 w_k1_DIS_s116
                                 + 0.893179 w_k1_FDX_s116
                                 + 0.593313 w_k1_IBM_s116
                                 + 0.862849 w_k1_INTC_s116
                                 + 0.578382 w_k1_KO_s116
                                 + 0.472359 w_k1_MAT_s116
                                 + 0.760816 w_k1_MSFT_s116
                                 + 0.876712 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.0842 w_k1_ADM_s116
                                 + 1.0202 w_k1_AXP_s116
                                 + 0.932202 w_k1_DIS_s116
                                 + 0.991865 w_k1_FDX_s116
                                 + 1.26783 w_k1_IBM_s116
                                 + 1.14537 w_k1_INTC_s116
                                 + 1.14584 w_k1_KO_s116 + 1.32983 w_k1_MAT_s116
                                 + 0.96104 w_k1_MSFT_s116
                                 + 1.04898 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.82022 w_k1_ADM_s117
                                 + 0.887887 w_k1_AXP_s117
                                 + 0.65551 w_k1_DIS_s117
                                 + 0.782714 w_k1_FDX_s117
                                 + 1.06194 w_k1_IBM_s117
                                 + 0.697296 w_k1_INTC_s117
                                 + 0.814189 w_k1_KO_s117
                                 + 1.12345 w_k1_MAT_s117
                                 + 0.747318 w_k1_MSFT_s117
                                 + 1.01806 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.15377 w_k1_ADM_s117
                                 + 0.873441 w_k1_AXP_s117
                                 + 0.791492 w_k1_DIS_s117
                                 + 0.888594 w_k1_FDX_s117
                                 + 0.760107 w_k1_IBM_s117
                                 + 0.802113 w_k1_INTC_s117
                                 + 0.804939 w_k1_KO_s117
                                 + 0.78521 w_k1_MAT_s117
                                 + 0.895135 w_k1_MSFT_s117
                                 + 1.02874 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.13482 w_k1_ADM_s117
                                 + 1.24424 w_k1_AXP_s117
                                 + 1.29915 w_k1_DIS_s117
                                 + 1.13441 w_k1_FDX_s117
                                 + 1.46429 w_k1_IBM_s117
                                 + 1.11899 w_k1_INTC_s117
                                 + 1.46057 w_k1_KO_s117 + 1.45066 w_k1_MAT_s117
                                 + 1.31376 w_k1_MSFT_s117 + 1.2493 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.596992 w_k1_ADM_s117
                                 + 0.66982 w_k1_AXP_s117
                                 + 0.686584 w_k1_DIS_s117
                                 + 0.724341 w_k1_FDX_s117
                                 + 0.714524 w_k1_IBM_s117
                                 + 0.773542 w_k1_INTC_s117
                                 + 0.674345 w_k1_KO_s117
                                 + 0.772842 w_k1_MAT_s117
                                 + 0.852472 w_k1_MSFT_s117
                                 + 0.930989 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.3136 w_k1_ADM_s117
                                 + 1.08608 w_k1_AXP_s117
                                 + 1.09627 w_k1_DIS_s117
                                 + 1.09477 w_k1_FDX_s117
                                 + 0.776566 w_k1_IBM_s117
                                 + 0.98907 w_k1_INTC_s117
                                 + 1.08926 w_k1_KO_s117
                                 + 0.829403 w_k1_MAT_s117
                                 + 1.06554 w_k1_MSFT_s117
                                 + 1.12348 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.978192 w_k1_ADM_s117
                                 + 0.997131 w_k1_AXP_s117
                                 + 0.981921 w_k1_DIS_s117
                                 + 0.872784 w_k1_FDX_s117
                                 + 1.3521 w_k1_IBM_s117
                                 + 0.750185 w_k1_INTC_s117
                                 + 1.00723 w_k1_KO_s117 + 1.36195 w_k1_MAT_s117
                                 + 0.973487 w_k1_MSFT_s117
                                 + 0.973087 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.1027 w_k1_ADM_s117
                                 + 0.996023 w_k1_AXP_s117
                                 + 0.963371 w_k1_DIS_s117
                                 + 0.948164 w_k1_FDX_s117
                                 + 1.04745 w_k1_IBM_s117
                                 + 0.857184 w_k1_INTC_s117
                                 + 1.01439 w_k1_KO_s117 + 1.33094 w_k1_MAT_s117
                                 + 1.07025 w_k1_MSFT_s117
                                 + 1.06593 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.39158 w_k1_ADM_s117
                                 + 1.33067 w_k1_AXP_s117
                                 + 1.53431 w_k1_DIS_s117
                                 + 1.36559 w_k1_FDX_s117
                                 + 1.03729 w_k1_IBM_s117
                                 + 1.29904 w_k1_INTC_s117
                                 + 1.14686 w_k1_KO_s117 + 1.07867 w_k1_MAT_s117
                                 + 1.4876 w_k1_MSFT_s117 + 1.26696 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0262197343508992 w_k1_ADM_s000
                                 + 0.0290219411617024 w_k1_ADM_s001
                                 + 0.018520179809436 w_k1_ADM_s002
                                 + 0.0321989946029182 w_k1_ADM_s003
                                 + 0.0391029925137594 w_k1_ADM_s004
                                 + 0.0380170141173945 w_k1_ADM_s005
                                 + 0.0335499951698451 w_k1_ADM_s006
                                 + 0.014399193634562 w_k1_ADM_s007
                                 + 0.0125474037332127 w_k1_ADM_s010
                                 + 0.012033662461935 w_k1_ADM_s011
                                 + 0.0337259127818873 w_k1_ADM_s012
                                 + 0.0123415184440646 w_k1_ADM_s013
                                 + 0.0316205376690634 w_k1_ADM_s014
                                 + 0.0149010575059012 w_k1_ADM_s015
                                 + 0.00952510394024199 w_k1_ADM_s016
                                 + 0.0151140827365523 w_k1_ADM_s017
                                 + 0.0294916877213567 w_k1_ADM_s100
                                 + 0.0472022685580829 w_k1_ADM_s101
                                 + 0.062254106395905 w_k1_ADM_s102
                                 + 0.0479708644906673 w_k1_ADM_s103
                                 + 0.0356501827359326 w_k1_ADM_s104
                                 + 0.0276660794840061 w_k1_ADM_s105
                                 + 0.0142127080289593 w_k1_ADM_s106
                                 + 0.0306110954798313 w_k1_ADM_s107
                                 + 0.0668023135513972 w_k1_ADM_s110
                                 + 0.0658996036224329 w_k1_ADM_s111
                                 + 0.0197643311399007 w_k1_ADM_s112
                                 + 0.0321460482959225 w_k1_ADM_s113
                                 + 0.0603705319985926 w_k1_ADM_s114
                                 + 0.0142758050812293 w_k1_ADM_s115
                                 + 0.0633043785434607 w_k1_ADM_s116
                                 + 0.0308825440733668 w_k1_ADM_s117
                                 + 0.0270983156708816 w_k1_AXP_s000
                                 + 0.0267557219021625 w_k1_AXP_s001
                                 + 0.0172590064364166 w_k1_AXP_s002
                                 + 0.0304350506045364 w_k1_AXP_s003
                                 + 0.0383414034586082 w_k1_AXP_s004
                                 + 0.0418983822742918 w_k1_AXP_s005
                                 + 0.0330415899024947 w_k1_AXP_s006
                                 + 0.0162562586475566 w_k1_AXP_s007
                                 + 0.0133802771430639 w_k1_AXP_s010
                                 + 0.0122511347037516 w_k1_AXP_s011
                                 + 0.0350030285185821 w_k1_AXP_s012
                                 + 0.0139766209385403 w_k1_AXP_s013
                                 + 0.0364009706097168 w_k1_AXP_s014
                                 + 0.0163480473521354 w_k1_AXP_s015
                                 + 0.00939606286198772 w_k1_AXP_s016
                                 + 0.0134541229690104 w_k1_AXP_s017
                                 + 0.028473801078046 w_k1_AXP_s100
                                 + 0.0442994866447223 w_k1_AXP_s101
                                 + 0.0601192175718997 w_k1_AXP_s102
                                 + 0.0461837415712021 w_k1_AXP_s103
                                 + 0.0320513178221879 w_k1_AXP_s104
                                 + 0.028651348707502 w_k1_AXP_s105
                                 + 0.0148458454389867 w_k1_AXP_s106
                                 + 0.0356261558528421 w_k1_AXP_s107
                                 + 0.0573747049535909 w_k1_AXP_s110
                                 + 0.0752500798659838 w_k1_AXP_s111
                                 + 0.0191742194498725 w_k1_AXP_s112
                                 + 0.0299676183860448 w_k1_AXP_s113
                                 + 0.0650570383569402 w_k1_AXP_s114
                                 + 0.0145131446954744 w_k1_AXP_s115
                                 + 0.0630953139013033 w_k1_AXP_s116
                                 + 0.0297667880958019 w_k1_AXP_s117
                                 + 0.02838170373961 w_k1_DIS_s000
                                 + 0.0288224559857553 w_k1_DIS_s001
                                 + 0.0177747374583055 w_k1_DIS_s002
                                 + 0.0323891707061816 w_k1_DIS_s003
                                 + 0.0377124937215815 w_k1_DIS_s004
                                 + 0.0402835403719941 w_k1_DIS_s005
                                 + 0.0342312547807951 w_k1_DIS_s006
                                 + 0.0157686751056915 w_k1_DIS_s007
                                 + 0.0148144382149078 w_k1_DIS_s010
                                 + 0.0125063349399794 w_k1_DIS_s011
                                 + 0.0379601703991741 w_k1_DIS_s012
                                 + 0.0147831723267978 w_k1_DIS_s013
                                 + 0.0351920240251286 w_k1_DIS_s014
                                 + 0.0167806017054662 w_k1_DIS_s015
                                 + 0.00917468538763134 w_k1_DIS_s016
                                 + 0.0134117323083822 w_k1_DIS_s017
                                 + 0.0285163852682871 w_k1_DIS_s100
                                 + 0.0461746875648676 w_k1_DIS_s101
                                 + 0.0615297303108758 w_k1_DIS_s102
                                 + 0.0534032145757603 w_k1_DIS_s103
                                 + 0.035071293950604 w_k1_DIS_s104
                                 + 0.0310607837922417 w_k1_DIS_s105
                                 + 0.0136755850928608 w_k1_DIS_s106
                                 + 0.0364950026450327 w_k1_DIS_s107
                                 + 0.0596810511878162 w_k1_DIS_s110
                                 + 0.0714584754302365 w_k1_DIS_s111
                                 + 0.0184877891933142 w_k1_DIS_s112
                                 + 0.0328742305377659 w_k1_DIS_s113
                                 + 0.0643221364607072 w_k1_DIS_s114
                                 + 0.0166180982631297 w_k1_DIS_s115
                                 + 0.0561366265828159 w_k1_DIS_s116
                                 + 0.0295326605729093 w_k1_DIS_s117
                                 + 0.0272019155606692 w_k1_FDX_s000
                                 + 0.0278150212517557 w_k1_FDX_s001
                                 + 0.0176626499249564 w_k1_FDX_s002
                                 + 0.0310246551968174 w_k1_FDX_s003
                                 + 0.0356065656833922 w_k1_FDX_s004
                                 + 0.0373235953481939 w_k1_FDX_s005
                                 + 0.0333328280254037 w_k1_FDX_s006
                                 + 0.0151638079825911 w_k1_FDX_s007
                                 + 0.0134990046326444 w_k1_FDX_s010
                                 + 0.0128453126540073 w_k1_FDX_s011
                                 + 0.0339288774883959 w_k1_FDX_s012
                                 + 0.0144350543387316 w_k1_FDX_s013
                                 + 0.0342344907071521 w_k1_FDX_s014
                                 + 0.0166535750888219 w_k1_FDX_s015
                                 + 0.0089961550800781 w_k1_FDX_s016
                                 + 0.0127923639669916 w_k1_FDX_s017
                                 + 0.028728085935728 w_k1_FDX_s100
                                 + 0.0437464193509117 w_k1_FDX_s101
                                 + 0.059627133438458 w_k1_FDX_s102
                                 + 0.0504111500053066 w_k1_FDX_s103
                                 + 0.03424298531386 w_k1_FDX_s104
                                 + 0.0282481199878678 w_k1_FDX_s105
                                 + 0.0136372787931061 w_k1_FDX_s106
                                 + 0.0357748970023661 w_k1_FDX_s107
                                 + 0.060610074788806 w_k1_FDX_s110
                                 + 0.0700141017493242 w_k1_FDX_s111
                                 + 0.0194883130820266 w_k1_FDX_s112
                                 + 0.0328587644114069 w_k1_FDX_s113
                                 + 0.0644630767461938 w_k1_FDX_s114
                                 + 0.0153099462546706 w_k1_FDX_s115
                                 + 0.061264740617329 w_k1_FDX_s116
                                 + 0.0285622314790071 w_k1_FDX_s117
                                 + 0.0281331410710499 w_k1_IBM_s000
                                 + 0.0275270520853355 w_k1_IBM_s001
                                 + 0.0174926842755454 w_k1_IBM_s002
                                 + 0.0322651722862519 w_k1_IBM_s003
                                 + 0.0373924617524607 w_k1_IBM_s004
                                 + 0.0379685071888786 w_k1_IBM_s005
                                 + 0.0325437690713369 w_k1_IBM_s006
                                 + 0.0157561888078201 w_k1_IBM_s007
                                 + 0.0138551835057987 w_k1_IBM_s010
                                 + 0.013440301232015 w_k1_IBM_s011
                                 + 0.0345110775494593 w_k1_IBM_s012
                                 + 0.014388826815904 w_k1_IBM_s013
                                 + 0.0342914983097742 w_k1_IBM_s014
                                 + 0.0159753199849199 w_k1_IBM_s015
                                 + 0.00892193080283901 w_k1_IBM_s016
                                 + 0.0139558828538817 w_k1_IBM_s017
                                 + 0.0295423853172122 w_k1_IBM_s100
                                 + 0.0451644335164173 w_k1_IBM_s101
                                 + 0.0642411683111656 w_k1_IBM_s102
                                 + 0.0504012829864955 w_k1_IBM_s103
                                 + 0.0331787418477265 w_k1_IBM_s104
                                 + 0.0287824747747787 w_k1_IBM_s105
                                 + 0.0137206690919196 w_k1_IBM_s106
                                 + 0.0380564494176433 w_k1_IBM_s107
                                 + 0.0623435076813533 w_k1_IBM_s110
                                 + 0.0720471978423657 w_k1_IBM_s111
                                 + 0.0198070655688738 w_k1_IBM_s112
                                 + 0.0324699027262933 w_k1_IBM_s113
                                 + 0.0637923821427017 w_k1_IBM_s114
                                 + 0.0148460530747398 w_k1_IBM_s115
                                 + 0.0592129048279485 w_k1_IBM_s116
                                 + 0.0306456696744747 w_k1_IBM_s117
                                 + 0.0266549555327146 w_k1_INTC_s000
                                 + 0.0289813471000988 w_k1_INTC_s001
                                 + 0.0186797829286185 w_k1_INTC_s002
                                 + 0.0314338729541233 w_k1_INTC_s003
                                 + 0.0316102703517846 w_k1_INTC_s004
                                 + 0.0364213116664614 w_k1_INTC_s005
                                 + 0.0315183787293078 w_k1_INTC_s006
                                 + 0.0165717189588586 w_k1_INTC_s007
                                 + 0.0119415085895452 w_k1_INTC_s010
                                 + 0.0135595821945519 w_k1_INTC_s011
                                 + 0.0369481698351347 w_k1_INTC_s012
                                 + 0.0137517934762833 w_k1_INTC_s013
                                 + 0.0341390982750558 w_k1_INTC_s014
                                 + 0.0172537441715775 w_k1_INTC_s015
                                 + 0.0095219691960181 w_k1_INTC_s016
                                 + 0.0125081472829559 w_k1_INTC_s017
                                 + 0.0251619000888479 w_k1_INTC_s100
                                 + 0.0455349550284982 w_k1_INTC_s101
                                 + 0.0630802666878909 w_k1_INTC_s102
                                 + 0.0448616926660523 w_k1_INTC_s103
                                 + 0.0296082823922614 w_k1_INTC_s104
                                 + 0.0258139372111088 w_k1_INTC_s105
                                 + 0.0127003184665158 w_k1_INTC_s106
                                 + 0.034901981177359 w_k1_INTC_s107
                                 + 0.0598807431287244 w_k1_INTC_s110
                                 + 0.0678726950294105 w_k1_INTC_s111
                                 + 0.0154960118584023 w_k1_INTC_s112
                                 + 0.0316814410107696 w_k1_INTC_s113
                                 + 0.0613017074921529 w_k1_INTC_s114
                                 + 0.0145386299171858 w_k1_INTC_s115
                                 + 0.0606186840546997 w_k1_INTC_s116
                                 + 0.0266809024809528 w_k1_INTC_s117
                                 + 0.0281685812119268 w_k1_KO_s000
                                 + 0.0288441739957932 w_k1_KO_s001
                                 + 0.0176211987535424 w_k1_KO_s002
                                 + 0.0312814227165756 w_k1_KO_s003
                                 + 0.0326941075798453 w_k1_KO_s004
                                 + 0.0381232219779261 w_k1_KO_s005
                                 + 0.0330966658841119 w_k1_KO_s006
                                 + 0.0145677286999132 w_k1_KO_s007
                                 + 0.0139046214681389 w_k1_KO_s010
                                 + 0.0130100638866726 w_k1_KO_s011
                                 + 0.0369164465852583 w_k1_KO_s012
                                 + 0.013180709555975 w_k1_KO_s013
                                 + 0.0349929870710056 w_k1_KO_s014
                                 + 0.0162987898521578 w_k1_KO_s015
                                 + 0.00924847819386263 w_k1_KO_s016
                                 + 0.015011270129152 w_k1_KO_s017
                                 + 0.0300335741931738 w_k1_KO_s100
                                 + 0.0420567102617011 w_k1_KO_s101
                                 + 0.0570304952527631 w_k1_KO_s102
                                 + 0.0525287715781472 w_k1_KO_s103
                                 + 0.0357775816127796 w_k1_KO_s104
                                 + 0.0292516705855546 w_k1_KO_s105
                                 + 0.013816737013765 w_k1_KO_s106
                                 + 0.0325950988549819 w_k1_KO_s107
                                 + 0.0633491326287939 w_k1_KO_s110
                                 + 0.0710666265260336 w_k1_KO_s111
                                 + 0.0180087324897103 w_k1_KO_s112
                                 + 0.0336645539891696 w_k1_KO_s113
                                 + 0.0671192123648453 w_k1_KO_s114
                                 + 0.0153404104916038 w_k1_KO_s115
                                 + 0.0560722102351941 w_k1_KO_s116
                                 + 0.0299126333933529 w_k1_KO_s117
                                 + 0.023009084560653 w_k1_MAT_s000
                                 + 0.0272110162622656 w_k1_MAT_s001
                                 + 0.0167924568813643 w_k1_MAT_s002
                                 + 0.0351791153765692 w_k1_MAT_s003
                                 + 0.0345866553002885 w_k1_MAT_s004
                                 + 0.038561064714566 w_k1_MAT_s005
                                 + 0.0366076209201624 w_k1_MAT_s006
                                 + 0.0136613453278478 w_k1_MAT_s007
                                 + 0.0122837574283788 w_k1_MAT_s010
                                 + 0.0134980980076278 w_k1_MAT_s011
                                 + 0.0352988480874232 w_k1_MAT_s012
                                 + 0.0102935725832431 w_k1_MAT_s013
                                 + 0.0430113256314404 w_k1_MAT_s014
                                 + 0.0186028046197563 w_k1_MAT_s015
                                 + 0.00680531703966017 w_k1_MAT_s016
                                 + 0.0177648247192161 w_k1_MAT_s017
                                 + 0.0287751202308143 w_k1_MAT_s100
                                 + 0.0418496625083817 w_k1_MAT_s101
                                 + 0.0540127947084846 w_k1_MAT_s102
                                 + 0.0701328304488283 w_k1_MAT_s103
                                 + 0.0368436433666502 w_k1_MAT_s104
                                 + 0.0266737899016269 w_k1_MAT_s105
                                 + 0.0145903610713146 w_k1_MAT_s106
                                 + 0.0368948634932993 w_k1_MAT_s107
                                 + 0.0554463545200055 w_k1_MAT_s110
                                 + 0.0660112258408336 w_k1_MAT_s111
                                 + 0.0201354157025628 w_k1_MAT_s112
                                 + 0.0307895816500707 w_k1_MAT_s113
                                 + 0.0692098096848587 w_k1_MAT_s114
                                 + 0.0120693565429494 w_k1_MAT_s115
                                 + 0.0576445848969163 w_k1_MAT_s116
                                 + 0.0323127267035518 w_k1_MAT_s117
                                 + 0.0260548475613241 w_k1_MSFT_s000
                                 + 0.02833992028427 w_k1_MSFT_s001
                                 + 0.0195287486190628 w_k1_MSFT_s002
                                 + 0.0308584874067347 w_k1_MSFT_s003
                                 + 0.0337379281795611 w_k1_MSFT_s004
                                 + 0.0391667546759744 w_k1_MSFT_s005
                                 + 0.0354567265502369 w_k1_MSFT_s006
                                 + 0.015653769084226 w_k1_MSFT_s007
                                 + 0.0141263494619126 w_k1_MSFT_s010
                                 + 0.0130723389551997 w_k1_MSFT_s011
                                 + 0.0341191732505724 w_k1_MSFT_s012
                                 + 0.0140324621498739 w_k1_MSFT_s013
                                 + 0.0330747313152109 w_k1_MSFT_s014
                                 + 0.0168301388944985 w_k1_MSFT_s015
                                 + 0.00921649277247743 w_k1_MSFT_s016
                                 + 0.0131773847657672 w_k1_MSFT_s017
                                 + 0.0273037845536047 w_k1_MSFT_s100
                                 + 0.0431895019357814 w_k1_MSFT_s101
                                 + 0.0630005187480497 w_k1_MSFT_s102
                                 + 0.0521831392914415 w_k1_MSFT_s103
                                 + 0.0343948035800498 w_k1_MSFT_s104
                                 + 0.0275717571549551 w_k1_MSFT_s105
                                 + 0.0146678454169149 w_k1_MSFT_s106
                                 + 0.0346382655715445 w_k1_MSFT_s107
                                 + 0.0627317243041442 w_k1_MSFT_s110
                                 + 0.0641834905969814 w_k1_MSFT_s111
                                 + 0.016763793169383 w_k1_MSFT_s112
                                 + 0.0307580942212727 w_k1_MSFT_s113
                                 + 0.0654548090068774 w_k1_MSFT_s114
                                 + 0.0153430991316389 w_k1_MSFT_s115
                                 + 0.0551420911900764 w_k1_MSFT_s116
                                 + 0.0307759682579366 w_k1_MSFT_s117
                                 + 0.0270069363945679 w_k1_PG_s000
                                 + 0.0289967051545488 w_k1_PG_s001
                                 + 0.0167817301317506 w_k1_PG_s002
                                 + 0.0268416442143707 w_k1_PG_s003
                                 + 0.0324941552611902 w_k1_PG_s004
                                 + 0.0382442637024924 w_k1_PG_s005
                                 + 0.0330184835434554 w_k1_PG_s006
                                 + 0.0145627758890326 w_k1_PG_s007
                                 + 0.0150214700867276 w_k1_PG_s010
                                 + 0.0132076454561531 w_k1_PG_s011
                                 + 0.0326314583379229 w_k1_PG_s012
                                 + 0.0145771002172627 w_k1_PG_s013
                                 + 0.0331834743994806 w_k1_PG_s014
                                 + 0.0161837730223877 w_k1_PG_s015
                                 + 0.00911738931416604 w_k1_PG_s016
                                 + 0.0132455281656812 w_k1_PG_s017
                                 + 0.0297358356030724 w_k1_PG_s100
                                 + 0.0440097070532724 w_k1_PG_s101
                                 + 0.0600492365580262 w_k1_PG_s102
                                 + 0.0464014760904432 w_k1_PG_s103
                                 + 0.0336885939603206 w_k1_PG_s104
                                 + 0.0278852584080702 w_k1_PG_s105
                                 + 0.0138991834915069 w_k1_PG_s106
                                 + 0.0389862958815896 w_k1_PG_s107
                                 + 0.061483125863629 w_k1_PG_s110
                                 + 0.0728714232073483 w_k1_PG_s111
                                 + 0.0185755368129384 w_k1_PG_s112
                                 + 0.0331310158184322 w_k1_PG_s113
                                 + 0.0672014506004038 w_k1_PG_s114
                                 + 0.0142234203748779 w_k1_PG_s115
                                 + 0.0605847437283873 w_k1_PG_s116
                                 + 0.0316180320672194 w_k1_PG_s117 >= 1.17
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
