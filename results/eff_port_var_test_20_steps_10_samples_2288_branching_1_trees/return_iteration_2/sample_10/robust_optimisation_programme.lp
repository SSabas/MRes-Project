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
 w_k1_ADM_s0_return_constraint:  1.10699 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.87541 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.13532 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.896832 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.22908 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.972141 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.07101 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.13184 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.841 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.41884 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.946767 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.48885 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.765248 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.10173 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.21542 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.945997 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.773039 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.925555 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.24017 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.20834 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.655636 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.28433 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.14004 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.157 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.5257 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.877077 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.792349 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.22478 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.09867 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.55772 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.771257 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.26361 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.884829 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.685945 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.06901 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.0872 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.991039 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.23467 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.06876 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.97565 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.35172 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.905329 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.39759 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.862015 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.15294 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.1286 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.820825 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.47785 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.853176 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.70511 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.707906 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.09916 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.962824 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.944412 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.734774 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.811713 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.09814 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.26621 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.6236 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.47608 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.756671 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.00706 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.22881 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.930368 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.83322 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.38435 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.26182 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.40885 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.497223 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.36994 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.852786 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.697176 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.30166 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.994536 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.802243 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.22833 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.09332 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.914596 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.38076 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.873981 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.6048 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.915331 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.37619 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.33936 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.954004 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.58141 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.71627 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 2.09071 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.62022 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.02873 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.36985 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.749508 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.77659 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.817671 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.25524 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.44718 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.523119 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.14523 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.747722 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.00634 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.20444 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.979431 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.799775 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.13825 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.32305 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.63467 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.439894 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.41112 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.827196 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.695571 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.27001 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.01341 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.666235 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.16359 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.07314 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.883818 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.19901 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.935493 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.36579 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.932682 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.08501 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.23019 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.921227 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.54536 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.874699 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.59424 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.678475 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.11919 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.17998 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.01342 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.749382 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.868952 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.19561 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.13073 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.663121 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.11903 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.895924 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.921603 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.27773 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.909685 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.763592 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.09473 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.987619 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.28174 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.661437 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.33014 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.892146 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.776248 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.1479 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.159 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.894315 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.972271 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.959609 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.17344 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.45783 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.875116 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.45985 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.918122 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.38156 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.966253 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.989657 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.819095 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.647233 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.41386 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.969221 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.819492 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.09816 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.608286 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.25874 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.815379 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.806485 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.65734 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.782953 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.17902 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.589339 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.04245 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.698467 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.784361 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.15243 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.23641 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 2.09534 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.25332 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.584388 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.988271 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.02743 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.776576 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.19786 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.625174 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.600884 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.66594 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.02358 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.871733 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.23375 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.868686 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.55014 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.07364 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.13616 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.08679 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.984072 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.19928 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.719331 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.23797 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.663495 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.04799 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.16594 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.11308 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.01107 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.03559 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.23267 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.28154 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.932162 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.10411 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.701313 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.879776 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.04866 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.759736 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.727739 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.879706 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.695917 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.1797 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.581122 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.17286 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.94513 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.813843 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.987491 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.10208 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.855802 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.945677 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.983367 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.02029 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.3458 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.948512 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.41735 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.955639 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.34151 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.14265 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.08133 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.08344 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.763956 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.71395 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.785637 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.00127 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.19184 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.739278 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.998887 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.888232 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.05333 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.3 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.637914 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.08596 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.729423 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.0393 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.991813 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.833833 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.948296 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.10115 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.44254 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.47275 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.548572 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.16461 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.973592 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.756958 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.24148 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.822366 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.640625 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.40242 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.93976 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.16201 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.52994 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.915883 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.47617 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.853548 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.51968 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.910916 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.920958 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.859711 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.769112 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.74475 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.952604 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.943857 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.02502 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.650597 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.20244 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.74526 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.808676 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.23608 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.657544 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.2909 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.463763 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.909815 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.687987 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.87024 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.00197 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.70816 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 2.24415 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.58625 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.506366 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.86113 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.94546 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.745006 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.41979 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.673256 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.576248 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.75144 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.0433 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.923977 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.1924 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.913245 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.4252 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.944975 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.13448 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.11379 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.924006 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.36464 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.843986 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.52745 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.741841 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.08194 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.29975 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01662 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.879246 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.852964 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.14681 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.19365 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.763038 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.04641 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.787498 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.930067 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.04141 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.906707 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.719882 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.860879 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.86022 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.16833 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.705562 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.22107 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.841218 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.791155 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.1279 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.12139 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.915781 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.02695 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.997022 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.9381 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.21618 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.965169 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.3361 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.936985 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.09424 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.11885 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.960882 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.16732 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.899527 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.37149 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.782326 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.01608 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.1893 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.875689 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.92745 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.872888 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.02043 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.18699 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.712994 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.10041 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.775039 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.968594 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.07553 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.949908 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.855299 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.05929 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.13084 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.29787 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.72248 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.26437 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.02813 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.877401 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.15831 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.04505 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.881249 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.09778 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.384999014806646 z_k1_s0000
                                 - 0.680997718684594 z_k1_s0001
                                 - 0.439000631660591 z_k1_s0002
                                 - 0.388999630128581 z_k1_s0003
                                 - 0.248999836335433 z_k1_s0004
                                 - 0.414000130894586 z_k1_s0005
                                 - 0.527000788758806 z_k1_s0006
                                 - 0.260999675303585 z_k1_s0007
                                 - 0.11500015210022 z_k1_s0010
                                 - 0.279000202181141 z_k1_s0011
                                 - 0.430001519793616 z_k1_s0012
                                 - 0.153000157144569 z_k1_s0013
                                 - 0.201999696524709 z_k1_s0014
                                 - 0.368999978206862 z_k1_s0015
                                 - 0.100000069895745 z_k1_s0016
                                 - 0.145000316509168 z_k1_s0017
                                 - 0.347000470198453 z_k1_s0020
                                 - 0.717001546670034 z_k1_s0021
                                 - 0.275000123014649 z_k1_s0022
                                 - 0.612002101861439 z_k1_s0023
                                 - 0.913001359513639 z_k1_s0024
                                 - 0.363000264283445 z_k1_s0025
                                 - 0.415000231960338 z_k1_s0026
                                 - 0.604002629419392 z_k1_s0027
                                 - 0.163999951775672 z_k1_s0030
                                 - 0.0420001263574497 z_k1_s0031
                                 - 0.151000478904212 z_k1_s0032
                                 - 0.103000245866957 z_k1_s0033
                                 - 0.208000871961174 z_k1_s0034
                                 - 0.137999972030773 z_k1_s0035
                                 - 0.162000191949598 z_k1_s0036
                                 - 0.0660001394758794 z_k1_s0037
                                 - 0.118000049368386 z_k1_s0040
                                 - 0.139000145205212 z_k1_s0041
                                 - 0.447000181210746 z_k1_s0042
                                 - 0.269999764757856 z_k1_s0043
                                 - 0.129000172940123 z_k1_s0044
                                 - 0.186000408611485 z_k1_s0045
                                 - 0.223999995058542 z_k1_s0046
                                 - 0.198999636395433 z_k1_s0047
                                 - 0.1530001201147 z_k1_s0050
                                 - 0.113000079312159 z_k1_s0051
                                 - 0.368000439303419 z_k1_s0052
                                 - 0.113000079312159 z_k1_s0053
                                 - 0.0940000079959211 z_k1_s0054
                                 - 0.171000482046045 z_k1_s0055
                                 - 0.37899947973855 z_k1_s0056
                                 - 0.207000247108173 z_k1_s0057
                                 - 0.245999749197307 z_k1_s0060
                                 - 0.695000941653264 z_k1_s0061
                                 - 0.555000293371072 z_k1_s0062
                                 - 0.32399972695493 z_k1_s0063
                                 - 0.159999782323509 z_k1_s0064
                                 - 0.410000700327673 z_k1_s0065
                                 - 0.36799949934407 z_k1_s0066
                                 - 0.596998139358191 z_k1_s0067
                                 - 0.548002063479604 z_k1_s0070
                                 - 0.214001081219355 z_k1_s0071
                                 - 0.4210032004716 z_k1_s0072
                                 - 0.235001289337664 z_k1_s0073
                                 - 0.341001842637061 z_k1_s0074
                                 - 0.125000549495933 z_k1_s0075
                                 - 0.172000902283951 z_k1_s0076
                                 - 0.317001197985487 z_k1_s0077
                                 - 0.894001848983012 z_k1_s0100
                                 - 0.698000720235156 z_k1_s0101
                                 - 0.912000852813746 z_k1_s0102
                                 - 0.47100013653308 z_k1_s0103
                                 - 2.30700063778047 z_k1_s0104
                                 - 1.40900351239559 z_k1_s0105
                                 - 0.613001012185162 z_k1_s0106
                                 - 0.800001018215864 z_k1_s0107
                                 - 0.292999675973658 z_k1_s0110
                                 - 0.700001189768437 z_k1_s0111
                                 - 0.215999830768927 z_k1_s0112
                                 - 0.477999028463143 z_k1_s0113
                                 - 0.371999617857662 z_k1_s0114
                                 - 0.767998763467748 z_k1_s0115
                                 - 0.786000041680907 z_k1_s0116
                                 - 0.468000585565333 z_k1_s0117
                                 - 0.309999353664154 z_k1_s0120
                                 - 0.402999415363405 z_k1_s0121
                                 - 0.19399989234467 z_k1_s0122
                                 - 0.347000010340212 z_k1_s0123
                                 - 0.54799874201276 z_k1_s0124
                                 - 0.215999895946643 z_k1_s0125
                                 - 0.189000100653316 z_k1_s0126
                                 - 0.348998802376733 z_k1_s0127
                                 - 1.17999820070786 z_k1_s0130
                                 - 0.344998804001979 z_k1_s0131
                                 - 0.289999185615643 z_k1_s0132
                                 - 0.965998909839472 z_k1_s0133
                                 - 1.07699821952441 z_k1_s0134
                                 - 1.24399910270289 z_k1_s0135
                                 - 0.431998873385829 z_k1_s0136
                                 - 0.739997489227287 z_k1_s0137
                                 - 0.881994912287406 z_k1_s0140
                                 - 0.994994321150592 z_k1_s0141
                                 - 0.772000766176012 z_k1_s0142
                                 - 0.520998621173471 z_k1_s0143
                                 - 0.697997891483409 z_k1_s0144
                                 - 1.2519939917924 z_k1_s0145
                                 - 0.879999050765223 z_k1_s0146
                                 - 1.64699869266726 z_k1_s0147
                                 - 1.29999807604761 z_k1_s0150
                                 - 0.607998252691283 z_k1_s0151
                                 - 0.353999128927984 z_k1_s0152
                                 - 0.855997548706119 z_k1_s0153
                                 - 1.56299495034033 z_k1_s0154
                                 - 0.383998931368061 z_k1_s0155
                                 - 1.01399743740072 z_k1_s0156
                                 - 0.557997918260061 z_k1_s0157
                                 - 0.425000443910806 z_k1_s0160
                                 - 0.77700222480371 z_k1_s0161
                                 - 1.53099844819664 z_k1_s0162
                                 - 0.798996973405224 z_k1_s0163
                                 - 1.06399883647558 z_k1_s0164
                                 - 1.14499865429649 z_k1_s0165
                                 - 1.06100338307844 z_k1_s0166
                                 - 0.858997973421618 z_k1_s0167
                                 - 0.508999638548137 z_k1_s0170
                                 - 1.14999952036396 z_k1_s0171
                                 - 0.468999475209499 z_k1_s0172
                                 - 0.440000022226085 z_k1_s0173
                                 - 0.904000733954996 z_k1_s0174
                                 - 0.481999454357101 z_k1_s0175
                                 - 0.346999716401208 z_k1_s0176
                                 - 1.61400141509215 z_k1_s0177
                                 - 0.144999825345779 z_k1_s1000
                                 - 0.0559998578992894 z_k1_s1001
                                 - 0.0379998637460691 z_k1_s1002
                                 - 0.0419998770245176 z_k1_s1003
                                 - 0.175999665854563 z_k1_s1004
                                 - 0.0609998088975521 z_k1_s1005
                                 - 0.0949996593651727 z_k1_s1006
                                 - 0.0429998803441297 z_k1_s1007
                                 - 0.145000225880932 z_k1_s1010
                                 - 0.13800076768209 z_k1_s1011
                                 - 0.118000923251016 z_k1_s1012
                                 - 0.143000999540988 z_k1_s1013
                                 - 0.0660002447257061 z_k1_s1014
                                 - 0.115001000736581 z_k1_s1015
                                 - 0.214000609721977 z_k1_s1016
                                 - 0.14400061271096 z_k1_s1017
                                 - 0.0780003960860379 z_k1_s1020
                                 - 0.0940000700004119 z_k1_s1021
                                 - 0.0890002482773033 z_k1_s1022
                                 - 0.054000090913091 z_k1_s1023
                                 - 0.110000354915852 z_k1_s1024
                                 - 0.0650001264046762 z_k1_s1025
                                 - 0.0310000555852989 z_k1_s1026
                                 - 0.0900004570223514 z_k1_s1027
                                 - 0.0679999483576693 z_k1_s1030
                                 - 0.147999444543317 z_k1_s1031
                                 - 0.206999732030003 z_k1_s1032
                                 - 0.151000407499937 z_k1_s1033
                                 - 0.0599999987391046 z_k1_s1034
                                 - 0.0890000178563316 z_k1_s1035
                                 - 0.279000246996749 z_k1_s1036
                                 - 0.0739999374715836 z_k1_s1037
                                 - 0.0850002446489817 z_k1_s1040
                                 - 0.136000211757816 z_k1_s1041
                                 - 0.161000514919995 z_k1_s1042
                                 - 0.239000966664468 z_k1_s1043
                                 - 0.0780002271437796 z_k1_s1044
                                 - 0.142999892361978 z_k1_s1045
                                 - 0.0830002877762154 z_k1_s1046
                                 - 0.198000110117385 z_k1_s1047
                                 - 0.10800012225148 z_k1_s1050
                                 - 0.0709999428968604 z_k1_s1051
                                 - 0.112999590272594 z_k1_s1052
                                 - 0.0600000107508446 z_k1_s1053
                                 - 0.072999950605219 z_k1_s1054
                                 - 0.0849997763554577 z_k1_s1055
                                 - 0.0959997820014444 z_k1_s1056
                                 - 0.128999798939405 z_k1_s1057
                                 - 0.0390000561254308 z_k1_s1060
                                 - 0.0399999939746832 z_k1_s1061
                                 - 0.0410001798243943 z_k1_s1062
                                 - 0.0180000220886533 z_k1_s1063
                                 - 0.0300001194812417 z_k1_s1064
                                 - 0.0450001792218626 z_k1_s1065
                                 - 0.0160000223899192 z_k1_s1066
                                 - 0.0190000343380433 z_k1_s1067
                                 - 0.09299976027642 z_k1_s1070
                                 - 0.184999531872437 z_k1_s1071
                                 - 0.0619999485841369 z_k1_s1072
                                 - 0.0839998621527916 z_k1_s1073
                                 - 0.170999690346793 z_k1_s1074
                                 - 0.0949997376372263 z_k1_s1075
                                 - 0.053999957841019 z_k1_s1076
                                 - 0.068999869346959 z_k1_s1077
                                 - 0.206999698771142 z_k1_s1100
                                 - 0.238999593972534 z_k1_s1101
                                 - 0.233000151746755 z_k1_s1102
                                 - 0.134999848468093 z_k1_s1103
                                 - 0.295999461112461 z_k1_s1104
                                 - 0.129999854080386 z_k1_s1105
                                 - 0.202999496621175 z_k1_s1106
                                 - 0.278999893473859 z_k1_s1107
                                 - 0.587998252642231 z_k1_s1110
                                 - 0.555997338963175 z_k1_s1111
                                 - 1.2499980002936 z_k1_s1112
                                 - 0.359999404312599 z_k1_s1113
                                 - 1.0289969421239 z_k1_s1114
                                 - 0.335999213352251 z_k1_s1115
                                 - 0.313999450221053 z_k1_s1116
                                 - 0.509997879170574 z_k1_s1117
                                 - 0.280000867723482 z_k1_s1120
                                 - 0.321000772094911 z_k1_s1121
                                 - 0.192000595010388 z_k1_s1122
                                 - 0.483001541360623 z_k1_s1123
                                 - 0.324000825929564 z_k1_s1124
                                 - 0.752001617855491 z_k1_s1125
                                 - 0.602002489132115 z_k1_s1126
                                 - 0.60900023940677 z_k1_s1127
                                 - 0.344999783762665 z_k1_s1130
                                 - 0.300000771966933 z_k1_s1131
                                 - 0.251998882053331 z_k1_s1132
                                 - 0.302999234087571 z_k1_s1133
                                 - 0.497000616492301 z_k1_s1134
                                 - 0.122999874906719 z_k1_s1135
                                 - 0.192999834232476 z_k1_s1136
                                 - 0.194999839378922 z_k1_s1137
                                 - 0.628995774464383 z_k1_s1140
                                 - 0.611999392036621 z_k1_s1141
                                 - 0.472998745594225 z_k1_s1142
                                 - 0.302999144916741 z_k1_s1143
                                 - 0.353999180669551 z_k1_s1144
                                 - 0.756997511192316 z_k1_s1145
                                 - 0.829996921869699 z_k1_s1146
                                 - 1.22699570568497 z_k1_s1147
                                 - 0.154000602048966 z_k1_s1150
                                 - 0.150999453048011 z_k1_s1151
                                 - 0.0770000508244827 z_k1_s1152
                                 - 0.0760000014241647 z_k1_s1153
                                 - 0.308999502098249 z_k1_s1154
                                 - 0.271999926086484 z_k1_s1155
                                 - 0.141000210044832 z_k1_s1156
                                 - 0.0710000046225749 z_k1_s1157
                                 - 0.17499997175244 z_k1_s1160
                                 - 0.503000655503209 z_k1_s1161
                                 - 0.269000425646836 z_k1_s1162
                                 - 0.17499997175244 z_k1_s1163
                                 - 0.335999361588754 z_k1_s1164
                                 - 0.438001011320833 z_k1_s1165
                                 - 0.187999900588915 z_k1_s1166
                                 - 0.433999909802053 z_k1_s1167
                                 - 0.393000486236948 z_k1_s1170
                                 - 0.42100155893677 z_k1_s1171
                                 - 0.310000439602763 z_k1_s1172
                                 - 0.416001437633333 z_k1_s1173
                                 - 0.732999686657461 z_k1_s1174
                                 - 0.810999218587633 z_k1_s1175
                                 - 0.233000145132129 z_k1_s1176
                                 - 0.617001593224613 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.768313 w_k1_ADM_s000
                                 + 0.819646 w_k1_AXP_s000
                                 + 0.646561 w_k1_DIS_s000
                                 + 0.85091 w_k1_FDX_s000
                                 + 0.691253 w_k1_IBM_s000
                                 + 0.844454 w_k1_INTC_s000
                                 + 0.752176 w_k1_KO_s000
                                 + 0.753007 w_k1_MAT_s000
                                 + 0.876005 w_k1_MSFT_s000
                                 + 0.92691 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.12206 w_k1_ADM_s000
                                 + 1.22818 w_k1_AXP_s000 + 1.0772 w_k1_DIS_s000
                                 + 1.23918 w_k1_FDX_s000
                                 + 0.903726 w_k1_IBM_s000
                                 + 1.10361 w_k1_INTC_s000
                                 + 0.948967 w_k1_KO_s000
                                 + 0.938479 w_k1_MAT_s000
                                 + 1.26665 w_k1_MSFT_s000
                                 + 1.11027 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.735047 w_k1_ADM_s000
                                 + 0.900714 w_k1_AXP_s000
                                 + 0.768736 w_k1_DIS_s000
                                 + 0.927713 w_k1_FDX_s000
                                 + 0.991533 w_k1_IBM_s000
                                 + 0.990131 w_k1_INTC_s000
                                 + 0.947689 w_k1_KO_s000
                                 + 1.09692 w_k1_MAT_s000
                                 + 0.921501 w_k1_MSFT_s000
                                 + 0.933384 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.802483 w_k1_ADM_s000
                                 + 0.941369 w_k1_AXP_s000
                                 + 0.898349 w_k1_DIS_s000
                                 + 0.901489 w_k1_FDX_s000
                                 + 0.8165 w_k1_IBM_s000
                                 + 0.836046 w_k1_INTC_s000
                                 + 0.918881 w_k1_KO_s000
                                 + 1.11462 w_k1_MAT_s000
                                 + 0.888234 w_k1_MSFT_s000
                                 + 1.04002 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.81849 w_k1_ADM_s000
                                 + 0.9442 w_k1_AXP_s000 + 1.17016 w_k1_DIS_s000
                                 + 1.04933 w_k1_FDX_s000
                                 + 1.08971 w_k1_IBM_s000
                                 + 0.908412 w_k1_INTC_s000
                                 + 1.17006 w_k1_KO_s000 + 1.1058 w_k1_MAT_s000
                                 + 1.12816 w_k1_MSFT_s000 + 1.1523 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.29443 w_k1_ADM_s000
                                 + 1.16201 w_k1_AXP_s000
                                 + 1.01837 w_k1_DIS_s000
                                 + 1.05583 w_k1_FDX_s000
                                 + 1.63443 w_k1_IBM_s000
                                 + 0.755099 w_k1_INTC_s000
                                 + 1.26592 w_k1_KO_s000 + 2.04685 w_k1_MAT_s000
                                 + 0.93053 w_k1_MSFT_s000
                                 + 1.12766 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.732568 w_k1_ADM_s000
                                 + 0.892309 w_k1_AXP_s000
                                 + 0.664249 w_k1_DIS_s000
                                 + 0.811763 w_k1_FDX_s000
                                 + 1.21723 w_k1_IBM_s000
                                 + 0.698131 w_k1_INTC_s000
                                 + 0.81037 w_k1_KO_s000 + 1.20473 w_k1_MAT_s000
                                 + 0.772711 w_k1_MSFT_s000
                                 + 0.928891 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.10063 w_k1_ADM_s000
                                 + 1.26619 w_k1_AXP_s000
                                 + 1.75904 w_k1_DIS_s000
                                 + 1.33935 w_k1_FDX_s000
                                 + 1.41096 w_k1_IBM_s000
                                 + 1.10916 w_k1_INTC_s000
                                 + 1.37978 w_k1_KO_s000 + 1.2425 w_k1_MAT_s000
                                 + 1.29224 w_k1_MSFT_s000
                                 + 1.29155 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.54366 w_k1_ADM_s001
                                 + 1.40455 w_k1_AXP_s001
                                 + 1.70836 w_k1_DIS_s001
                                 + 1.35701 w_k1_FDX_s001
                                 + 1.34498 w_k1_IBM_s001
                                 + 1.27625 w_k1_INTC_s001
                                 + 1.28218 w_k1_KO_s001 + 1.21706 w_k1_MAT_s001
                                 + 1.29393 w_k1_MSFT_s001
                                 + 1.20732 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.06866 w_k1_ADM_s001
                                 + 1.00141 w_k1_AXP_s001
                                 + 1.19166 w_k1_DIS_s001
                                 + 1.04109 w_k1_FDX_s001
                                 + 0.964143 w_k1_IBM_s001
                                 + 1.08445 w_k1_INTC_s001
                                 + 0.996668 w_k1_KO_s001
                                 + 0.927149 w_k1_MAT_s001
                                 + 1.19097 w_k1_MSFT_s001
                                 + 1.06084 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.915034 w_k1_ADM_s001
                                 + 0.977535 w_k1_AXP_s001
                                 + 0.796383 w_k1_DIS_s001
                                 + 0.874758 w_k1_FDX_s001
                                 + 1.01116 w_k1_IBM_s001
                                 + 0.984871 w_k1_INTC_s001
                                 + 0.817803 w_k1_KO_s001 + 1.1527 w_k1_MAT_s001
                                 + 0.851827 w_k1_MSFT_s001
                                 + 0.945611 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.21988 w_k1_ADM_s001
                                 + 1.44966 w_k1_AXP_s001
                                 + 1.28706 w_k1_DIS_s001
                                 + 1.11697 w_k1_FDX_s001
                                 + 1.40938 w_k1_IBM_s001
                                 + 0.928727 w_k1_INTC_s001
                                 + 1.02943 w_k1_KO_s001 + 1.98319 w_k1_MAT_s001
                                 + 1.07651 w_k1_MSFT_s001
                                 + 0.988322 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.753442 w_k1_ADM_s001
                                 + 0.712864 w_k1_AXP_s001
                                 + 0.65746 w_k1_DIS_s001
                                 + 0.777718 w_k1_FDX_s001
                                 + 0.691438 w_k1_IBM_s001
                                 + 0.805211 w_k1_INTC_s001
                                 + 0.794548 w_k1_KO_s001
                                 + 0.621242 w_k1_MAT_s001
                                 + 0.825021 w_k1_MSFT_s001
                                 + 0.874289 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.797055 w_k1_ADM_s001
                                 + 0.688491 w_k1_AXP_s001
                                 + 0.931478 w_k1_DIS_s001
                                 + 0.883091 w_k1_FDX_s001
                                 + 0.812585 w_k1_IBM_s001
                                 + 0.815996 w_k1_INTC_s001
                                 + 0.8972 w_k1_KO_s001 + 0.821079 w_k1_MAT_s001
                                 + 1.06494 w_k1_MSFT_s001
                                 + 0.895463 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.21845 w_k1_ADM_s001
                                 + 1.32824 w_k1_AXP_s001
                                 + 1.64118 w_k1_DIS_s001
                                 + 1.20872 w_k1_FDX_s001
                                 + 0.939643 w_k1_IBM_s001
                                 + 1.348 w_k1_INTC_s001 + 1.19119 w_k1_KO_s001
                                 + 1.14849 w_k1_MAT_s001
                                 + 1.46992 w_k1_MSFT_s001
                                 + 1.14346 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.98873 w_k1_ADM_s001
                                 + 1.10356 w_k1_AXP_s001
                                 + 0.949852 w_k1_DIS_s001
                                 + 0.932944 w_k1_FDX_s001
                                 + 1.45459 w_k1_IBM_s001
                                 + 1.05277 w_k1_INTC_s001 + 1.1957 w_k1_KO_s001
                                 + 2.29262 w_k1_MAT_s001
                                 + 1.00331 w_k1_MSFT_s001
                                 + 0.92227 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.21054 w_k1_ADM_s002
                                 + 1.31901 w_k1_AXP_s002 + 1.1433 w_k1_DIS_s002
                                 + 1.07279 w_k1_FDX_s002
                                 + 1.30824 w_k1_IBM_s002
                                 + 1.21986 w_k1_INTC_s002
                                 + 1.14622 w_k1_KO_s002 + 1.51615 w_k1_MAT_s002
                                 + 1.13264 w_k1_MSFT_s002
                                 + 1.07476 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.21437 w_k1_ADM_s002
                                 + 1.02625 w_k1_AXP_s002
                                 + 0.773171 w_k1_DIS_s002
                                 + 0.861389 w_k1_FDX_s002
                                 + 1.05171 w_k1_IBM_s002
                                 + 0.77022 w_k1_INTC_s002 + 0.8114 w_k1_KO_s002
                                 + 1.02551 w_k1_MAT_s002
                                 + 0.827563 w_k1_MSFT_s002
                                 + 0.948656 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.09222 w_k1_ADM_s002
                                 + 1.27815 w_k1_AXP_s002
                                 + 1.08497 w_k1_DIS_s002
                                 + 1.11851 w_k1_FDX_s002
                                 + 1.05638 w_k1_IBM_s002
                                 + 1.10886 w_k1_INTC_s002
                                 + 0.838023 w_k1_KO_s002
                                 + 0.904948 w_k1_MAT_s002
                                 + 1.0976 w_k1_MSFT_s002
                                 + 0.999231 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.860759 w_k1_ADM_s002
                                 + 0.696799 w_k1_AXP_s002
                                 + 0.780638 w_k1_DIS_s002
                                 + 0.877916 w_k1_FDX_s002
                                 + 0.6303 w_k1_IBM_s002
                                 + 0.853962 w_k1_INTC_s002
                                 + 0.71678 w_k1_KO_s002
                                 + 0.659574 w_k1_MAT_s002
                                 + 0.941349 w_k1_MSFT_s002
                                 + 0.834727 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.06069 w_k1_ADM_s002
                                 + 0.878195 w_k1_AXP_s002
                                 + 0.918459 w_k1_DIS_s002
                                 + 0.870889 w_k1_FDX_s002
                                 + 0.890763 w_k1_IBM_s002
                                 + 0.873216 w_k1_INTC_s002
                                 + 0.873337 w_k1_KO_s002
                                 + 0.814232 w_k1_MAT_s002
                                 + 0.944634 w_k1_MSFT_s002
                                 + 0.924526 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.2734 w_k1_ADM_s002
                                 + 1.46857 w_k1_AXP_s002
                                 + 1.18249 w_k1_DIS_s002
                                 + 0.997353 w_k1_FDX_s002
                                 + 1.67702 w_k1_IBM_s002
                                 + 1.20068 w_k1_INTC_s002
                                 + 1.06512 w_k1_KO_s002 + 1.46747 w_k1_MAT_s002
                                 + 1.084 w_k1_MSFT_s002 + 1.04085 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.26302 w_k1_ADM_s002
                                 + 1.51244 w_k1_AXP_s002
                                 + 1.55698 w_k1_DIS_s002
                                 + 1.36921 w_k1_FDX_s002
                                 + 1.18294 w_k1_IBM_s002
                                 + 1.32704 w_k1_INTC_s002
                                 + 1.20208 w_k1_KO_s002 + 1.20966 w_k1_MAT_s002
                                 + 1.33145 w_k1_MSFT_s002
                                 + 1.27622 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.4747 w_k1_ADM_s002
                                 + 1.12976 w_k1_AXP_s002
                                 + 1.22089 w_k1_DIS_s002
                                 + 1.15151 w_k1_FDX_s002
                                 + 0.861827 w_k1_IBM_s002
                                 + 1.04824 w_k1_INTC_s002
                                 + 0.967066 w_k1_KO_s002
                                 + 0.817628 w_k1_MAT_s002
                                 + 1.16281 w_k1_MSFT_s002
                                 + 1.10948 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.711845 w_k1_ADM_s003
                                 + 0.665392 w_k1_AXP_s003
                                 + 0.684249 w_k1_DIS_s003
                                 + 0.796953 w_k1_FDX_s003
                                 + 0.699046 w_k1_IBM_s003
                                 + 0.956638 w_k1_INTC_s003
                                 + 0.869448 w_k1_KO_s003
                                 + 0.914541 w_k1_MAT_s003
                                 + 0.927051 w_k1_MSFT_s003
                                 + 0.797353 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.05341 w_k1_ADM_s003
                                 + 1.59839 w_k1_AXP_s003
                                 + 0.96726 w_k1_DIS_s003
                                 + 1.09259 w_k1_FDX_s003
                                 + 0.956725 w_k1_IBM_s003
                                 + 1.11292 w_k1_INTC_s003
                                 + 0.99652 w_k1_KO_s003 + 1.59985 w_k1_MAT_s003
                                 + 0.995237 w_k1_MSFT_s003
                                 + 0.956722 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.0681 w_k1_ADM_s003
                                 + 1.07734 w_k1_AXP_s003
                                 + 0.888824 w_k1_DIS_s003
                                 + 1.01161 w_k1_FDX_s003
                                 + 0.649745 w_k1_IBM_s003
                                 + 0.968995 w_k1_INTC_s003
                                 + 0.895685 w_k1_KO_s003 + 1.148 w_k1_MAT_s003
                                 + 0.849922 w_k1_MSFT_s003
                                 + 0.972057 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.02226 w_k1_ADM_s003
                                 + 1.24868 w_k1_AXP_s003
                                 + 0.958449 w_k1_DIS_s003
                                 + 1.00654 w_k1_FDX_s003
                                 + 1.48572 w_k1_IBM_s003
                                 + 1.13885 w_k1_INTC_s003
                                 + 1.11489 w_k1_KO_s003 + 2.84585 w_k1_MAT_s003
                                 + 0.94905 w_k1_MSFT_s003
                                 + 1.12532 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.0423 w_k1_ADM_s003
                                 + 0.98664 w_k1_AXP_s003
                                 + 0.880863 w_k1_DIS_s003
                                 + 0.932685 w_k1_FDX_s003
                                 + 1.13878 w_k1_IBM_s003
                                 + 0.885302 w_k1_INTC_s003
                                 + 1.05445 w_k1_KO_s003 + 1.57457 w_k1_MAT_s003
                                 + 0.910234 w_k1_MSFT_s003 + 1.04 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.0394 w_k1_ADM_s003
                                 + 0.709094 w_k1_AXP_s003
                                 + 0.963079 w_k1_DIS_s003
                                 + 0.930093 w_k1_FDX_s003
                                 + 0.880997 w_k1_IBM_s003
                                 + 0.88807 w_k1_INTC_s003
                                 + 1.19598 w_k1_KO_s003 + 1.37495 w_k1_MAT_s003
                                 + 1.01712 w_k1_MSFT_s003
                                 + 0.999466 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.24692 w_k1_ADM_s003
                                 + 1.17332 w_k1_AXP_s003
                                 + 1.45702 w_k1_DIS_s003
                                 + 1.26499 w_k1_FDX_s003
                                 + 1.03383 w_k1_IBM_s003
                                 + 1.04974 w_k1_INTC_s003
                                 + 1.32498 w_k1_KO_s003 + 1.09537 w_k1_MAT_s003
                                 + 1.07571 w_k1_MSFT_s003 + 1.1105 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.721214 w_k1_ADM_s003
                                 + 0.931729 w_k1_AXP_s003
                                 + 0.502976 w_k1_DIS_s003
                                 + 0.722313 w_k1_FDX_s003
                                 + 1.0162 w_k1_IBM_s003
                                 + 0.894685 w_k1_INTC_s003
                                 + 0.749524 w_k1_KO_s003
                                 + 1.34746 w_k1_MAT_s003
                                 + 0.642423 w_k1_MSFT_s003
                                 + 0.755082 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.00551 w_k1_ADM_s004
                                 + 1.52471 w_k1_AXP_s004
                                 + 1.82531 w_k1_DIS_s004
                                 + 1.40816 w_k1_FDX_s004
                                 + 1.45279 w_k1_IBM_s004
                                 + 1.59666 w_k1_INTC_s004
                                 + 1.78747 w_k1_KO_s004 + 2.02161 w_k1_MAT_s004
                                 + 1.53679 w_k1_MSFT_s004
                                 + 1.28638 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.948403 w_k1_ADM_s004
                                 + 0.980145 w_k1_AXP_s004
                                 + 1.25374 w_k1_DIS_s004 + 1.178 w_k1_FDX_s004
                                 + 1.08875 w_k1_IBM_s004
                                 + 1.18096 w_k1_INTC_s004
                                 + 0.94841 w_k1_KO_s004 + 1.00829 w_k1_MAT_s004
                                 + 1.11755 w_k1_MSFT_s004
                                 + 1.07186 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.796848 w_k1_ADM_s004
                                 + 0.632866 w_k1_AXP_s004
                                 + 0.763983 w_k1_DIS_s004
                                 + 0.865322 w_k1_FDX_s004
                                 + 0.769383 w_k1_IBM_s004
                                 + 1.14463 w_k1_INTC_s004
                                 + 0.765842 w_k1_KO_s004
                                 + 0.703574 w_k1_MAT_s004
                                 + 0.914938 w_k1_MSFT_s004
                                 + 0.804272 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.961139 w_k1_ADM_s004
                                 + 1.05119 w_k1_AXP_s004
                                 + 1.01784 w_k1_DIS_s004
                                 + 0.97398 w_k1_FDX_s004
                                 + 1.13274 w_k1_IBM_s004
                                 + 1.24642 w_k1_INTC_s004
                                 + 0.999489 w_k1_KO_s004
                                 + 1.12687 w_k1_MAT_s004
                                 + 0.912819 w_k1_MSFT_s004
                                 + 0.95356 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.07762 w_k1_ADM_s004
                                 + 1.47854 w_k1_AXP_s004
                                 + 1.16444 w_k1_DIS_s004
                                 + 1.22496 w_k1_FDX_s004
                                 + 1.20558 w_k1_IBM_s004
                                 + 1.40295 w_k1_INTC_s004
                                 + 1.22801 w_k1_KO_s004 + 1.80623 w_k1_MAT_s004
                                 + 1.18848 w_k1_MSFT_s004 + 1.2037 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.779909 w_k1_ADM_s004
                                 + 0.902151 w_k1_AXP_s004
                                 + 1.23087 w_k1_DIS_s004
                                 + 0.947324 w_k1_FDX_s004
                                 + 1.42875 w_k1_IBM_s004
                                 + 1.26459 w_k1_INTC_s004
                                 + 1.25766 w_k1_KO_s004 + 1.2058 w_k1_MAT_s004
                                 + 1.28451 w_k1_MSFT_s004
                                 + 0.917716 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.11653 w_k1_ADM_s004
                                 + 1.01268 w_k1_AXP_s004
                                 + 0.910549 w_k1_DIS_s004
                                 + 1.10051 w_k1_FDX_s004
                                 + 0.724881 w_k1_IBM_s004
                                 + 1.21038 w_k1_INTC_s004
                                 + 0.758015 w_k1_KO_s004
                                 + 0.673346 w_k1_MAT_s004
                                 + 0.99181 w_k1_MSFT_s004
                                 + 0.897691 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.94886 w_k1_ADM_s004
                                 + 1.15188 w_k1_AXP_s004
                                 + 1.39925 w_k1_DIS_s004
                                 + 1.26052 w_k1_FDX_s004
                                 + 0.821225 w_k1_IBM_s004
                                 + 1.75471 w_k1_INTC_s004
                                 + 1.03752 w_k1_KO_s004
                                 + 0.657811 w_k1_MAT_s004
                                 + 1.37129 w_k1_MSFT_s004
                                 + 0.970357 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.17564 w_k1_ADM_s005
                                 + 1.38279 w_k1_AXP_s005
                                 + 1.89258 w_k1_DIS_s005
                                 + 1.27837 w_k1_FDX_s005
                                 + 1.47484 w_k1_IBM_s005
                                 + 1.32314 w_k1_INTC_s005
                                 + 1.54976 w_k1_KO_s005 + 1.87406 w_k1_MAT_s005
                                 + 1.37147 w_k1_MSFT_s005
                                 + 1.37863 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.18553 w_k1_ADM_s005
                                 + 1.24002 w_k1_AXP_s005
                                 + 1.24814 w_k1_DIS_s005
                                 + 1.21871 w_k1_FDX_s005
                                 + 0.676147 w_k1_IBM_s005
                                 + 1.3866 w_k1_INTC_s005
                                 + 0.989753 w_k1_KO_s005
                                 + 0.888228 w_k1_MAT_s005
                                 + 1.31278 w_k1_MSFT_s005
                                 + 1.10593 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.86303 w_k1_ADM_s005
                                 + 0.966124 w_k1_AXP_s005
                                 + 0.794633 w_k1_DIS_s005
                                 + 0.761425 w_k1_FDX_s005
                                 + 1.31863 w_k1_IBM_s005
                                 + 1.15483 w_k1_INTC_s005
                                 + 0.963461 w_k1_KO_s005
                                 + 1.33755 w_k1_MAT_s005
                                 + 0.856528 w_k1_MSFT_s005
                                 + 0.847645 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.18311 w_k1_ADM_s005
                                 + 1.58667 w_k1_AXP_s005
                                 + 1.32118 w_k1_DIS_s005
                                 + 1.24677 w_k1_FDX_s005
                                 + 1.09969 w_k1_IBM_s005
                                 + 1.23218 w_k1_INTC_s005
                                 + 1.07589 w_k1_KO_s005 + 1.45245 w_k1_MAT_s005
                                 + 1.09093 w_k1_MSFT_s005 + 1.2298 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.639041 w_k1_ADM_s005
                                 + 0.749064 w_k1_AXP_s005
                                 + 0.487057 w_k1_DIS_s005
                                 + 0.668022 w_k1_FDX_s005
                                 + 0.672967 w_k1_IBM_s005
                                 + 0.670415 w_k1_INTC_s005
                                 + 0.574981 w_k1_KO_s005
                                 + 0.883277 w_k1_MAT_s005
                                 + 0.60666 w_k1_MSFT_s005
                                 + 0.74273 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.860568 w_k1_ADM_s005
                                 + 0.708396 w_k1_AXP_s005
                                 + 0.962203 w_k1_DIS_s005
                                 + 1.00687 w_k1_FDX_s005
                                 + 0.709605 w_k1_IBM_s005
                                 + 1.04194 w_k1_INTC_s005
                                 + 0.763582 w_k1_KO_s005
                                 + 0.503886 w_k1_MAT_s005
                                 + 1.00672 w_k1_MSFT_s005
                                 + 0.854796 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.736352 w_k1_ADM_s005
                                 + 1.11719 w_k1_AXP_s005
                                 + 1.09326 w_k1_DIS_s005
                                 + 0.990352 w_k1_FDX_s005
                                 + 1.08287 w_k1_IBM_s005
                                 + 1.18551 w_k1_INTC_s005
                                 + 1.00651 w_k1_KO_s005 + 1.12898 w_k1_MAT_s005
                                 + 1.05859 w_k1_MSFT_s005
                                 + 0.972372 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.829622 w_k1_ADM_s005
                                 + 0.805841 w_k1_AXP_s005
                                 + 0.701984 w_k1_DIS_s005
                                 + 0.848903 w_k1_FDX_s005
                                 + 0.638503 w_k1_IBM_s005
                                 + 0.984256 w_k1_INTC_s005
                                 + 0.727052 w_k1_KO_s005
                                 + 0.691306 w_k1_MAT_s005
                                 + 0.90232 w_k1_MSFT_s005
                                 + 0.869249 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.17389 w_k1_ADM_s006
                                 + 0.880458 w_k1_AXP_s006
                                 + 1.22696 w_k1_DIS_s006
                                 + 0.993093 w_k1_FDX_s006
                                 + 1.12723 w_k1_IBM_s006
                                 + 1.03018 w_k1_INTC_s006
                                 + 1.16793 w_k1_KO_s006 + 1.17945 w_k1_MAT_s006
                                 + 1.13513 w_k1_MSFT_s006
                                 + 1.09012 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.807172 w_k1_ADM_s006
                                 + 0.787104 w_k1_AXP_s006
                                 + 0.810598 w_k1_DIS_s006
                                 + 0.946385 w_k1_FDX_s006
                                 + 0.907632 w_k1_IBM_s006
                                 + 1.1133 w_k1_INTC_s006 + 0.89596 w_k1_KO_s006
                                 + 0.757937 w_k1_MAT_s006
                                 + 0.945923 w_k1_MSFT_s006
                                 + 0.966554 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.13428 w_k1_ADM_s006
                                 + 1.26158 w_k1_AXP_s006
                                 + 1.16803 w_k1_DIS_s006
                                 + 1.19339 w_k1_FDX_s006
                                 + 0.947602 w_k1_IBM_s006
                                 + 1.1346 w_k1_INTC_s006
                                 + 0.984667 w_k1_KO_s006
                                 + 1.11388 w_k1_MAT_s006
                                 + 1.01623 w_k1_MSFT_s006 + 1.1747 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.30686 w_k1_ADM_s006
                                 + 1.54622 w_k1_AXP_s006 + 1.5012 w_k1_DIS_s006
                                 + 1.26142 w_k1_FDX_s006
                                 + 1.55059 w_k1_IBM_s006
                                 + 1.46279 w_k1_INTC_s006
                                 + 1.79031 w_k1_KO_s006 + 1.49691 w_k1_MAT_s006
                                 + 1.24001 w_k1_MSFT_s006
                                 + 1.25938 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.25721 w_k1_ADM_s006
                                 + 1.51191 w_k1_AXP_s006
                                 + 1.47158 w_k1_DIS_s006
                                 + 1.30774 w_k1_FDX_s006
                                 + 1.12786 w_k1_IBM_s006
                                 + 1.29029 w_k1_INTC_s006
                                 + 1.21497 w_k1_KO_s006 + 1.54082 w_k1_MAT_s006
                                 + 1.27479 w_k1_MSFT_s006
                                 + 1.22534 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.950255 w_k1_ADM_s006
                                 + 0.918762 w_k1_AXP_s006
                                 + 1.00406 w_k1_DIS_s006
                                 + 1.14796 w_k1_FDX_s006
                                 + 0.627729 w_k1_IBM_s006
                                 + 1.23722 w_k1_INTC_s006
                                 + 0.984216 w_k1_KO_s006
                                 + 0.764075 w_k1_MAT_s006
                                 + 1.04083 w_k1_MSFT_s006
                                 + 1.06164 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.845575 w_k1_ADM_s006
                                 + 1.0378 w_k1_AXP_s006 + 1.17315 w_k1_DIS_s006
                                 + 1.02364 w_k1_FDX_s006
                                 + 1.67428 w_k1_IBM_s006
                                 + 1.03209 w_k1_INTC_s006 + 1.304 w_k1_KO_s006
                                 + 1.87278 w_k1_MAT_s006
                                 + 0.987573 w_k1_MSFT_s006
                                 + 1.16548 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.644694 w_k1_ADM_s006
                                 + 0.854611 w_k1_AXP_s006
                                 + 0.638683 w_k1_DIS_s006
                                 + 0.789295 w_k1_FDX_s006
                                 + 0.796511 w_k1_IBM_s006
                                 + 0.91459 w_k1_INTC_s006
                                 + 0.784351 w_k1_KO_s006
                                 + 0.825541 w_k1_MAT_s006
                                 + 0.782932 w_k1_MSFT_s006
                                 + 0.816556 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.31686 w_k1_ADM_s007
                                 + 1.26527 w_k1_AXP_s007
                                 + 1.11511 w_k1_DIS_s007
                                 + 1.19114 w_k1_FDX_s007
                                 + 0.83409 w_k1_IBM_s007
                                 + 0.953004 w_k1_INTC_s007
                                 + 0.906675 w_k1_KO_s007
                                 + 1.00749 w_k1_MAT_s007
                                 + 0.994221 w_k1_MSFT_s007
                                 + 1.01634 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.3598 w_k1_ADM_s007
                                 + 1.01967 w_k1_AXP_s007
                                 + 1.45362 w_k1_DIS_s007
                                 + 1.27047 w_k1_FDX_s007
                                 + 0.844183 w_k1_IBM_s007
                                 + 1.23705 w_k1_INTC_s007
                                 + 1.15598 w_k1_KO_s007
                                 + 0.729634 w_k1_MAT_s007
                                 + 1.35829 w_k1_MSFT_s007
                                 + 1.10904 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.917259 w_k1_ADM_s007
                                 + 0.838821 w_k1_AXP_s007
                                 + 0.963006 w_k1_DIS_s007
                                 + 0.949206 w_k1_FDX_s007
                                 + 0.845924 w_k1_IBM_s007
                                 + 0.888588 w_k1_INTC_s007
                                 + 0.814711 w_k1_KO_s007
                                 + 0.596586 w_k1_MAT_s007
                                 + 0.885621 w_k1_MSFT_s007
                                 + 0.872412 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.23021 w_k1_ADM_s007
                                 + 1.32839 w_k1_AXP_s007
                                 + 1.52787 w_k1_DIS_s007
                                 + 1.25147 w_k1_FDX_s007
                                 + 1.56652 w_k1_IBM_s007
                                 + 1.15545 w_k1_INTC_s007
                                 + 1.30125 w_k1_KO_s007 + 1.39331 w_k1_MAT_s007
                                 + 1.07461 w_k1_MSFT_s007
                                 + 1.04865 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.02875 w_k1_ADM_s007
                                 + 1.21233 w_k1_AXP_s007
                                 + 1.08857 w_k1_DIS_s007
                                 + 1.02975 w_k1_FDX_s007
                                 + 1.36734 w_k1_IBM_s007
                                 + 1.08454 w_k1_INTC_s007
                                 + 0.986367 w_k1_KO_s007
                                 + 1.58059 w_k1_MAT_s007
                                 + 1.03011 w_k1_MSFT_s007
                                 + 1.02789 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.93231 w_k1_ADM_s007
                                 + 0.876225 w_k1_AXP_s007
                                 + 0.652961 w_k1_DIS_s007
                                 + 0.972449 w_k1_FDX_s007
                                 + 0.517927 w_k1_IBM_s007
                                 + 0.796358 w_k1_INTC_s007
                                 + 0.506498 w_k1_KO_s007
                                 + 0.58532 w_k1_MAT_s007
                                 + 0.781321 w_k1_MSFT_s007
                                 + 0.759586 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.91298 w_k1_ADM_s007
                                 + 0.865068 w_k1_AXP_s007
                                 + 0.874794 w_k1_DIS_s007
                                 + 0.796989 w_k1_FDX_s007 + 1.199 w_k1_IBM_s007
                                 + 0.924814 w_k1_INTC_s007
                                 + 1.0551 w_k1_KO_s007 + 1.06924 w_k1_MAT_s007
                                 + 0.819286 w_k1_MSFT_s007
                                 + 0.867447 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.881053 w_k1_ADM_s007
                                 + 1.06499 w_k1_AXP_s007
                                 + 0.709839 w_k1_DIS_s007
                                 + 0.85016 w_k1_FDX_s007
                                 + 0.897812 w_k1_IBM_s007
                                 + 0.904189 w_k1_INTC_s007
                                 + 0.745463 w_k1_KO_s007
                                 + 1.09175 w_k1_MAT_s007
                                 + 0.825232 w_k1_MSFT_s007
                                 + 0.930026 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.683609 w_k1_ADM_s010
                                 + 0.775126 w_k1_AXP_s010
                                 + 0.614105 w_k1_DIS_s010
                                 + 0.634516 w_k1_FDX_s010
                                 + 0.956038 w_k1_IBM_s010
                                 + 0.902294 w_k1_INTC_s010
                                 + 0.854699 w_k1_KO_s010
                                 + 0.883681 w_k1_MAT_s010
                                 + 0.701699 w_k1_MSFT_s010
                                 + 0.749074 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.27997 w_k1_ADM_s010
                                 + 1.77876 w_k1_AXP_s010 + 1.944 w_k1_DIS_s010
                                 + 1.56974 w_k1_FDX_s010
                                 + 1.21684 w_k1_IBM_s010
                                 + 1.58019 w_k1_INTC_s010
                                 + 1.49972 w_k1_KO_s010 + 1.35697 w_k1_MAT_s010
                                 + 1.40974 w_k1_MSFT_s010
                                 + 1.18355 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.72464 w_k1_ADM_s010
                                 + 1.1097 w_k1_AXP_s010 + 1.08522 w_k1_DIS_s010
                                 + 0.897281 w_k1_FDX_s010
                                 + 1.52212 w_k1_IBM_s010
                                 + 1.14957 w_k1_INTC_s010
                                 + 1.14888 w_k1_KO_s010 + 1.54067 w_k1_MAT_s010
                                 + 0.988004 w_k1_MSFT_s010
                                 + 1.01637 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.942314 w_k1_ADM_s010
                                 + 1.10535 w_k1_AXP_s010
                                 + 1.42228 w_k1_DIS_s010
                                 + 1.10595 w_k1_FDX_s010
                                 + 1.00789 w_k1_IBM_s010
                                 + 1.27778 w_k1_INTC_s010 + 1.2893 w_k1_KO_s010
                                 + 0.986958 w_k1_MAT_s010
                                 + 1.05591 w_k1_MSFT_s010 + 1.0839 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.840293 w_k1_ADM_s010
                                 + 0.89312 w_k1_AXP_s010
                                 + 0.792174 w_k1_DIS_s010
                                 + 0.79039 w_k1_FDX_s010
                                 + 0.877882 w_k1_IBM_s010
                                 + 0.86874 w_k1_INTC_s010
                                 + 0.793901 w_k1_KO_s010
                                 + 0.806839 w_k1_MAT_s010
                                 + 0.822673 w_k1_MSFT_s010
                                 + 0.866731 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.04335 w_k1_ADM_s010
                                 + 1.11513 w_k1_AXP_s010
                                 + 0.978117 w_k1_DIS_s010
                                 + 1.00898 w_k1_FDX_s010
                                 + 0.926635 w_k1_IBM_s010
                                 + 1.03189 w_k1_INTC_s010
                                 + 0.871331 w_k1_KO_s010
                                 + 0.964676 w_k1_MAT_s010
                                 + 0.953044 w_k1_MSFT_s010
                                 + 0.939761 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.15696 w_k1_ADM_s010
                                 + 1.38038 w_k1_AXP_s010
                                 + 1.48412 w_k1_DIS_s010
                                 + 1.29655 w_k1_FDX_s010
                                 + 0.89102 w_k1_IBM_s010
                                 + 1.26825 w_k1_INTC_s010
                                 + 1.08482 w_k1_KO_s010
                                 + 0.791398 w_k1_MAT_s010
                                 + 1.2121 w_k1_MSFT_s010 + 1.04563 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.06 w_k1_ADM_s010
                                 + 1.43106 w_k1_AXP_s010 + 1.1475 w_k1_DIS_s010
                                 + 1.12746 w_k1_FDX_s010
                                 + 1.24042 w_k1_IBM_s010
                                 + 1.18455 w_k1_INTC_s010 + 1.2864 w_k1_KO_s010
                                 + 1.62473 w_k1_MAT_s010
                                 + 1.01518 w_k1_MSFT_s010 + 1.0627 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.759808 w_k1_ADM_s011
                                 + 0.634733 w_k1_AXP_s011
                                 + 0.540163 w_k1_DIS_s011
                                 + 0.690303 w_k1_FDX_s011
                                 + 0.788396 w_k1_IBM_s011
                                 + 0.654995 w_k1_INTC_s011
                                 + 0.645468 w_k1_KO_s011
                                 + 0.842727 w_k1_MAT_s011
                                 + 0.706107 w_k1_MSFT_s011
                                 + 0.857917 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.875996 w_k1_ADM_s011
                                 + 0.826702 w_k1_AXP_s011
                                 + 0.903148 w_k1_DIS_s011
                                 + 0.836263 w_k1_FDX_s011
                                 + 1.06462 w_k1_IBM_s011
                                 + 0.832984 w_k1_INTC_s011
                                 + 0.886273 w_k1_KO_s011
                                 + 0.893674 w_k1_MAT_s011
                                 + 0.796236 w_k1_MSFT_s011
                                 + 0.962129 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.876171 w_k1_ADM_s011
                                 + 0.836357 w_k1_AXP_s011
                                 + 1.25472 w_k1_DIS_s011
                                 + 1.04044 w_k1_FDX_s011
                                 + 0.783911 w_k1_IBM_s011
                                 + 0.999872 w_k1_INTC_s011
                                 + 0.881867 w_k1_KO_s011
                                 + 0.595041 w_k1_MAT_s011
                                 + 1.07642 w_k1_MSFT_s011
                                 + 0.871937 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.0621 w_k1_ADM_s011
                                 + 1.01389 w_k1_AXP_s011
                                 + 0.999964 w_k1_DIS_s011
                                 + 0.993158 w_k1_FDX_s011
                                 + 0.954797 w_k1_IBM_s011
                                 + 0.914862 w_k1_INTC_s011
                                 + 0.92494 w_k1_KO_s011
                                 + 0.958491 w_k1_MAT_s011
                                 + 0.841465 w_k1_MSFT_s011
                                 + 1.04216 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.21702 w_k1_ADM_s011
                                 + 1.57436 w_k1_AXP_s011
                                 + 1.45897 w_k1_DIS_s011
                                 + 1.28465 w_k1_FDX_s011
                                 + 1.63835 w_k1_IBM_s011
                                 + 1.12157 w_k1_INTC_s011
                                 + 1.31706 w_k1_KO_s011 + 2.08419 w_k1_MAT_s011
                                 + 1.16576 w_k1_MSFT_s011
                                 + 1.15101 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.01261 w_k1_ADM_s011
                                 + 1.11603 w_k1_AXP_s011 + 1.0772 w_k1_DIS_s011
                                 + 1.04588 w_k1_FDX_s011
                                 + 1.37962 w_k1_IBM_s011
                                 + 1.15891 w_k1_INTC_s011
                                 + 1.10626 w_k1_KO_s011 + 1.19398 w_k1_MAT_s011
                                 + 1.0535 w_k1_MSFT_s011 + 1.17419 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.05391 w_k1_ADM_s011
                                 + 1.25789 w_k1_AXP_s011
                                 + 1.51143 w_k1_DIS_s011
                                 + 1.31622 w_k1_FDX_s011
                                 + 1.05641 w_k1_IBM_s011
                                 + 1.14203 w_k1_INTC_s011 + 1.07 w_k1_KO_s011
                                 + 1.44739 w_k1_MAT_s011
                                 + 1.23413 w_k1_MSFT_s011
                                 + 1.19777 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.776107 w_k1_ADM_s011
                                 + 0.65153 w_k1_AXP_s011
                                 + 0.716702 w_k1_DIS_s011
                                 + 0.878059 w_k1_FDX_s011
                                 + 0.664489 w_k1_IBM_s011
                                 + 0.823973 w_k1_INTC_s011
                                 + 0.797054 w_k1_KO_s011
                                 + 0.690531 w_k1_MAT_s011
                                 + 0.821879 w_k1_MSFT_s011
                                 + 0.874414 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.04978 w_k1_ADM_s012
                                 + 1.02377 w_k1_AXP_s012
                                 + 1.09474 w_k1_DIS_s012
                                 + 1.01271 w_k1_FDX_s012
                                 + 1.05799 w_k1_IBM_s012
                                 + 0.985928 w_k1_INTC_s012
                                 + 1.15289 w_k1_KO_s012
                                 + 0.874331 w_k1_MAT_s012
                                 + 1.01066 w_k1_MSFT_s012
                                 + 1.06137 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.957397 w_k1_ADM_s012
                                 + 0.689999 w_k1_AXP_s012
                                 + 0.70339 w_k1_DIS_s012
                                 + 0.830274 w_k1_FDX_s012
                                 + 0.726954 w_k1_IBM_s012
                                 + 0.822138 w_k1_INTC_s012
                                 + 0.70069 w_k1_KO_s012
                                 + 0.448971 w_k1_MAT_s012
                                 + 0.869127 w_k1_MSFT_s012
                                 + 0.801433 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.11885 w_k1_ADM_s012
                                 + 1.6006 w_k1_AXP_s012 + 1.42213 w_k1_DIS_s012
                                 + 1.20724 w_k1_FDX_s012
                                 + 1.60223 w_k1_IBM_s012
                                 + 1.19946 w_k1_INTC_s012
                                 + 1.10651 w_k1_KO_s012 + 1.4505 w_k1_MAT_s012
                                 + 1.0875 w_k1_MSFT_s012 + 1.03803 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.01407 w_k1_ADM_s012
                                 + 0.973144 w_k1_AXP_s012
                                 + 0.702101 w_k1_DIS_s012
                                 + 0.863087 w_k1_FDX_s012
                                 + 1.28227 w_k1_IBM_s012
                                 + 0.882309 w_k1_INTC_s012
                                 + 0.852969 w_k1_KO_s012
                                 + 1.11869 w_k1_MAT_s012
                                 + 0.898616 w_k1_MSFT_s012
                                 + 0.911283 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.27269 w_k1_ADM_s012
                                 + 1.09007 w_k1_AXP_s012
                                 + 1.08551 w_k1_DIS_s012
                                 + 1.06544 w_k1_FDX_s012
                                 + 0.882781 w_k1_IBM_s012
                                 + 0.920605 w_k1_INTC_s012
                                 + 0.951887 w_k1_KO_s012
                                 + 0.687078 w_k1_MAT_s012
                                 + 1.04389 w_k1_MSFT_s012
                                 + 0.986026 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.23904 w_k1_ADM_s012
                                 + 1.15308 w_k1_AXP_s012
                                 + 1.14228 w_k1_DIS_s012
                                 + 0.968922 w_k1_FDX_s012
                                 + 1.39088 w_k1_IBM_s012
                                 + 0.988477 w_k1_INTC_s012
                                 + 1.24043 w_k1_KO_s012 + 1.34287 w_k1_MAT_s012
                                 + 1.07091 w_k1_MSFT_s012
                                 + 1.17008 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.728548 w_k1_ADM_s012
                                 + 0.775002 w_k1_AXP_s012
                                 + 0.764701 w_k1_DIS_s012
                                 + 0.763844 w_k1_FDX_s012
                                 + 0.968414 w_k1_IBM_s012
                                 + 0.870613 w_k1_INTC_s012
                                 + 0.810656 w_k1_KO_s012
                                 + 0.761054 w_k1_MAT_s012
                                 + 0.860237 w_k1_MSFT_s012
                                 + 0.857201 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.03837 w_k1_ADM_s012
                                 + 1.38102 w_k1_AXP_s012
                                 + 1.56752 w_k1_DIS_s012
                                 + 1.26961 w_k1_FDX_s012 + 1.2011 w_k1_IBM_s012
                                 + 1.07646 w_k1_INTC_s012
                                 + 1.15778 w_k1_KO_s012 + 1.35331 w_k1_MAT_s012
                                 + 1.35481 w_k1_MSFT_s012
                                 + 1.13533 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.823669 w_k1_ADM_s013
                                 + 1.02747 w_k1_AXP_s013
                                 + 0.823963 w_k1_DIS_s013
                                 + 0.86179 w_k1_FDX_s013
                                 + 1.23507 w_k1_IBM_s013
                                 + 0.773363 w_k1_INTC_s013
                                 + 0.840788 w_k1_KO_s013 + 1.2733 w_k1_MAT_s013
                                 + 0.935368 w_k1_MSFT_s013
                                 + 0.996108 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.01734 w_k1_ADM_s013
                                 + 1.39024 w_k1_AXP_s013
                                 + 1.54994 w_k1_DIS_s013
                                 + 1.14274 w_k1_FDX_s013
                                 + 1.73089 w_k1_IBM_s013
                                 + 1.23074 w_k1_INTC_s013
                                 + 1.46205 w_k1_KO_s013 + 1.96228 w_k1_MAT_s013
                                 + 1.2566 w_k1_MSFT_s013 + 1.34835 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.687697 w_k1_ADM_s013
                                 + 0.631931 w_k1_AXP_s013
                                 + 0.540666 w_k1_DIS_s013
                                 + 0.664047 w_k1_FDX_s013
                                 + 0.819323 w_k1_IBM_s013
                                 + 0.78611 w_k1_INTC_s013
                                 + 0.726494 w_k1_KO_s013
                                 + 0.838002 w_k1_MAT_s013
                                 + 0.803975 w_k1_MSFT_s013
                                 + 0.845119 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.40783 w_k1_ADM_s013
                                 + 1.35749 w_k1_AXP_s013
                                 + 1.38618 w_k1_DIS_s013
                                 + 1.28599 w_k1_FDX_s013
                                 + 1.06421 w_k1_IBM_s013
                                 + 1.05553 w_k1_INTC_s013 + 1.185 w_k1_KO_s013
                                 + 1.26604 w_k1_MAT_s013
                                 + 1.43217 w_k1_MSFT_s013
                                 + 1.26277 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.9969 w_k1_ADM_s013
                                 + 0.800258 w_k1_AXP_s013
                                 + 0.922794 w_k1_DIS_s013
                                 + 0.993693 w_k1_FDX_s013
                                 + 0.655524 w_k1_IBM_s013
                                 + 0.961698 w_k1_INTC_s013
                                 + 0.827234 w_k1_KO_s013
                                 + 0.62179 w_k1_MAT_s013
                                 + 1.14576 w_k1_MSFT_s013
                                 + 0.94668 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.01073 w_k1_ADM_s013
                                 + 1.0603 w_k1_AXP_s013 + 1.19771 w_k1_DIS_s013
                                 + 1.06979 w_k1_FDX_s013 + 1.238 w_k1_IBM_s013
                                 + 0.999075 w_k1_INTC_s013
                                 + 1.15593 w_k1_KO_s013 + 1.61279 w_k1_MAT_s013
                                 + 1.24972 w_k1_MSFT_s013
                                 + 1.12621 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.823861 w_k1_ADM_s013
                                 + 0.742855 w_k1_AXP_s013
                                 + 0.816103 w_k1_DIS_s013
                                 + 0.857932 w_k1_FDX_s013
                                 + 0.915172 w_k1_IBM_s013
                                 + 0.953238 w_k1_INTC_s013
                                 + 0.901265 w_k1_KO_s013
                                 + 0.973251 w_k1_MAT_s013
                                 + 1.10172 w_k1_MSFT_s013
                                 + 1.02889 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.984303 w_k1_ADM_s013
                                 + 0.908672 w_k1_AXP_s013
                                 + 0.798404 w_k1_DIS_s013
                                 + 0.963794 w_k1_FDX_s013
                                 + 0.790514 w_k1_IBM_s013
                                 + 0.792221 w_k1_INTC_s013
                                 + 0.744355 w_k1_KO_s013
                                 + 0.719243 w_k1_MAT_s013
                                 + 0.92035 w_k1_MSFT_s013
                                 + 0.980465 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.07729 w_k1_ADM_s014
                                 + 1.24353 w_k1_AXP_s014
                                 + 1.26224 w_k1_DIS_s014 + 1.1508 w_k1_FDX_s014
                                 + 1.4251 w_k1_IBM_s014
                                 + 1.16654 w_k1_INTC_s014
                                 + 1.21994 w_k1_KO_s014 + 1.81275 w_k1_MAT_s014
                                 + 1.16915 w_k1_MSFT_s014
                                 + 1.25835 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.19788 w_k1_ADM_s014
                                 + 1.22836 w_k1_AXP_s014
                                 + 1.25394 w_k1_DIS_s014
                                 + 1.27215 w_k1_FDX_s014
                                 + 0.831381 w_k1_IBM_s014
                                 + 1.14249 w_k1_INTC_s014
                                 + 0.953711 w_k1_KO_s014
                                 + 0.787156 w_k1_MAT_s014
                                 + 1.21116 w_k1_MSFT_s014
                                 + 1.08203 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.40284 w_k1_ADM_s014
                                 + 1.79324 w_k1_AXP_s014
                                 + 1.80423 w_k1_DIS_s014
                                 + 1.55444 w_k1_FDX_s014
                                 + 1.19074 w_k1_IBM_s014
                                 + 1.56303 w_k1_INTC_s014
                                 + 1.30667 w_k1_KO_s014 + 1.30353 w_k1_MAT_s014
                                 + 1.68939 w_k1_MSFT_s014
                                 + 1.23457 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.624309 w_k1_ADM_s014
                                 + 0.909549 w_k1_AXP_s014
                                 + 0.806587 w_k1_DIS_s014
                                 + 0.784021 w_k1_FDX_s014
                                 + 1.28819 w_k1_IBM_s014
                                 + 0.99147 w_k1_INTC_s014
                                 + 0.916741 w_k1_KO_s014
                                 + 1.34325 w_k1_MAT_s014
                                 + 0.897197 w_k1_MSFT_s014
                                 + 0.99813 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.980987 w_k1_ADM_s014
                                 + 0.950452 w_k1_AXP_s014
                                 + 0.959905 w_k1_DIS_s014
                                 + 1.04517 w_k1_FDX_s014
                                 + 0.726072 w_k1_IBM_s014
                                 + 1.01539 w_k1_INTC_s014
                                 + 0.780946 w_k1_KO_s014
                                 + 0.761075 w_k1_MAT_s014
                                 + 0.996075 w_k1_MSFT_s014
                                 + 1.01663 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.890043 w_k1_ADM_s014
                                 + 0.935152 w_k1_AXP_s014
                                 + 1.02979 w_k1_DIS_s014
                                 + 1.03916 w_k1_FDX_s014
                                 + 1.15388 w_k1_IBM_s014
                                 + 0.913368 w_k1_INTC_s014
                                 + 1.01029 w_k1_KO_s014 + 1.14956 w_k1_MAT_s014
                                 + 1.03859 w_k1_MSFT_s014
                                 + 1.07974 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.702544 w_k1_ADM_s014
                                 + 0.689023 w_k1_AXP_s014
                                 + 0.596148 w_k1_DIS_s014
                                 + 0.699858 w_k1_FDX_s014
                                 + 0.733577 w_k1_IBM_s014
                                 + 0.854718 w_k1_INTC_s014
                                 + 0.721154 w_k1_KO_s014
                                 + 0.841998 w_k1_MAT_s014
                                 + 0.837547 w_k1_MSFT_s014
                                 + 0.823139 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.789262 w_k1_ADM_s014
                                 + 0.872577 w_k1_AXP_s014
                                 + 0.794583 w_k1_DIS_s014
                                 + 0.879218 w_k1_FDX_s014
                                 + 0.911438 w_k1_IBM_s014
                                 + 0.879721 w_k1_INTC_s014
                                 + 0.832137 w_k1_KO_s014
                                 + 0.898735 w_k1_MAT_s014
                                 + 0.889649 w_k1_MSFT_s014
                                 + 0.919663 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.746398 w_k1_ADM_s015
                                 + 0.876143 w_k1_AXP_s015
                                 + 0.722679 w_k1_DIS_s015
                                 + 0.765175 w_k1_FDX_s015
                                 + 1.15456 w_k1_IBM_s015
                                 + 0.788771 w_k1_INTC_s015
                                 + 0.886904 w_k1_KO_s015
                                 + 1.24234 w_k1_MAT_s015
                                 + 0.824743 w_k1_MSFT_s015
                                 + 0.863702 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.37261 w_k1_ADM_s015
                                 + 1.68139 w_k1_AXP_s015
                                 + 1.41935 w_k1_DIS_s015 + 1.2825 w_k1_FDX_s015
                                 + 1.66997 w_k1_IBM_s015
                                 + 1.14059 w_k1_INTC_s015
                                 + 1.41135 w_k1_KO_s015 + 2.53175 w_k1_MAT_s015
                                 + 1.31521 w_k1_MSFT_s015
                                 + 1.13739 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.08326 w_k1_ADM_s015
                                 + 1.34907 w_k1_AXP_s015
                                 + 1.71223 w_k1_DIS_s015 + 1.4404 w_k1_FDX_s015
                                 + 0.891078 w_k1_IBM_s015
                                 + 1.29172 w_k1_INTC_s015
                                 + 1.40524 w_k1_KO_s015 + 1.40221 w_k1_MAT_s015
                                 + 1.41592 w_k1_MSFT_s015
                                 + 1.31694 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.813627 w_k1_ADM_s015
                                 + 0.611277 w_k1_AXP_s015
                                 + 0.606548 w_k1_DIS_s015
                                 + 0.7103 w_k1_FDX_s015
                                 + 0.703177 w_k1_IBM_s015
                                 + 0.754794 w_k1_INTC_s015
                                 + 0.762984 w_k1_KO_s015
                                 + 0.652164 w_k1_MAT_s015
                                 + 0.711317 w_k1_MSFT_s015
                                 + 0.795397 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.902051 w_k1_ADM_s015
                                 + 0.78196 w_k1_AXP_s015
                                 + 0.854137 w_k1_DIS_s015
                                 + 0.938888 w_k1_FDX_s015
                                 + 0.868925 w_k1_IBM_s015
                                 + 0.964037 w_k1_INTC_s015
                                 + 1.01677 w_k1_KO_s015 + 1.25313 w_k1_MAT_s015
                                 + 0.969136 w_k1_MSFT_s015
                                 + 0.889687 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.32609 w_k1_ADM_s015
                                 + 1.67182 w_k1_AXP_s015
                                 + 1.62356 w_k1_DIS_s015
                                 + 1.63803 w_k1_FDX_s015
                                 + 0.911142 w_k1_IBM_s015
                                 + 1.11597 w_k1_INTC_s015
                                 + 1.31033 w_k1_KO_s015 + 1.91951 w_k1_MAT_s015
                                 + 1.60079 w_k1_MSFT_s015
                                 + 1.42991 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.06835 w_k1_ADM_s015
                                 + 1.2336 w_k1_AXP_s015 + 1.12701 w_k1_DIS_s015
                                 + 1.1725 w_k1_FDX_s015
                                 + 0.944601 w_k1_IBM_s015
                                 + 1.046 w_k1_INTC_s015 + 1.02313 w_k1_KO_s015
                                 + 1.65343 w_k1_MAT_s015
                                 + 1.09431 w_k1_MSFT_s015
                                 + 1.03588 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.00842 w_k1_ADM_s015
                                 + 1.23492 w_k1_AXP_s015
                                 + 1.32454 w_k1_DIS_s015
                                 + 1.08898 w_k1_FDX_s015
                                 + 1.30242 w_k1_IBM_s015
                                 + 0.952588 w_k1_INTC_s015
                                 + 1.37984 w_k1_KO_s015 + 1.91831 w_k1_MAT_s015
                                 + 1.17161 w_k1_MSFT_s015
                                 + 1.20157 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.52874 w_k1_ADM_s016
                                 + 1.51553 w_k1_AXP_s016
                                 + 1.88316 w_k1_DIS_s016
                                 + 1.49642 w_k1_FDX_s016
                                 + 1.26968 w_k1_IBM_s016
                                 + 1.25778 w_k1_INTC_s016
                                 + 1.58384 w_k1_KO_s016 + 1.03955 w_k1_MAT_s016
                                 + 1.24098 w_k1_MSFT_s016 + 1.3524 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.691892 w_k1_ADM_s016
                                 + 0.594891 w_k1_AXP_s016
                                 + 0.566361 w_k1_DIS_s016
                                 + 0.716364 w_k1_FDX_s016
                                 + 0.629058 w_k1_IBM_s016
                                 + 0.650711 w_k1_INTC_s016
                                 + 0.655356 w_k1_KO_s016
                                 + 0.656592 w_k1_MAT_s016
                                 + 0.769526 w_k1_MSFT_s016
                                 + 0.872389 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.950656 w_k1_ADM_s016
                                 + 0.873611 w_k1_AXP_s016
                                 + 0.940905 w_k1_DIS_s016
                                 + 0.904051 w_k1_FDX_s016
                                 + 1.01621 w_k1_IBM_s016
                                 + 0.797425 w_k1_INTC_s016
                                 + 1.08118 w_k1_KO_s016 + 1.15269 w_k1_MAT_s016
                                 + 0.932159 w_k1_MSFT_s016
                                 + 1.03016 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.37404 w_k1_ADM_s016
                                 + 1.37916 w_k1_AXP_s016
                                 + 1.22664 w_k1_DIS_s016
                                 + 1.18079 w_k1_FDX_s016
                                 + 1.37284 w_k1_IBM_s016
                                 + 1.06304 w_k1_INTC_s016
                                 + 1.33434 w_k1_KO_s016 + 1.87101 w_k1_MAT_s016
                                 + 0.991271 w_k1_MSFT_s016
                                 + 1.20822 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.09013 w_k1_ADM_s016
                                 + 0.971568 w_k1_AXP_s016
                                 + 1.26681 w_k1_DIS_s016
                                 + 0.925948 w_k1_FDX_s016
                                 + 1.62039 w_k1_IBM_s016
                                 + 0.881946 w_k1_INTC_s016
                                 + 1.41513 w_k1_KO_s016 + 1.61039 w_k1_MAT_s016
                                 + 0.902861 w_k1_MSFT_s016
                                 + 1.14413 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.13955 w_k1_ADM_s016
                                 + 0.992987 w_k1_AXP_s016
                                 + 1.13578 w_k1_DIS_s016
                                 + 1.14942 w_k1_FDX_s016
                                 + 0.698484 w_k1_IBM_s016
                                 + 0.887626 w_k1_INTC_s016
                                 + 0.995806 w_k1_KO_s016
                                 + 0.648909 w_k1_MAT_s016
                                 + 1.00506 w_k1_MSFT_s016 + 1.1225 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.00344 w_k1_ADM_s016
                                 + 0.798769 w_k1_AXP_s016
                                 + 0.877315 w_k1_DIS_s016
                                 + 0.938413 w_k1_FDX_s016
                                 + 0.66811 w_k1_IBM_s016
                                 + 0.853595 w_k1_INTC_s016
                                 + 0.843462 w_k1_KO_s016
                                 + 0.789451 w_k1_MAT_s016
                                 + 0.964252 w_k1_MSFT_s016
                                 + 0.991549 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.26819 w_k1_ADM_s016
                                 + 1.19682 w_k1_AXP_s016
                                 + 1.47164 w_k1_DIS_s016
                                 + 1.20269 w_k1_FDX_s016
                                 + 1.05319 w_k1_IBM_s016
                                 + 0.905084 w_k1_INTC_s016
                                 + 1.41349 w_k1_KO_s016 + 1.16614 w_k1_MAT_s016
                                 + 1.18064 w_k1_MSFT_s016
                                 + 1.25506 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.674444 w_k1_ADM_s017
                                 + 0.748431 w_k1_AXP_s017
                                 + 0.646667 w_k1_DIS_s017
                                 + 0.814758 w_k1_FDX_s017
                                 + 0.949729 w_k1_IBM_s017
                                 + 1.07576 w_k1_INTC_s017
                                 + 0.814644 w_k1_KO_s017
                                 + 0.82872 w_k1_MAT_s017
                                 + 0.80319 w_k1_MSFT_s017
                                 + 0.874778 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.15538 w_k1_ADM_s017
                                 + 1.22311 w_k1_AXP_s017
                                 + 1.53275 w_k1_DIS_s017
                                 + 1.40986 w_k1_FDX_s017
                                 + 1.03565 w_k1_IBM_s017
                                 + 1.46961 w_k1_INTC_s017
                                 + 1.11328 w_k1_KO_s017
                                 + 0.923589 w_k1_MAT_s017
                                 + 1.4162 w_k1_MSFT_s017 + 1.14936 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.822547 w_k1_ADM_s017
                                 + 1.12642 w_k1_AXP_s017
                                 + 0.944674 w_k1_DIS_s017
                                 + 1.10916 w_k1_FDX_s017
                                 + 0.827346 w_k1_IBM_s017
                                 + 1.46896 w_k1_INTC_s017
                                 + 0.804322 w_k1_KO_s017
                                 + 1.00152 w_k1_MAT_s017
                                 + 0.965088 w_k1_MSFT_s017
                                 + 0.965984 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.845148 w_k1_ADM_s017
                                 + 0.710264 w_k1_AXP_s017
                                 + 0.798324 w_k1_DIS_s017
                                 + 0.904195 w_k1_FDX_s017
                                 + 0.641553 w_k1_IBM_s017
                                 + 0.978045 w_k1_INTC_s017
                                 + 0.767237 w_k1_KO_s017
                                 + 0.673523 w_k1_MAT_s017
                                 + 0.971224 w_k1_MSFT_s017
                                 + 0.85276 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.935002 w_k1_ADM_s017
                                 + 1.39908 w_k1_AXP_s017 + 1.4926 w_k1_DIS_s017
                                 + 1.29733 w_k1_FDX_s017
                                 + 1.53057 w_k1_IBM_s017
                                 + 1.34116 w_k1_INTC_s017
                                 + 1.39402 w_k1_KO_s017 + 2.01768 w_k1_MAT_s017
                                 + 1.29197 w_k1_MSFT_s017
                                 + 1.26446 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.561038 w_k1_ADM_s017
                                 + 0.581134 w_k1_AXP_s017
                                 + 0.486485 w_k1_DIS_s017
                                 + 0.675219 w_k1_FDX_s017
                                 + 0.890019 w_k1_IBM_s017
                                 + 0.92835 w_k1_INTC_s017
                                 + 0.688889 w_k1_KO_s017
                                 + 0.836351 w_k1_MAT_s017
                                 + 0.747728 w_k1_MSFT_s017
                                 + 0.77568 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.946892 w_k1_ADM_s017
                                 + 0.925341 w_k1_AXP_s017
                                 + 0.812231 w_k1_DIS_s017
                                 + 0.958853 w_k1_FDX_s017
                                 + 0.809875 w_k1_IBM_s017
                                 + 0.98551 w_k1_INTC_s017
                                 + 0.795064 w_k1_KO_s017
                                 + 1.19553 w_k1_MAT_s017
                                 + 1.00016 w_k1_MSFT_s017
                                 + 0.999594 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.801194 w_k1_ADM_s017
                                 + 0.866599 w_k1_AXP_s017
                                 + 0.943731 w_k1_DIS_s017
                                 + 0.915006 w_k1_FDX_s017
                                 + 1.15298 w_k1_IBM_s017
                                 + 0.898791 w_k1_INTC_s017
                                 + 0.995138 w_k1_KO_s017
                                 + 0.982451 w_k1_MAT_s017
                                 + 0.898981 w_k1_MSFT_s017
                                 + 1.00282 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.605703 w_k1_ADM_s100
                                 + 0.727918 w_k1_AXP_s100
                                 + 0.621708 w_k1_DIS_s100
                                 + 0.762817 w_k1_FDX_s100
                                 + 0.772976 w_k1_IBM_s100
                                 + 0.732341 w_k1_INTC_s100
                                 + 0.678947 w_k1_KO_s100
                                 + 0.837108 w_k1_MAT_s100
                                 + 0.65985 w_k1_MSFT_s100
                                 + 0.85524 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.19189 w_k1_ADM_s100
                                 + 1.72597 w_k1_AXP_s100
                                 + 1.51841 w_k1_DIS_s100
                                 + 1.48937 w_k1_FDX_s100
                                 + 1.45007 w_k1_IBM_s100
                                 + 1.42772 w_k1_INTC_s100
                                 + 1.24496 w_k1_KO_s100 + 2.16441 w_k1_MAT_s100
                                 + 1.43678 w_k1_MSFT_s100
                                 + 1.36728 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.15776 w_k1_ADM_s100
                                 + 1.19369 w_k1_AXP_s100
                                 + 1.46486 w_k1_DIS_s100
                                 + 1.32331 w_k1_FDX_s100
                                 + 0.928779 w_k1_IBM_s100
                                 + 1.43655 w_k1_INTC_s100
                                 + 1.20388 w_k1_KO_s100 + 1.13359 w_k1_MAT_s100
                                 + 1.29431 w_k1_MSFT_s100
                                 + 1.24606 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.17242 w_k1_ADM_s100
                                 + 1.12375 w_k1_AXP_s100
                                 + 1.33026 w_k1_DIS_s100
                                 + 1.26561 w_k1_FDX_s100
                                 + 1.02381 w_k1_IBM_s100
                                 + 1.2614 w_k1_INTC_s100 + 1.21389 w_k1_KO_s100
                                 + 1.23148 w_k1_MAT_s100
                                 + 1.21968 w_k1_MSFT_s100
                                 + 1.10715 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.788142 w_k1_ADM_s100
                                 + 1.01919 w_k1_AXP_s100
                                 + 0.738293 w_k1_DIS_s100
                                 + 0.860607 w_k1_FDX_s100
                                 + 1.09435 w_k1_IBM_s100
                                 + 1.03624 w_k1_INTC_s100
                                 + 0.879156 w_k1_KO_s100
                                 + 1.42077 w_k1_MAT_s100
                                 + 0.909515 w_k1_MSFT_s100
                                 + 1.00632 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.860853 w_k1_ADM_s100
                                 + 1.05109 w_k1_AXP_s100
                                 + 1.14991 w_k1_DIS_s100
                                 + 1.06029 w_k1_FDX_s100
                                 + 0.755253 w_k1_IBM_s100
                                 + 1.08453 w_k1_INTC_s100
                                 + 0.89318 w_k1_KO_s100 + 1.07486 w_k1_MAT_s100
                                 + 1.04772 w_k1_MSFT_s100
                                 + 1.16573 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.885121 w_k1_ADM_s100
                                 + 1.17454 w_k1_AXP_s100
                                 + 1.24755 w_k1_DIS_s100
                                 + 1.10512 w_k1_FDX_s100
                                 + 1.27164 w_k1_IBM_s100
                                 + 1.28465 w_k1_INTC_s100
                                 + 1.05895 w_k1_KO_s100 + 1.28432 w_k1_MAT_s100
                                 + 1.00176 w_k1_MSFT_s100
                                 + 1.12052 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.949027 w_k1_ADM_s100
                                 + 1.30335 w_k1_AXP_s100
                                 + 1.05629 w_k1_DIS_s100
                                 + 1.00556 w_k1_FDX_s100
                                 + 1.66624 w_k1_IBM_s100
                                 + 1.00484 w_k1_INTC_s100
                                 + 1.33908 w_k1_KO_s100 + 2.67579 w_k1_MAT_s100
                                 + 0.982646 w_k1_MSFT_s100
                                 + 1.18399 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.849561 w_k1_ADM_s101
                                 + 1.21803 w_k1_AXP_s101
                                 + 1.46431 w_k1_DIS_s101
                                 + 1.12008 w_k1_FDX_s101
                                 + 1.52411 w_k1_IBM_s101
                                 + 1.08295 w_k1_INTC_s101
                                 + 1.15329 w_k1_KO_s101 + 1.70653 w_k1_MAT_s101
                                 + 0.980488 w_k1_MSFT_s101
                                 + 1.07458 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.977401 w_k1_ADM_s101
                                 + 1.18069 w_k1_AXP_s101
                                 + 0.908435 w_k1_DIS_s101
                                 + 1.03157 w_k1_FDX_s101
                                 + 0.934847 w_k1_IBM_s101
                                 + 1.19218 w_k1_INTC_s101
                                 + 0.985164 w_k1_KO_s101
                                 + 1.39846 w_k1_MAT_s101
                                 + 1.14311 w_k1_MSFT_s101
                                 + 1.02157 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.679721 w_k1_ADM_s101
                                 + 0.751927 w_k1_AXP_s101
                                 + 0.7296 w_k1_DIS_s101
                                 + 0.776388 w_k1_FDX_s101
                                 + 0.817439 w_k1_IBM_s101
                                 + 0.866719 w_k1_INTC_s101
                                 + 0.678423 w_k1_KO_s101 + 1.0379 w_k1_MAT_s101
                                 + 0.894466 w_k1_MSFT_s101
                                 + 0.840594 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.995616 w_k1_ADM_s101
                                 + 1.14429 w_k1_AXP_s101
                                 + 1.21454 w_k1_DIS_s101
                                 + 1.10023 w_k1_FDX_s101
                                 + 0.898172 w_k1_IBM_s101
                                 + 1.03655 w_k1_INTC_s101
                                 + 1.05076 w_k1_KO_s101 + 1.05368 w_k1_MAT_s101
                                 + 0.915567 w_k1_MSFT_s101
                                 + 1.07516 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.802895 w_k1_ADM_s101
                                 + 0.877727 w_k1_AXP_s101
                                 + 0.773679 w_k1_DIS_s101
                                 + 1.00898 w_k1_FDX_s101
                                 + 0.651034 w_k1_IBM_s101
                                 + 0.876148 w_k1_INTC_s101
                                 + 0.735717 w_k1_KO_s101
                                 + 1.12464 w_k1_MAT_s101
                                 + 0.903615 w_k1_MSFT_s101
                                 + 0.884382 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.07892 w_k1_ADM_s101
                                 + 1.15297 w_k1_AXP_s101
                                 + 1.50423 w_k1_DIS_s101
                                 + 1.46716 w_k1_FDX_s101
                                 + 1.06142 w_k1_IBM_s101
                                 + 1.34185 w_k1_INTC_s101
                                 + 1.21401 w_k1_KO_s101 + 1.36181 w_k1_MAT_s101
                                 + 1.34846 w_k1_MSFT_s101
                                 + 1.18329 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.959593 w_k1_ADM_s101
                                 + 0.969112 w_k1_AXP_s101
                                 + 0.991061 w_k1_DIS_s101
                                 + 0.956173 w_k1_FDX_s101
                                 + 1.16231 w_k1_IBM_s101
                                 + 0.962695 w_k1_INTC_s101
                                 + 0.933297 w_k1_KO_s101 + 1.3393 w_k1_MAT_s101
                                 + 0.912488 w_k1_MSFT_s101
                                 + 1.05121 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.765577 w_k1_ADM_s101
                                 + 0.756897 w_k1_AXP_s101
                                 + 0.726134 w_k1_DIS_s101
                                 + 0.92829 w_k1_FDX_s101
                                 + 0.783599 w_k1_IBM_s101
                                 + 0.911302 w_k1_INTC_s101
                                 + 0.698904 w_k1_KO_s101
                                 + 0.761004 w_k1_MAT_s101
                                 + 0.842229 w_k1_MSFT_s101
                                 + 0.894095 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.810109 w_k1_ADM_s102
                                 + 0.700463 w_k1_AXP_s102
                                 + 0.838336 w_k1_DIS_s102
                                 + 0.855448 w_k1_FDX_s102
                                 + 0.619141 w_k1_IBM_s102
                                 + 0.792398 w_k1_INTC_s102
                                 + 0.724004 w_k1_KO_s102
                                 + 0.474407 w_k1_MAT_s102
                                 + 0.887682 w_k1_MSFT_s102
                                 + 0.799071 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.875475 w_k1_ADM_s102
                                 + 1.27068 w_k1_AXP_s102
                                 + 1.64985 w_k1_DIS_s102
                                 + 1.28229 w_k1_FDX_s102
                                 + 1.24464 w_k1_IBM_s102
                                 + 0.981578 w_k1_INTC_s102
                                 + 1.31741 w_k1_KO_s102 + 1.58741 w_k1_MAT_s102
                                 + 1.19576 w_k1_MSFT_s102
                                 + 1.24302 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.02436 w_k1_ADM_s102
                                 + 0.856338 w_k1_AXP_s102
                                 + 0.899115 w_k1_DIS_s102
                                 + 0.866139 w_k1_FDX_s102
                                 + 1.03044 w_k1_IBM_s102
                                 + 0.754326 w_k1_INTC_s102
                                 + 0.979829 w_k1_KO_s102
                                 + 1.48737 w_k1_MAT_s102
                                 + 0.887743 w_k1_MSFT_s102
                                 + 0.955407 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.761968 w_k1_ADM_s102
                                 + 1.19556 w_k1_AXP_s102
                                 + 0.931646 w_k1_DIS_s102
                                 + 0.963746 w_k1_FDX_s102
                                 + 1.22728 w_k1_IBM_s102
                                 + 1.03234 w_k1_INTC_s102
                                 + 1.04361 w_k1_KO_s102 + 1.80569 w_k1_MAT_s102
                                 + 1.05372 w_k1_MSFT_s102
                                 + 0.952238 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.02482 w_k1_ADM_s102
                                 + 0.960264 w_k1_AXP_s102
                                 + 1.20318 w_k1_DIS_s102 + 1.0416 w_k1_FDX_s102
                                 + 0.948967 w_k1_IBM_s102
                                 + 0.892182 w_k1_INTC_s102
                                 + 1.17342 w_k1_KO_s102 + 1.043 w_k1_MAT_s102
                                 + 1.10325 w_k1_MSFT_s102
                                 + 1.06683 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.546528 w_k1_ADM_s102
                                 + 0.712145 w_k1_AXP_s102
                                 + 0.597484 w_k1_DIS_s102
                                 + 0.640208 w_k1_FDX_s102
                                 + 0.824824 w_k1_IBM_s102
                                 + 0.837993 w_k1_INTC_s102
                                 + 0.756332 w_k1_KO_s102
                                 + 0.93769 w_k1_MAT_s102
                                 + 0.807663 w_k1_MSFT_s102
                                 + 0.713339 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.801677 w_k1_ADM_s102
                                 + 1.07379 w_k1_AXP_s102
                                 + 1.26277 w_k1_DIS_s102
                                 + 1.05696 w_k1_FDX_s102
                                 + 1.63599 w_k1_IBM_s102
                                 + 0.988973 w_k1_INTC_s102
                                 + 1.33556 w_k1_KO_s102 + 1.88856 w_k1_MAT_s102
                                 + 1.12837 w_k1_MSFT_s102
                                 + 1.24481 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.749064 w_k1_ADM_s102
                                 + 1.12118 w_k1_AXP_s102
                                 + 0.795699 w_k1_DIS_s102
                                 + 0.902656 w_k1_FDX_s102
                                 + 0.946838 w_k1_IBM_s102
                                 + 0.847663 w_k1_INTC_s102
                                 + 0.770986 w_k1_KO_s102
                                 + 1.19873 w_k1_MAT_s102
                                 + 0.8828 w_k1_MSFT_s102 + 0.8936 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.36045 w_k1_ADM_s103
                                 + 1.29892 w_k1_AXP_s103
                                 + 1.12098 w_k1_DIS_s103
                                 + 1.12914 w_k1_FDX_s103
                                 + 1.32626 w_k1_IBM_s103
                                 + 1.06577 w_k1_INTC_s103
                                 + 1.42758 w_k1_KO_s103 + 1.26829 w_k1_MAT_s103
                                 + 1.16627 w_k1_MSFT_s103 + 1.1248 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.0132 w_k1_ADM_s103
                                 + 0.834211 w_k1_AXP_s103
                                 + 1.00761 w_k1_DIS_s103
                                 + 0.990731 w_k1_FDX_s103
                                 + 0.743974 w_k1_IBM_s103
                                 + 1.07418 w_k1_INTC_s103
                                 + 1.05313 w_k1_KO_s103
                                 + 0.765673 w_k1_MAT_s103
                                 + 1.05305 w_k1_MSFT_s103
                                 + 0.861522 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.922219 w_k1_ADM_s103
                                 + 0.835953 w_k1_AXP_s103
                                 + 0.881553 w_k1_DIS_s103
                                 + 0.984168 w_k1_FDX_s103
                                 + 1.01544 w_k1_IBM_s103
                                 + 1.1219 w_k1_INTC_s103 + 1.08872 w_k1_KO_s103
                                 + 0.887216 w_k1_MAT_s103
                                 + 0.877113 w_k1_MSFT_s103
                                 + 0.924901 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.06376 w_k1_ADM_s103
                                 + 0.970238 w_k1_AXP_s103
                                 + 1.28872 w_k1_DIS_s103
                                 + 1.21195 w_k1_FDX_s103 + 1.0165 w_k1_IBM_s103
                                 + 1.41596 w_k1_INTC_s103
                                 + 1.34361 w_k1_KO_s103 + 1.10476 w_k1_MAT_s103
                                 + 1.14596 w_k1_MSFT_s103
                                 + 1.04338 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.17975 w_k1_ADM_s103
                                 + 0.967509 w_k1_AXP_s103
                                 + 0.991019 w_k1_DIS_s103
                                 + 0.897177 w_k1_FDX_s103
                                 + 1.38942 w_k1_IBM_s103
                                 + 1.11635 w_k1_INTC_s103
                                 + 1.26477 w_k1_KO_s103 + 1.34281 w_k1_MAT_s103
                                 + 0.910364 w_k1_MSFT_s103
                                 + 1.03923 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.47003 w_k1_ADM_s103
                                 + 1.17495 w_k1_AXP_s103
                                 + 1.82221 w_k1_DIS_s103
                                 + 1.39727 w_k1_FDX_s103
                                 + 1.07278 w_k1_IBM_s103
                                 + 1.43045 w_k1_INTC_s103
                                 + 1.37922 w_k1_KO_s103 + 1.00355 w_k1_MAT_s103
                                 + 1.44886 w_k1_MSFT_s103 + 1.2984 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.853758 w_k1_ADM_s103
                                 + 0.781664 w_k1_AXP_s103
                                 + 0.706047 w_k1_DIS_s103
                                 + 0.747444 w_k1_FDX_s103
                                 + 0.914608 w_k1_IBM_s103
                                 + 0.800907 w_k1_INTC_s103
                                 + 0.775474 w_k1_KO_s103
                                 + 0.828411 w_k1_MAT_s103
                                 + 0.708126 w_k1_MSFT_s103
                                 + 0.831705 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.31776 w_k1_ADM_s103
                                 + 1.31868 w_k1_AXP_s103
                                 + 1.43799 w_k1_DIS_s103
                                 + 1.26027 w_k1_FDX_s103 + 1.5112 w_k1_IBM_s103
                                 + 1.37337 w_k1_INTC_s103
                                 + 1.39669 w_k1_KO_s103 + 1.41832 w_k1_MAT_s103
                                 + 1.22781 w_k1_MSFT_s103
                                 + 1.20519 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.55039 w_k1_ADM_s104
                                 + 1.24066 w_k1_AXP_s104
                                 + 1.33951 w_k1_DIS_s104
                                 + 1.27697 w_k1_FDX_s104
                                 + 0.989032 w_k1_IBM_s104
                                 + 1.08831 w_k1_INTC_s104
                                 + 1.08851 w_k1_KO_s104
                                 + 0.911739 w_k1_MAT_s104
                                 + 1.16988 w_k1_MSFT_s104 + 1.2901 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.17658 w_k1_ADM_s104
                                 + 1.19782 w_k1_AXP_s104
                                 + 1.26067 w_k1_DIS_s104
                                 + 1.10443 w_k1_FDX_s104
                                 + 1.19409 w_k1_IBM_s104
                                 + 1.23358 w_k1_INTC_s104
                                 + 1.37337 w_k1_KO_s104 + 1.29785 w_k1_MAT_s104
                                 + 1.25029 w_k1_MSFT_s104
                                 + 1.31377 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.22508 w_k1_ADM_s104
                                 + 1.3118 w_k1_AXP_s104 + 1.31256 w_k1_DIS_s104
                                 + 1.05406 w_k1_FDX_s104
                                 + 1.57232 w_k1_IBM_s104
                                 + 1.02095 w_k1_INTC_s104
                                 + 1.27866 w_k1_KO_s104 + 1.67804 w_k1_MAT_s104
                                 + 1.04988 w_k1_MSFT_s104
                                 + 1.10317 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.12313 w_k1_ADM_s104
                                 + 0.847444 w_k1_AXP_s104
                                 + 0.798831 w_k1_DIS_s104
                                 + 0.877544 w_k1_FDX_s104
                                 + 0.806393 w_k1_IBM_s104
                                 + 0.863293 w_k1_INTC_s104
                                 + 0.88101 w_k1_KO_s104
                                 + 0.659249 w_k1_MAT_s104
                                 + 0.918523 w_k1_MSFT_s104
                                 + 0.889489 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.778657 w_k1_ADM_s104
                                 + 0.912711 w_k1_AXP_s104
                                 + 0.873366 w_k1_DIS_s104
                                 + 0.821174 w_k1_FDX_s104
                                 + 1.25234 w_k1_IBM_s104
                                 + 0.828391 w_k1_INTC_s104
                                 + 1.00133 w_k1_KO_s104 + 1.2266 w_k1_MAT_s104
                                 + 0.849222 w_k1_MSFT_s104
                                 + 0.885399 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.944478 w_k1_ADM_s104
                                 + 0.668856 w_k1_AXP_s104
                                 + 0.876055 w_k1_DIS_s104
                                 + 0.882777 w_k1_FDX_s104
                                 + 1.01864 w_k1_IBM_s104
                                 + 0.821684 w_k1_INTC_s104
                                 + 1.10611 w_k1_KO_s104 + 1.0305 w_k1_MAT_s104
                                 + 1.04423 w_k1_MSFT_s104
                                 + 0.934648 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.910801 w_k1_ADM_s104
                                 + 1.06065 w_k1_AXP_s104
                                 + 0.802692 w_k1_DIS_s104
                                 + 0.932873 w_k1_FDX_s104
                                 + 0.996826 w_k1_IBM_s104
                                 + 0.879323 w_k1_INTC_s104
                                 + 0.705974 w_k1_KO_s104
                                 + 1.27459 w_k1_MAT_s104
                                 + 0.938561 w_k1_MSFT_s104
                                 + 0.844677 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.19352 w_k1_ADM_s104
                                 + 0.679367 w_k1_AXP_s104
                                 + 0.86478 w_k1_DIS_s104
                                 + 0.887266 w_k1_FDX_s104
                                 + 0.639456 w_k1_IBM_s104
                                 + 0.914447 w_k1_INTC_s104
                                 + 0.740439 w_k1_KO_s104
                                 + 0.638441 w_k1_MAT_s104
                                 + 1.11652 w_k1_MSFT_s104
                                 + 0.876972 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.94016 w_k1_ADM_s105
                                 + 0.86052 w_k1_AXP_s105
                                 + 0.963946 w_k1_DIS_s105
                                 + 0.87098 w_k1_FDX_s105
                                 + 1.08204 w_k1_IBM_s105
                                 + 0.994269 w_k1_INTC_s105
                                 + 1.02457 w_k1_KO_s105
                                 + 0.836165 w_k1_MAT_s105
                                 + 1.09199 w_k1_MSFT_s105
                                 + 1.01107 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.67145 w_k1_ADM_s105
                                 + 1.4396 w_k1_AXP_s105 + 1.60625 w_k1_DIS_s105
                                 + 1.25272 w_k1_FDX_s105
                                 + 1.14401 w_k1_IBM_s105
                                 + 1.17189 w_k1_INTC_s105
                                 + 1.22413 w_k1_KO_s105 + 1.00241 w_k1_MAT_s105
                                 + 1.35253 w_k1_MSFT_s105
                                 + 1.10062 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.825555 w_k1_ADM_s105
                                 + 0.889761 w_k1_AXP_s105
                                 + 0.832517 w_k1_DIS_s105
                                 + 0.923525 w_k1_FDX_s105
                                 + 0.750648 w_k1_IBM_s105
                                 + 0.925345 w_k1_INTC_s105
                                 + 0.738485 w_k1_KO_s105
                                 + 0.729691 w_k1_MAT_s105
                                 + 0.982139 w_k1_MSFT_s105
                                 + 0.789901 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.642189 w_k1_ADM_s105
                                 + 0.623552 w_k1_AXP_s105
                                 + 0.586191 w_k1_DIS_s105
                                 + 0.633956 w_k1_FDX_s105
                                 + 0.982989 w_k1_IBM_s105
                                 + 0.817081 w_k1_INTC_s105
                                 + 0.790365 w_k1_KO_s105
                                 + 0.753046 w_k1_MAT_s105
                                 + 0.740969 w_k1_MSFT_s105
                                 + 0.771794 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.09657 w_k1_ADM_s105
                                 + 0.81362 w_k1_AXP_s105
                                 + 0.933878 w_k1_DIS_s105
                                 + 0.846011 w_k1_FDX_s105
                                 + 1.30386 w_k1_IBM_s105
                                 + 0.842806 w_k1_INTC_s105
                                 + 1.11526 w_k1_KO_s105
                                 + 0.959237 w_k1_MAT_s105
                                 + 0.958229 w_k1_MSFT_s105
                                 + 0.93092 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.22646 w_k1_ADM_s105
                                 + 1.19891 w_k1_AXP_s105
                                 + 0.950241 w_k1_DIS_s105
                                 + 1.10813 w_k1_FDX_s105
                                 + 0.751839 w_k1_IBM_s105
                                 + 1.11538 w_k1_INTC_s105
                                 + 0.834935 w_k1_KO_s105
                                 + 0.546258 w_k1_MAT_s105
                                 + 1.19284 w_k1_MSFT_s105
                                 + 0.933252 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.988657 w_k1_ADM_s105
                                 + 1.27526 w_k1_AXP_s105
                                 + 1.20813 w_k1_DIS_s105
                                 + 1.32381 w_k1_FDX_s105
                                 + 0.910042 w_k1_IBM_s105
                                 + 1.42086 w_k1_INTC_s105
                                 + 1.17137 w_k1_KO_s105
                                 + 0.907592 w_k1_MAT_s105
                                 + 1.27153 w_k1_MSFT_s105
                                 + 1.06078 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.894582 w_k1_ADM_s105
                                 + 1.40481 w_k1_AXP_s105
                                 + 1.39493 w_k1_DIS_s105
                                 + 1.24325 w_k1_FDX_s105
                                 + 1.50122 w_k1_IBM_s105
                                 + 1.2657 w_k1_INTC_s105 + 1.34386 w_k1_KO_s105
                                 + 1.3952 w_k1_MAT_s105
                                 + 1.27549 w_k1_MSFT_s105
                                 + 1.25211 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.790971 w_k1_ADM_s106
                                 + 0.782234 w_k1_AXP_s106
                                 + 0.626329 w_k1_DIS_s106
                                 + 0.733159 w_k1_FDX_s106
                                 + 1.02127 w_k1_IBM_s106
                                 + 1.47294 w_k1_INTC_s106
                                 + 0.97365 w_k1_KO_s106
                                 + 0.715739 w_k1_MAT_s106
                                 + 0.939901 w_k1_MSFT_s106
                                 + 0.871575 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.900549 w_k1_ADM_s106
                                 + 0.821677 w_k1_AXP_s106
                                 + 0.888153 w_k1_DIS_s106
                                 + 0.870806 w_k1_FDX_s106
                                 + 0.834006 w_k1_IBM_s106
                                 + 1.39921 w_k1_INTC_s106
                                 + 0.853892 w_k1_KO_s106
                                 + 0.646161 w_k1_MAT_s106
                                 + 1.00662 w_k1_MSFT_s106
                                 + 0.974806 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.830625 w_k1_ADM_s106
                                 + 1.03772 w_k1_AXP_s106
                                 + 1.02309 w_k1_DIS_s106
                                 + 1.10678 w_k1_FDX_s106
                                 + 1.43599 w_k1_IBM_s106
                                 + 1.15357 w_k1_INTC_s106
                                 + 1.01378 w_k1_KO_s106 + 1.45522 w_k1_MAT_s106
                                 + 1.01333 w_k1_MSFT_s106
                                 + 0.838653 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.33523 w_k1_ADM_s106
                                 + 1.88238 w_k1_AXP_s106
                                 + 1.89302 w_k1_DIS_s106
                                 + 1.47588 w_k1_FDX_s106
                                 + 1.41631 w_k1_IBM_s106
                                 + 1.63742 w_k1_INTC_s106
                                 + 1.48301 w_k1_KO_s106 + 1.64216 w_k1_MAT_s106
                                 + 1.3742 w_k1_MSFT_s106 + 1.16928 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.12594 w_k1_ADM_s106
                                 + 0.990389 w_k1_AXP_s106
                                 + 0.851315 w_k1_DIS_s106
                                 + 1.1036 w_k1_FDX_s106
                                 + 0.719114 w_k1_IBM_s106
                                 + 1.22346 w_k1_INTC_s106
                                 + 0.92343 w_k1_KO_s106
                                 + 0.783689 w_k1_MAT_s106
                                 + 1.07134 w_k1_MSFT_s106
                                 + 0.984653 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.46988 w_k1_ADM_s106
                                 + 1.78985 w_k1_AXP_s106
                                 + 1.34108 w_k1_DIS_s106
                                 + 1.26987 w_k1_FDX_s106
                                 + 0.98872 w_k1_IBM_s106
                                 + 1.73644 w_k1_INTC_s106
                                 + 1.27028 w_k1_KO_s106 + 1.29411 w_k1_MAT_s106
                                 + 1.38447 w_k1_MSFT_s106
                                 + 1.25198 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.682515 w_k1_ADM_s106
                                 + 0.765561 w_k1_AXP_s106
                                 + 0.604016 w_k1_DIS_s106
                                 + 0.722082 w_k1_FDX_s106
                                 + 0.653559 w_k1_IBM_s106
                                 + 1.39981 w_k1_INTC_s106
                                 + 0.689404 w_k1_KO_s106
                                 + 0.578863 w_k1_MAT_s106
                                 + 0.828356 w_k1_MSFT_s106
                                 + 0.72707 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.26763 w_k1_ADM_s106
                                 + 1.22822 w_k1_AXP_s106
                                 + 0.67722 w_k1_DIS_s106
                                 + 1.04031 w_k1_FDX_s106
                                 + 0.791815 w_k1_IBM_s106
                                 + 1.27996 w_k1_INTC_s106
                                 + 0.900471 w_k1_KO_s106
                                 + 0.808327 w_k1_MAT_s106
                                 + 1.01397 w_k1_MSFT_s106
                                 + 1.08772 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.01932 w_k1_ADM_s107
                                 + 1.38535 w_k1_AXP_s107 + 1.4385 w_k1_DIS_s107
                                 + 1.35424 w_k1_FDX_s107
                                 + 0.812463 w_k1_IBM_s107
                                 + 1.05637 w_k1_INTC_s107
                                 + 0.966418 w_k1_KO_s107
                                 + 0.881127 w_k1_MAT_s107
                                 + 1.27741 w_k1_MSFT_s107
                                 + 0.954468 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.644789 w_k1_ADM_s107
                                 + 0.840191 w_k1_AXP_s107
                                 + 0.834633 w_k1_DIS_s107
                                 + 0.877181 w_k1_FDX_s107
                                 + 0.810873 w_k1_IBM_s107
                                 + 0.769622 w_k1_INTC_s107
                                 + 0.845536 w_k1_KO_s107
                                 + 0.88594 w_k1_MAT_s107
                                 + 0.865758 w_k1_MSFT_s107
                                 + 0.83007 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.08298 w_k1_ADM_s107
                                 + 1.08364 w_k1_AXP_s107
                                 + 1.54193 w_k1_DIS_s107
                                 + 1.20396 w_k1_FDX_s107
                                 + 1.30319 w_k1_IBM_s107
                                 + 1.04501 w_k1_INTC_s107
                                 + 1.07163 w_k1_KO_s107 + 1.05784 w_k1_MAT_s107
                                 + 1.30462 w_k1_MSFT_s107
                                 + 1.12771 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.65354 w_k1_ADM_s107
                                 + 0.650916 w_k1_AXP_s107
                                 + 0.684279 w_k1_DIS_s107
                                 + 0.750524 w_k1_FDX_s107
                                 + 1.1145 w_k1_IBM_s107
                                 + 0.613646 w_k1_INTC_s107
                                 + 0.849282 w_k1_KO_s107
                                 + 0.976224 w_k1_MAT_s107
                                 + 0.75554 w_k1_MSFT_s107
                                 + 0.851374 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.74386 w_k1_ADM_s107
                                 + 0.796166 w_k1_AXP_s107
                                 + 1.00002 w_k1_DIS_s107
                                 + 0.913971 w_k1_FDX_s107
                                 + 1.15558 w_k1_IBM_s107
                                 + 1.03094 w_k1_INTC_s107
                                 + 1.10286 w_k1_KO_s107 + 1.00406 w_k1_MAT_s107
                                 + 1.0617 w_k1_MSFT_s107
                                 + 0.949503 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.26921 w_k1_ADM_s107
                                 + 1.41471 w_k1_AXP_s107
                                 + 1.27066 w_k1_DIS_s107
                                 + 1.16514 w_k1_FDX_s107
                                 + 1.35851 w_k1_IBM_s107
                                 + 1.1117 w_k1_INTC_s107 + 1.12525 w_k1_KO_s107
                                 + 1.33586 w_k1_MAT_s107
                                 + 1.13745 w_k1_MSFT_s107
                                 + 1.07065 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.975913 w_k1_ADM_s107
                                 + 1.42393 w_k1_AXP_s107
                                 + 1.33959 w_k1_DIS_s107
                                 + 1.16418 w_k1_FDX_s107
                                 + 1.90526 w_k1_IBM_s107
                                 + 1.02882 w_k1_INTC_s107
                                 + 1.25095 w_k1_KO_s107 + 1.61667 w_k1_MAT_s107
                                 + 0.964753 w_k1_MSFT_s107
                                 + 1.02564 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.771518 w_k1_ADM_s107
                                 + 0.741781 w_k1_AXP_s107
                                 + 0.803963 w_k1_DIS_s107
                                 + 0.929953 w_k1_FDX_s107
                                 + 0.610981 w_k1_IBM_s107
                                 + 1.14902 w_k1_INTC_s107
                                 + 0.729921 w_k1_KO_s107
                                 + 0.494339 w_k1_MAT_s107
                                 + 1.0643 w_k1_MSFT_s107
                                 + 0.891198 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.805552 w_k1_ADM_s110
                                 + 0.909189 w_k1_AXP_s110
                                 + 0.662114 w_k1_DIS_s110
                                 + 0.862417 w_k1_FDX_s110
                                 + 0.711935 w_k1_IBM_s110
                                 + 1.19473 w_k1_INTC_s110
                                 + 0.749575 w_k1_KO_s110
                                 + 0.879612 w_k1_MAT_s110
                                 + 0.864995 w_k1_MSFT_s110
                                 + 0.940225 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.877179 w_k1_ADM_s110
                                 + 0.981047 w_k1_AXP_s110
                                 + 0.895368 w_k1_DIS_s110
                                 + 0.787969 w_k1_FDX_s110
                                 + 1.30884 w_k1_IBM_s110
                                 + 0.940269 w_k1_INTC_s110
                                 + 0.995082 w_k1_KO_s110
                                 + 0.968901 w_k1_MAT_s110
                                 + 0.824001 w_k1_MSFT_s110
                                 + 1.02475 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.942854 w_k1_ADM_s110
                                 + 1.6364 w_k1_AXP_s110 + 1.37206 w_k1_DIS_s110
                                 + 1.03027 w_k1_FDX_s110
                                 + 1.52934 w_k1_IBM_s110
                                 + 1.37383 w_k1_INTC_s110
                                 + 1.38885 w_k1_KO_s110 + 1.67434 w_k1_MAT_s110
                                 + 0.952039 w_k1_MSFT_s110
                                 + 1.29368 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.05094 w_k1_ADM_s110
                                 + 0.994225 w_k1_AXP_s110
                                 + 1.07766 w_k1_DIS_s110
                                 + 1.06075 w_k1_FDX_s110
                                 + 0.683778 w_k1_IBM_s110
                                 + 1.22642 w_k1_INTC_s110
                                 + 0.78053 w_k1_KO_s110
                                 + 0.661362 w_k1_MAT_s110
                                 + 0.946603 w_k1_MSFT_s110
                                 + 0.858148 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.14187 w_k1_ADM_s110
                                 + 1.35148 w_k1_AXP_s110
                                 + 1.45811 w_k1_DIS_s110
                                 + 1.14307 w_k1_FDX_s110 + 1.1283 w_k1_IBM_s110
                                 + 1.37248 w_k1_INTC_s110 + 1.1759 w_k1_KO_s110
                                 + 1.48138 w_k1_MAT_s110
                                 + 1.1053 w_k1_MSFT_s110 + 1.1832 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.680394 w_k1_ADM_s110
                                 + 0.571732 w_k1_AXP_s110
                                 + 0.632148 w_k1_DIS_s110
                                 + 0.779404 w_k1_FDX_s110
                                 + 0.528563 w_k1_IBM_s110
                                 + 1.01281 w_k1_INTC_s110
                                 + 0.713254 w_k1_KO_s110
                                 + 0.464246 w_k1_MAT_s110
                                 + 0.752264 w_k1_MSFT_s110
                                 + 0.868254 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.608827 w_k1_ADM_s110
                                 + 0.826835 w_k1_AXP_s110
                                 + 0.655008 w_k1_DIS_s110
                                 + 0.737212 w_k1_FDX_s110
                                 + 1.06177 w_k1_IBM_s110
                                 + 0.880121 w_k1_INTC_s110
                                 + 0.828533 w_k1_KO_s110
                                 + 1.11766 w_k1_MAT_s110
                                 + 0.746129 w_k1_MSFT_s110
                                 + 0.860056 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.06305 w_k1_ADM_s110
                                 + 1.18154 w_k1_AXP_s110 + 1.1362 w_k1_DIS_s110
                                 + 1.0784 w_k1_FDX_s110
                                 + 0.914549 w_k1_IBM_s110
                                 + 1.18557 w_k1_INTC_s110
                                 + 1.16414 w_k1_KO_s110 + 1.20143 w_k1_MAT_s110
                                 + 1.05409 w_k1_MSFT_s110
                                 + 1.10799 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.25087 w_k1_ADM_s111
                                 + 1.13613 w_k1_AXP_s111
                                 + 0.965271 w_k1_DIS_s111
                                 + 1.12336 w_k1_FDX_s111
                                 + 0.763069 w_k1_IBM_s111
                                 + 0.98457 w_k1_INTC_s111
                                 + 0.853309 w_k1_KO_s111
                                 + 0.96943 w_k1_MAT_s111
                                 + 1.03716 w_k1_MSFT_s111
                                 + 0.980626 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.824656 w_k1_ADM_s111
                                 + 0.698864 w_k1_AXP_s111
                                 + 0.941019 w_k1_DIS_s111
                                 + 0.912754 w_k1_FDX_s111
                                 + 0.907385 w_k1_IBM_s111
                                 + 1.04302 w_k1_INTC_s111
                                 + 0.905992 w_k1_KO_s111
                                 + 1.00565 w_k1_MAT_s111
                                 + 1.05547 w_k1_MSFT_s111
                                 + 0.872849 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.48656 w_k1_ADM_s111
                                 + 1.28378 w_k1_AXP_s111
                                 + 1.54462 w_k1_DIS_s111
                                 + 1.33821 w_k1_FDX_s111
                                 + 1.26486 w_k1_IBM_s111
                                 + 1.04794 w_k1_INTC_s111
                                 + 1.41302 w_k1_KO_s111 + 1.67754 w_k1_MAT_s111
                                 + 1.37931 w_k1_MSFT_s111
                                 + 1.25439 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.902851 w_k1_ADM_s111
                                 + 0.89726 w_k1_AXP_s111
                                 + 0.949738 w_k1_DIS_s111
                                 + 0.865293 w_k1_FDX_s111
                                 + 1.41365 w_k1_IBM_s111
                                 + 0.944668 w_k1_INTC_s111
                                 + 1.18093 w_k1_KO_s111 + 1.60698 w_k1_MAT_s111
                                 + 0.995914 w_k1_MSFT_s111
                                 + 1.00373 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.12377 w_k1_ADM_s111
                                 + 1.04299 w_k1_AXP_s111
                                 + 0.837095 w_k1_DIS_s111
                                 + 0.988432 w_k1_FDX_s111
                                 + 1.08359 w_k1_IBM_s111
                                 + 0.935174 w_k1_INTC_s111
                                 + 0.935331 w_k1_KO_s111 + 1.4554 w_k1_MAT_s111
                                 + 0.972273 w_k1_MSFT_s111
                                 + 0.887704 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.791146 w_k1_ADM_s111
                                 + 0.711356 w_k1_AXP_s111
                                 + 0.740848 w_k1_DIS_s111
                                 + 0.856652 w_k1_FDX_s111
                                 + 0.739099 w_k1_IBM_s111
                                 + 1.00431 w_k1_INTC_s111
                                 + 0.790332 w_k1_KO_s111
                                 + 0.860455 w_k1_MAT_s111
                                 + 0.884391 w_k1_MSFT_s111
                                 + 0.850646 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.816602 w_k1_ADM_s111
                                 + 0.848941 w_k1_AXP_s111
                                 + 0.847379 w_k1_DIS_s111
                                 + 0.908926 w_k1_FDX_s111
                                 + 0.682799 w_k1_IBM_s111
                                 + 1.05102 w_k1_INTC_s111
                                 + 0.726908 w_k1_KO_s111
                                 + 0.905276 w_k1_MAT_s111
                                 + 0.912393 w_k1_MSFT_s111
                                 + 0.728253 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.741457 w_k1_ADM_s111
                                 + 0.690442 w_k1_AXP_s111
                                 + 0.511394 w_k1_DIS_s111
                                 + 0.698466 w_k1_FDX_s111
                                 + 0.743143 w_k1_IBM_s111
                                 + 0.873406 w_k1_INTC_s111
                                 + 0.704173 w_k1_KO_s111
                                 + 0.851869 w_k1_MAT_s111
                                 + 0.717996 w_k1_MSFT_s111
                                 + 0.745037 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.2871 w_k1_ADM_s112
                                 + 1.04715 w_k1_AXP_s112
                                 + 1.11069 w_k1_DIS_s112
                                 + 0.931723 w_k1_FDX_s112
                                 + 1.56621 w_k1_IBM_s112
                                 + 0.96189 w_k1_INTC_s112
                                 + 1.37574 w_k1_KO_s112 + 1.71013 w_k1_MAT_s112
                                 + 1.1421 w_k1_MSFT_s112 + 1.14916 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.817021 w_k1_ADM_s112
                                 + 0.636551 w_k1_AXP_s112
                                 + 0.742182 w_k1_DIS_s112
                                 + 0.71776 w_k1_FDX_s112 + 1.1234 w_k1_IBM_s112
                                 + 0.670738 w_k1_INTC_s112
                                 + 0.939781 w_k1_KO_s112
                                 + 1.02185 w_k1_MAT_s112
                                 + 0.870319 w_k1_MSFT_s112
                                 + 0.840817 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.22458 w_k1_ADM_s112
                                 + 0.76623 w_k1_AXP_s112
                                 + 0.897593 w_k1_DIS_s112
                                 + 0.978906 w_k1_FDX_s112
                                 + 0.637677 w_k1_IBM_s112
                                 + 0.958441 w_k1_INTC_s112
                                 + 0.769919 w_k1_KO_s112
                                 + 0.729298 w_k1_MAT_s112
                                 + 1.11292 w_k1_MSFT_s112
                                 + 0.948052 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.13348 w_k1_ADM_s112
                                 + 1.04112 w_k1_AXP_s112
                                 + 0.993804 w_k1_DIS_s112
                                 + 1.01579 w_k1_FDX_s112
                                 + 1.13101 w_k1_IBM_s112
                                 + 0.825077 w_k1_INTC_s112
                                 + 1.03064 w_k1_KO_s112 + 1.0343 w_k1_MAT_s112
                                 + 0.950667 w_k1_MSFT_s112
                                 + 1.05018 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.19876 w_k1_ADM_s112
                                 + 1.02352 w_k1_AXP_s112
                                 + 1.24297 w_k1_DIS_s112
                                 + 1.22863 w_k1_FDX_s112
                                 + 0.761028 w_k1_IBM_s112
                                 + 1.0127 w_k1_INTC_s112
                                 + 0.941425 w_k1_KO_s112
                                 + 0.836139 w_k1_MAT_s112
                                 + 1.29046 w_k1_MSFT_s112
                                 + 0.979464 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.953397 w_k1_ADM_s112
                                 + 0.933687 w_k1_AXP_s112
                                 + 0.890198 w_k1_DIS_s112
                                 + 0.950233 w_k1_FDX_s112
                                 + 0.956146 w_k1_IBM_s112
                                 + 0.966752 w_k1_INTC_s112
                                 + 0.955228 w_k1_KO_s112
                                 + 1.30975 w_k1_MAT_s112
                                 + 1.00038 w_k1_MSFT_s112
                                 + 0.904648 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.17272 w_k1_ADM_s112
                                 + 1.35837 w_k1_AXP_s112
                                 + 1.61862 w_k1_DIS_s112
                                 + 1.23369 w_k1_FDX_s112 + 1.229 w_k1_IBM_s112
                                 + 1.1401 w_k1_INTC_s112 + 1.25528 w_k1_KO_s112
                                 + 1.41037 w_k1_MAT_s112
                                 + 1.2989 w_k1_MSFT_s112 + 1.26513 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.921333 w_k1_ADM_s112
                                 + 0.715154 w_k1_AXP_s112
                                 + 0.794581 w_k1_DIS_s112
                                 + 0.789814 w_k1_FDX_s112
                                 + 0.758348 w_k1_IBM_s112
                                 + 0.834955 w_k1_INTC_s112
                                 + 0.722035 w_k1_KO_s112
                                 + 0.556887 w_k1_MAT_s112
                                 + 0.872648 w_k1_MSFT_s112
                                 + 0.818036 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.51368 w_k1_ADM_s113
                                 + 1.25076 w_k1_AXP_s113
                                 + 1.59341 w_k1_DIS_s113
                                 + 1.29696 w_k1_FDX_s113
                                 + 1.28131 w_k1_IBM_s113
                                 + 1.29832 w_k1_INTC_s113
                                 + 1.32078 w_k1_KO_s113
                                 + 0.904599 w_k1_MAT_s113
                                 + 1.36721 w_k1_MSFT_s113
                                 + 1.17157 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.15732 w_k1_ADM_s113
                                 + 1.21163 w_k1_AXP_s113
                                 + 0.934362 w_k1_DIS_s113
                                 + 1.0448 w_k1_FDX_s113 + 1.17714 w_k1_IBM_s113
                                 + 0.98258 w_k1_INTC_s113
                                 + 0.924863 w_k1_KO_s113 + 1.0031 w_k1_MAT_s113
                                 + 1.02979 w_k1_MSFT_s113
                                 + 0.966233 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.22962 w_k1_ADM_s113
                                 + 0.892551 w_k1_AXP_s113
                                 + 1.16154 w_k1_DIS_s113
                                 + 1.06788 w_k1_FDX_s113
                                 + 0.75124 w_k1_IBM_s113
                                 + 1.18038 w_k1_INTC_s113
                                 + 1.00171 w_k1_KO_s113
                                 + 0.525725 w_k1_MAT_s113
                                 + 1.10539 w_k1_MSFT_s113
                                 + 0.985839 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.04428 w_k1_ADM_s113
                                 + 0.887444 w_k1_AXP_s113
                                 + 0.738664 w_k1_DIS_s113
                                 + 0.950196 w_k1_FDX_s113
                                 + 0.704869 w_k1_IBM_s113
                                 + 0.950091 w_k1_INTC_s113
                                 + 0.754352 w_k1_KO_s113
                                 + 0.679825 w_k1_MAT_s113
                                 + 0.854973 w_k1_MSFT_s113
                                 + 0.933463 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.03851 w_k1_ADM_s113
                                 + 0.813067 w_k1_AXP_s113
                                 + 0.658854 w_k1_DIS_s113
                                 + 0.78448 w_k1_FDX_s113
                                 + 0.968232 w_k1_IBM_s113
                                 + 0.723474 w_k1_INTC_s113
                                 + 0.766486 w_k1_KO_s113
                                 + 0.784469 w_k1_MAT_s113
                                 + 0.844203 w_k1_MSFT_s113
                                 + 0.907255 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.50872 w_k1_ADM_s113
                                 + 1.28094 w_k1_AXP_s113
                                 + 1.40333 w_k1_DIS_s113
                                 + 1.32277 w_k1_FDX_s113
                                 + 0.967517 w_k1_IBM_s113
                                 + 0.925925 w_k1_INTC_s113
                                 + 1.30148 w_k1_KO_s113 + 1.01924 w_k1_MAT_s113
                                 + 1.33571 w_k1_MSFT_s113
                                 + 1.18227 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.980874 w_k1_ADM_s113
                                 + 0.994871 w_k1_AXP_s113
                                 + 0.89056 w_k1_DIS_s113
                                 + 0.908897 w_k1_FDX_s113
                                 + 1.56489 w_k1_IBM_s113
                                 + 0.94795 w_k1_INTC_s113
                                 + 1.17395 w_k1_KO_s113 + 1.36653 w_k1_MAT_s113
                                 + 0.893535 w_k1_MSFT_s113
                                 + 0.969473 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.19796 w_k1_ADM_s113
                                 + 0.905986 w_k1_AXP_s113
                                 + 0.986252 w_k1_DIS_s113
                                 + 0.991774 w_k1_FDX_s113
                                 + 0.939972 w_k1_IBM_s113
                                 + 0.900084 w_k1_INTC_s113
                                 + 0.992725 w_k1_KO_s113
                                 + 0.813186 w_k1_MAT_s113
                                 + 0.94691 w_k1_MSFT_s113
                                 + 1.08585 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.4006 w_k1_ADM_s114
                                 + 1.42954 w_k1_AXP_s114
                                 + 1.39549 w_k1_DIS_s114
                                 + 1.33719 w_k1_FDX_s114
                                 + 1.02997 w_k1_IBM_s114
                                 + 1.20401 w_k1_INTC_s114
                                 + 1.04608 w_k1_KO_s114 + 1.12734 w_k1_MAT_s114
                                 + 1.25907 w_k1_MSFT_s114
                                 + 1.11105 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.974015 w_k1_ADM_s114
                                 + 0.880731 w_k1_AXP_s114
                                 + 1.0416 w_k1_DIS_s114
                                 + 0.979844 w_k1_FDX_s114
                                 + 0.969398 w_k1_IBM_s114
                                 + 1.09614 w_k1_INTC_s114
                                 + 0.998654 w_k1_KO_s114
                                 + 0.819032 w_k1_MAT_s114
                                 + 1.08989 w_k1_MSFT_s114
                                 + 1.13526 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.26343 w_k1_ADM_s114
                                 + 1.3847 w_k1_AXP_s114
                                 + 0.967561 w_k1_DIS_s114
                                 + 1.02459 w_k1_FDX_s114
                                 + 1.13247 w_k1_IBM_s114
                                 + 1.06622 w_k1_INTC_s114
                                 + 1.00855 w_k1_KO_s114 + 1.47236 w_k1_MAT_s114
                                 + 0.92183 w_k1_MSFT_s114
                                 + 1.06769 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.11993 w_k1_ADM_s114
                                 + 0.996391 w_k1_AXP_s114
                                 + 0.801495 w_k1_DIS_s114
                                 + 1.06605 w_k1_FDX_s114
                                 + 0.814438 w_k1_IBM_s114
                                 + 1.11494 w_k1_INTC_s114
                                 + 0.756506 w_k1_KO_s114
                                 + 1.05881 w_k1_MAT_s114
                                 + 0.99311 w_k1_MSFT_s114
                                 + 0.95225 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.886955 w_k1_ADM_s114
                                 + 0.769215 w_k1_AXP_s114
                                 + 1.08496 w_k1_DIS_s114
                                 + 1.01623 w_k1_FDX_s114
                                 + 0.730376 w_k1_IBM_s114
                                 + 1.24198 w_k1_INTC_s114
                                 + 0.842722 w_k1_KO_s114
                                 + 0.600605 w_k1_MAT_s114
                                 + 1.00631 w_k1_MSFT_s114
                                 + 0.987869 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.753222 w_k1_ADM_s114
                                 + 0.688336 w_k1_AXP_s114
                                 + 0.591776 w_k1_DIS_s114
                                 + 0.788539 w_k1_FDX_s114
                                 + 0.629875 w_k1_IBM_s114
                                 + 0.813731 w_k1_INTC_s114
                                 + 0.631954 w_k1_KO_s114
                                 + 0.731987 w_k1_MAT_s114
                                 + 0.702857 w_k1_MSFT_s114
                                 + 0.751258 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.14909 w_k1_ADM_s114
                                 + 1.15019 w_k1_AXP_s114
                                 + 1.36367 w_k1_DIS_s114
                                 + 1.12366 w_k1_FDX_s114
                                 + 1.56209 w_k1_IBM_s114
                                 + 1.32653 w_k1_INTC_s114
                                 + 1.44152 w_k1_KO_s114 + 1.41383 w_k1_MAT_s114
                                 + 1.15659 w_k1_MSFT_s114
                                 + 1.23242 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.0138 w_k1_ADM_s114
                                 + 0.748519 w_k1_AXP_s114
                                 + 0.719495 w_k1_DIS_s114
                                 + 0.830621 w_k1_FDX_s114
                                 + 1.03518 w_k1_IBM_s114
                                 + 0.857552 w_k1_INTC_s114
                                 + 0.846698 w_k1_KO_s114 + 1.027 w_k1_MAT_s114
                                 + 0.825517 w_k1_MSFT_s114
                                 + 0.939797 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.956154 w_k1_ADM_s115
                                 + 0.819585 w_k1_AXP_s115
                                 + 0.665115 w_k1_DIS_s115
                                 + 0.827064 w_k1_FDX_s115
                                 + 0.749003 w_k1_IBM_s115
                                 + 0.854596 w_k1_INTC_s115
                                 + 0.768814 w_k1_KO_s115
                                 + 0.674766 w_k1_MAT_s115
                                 + 0.823838 w_k1_MSFT_s115
                                 + 0.924586 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.02505 w_k1_ADM_s115
                                 + 0.845485 w_k1_AXP_s115
                                 + 1.14535 w_k1_DIS_s115
                                 + 1.07087 w_k1_FDX_s115
                                 + 0.825301 w_k1_IBM_s115
                                 + 1.10004 w_k1_INTC_s115
                                 + 0.996393 w_k1_KO_s115
                                 + 0.587518 w_k1_MAT_s115
                                 + 1.04668 w_k1_MSFT_s115
                                 + 1.01091 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.87119 w_k1_ADM_s115
                                 + 0.814848 w_k1_AXP_s115
                                 + 0.513486 w_k1_DIS_s115
                                 + 0.686127 w_k1_FDX_s115
                                 + 0.910285 w_k1_IBM_s115
                                 + 0.743689 w_k1_INTC_s115
                                 + 0.743081 w_k1_KO_s115
                                 + 0.893048 w_k1_MAT_s115
                                 + 0.675138 w_k1_MSFT_s115
                                 + 0.771503 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.729779 w_k1_ADM_s115
                                 + 0.787026 w_k1_AXP_s115
                                 + 0.812803 w_k1_DIS_s115
                                 + 0.935869 w_k1_FDX_s115
                                 + 0.632373 w_k1_IBM_s115
                                 + 0.823468 w_k1_INTC_s115
                                 + 0.835899 w_k1_KO_s115
                                 + 0.755079 w_k1_MAT_s115
                                 + 0.942754 w_k1_MSFT_s115
                                 + 0.922786 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.984116 w_k1_ADM_s115
                                 + 0.919605 w_k1_AXP_s115
                                 + 0.924632 w_k1_DIS_s115
                                 + 0.905953 w_k1_FDX_s115
                                 + 1.12924 w_k1_IBM_s115
                                 + 0.826306 w_k1_INTC_s115
                                 + 1.02627 w_k1_KO_s115 + 1.23175 w_k1_MAT_s115
                                 + 1.02108 w_k1_MSFT_s115
                                 + 0.954339 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.14478 w_k1_ADM_s115
                                 + 1.25809 w_k1_AXP_s115
                                 + 1.43726 w_k1_DIS_s115
                                 + 1.35196 w_k1_FDX_s115
                                 + 0.982689 w_k1_IBM_s115
                                 + 1.15543 w_k1_INTC_s115
                                 + 1.04935 w_k1_KO_s115 + 0.97799 w_k1_MAT_s115
                                 + 1.05359 w_k1_MSFT_s115 + 1.1451 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.9946 w_k1_ADM_s115
                                 + 1.49197 w_k1_AXP_s115
                                 + 1.11163 w_k1_DIS_s115
                                 + 1.02426 w_k1_FDX_s115
                                 + 1.55049 w_k1_IBM_s115
                                 + 1.17303 w_k1_INTC_s115
                                 + 1.25358 w_k1_KO_s115 + 1.78095 w_k1_MAT_s115
                                 + 0.959241 w_k1_MSFT_s115
                                 + 0.953189 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.14544 w_k1_ADM_s115
                                 + 1.22559 w_k1_AXP_s115
                                 + 0.843353 w_k1_DIS_s115
                                 + 0.978093 w_k1_FDX_s115
                                 + 0.911064 w_k1_IBM_s115
                                 + 0.765076 w_k1_INTC_s115
                                 + 0.782869 w_k1_KO_s115
                                 + 1.32944 w_k1_MAT_s115
                                 + 0.855892 w_k1_MSFT_s115
                                 + 0.966128 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.19219 w_k1_ADM_s116
                                 + 1.04964 w_k1_AXP_s116
                                 + 1.26875 w_k1_DIS_s116
                                 + 1.24936 w_k1_FDX_s116
                                 + 0.790861 w_k1_IBM_s116
                                 + 0.991083 w_k1_INTC_s116
                                 + 0.987263 w_k1_KO_s116 + 1.0208 w_k1_MAT_s116
                                 + 1.20394 w_k1_MSFT_s116
                                 + 1.10358 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.931956 w_k1_ADM_s116
                                 + 0.82429 w_k1_AXP_s116
                                 + 0.794261 w_k1_DIS_s116
                                 + 0.884151 w_k1_FDX_s116
                                 + 0.820968 w_k1_IBM_s116
                                 + 0.827672 w_k1_INTC_s116
                                 + 0.925162 w_k1_KO_s116
                                 + 0.795074 w_k1_MAT_s116
                                 + 0.842468 w_k1_MSFT_s116
                                 + 0.936426 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.89834 w_k1_ADM_s116
                                 + 0.701408 w_k1_AXP_s116
                                 + 0.748734 w_k1_DIS_s116
                                 + 0.803051 w_k1_FDX_s116
                                 + 1.02344 w_k1_IBM_s116
                                 + 0.742963 w_k1_INTC_s116
                                 + 0.978097 w_k1_KO_s116
                                 + 1.10703 w_k1_MAT_s116
                                 + 0.768924 w_k1_MSFT_s116
                                 + 0.890637 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.18938 w_k1_ADM_s116
                                 + 1.27818 w_k1_AXP_s116
                                 + 1.65558 w_k1_DIS_s116
                                 + 1.18287 w_k1_FDX_s116
                                 + 1.67086 w_k1_IBM_s116
                                 + 1.45737 w_k1_INTC_s116
                                 + 1.82327 w_k1_KO_s116 + 1.58489 w_k1_MAT_s116
                                 + 1.16293 w_k1_MSFT_s116
                                 + 1.31185 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.92036 w_k1_ADM_s116
                                 + 1.05756 w_k1_AXP_s116
                                 + 1.00707 w_k1_DIS_s116
                                 + 1.03003 w_k1_FDX_s116
                                 + 0.767139 w_k1_IBM_s116
                                 + 1.07154 w_k1_INTC_s116
                                 + 0.881651 w_k1_KO_s116
                                 + 0.78624 w_k1_MAT_s116
                                 + 0.95736 w_k1_MSFT_s116
                                 + 0.900092 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.999469 w_k1_ADM_s116
                                 + 1.32039 w_k1_AXP_s116
                                 + 1.15284 w_k1_DIS_s116 + 1.1633 w_k1_FDX_s116
                                 + 1.37737 w_k1_IBM_s116
                                 + 1.10489 w_k1_INTC_s116
                                 + 1.20742 w_k1_KO_s116 + 1.5772 w_k1_MAT_s116
                                 + 1.01705 w_k1_MSFT_s116
                                 + 1.12465 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.02587 w_k1_ADM_s116
                                 + 1.45467 w_k1_AXP_s116
                                 + 1.16632 w_k1_DIS_s116
                                 + 1.22194 w_k1_FDX_s116
                                 + 0.97958 w_k1_IBM_s116
                                 + 1.22083 w_k1_INTC_s116
                                 + 1.08377 w_k1_KO_s116 + 1.12966 w_k1_MAT_s116
                                 + 1.07892 w_k1_MSFT_s116
                                 + 1.15291 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.824628 w_k1_ADM_s116
                                 + 1.06372 w_k1_AXP_s116
                                 + 1.21505 w_k1_DIS_s116
                                 + 1.06941 w_k1_FDX_s116
                                 + 1.10371 w_k1_IBM_s116
                                 + 1.11982 w_k1_INTC_s116
                                 + 1.16282 w_k1_KO_s116 + 0.89806 w_k1_MAT_s116
                                 + 0.981459 w_k1_MSFT_s116
                                 + 1.04778 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.02326 w_k1_ADM_s117
                                 + 1.07253 w_k1_AXP_s117
                                 + 0.961873 w_k1_DIS_s117
                                 + 1.05742 w_k1_FDX_s117
                                 + 0.896975 w_k1_IBM_s117
                                 + 0.91412 w_k1_INTC_s117
                                 + 0.770121 w_k1_KO_s117
                                 + 1.03293 w_k1_MAT_s117
                                 + 0.906007 w_k1_MSFT_s117
                                 + 0.994289 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.96816 w_k1_ADM_s117
                                 + 1.1603 w_k1_AXP_s117 + 1.52508 w_k1_DIS_s117
                                 + 1.26574 w_k1_FDX_s117
                                 + 1.21396 w_k1_IBM_s117
                                 + 1.11995 w_k1_INTC_s117
                                 + 1.21752 w_k1_KO_s117 + 1.08986 w_k1_MAT_s117
                                 + 1.13944 w_k1_MSFT_s117
                                 + 1.35291 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.07659 w_k1_ADM_s117
                                 + 1.08523 w_k1_AXP_s117
                                 + 1.33659 w_k1_DIS_s117
                                 + 1.37761 w_k1_FDX_s117
                                 + 0.928471 w_k1_IBM_s117
                                 + 1.41449 w_k1_INTC_s117
                                 + 0.91207 w_k1_KO_s117
                                 + 0.619601 w_k1_MAT_s117
                                 + 1.28854 w_k1_MSFT_s117
                                 + 1.09601 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.707767 w_k1_ADM_s117
                                 + 0.633908 w_k1_AXP_s117
                                 + 0.494258 w_k1_DIS_s117
                                 + 0.790171 w_k1_FDX_s117
                                 + 0.591165 w_k1_IBM_s117
                                 + 0.898447 w_k1_INTC_s117
                                 + 0.54245 w_k1_KO_s117
                                 + 0.550328 w_k1_MAT_s117
                                 + 0.726898 w_k1_MSFT_s117
                                 + 0.863949 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.685586 w_k1_ADM_s117
                                 + 0.728306 w_k1_AXP_s117
                                 + 0.765398 w_k1_DIS_s117
                                 + 0.84609 w_k1_FDX_s117
                                 + 1.05212 w_k1_IBM_s117
                                 + 1.05911 w_k1_INTC_s117
                                 + 0.834094 w_k1_KO_s117
                                 + 0.967426 w_k1_MAT_s117
                                 + 0.898876 w_k1_MSFT_s117
                                 + 1.01521 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.677114 w_k1_ADM_s117
                                 + 0.786519 w_k1_AXP_s117
                                 + 0.679764 w_k1_DIS_s117
                                 + 0.853653 w_k1_FDX_s117
                                 + 0.805151 w_k1_IBM_s117
                                 + 0.863624 w_k1_INTC_s117
                                 + 0.734274 w_k1_KO_s117
                                 + 0.763193 w_k1_MAT_s117
                                 + 0.883424 w_k1_MSFT_s117
                                 + 0.899532 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.27888 w_k1_ADM_s117
                                 + 1.81394 w_k1_AXP_s117
                                 + 1.61836 w_k1_DIS_s117
                                 + 1.52579 w_k1_FDX_s117
                                 + 1.23092 w_k1_IBM_s117
                                 + 1.24838 w_k1_INTC_s117
                                 + 1.19342 w_k1_KO_s117 + 1.79156 w_k1_MAT_s117
                                 + 1.36746 w_k1_MSFT_s117
                                 + 1.39956 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.711998 w_k1_ADM_s117
                                 + 1.30288 w_k1_AXP_s117 + 1.1983 w_k1_DIS_s117
                                 + 1.06942 w_k1_FDX_s117
                                 + 1.61482 w_k1_IBM_s117
                                 + 1.29212 w_k1_INTC_s117
                                 + 1.11536 w_k1_KO_s117 + 1.67865 w_k1_MAT_s117
                                 + 1.09487 w_k1_MSFT_s117 + 1.0332 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0310779778769999 w_k1_ADM_s000
                                 + 0.0176730491429949 w_k1_ADM_s001
                                 + 0.0496344228811532 w_k1_ADM_s002
                                 + 0.0103740052534717 w_k1_ADM_s003
                                 + 0.0158917895716688 w_k1_ADM_s004
                                 + 0.0142316137405957 w_k1_ADM_s005
                                 + 0.031895207985327 w_k1_ADM_s006
                                 + 0.0259158372654264 w_k1_ADM_s007
                                 + 0.0757512904444777 w_k1_ADM_s010
                                 + 0.0395476333307942 w_k1_ADM_s011
                                 + 0.0274535119459642 w_k1_ADM_s012
                                 + 0.0629759834701602 w_k1_ADM_s013
                                 + 0.0726751266078879 w_k1_ADM_s014
                                 + 0.064499149833822 w_k1_ADM_s015
                                 + 0.0835932960679513 w_k1_ADM_s016
                                 + 0.0516697945752745 w_k1_ADM_s017
                                 + 0.00563927555989874 w_k1_ADM_s100
                                 + 0.00973315604311095 w_k1_ADM_s101
                                 + 0.00518321256463684 w_k1_ADM_s102
                                 + 0.011313213915494 w_k1_ADM_s103
                                 + 0.0126517583669892 w_k1_ADM_s104
                                 + 0.00746639109295567 w_k1_ADM_s105
                                 + 0.00259888278600296 w_k1_ADM_s106
                                 + 0.0068983250175368 w_k1_ADM_s107
                                 + 0.0158458274867424 w_k1_ADM_s110
                                 + 0.0543397316918504 w_k1_ADM_s111
                                 + 0.0377767385758218 w_k1_ADM_s112
                                 + 0.0262031808011003 w_k1_ADM_s113
                                 + 0.0549584463910282 w_k1_ADM_s114
                                 + 0.0126161164391318 w_k1_ADM_s115
                                 + 0.0242496374485523 w_k1_ADM_s116
                                 + 0.0322687038367922 w_k1_ADM_s117
                                 + 0.0343045755164227 w_k1_AXP_s000
                                 + 0.0177394891734042 w_k1_AXP_s001
                                 + 0.0461637927574848 w_k1_AXP_s002
                                 + 0.010221992824832 w_k1_AXP_s003
                                 + 0.0169746480848777 w_k1_AXP_s004
                                 + 0.0166828908107383 w_k1_AXP_s005
                                 + 0.0347549198790265 w_k1_AXP_s006
                                 + 0.0258623169445763 w_k1_AXP_s007
                                 + 0.0908988147632488 w_k1_AXP_s010
                                 + 0.0416634934465132 w_k1_AXP_s011
                                 + 0.0271850636844032 w_k1_AXP_s012
                                 + 0.0636085343541649 w_k1_AXP_s013
                                 + 0.0805494572800825 w_k1_AXP_s014
                                 + 0.0696621426768161 w_k1_AXP_s015
                                 + 0.0759205287058438 w_k1_AXP_s016
                                 + 0.0589299428114443 w_k1_AXP_s017
                                 + 0.00705877659299565 w_k1_AXP_s100
                                 + 0.0109882053677603 w_k1_AXP_s101
                                 + 0.00600968201615715 w_k1_AXP_s102
                                 + 0.0100962543428813 w_k1_AXP_s103
                                 + 0.0107148758520226 w_k1_AXP_s104
                                 + 0.00798049729363375 w_k1_AXP_s105
                                 + 0.00285644407567645 w_k1_AXP_s106
                                 + 0.00804750808489158 w_k1_AXP_s107
                                 + 0.0191003240765682 w_k1_AXP_s110
                                 + 0.0491528401744164 w_k1_AXP_s111
                                 + 0.0343454199007178 w_k1_AXP_s112
                                 + 0.0221914589392329 w_k1_AXP_s113
                                 + 0.050615023168144 w_k1_AXP_s114
                                 + 0.0130018452370886 w_k1_AXP_s115
                                 + 0.0267946973474564 w_k1_AXP_s116
                                 + 0.0390836301997196 w_k1_AXP_s117
                                 + 0.0319157268644972 w_k1_DIS_s000
                                 + 0.0184667036089107 w_k1_DIS_s001
                                 + 0.0437857575023794 w_k1_DIS_s002
                                 + 0.00971134330504738 w_k1_DIS_s003
                                 + 0.0186754230921815 w_k1_DIS_s004
                                 + 0.0164229999559556 w_k1_DIS_s005
                                 + 0.0345997256648617 w_k1_DIS_s006
                                 + 0.0251494889103076 w_k1_DIS_s007
                                 + 0.0859902448765955 w_k1_DIS_s010
                                 + 0.0443289893581732 w_k1_DIS_s011
                                 + 0.0267553821661823 w_k1_DIS_s012
                                 + 0.0643001356449793 w_k1_DIS_s013
                                 + 0.0796663948952692 w_k1_DIS_s014
                                 + 0.0676810809674229 w_k1_DIS_s015
                                 + 0.0850433423359953 w_k1_DIS_s016
                                 + 0.0627495331825929 w_k1_DIS_s017
                                 + 0.00650734312668505 w_k1_DIS_s100
                                 + 0.011381662735945 w_k1_DIS_s101
                                 + 0.00632752958868132 w_k1_DIS_s102
                                 + 0.0112746710111433 w_k1_DIS_s103
                                 + 0.0111880300103722 w_k1_DIS_s104
                                 + 0.00792264716837961 w_k1_DIS_s105
                                 + 0.00244394121437692 w_k1_DIS_s106
                                 + 0.00860792414288064 w_k1_DIS_s107
                                 + 0.0177996900716167 w_k1_DIS_s110
                                 + 0.0500063768273677 w_k1_DIS_s111
                                 + 0.0373204934369874 w_k1_DIS_s112
                                 + 0.0221081400704261 w_k1_DIS_s113
                                 + 0.0506243040694403 w_k1_DIS_s114
                                 + 0.0126995038301733 w_k1_DIS_s115
                                 + 0.0270260083315785 w_k1_DIS_s116
                                 + 0.0386878883136125 w_k1_DIS_s117
                                 + 0.0340519015816079 w_k1_FDX_s000
                                 + 0.0173267315137387 w_k1_FDX_s001
                                 + 0.0425565248353416 w_k1_FDX_s002
                                 + 0.0100797011637427 w_k1_FDX_s003
                                 + 0.0181125854141808 w_k1_FDX_s004
                                 + 0.0154043028581823 w_k1_FDX_s005
                                 + 0.0350087980397491 w_k1_FDX_s006
                                 + 0.0249763283142611 w_k1_FDX_s007
                                 + 0.0794400397438211 w_k1_FDX_s010
                                 + 0.0421371203015724 w_k1_FDX_s011
                                 + 0.0256284080323879 w_k1_FDX_s012
                                 + 0.0633084679935659 w_k1_FDX_s013
                                 + 0.0798377830665044 w_k1_FDX_s014
                                 + 0.0678543873193328 w_k1_FDX_s015
                                 + 0.0785020348860168 w_k1_FDX_s016
                                 + 0.0626187628894626 w_k1_FDX_s017
                                 + 0.00661822919229633 w_k1_FDX_s100
                                 + 0.0112733152438569 w_k1_FDX_s101
                                 + 0.00586584832142662 w_k1_FDX_s102
                                 + 0.010901211911549 w_k1_FDX_s103
                                 + 0.0108157981099454 w_k1_FDX_s104
                                 + 0.00768818343193401 w_k1_FDX_s105
                                 + 0.0025694129186576 w_k1_FDX_s106
                                 + 0.00819920499078611 w_k1_FDX_s107
                                 + 0.016402961461895 w_k1_FDX_s110
                                 + 0.0509883364732336 w_k1_FDX_s111
                                 + 0.0350619868682045 w_k1_FDX_s112
                                 + 0.0223930528169089 w_k1_FDX_s113
                                 + 0.051568622042634 w_k1_FDX_s114
                                 + 0.0127456452608156 w_k1_FDX_s115
                                 + 0.026358542156292 w_k1_FDX_s116
                                 + 0.0403205582103905 w_k1_FDX_s117
                                 + 0.0359220199716875 w_k1_IBM_s000
                                 + 0.0181849918425921 w_k1_IBM_s001
                                 + 0.0431777982824711 w_k1_IBM_s002
                                 + 0.00998962207143162 w_k1_IBM_s003
                                 + 0.0171958450288004 w_k1_IBM_s004
                                 + 0.0163875598412117 w_k1_IBM_s005
                                 + 0.0346588998742788 w_k1_IBM_s006
                                 + 0.0238385322751802 w_k1_IBM_s007
                                 + 0.0843638573030138 w_k1_IBM_s010
                                 + 0.0441228329401395 w_k1_IBM_s011
                                 + 0.0276312402170538 w_k1_IBM_s012
                                 + 0.0654656477835864 w_k1_IBM_s013
                                 + 0.0777267953976418 w_k1_IBM_s014
                                 + 0.0682619682845727 w_k1_IBM_s015
                                 + 0.0781851632774754 w_k1_IBM_s016
                                 + 0.0629927971768984 w_k1_IBM_s017
                                 + 0.00702707478944159 w_k1_IBM_s100
                                 + 0.0110151049872603 w_k1_IBM_s101
                                 + 0.00617204142497266 w_k1_IBM_s102
                                 + 0.0110982827319601 w_k1_IBM_s103
                                 + 0.0114503490904891 w_k1_IBM_s104
                                 + 0.00781996138188736 w_k1_IBM_s105
                                 + 0.00249126835747866 w_k1_IBM_s106
                                 + 0.00871689033604257 w_k1_IBM_s107
                                 + 0.0178221609152741 w_k1_IBM_s110
                                 + 0.0499991985625585 w_k1_IBM_s111
                                 + 0.0363515857528593 w_k1_IBM_s112
                                 + 0.0228361525942236 w_k1_IBM_s113
                                 + 0.0532560629071476 w_k1_IBM_s114
                                 + 0.0125765014970113 w_k1_IBM_s115
                                 + 0.0264327299351287 w_k1_IBM_s116
                                 + 0.0410467408525431 w_k1_IBM_s117
                                 + 0.0303277108124447 w_k1_INTC_s000
                                 + 0.0176612907840437 w_k1_INTC_s001
                                 + 0.0422006027672044 w_k1_INTC_s002
                                 + 0.0100305654494837 w_k1_INTC_s003
                                 + 0.0223724930986555 w_k1_INTC_s004
                                 + 0.0181758104001662 w_k1_INTC_s005
                                 + 0.0377033978056912 w_k1_INTC_s006
                                 + 0.0234768199235179 w_k1_INTC_s007
                                 + 0.0874306974556573 w_k1_INTC_s010
                                 + 0.0401879897625277 w_k1_INTC_s011
                                 + 0.0243404502557528 w_k1_INTC_s012
                                 + 0.0586141759587336 w_k1_INTC_s013
                                 + 0.0794218996035471 w_k1_INTC_s014
                                 + 0.0634974676893698 w_k1_INTC_s015
                                 + 0.0674823573111633 w_k1_INTC_s016
                                 + 0.0680938899131462 w_k1_INTC_s017
                                 + 0.00707492112013929 w_k1_INTC_s100
                                 + 0.0112143651670174 w_k1_INTC_s101
                                 + 0.0053651567665296 w_k1_INTC_s102
                                 + 0.0119686750375053 w_k1_INTC_s103
                                 + 0.0106713517316056 w_k1_INTC_s104
                                 + 0.00800182913970673 w_k1_INTC_s105
                                 + 0.00351743640528624 w_k1_INTC_s106
                                 + 0.00773698707762919 w_k1_INTC_s107
                                 + 0.020050582882406 w_k1_INTC_s110
                                 + 0.0488403807193195 w_k1_INTC_s111
                                 + 0.0331711769278879 w_k1_INTC_s112
                                 + 0.0215995227496299 w_k1_INTC_s113
                                 + 0.0547918148751821 w_k1_INTC_s114
                                 + 0.0120688428991435 w_k1_INTC_s115
                                 + 0.0260415351856427 w_k1_INTC_s116
                                 + 0.0420783657296797 w_k1_INTC_s117
                                 + 0.0331193537519971 w_k1_KO_s000
                                 + 0.0171874215021908 w_k1_KO_s001
                                 + 0.0391561182404441 w_k1_KO_s002
                                 + 0.0108301363290983 w_k1_KO_s003
                                 + 0.0172354421888926 w_k1_KO_s004
                                 + 0.015416731789226 w_k1_KO_s005
                                 + 0.0358260557535286 w_k1_KO_s006
                                 + 0.0221048841587201 w_k1_KO_s007
                                 + 0.0821930376295307 w_k1_KO_s010
                                 + 0.0399571152555362 w_k1_KO_s011
                                 + 0.0249725941637338 w_k1_KO_s012
                                 + 0.0612099692871554 w_k1_KO_s013
                                 + 0.073264079983331 w_k1_KO_s014
                                 + 0.0706140873826834 w_k1_KO_s015
                                 + 0.0865875858118882 w_k1_KO_s016
                                 + 0.0588401620852167 w_k1_KO_s017
                                 + 0.00632290426798367 w_k1_KO_s100
                                 + 0.010220335676623 w_k1_KO_s101
                                 + 0.00612899108798829 w_k1_KO_s102
                                 + 0.0119953764032485 w_k1_KO_s103
                                 + 0.0113721001425748 w_k1_KO_s104
                                 + 0.00766629403902142 w_k1_KO_s105
                                 + 0.00253392751912247 w_k1_KO_s106
                                 + 0.00797483856505154 w_k1_KO_s107
                                 + 0.0175573508582112 w_k1_KO_s110
                                 + 0.0501225873108683 w_k1_KO_s111
                                 + 0.0355126108644918 w_k1_KO_s112
                                 + 0.0217530556940675 w_k1_KO_s113
                                 + 0.0498748157214598 w_k1_KO_s114
                                 + 0.012244770042258 w_k1_KO_s115
                                 + 0.0275380657899021 w_k1_KO_s116
                                 + 0.0349677003738535 w_k1_KO_s117
                                 + 0.0392607029553837 w_k1_MAT_s000
                                 + 0.0207347651156511 w_k1_MAT_s001
                                 + 0.0418586406793959 w_k1_MAT_s002
                                 + 0.0146728807423397 w_k1_MAT_s003
                                 + 0.0173647180218342 w_k1_MAT_s004
                                 + 0.0178362333175813 w_k1_MAT_s005
                                 + 0.0366194418400895 w_k1_MAT_s006
                                 + 0.024289890523765 w_k1_MAT_s007
                                 + 0.0861265158118269 w_k1_MAT_s010
                                 + 0.0461228868311814 w_k1_MAT_s011
                                 + 0.0240427989259867 w_k1_MAT_s012
                                 + 0.0727414650486797 w_k1_MAT_s013
                                 + 0.0827985335299943 w_k1_MAT_s014
                                 + 0.0965168351034586 w_k1_MAT_s015
                                 + 0.0850745568125184 w_k1_MAT_s016
                                 + 0.0647763837859781 w_k1_MAT_s017
                                 + 0.00890075352523124 w_k1_MAT_s100
                                 + 0.013406183274014 w_k1_MAT_s101
                                 + 0.00758216359994625 w_k1_MAT_s102
                                 + 0.0105600254880282 w_k1_MAT_s103
                                 + 0.0115697155800855 w_k1_MAT_s104
                                 + 0.00672679742180721 w_k1_MAT_s105
                                 + 0.00249349502520996 w_k1_MAT_s106
                                 + 0.00813441528946251 w_k1_MAT_s107
                                 + 0.0195397446186338 w_k1_MAT_s110
                                 + 0.0631001726046368 w_k1_MAT_s111
                                 + 0.0389048107955225 w_k1_MAT_s112
                                 + 0.0188904497231094 w_k1_MAT_s113
                                 + 0.0542790729305229 w_k1_MAT_s114
                                 + 0.0131090792800971 w_k1_MAT_s115
                                 + 0.0271083522057986 w_k1_MAT_s116
                                 + 0.04067023918795 w_k1_MAT_s117
                                 + 0.0336055376493947 w_k1_MSFT_s000
                                 + 0.0186416532008374 w_k1_MSFT_s001
                                 + 0.0437517503422333 w_k1_MSFT_s002
                                 + 0.00966284642745721 w_k1_MSFT_s003
                                 + 0.0187940443209915 w_k1_MSFT_s004
                                 + 0.016138183480161 w_k1_MSFT_s005
                                 + 0.0336397238883164 w_k1_MSFT_s006
                                 + 0.0231234712540034 w_k1_MSFT_s007
                                 + 0.0780562702321821 w_k1_MSFT_s010
                                 + 0.0399639631639175 w_k1_MSFT_s011
                                 + 0.0262513506626912 w_k1_MSFT_s012
                                 + 0.0709951533469784 w_k1_MSFT_s013
                                 + 0.0820578844259564 w_k1_MSFT_s014
                                 + 0.068747695504129 w_k1_MSFT_s015
                                 + 0.074931743786927 w_k1_MSFT_s016
                                 + 0.0624377524629456 w_k1_MSFT_s017
                                 + 0.00637953199295542 w_k1_MSFT_s100
                                 + 0.0106766338998 w_k1_MSFT_s101
                                 + 0.00605839419338929 w_k1_MSFT_s102
                                 + 0.0106175529464135 w_k1_MSFT_s103
                                 + 0.0117257509901682 w_k1_MSFT_s104
                                 + 0.00827350418074313 w_k1_MSFT_s105
                                 + 0.00270164233759721 w_k1_MSFT_s106
                                 + 0.00838456282351894 w_k1_MSFT_s107
                                 + 0.0159612332562363 w_k1_MSFT_s110
                                 + 0.0522963792944136 w_k1_MSFT_s111
                                 + 0.0375579834886639 w_k1_MSFT_s112
                                 + 0.0225919913686398 w_k1_MSFT_s113
                                 + 0.0505706351065995 w_k1_MSFT_s114
                                 + 0.0120666575333126 w_k1_MSFT_s115
                                 + 0.0244073627026451 w_k1_MSFT_s116
                                 + 0.0390709270016922 w_k1_MSFT_s117
                                 + 0.035076686541698 w_k1_PG_s000
                                 + 0.0174775160854832 w_k1_PG_s001
                                 + 0.0426045612431462 w_k1_PG_s002
                                 + 0.0101762152499942 w_k1_PG_s003
                                 + 0.0163790883346658 w_k1_PG_s004
                                 + 0.0155125235711957 w_k1_PG_s005
                                 + 0.0354762855345465 w_k1_PG_s006
                                 + 0.0229749659676493 w_k1_PG_s007
                                 + 0.077480484982476 w_k1_PG_s010
                                 + 0.0429197584184454 w_k1_PG_s011
                                 + 0.0252091201454848 w_k1_PG_s012
                                 + 0.0669610091324064 w_k1_PG_s013
                                 + 0.0796006422930022 w_k1_PG_s014
                                 + 0.0662190899043998 w_k1_PG_s015
                                 + 0.0842790109707638 w_k1_PG_s016
                                 + 0.0607764738096119 w_k1_PG_s017
                                 + 0.00700009148013804 w_k1_PG_s100
                                 + 0.0109789038107412 w_k1_PG_s101
                                 + 0.00598356611682125 w_k1_PG_s102
                                 + 0.0105213749065023 w_k1_PG_s103
                                 + 0.0112499635586145 w_k1_PG_s104
                                 + 0.007335456907115 w_k1_PG_s105
                                 + 0.00246594583341062 w_k1_PG_s106
                                 + 0.00764714367678137 w_k1_PG_s107
                                 + 0.0180363809470449 w_k1_PG_s110
                                 + 0.0479913470936902 w_k1_PG_s111
                                 + 0.0353837324362761 w_k1_PG_s112
                                 + 0.0222050500280057 w_k1_PG_s113
                                 + 0.0528160809399106 w_k1_PG_s114
                                 + 0.0123392349812525 w_k1_PG_s115
                                 + 0.0259981637133055 w_k1_PG_s116
                                 + 0.0409675297763899 w_k1_PG_s117 >= 1.05
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
