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
 w_k1_ADM_s0_return_constraint:  1.11481 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.939277 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.915897 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.02899 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.836682 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.02112 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.980829 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.31221 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.904307 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.01215 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.819893 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.35356 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.88927 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.10752 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.775003 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.37765 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.800427 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.07599 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.28246 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.792376 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.959703 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.66478 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.27927 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.02794 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.780349 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.02534 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.19069 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.17416 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.14834 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.948789 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.968564 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.797204 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.915529 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.14153 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.917776 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.765007 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.26902 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.49571 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20165 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.894992 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.994843 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.13011 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.865239 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.16683 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.691036 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.45312 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.85436 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.941531 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.731097 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.32923 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.590514 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.957002 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.758816 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.52214 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.766436 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.06693 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.14081 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.745367 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.02714 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.53757 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.44462 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.863607 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.543286 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.812607 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.05799 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.29894 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.843706 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.706378 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.840151 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.609359 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.547289 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.06142 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.11124 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.848201 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.32525 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.40707 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.27645 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.863501 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.951273 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.15545 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.765711 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.24826 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.542543 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.33987 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.733363 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.975834 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.745911 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.88024 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.621316 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.0207 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.615136 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.49809 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.684635 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.864682 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.22023 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.90527 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.01219 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.97642 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.25187 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.878152 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.54266 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.863589 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.25893 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.64281 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.934747 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.733441 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.754859 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.51943 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.735405 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.15765 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.10818 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.792513 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.20975 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.8518 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15028 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.879095 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.02681 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.06638 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.90074 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.18366 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.786335 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.23772 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.801127 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.947487 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.720848 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.51225 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.722121 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.05047 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.727774 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.19066 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.814036 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.951755 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.16058 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.835964 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.892463 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.57718 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.20806 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.863964 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.702423 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.841009 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.25108 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.29219 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.06276 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.817409 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.912568 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.65915 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.786263 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.13055 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.941757 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.805633 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.07656 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.57762 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.08641 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.95258 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.814367 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.35449 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.821785 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.2698 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.606995 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.47332 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.995041 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.79439 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.26201 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.03435 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.688925 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.958986 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.937852 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.6186 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.68684 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.12907 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.935274 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.05197 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.56268 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.22436 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.50961 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.50594 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.606531 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.959884 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.850251 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.31082 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.698649 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.665386 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.590177 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.742798 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.589608 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.787552 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.6721 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.938208 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.21093 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.893352 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16222 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.933578 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.04664 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.12717 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.855114 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.20749 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.738209 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.963579 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.754925 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.854548 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.792142 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.21472 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.651155 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.952689 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.687333 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.09788 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.766342 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.866782 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.944565 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.798978 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.01304 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.21688 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.935911 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.05192 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.72901 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.895049 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.20814 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.33714 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.00824 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.805513 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.953496 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.726834 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.786555 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.986296 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.953659 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.78369 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.09335 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.50706 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.10119 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.938358 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.942215 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.25408 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.817589 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.12042 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.599838 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.28367 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.864504 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.843781 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.865652 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.33428 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.725816 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.913555 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.815132 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.42513 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.799548 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.972149 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.03481 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.984225 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.18422 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.36419 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.24943 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.16053 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.642581 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.858766 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.06909 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.33398 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.815011 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.783659 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.75329 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.789314 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.808055 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.02619 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.33526 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.921243 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.26294 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.15615 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.07824 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.975034 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.949079 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.29817 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.772663 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.20311 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.61464 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.28105 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.972209 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.766146 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.04277 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.49863 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.535578 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.964049 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.966447 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.8966 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.928305 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.47602 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.1894 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.921546 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.51159 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.19935 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.90173 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.25233 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.539565 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.880374 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.88517 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.3831 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.776677 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.685333 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.505673 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.641479 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.561012 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.878926 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.79898 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.02622 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.36287 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.0491 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16131 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.944707 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.967903 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.03802 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.824913 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.1056 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.806592 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.15505 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.861888 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.939597 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.846814 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.4932 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.770722 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.03402 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.728699 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.22922 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.835761 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.970929 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.10215 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.928518 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.01732 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.47057 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.14147 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.987232 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.727057 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.930288 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.20318 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.30569 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.00871 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.879666 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.896372 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.702876 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.848993 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.08231 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.925362 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.891889 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.11216 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.53796 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09215 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.927657 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.956869 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.06568 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.891006 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.06993 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.815808 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.1104 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.932134 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.940392 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.923387 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.34932 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.825495 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.986342 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.867272 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.16518 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.849809 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.03709 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.08994 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.961719 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.04901 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.29038 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.15385 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.05456 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.754997 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.953891 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.03816 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.2677 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.918786 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.853215 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.893217 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.787086 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.789324 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.04411 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.12359 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.93236 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.26093 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.33645 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.295999582904909 z_k1_s0000
                                 - 0.237000841185526 z_k1_s0001
                                 - 0.2050011285745 z_k1_s0002
                                 - 0.194001059270491 z_k1_s0003
                                 - 0.263001167995214 z_k1_s0004
                                 - 0.308000147509918 z_k1_s0005
                                 - 0.190000871068821 z_k1_s0006
                                 - 0.100000043235682 z_k1_s0007
                                 - 0.933000811914302 z_k1_s0010
                                 - 0.40900056082753 z_k1_s0011
                                 - 0.364000411717058 z_k1_s0012
                                 - 1.05100341347782 z_k1_s0013
                                 - 0.224000293218239 z_k1_s0014
                                 - 0.884998787700831 z_k1_s0015
                                 - 0.924001403812974 z_k1_s0016
                                 - 0.391000708423597 z_k1_s0017
                                 - 0.532000825536379 z_k1_s0020
                                 - 0.341000856342278 z_k1_s0021
                                 - 0.541000307204655 z_k1_s0022
                                 - 0.354001262308807 z_k1_s0023
                                 - 0.446001737146282 z_k1_s0024
                                 - 0.129000191386819 z_k1_s0025
                                 - 0.146000222650205 z_k1_s0026
                                 - 0.109000246300582 z_k1_s0027
                                 - 0.349000934225722 z_k1_s0030
                                 - 0.30899965639116 z_k1_s0031
                                 - 0.322000413812588 z_k1_s0032
                                 - 0.222999157298135 z_k1_s0033
                                 - 0.26799937298632 z_k1_s0034
                                 - 0.137000026705892 z_k1_s0035
                                 - 0.196999597456395 z_k1_s0036
                                 - 0.150000002126873 z_k1_s0037
                                 - 0.19100021878397 z_k1_s0040
                                 - 0.263999576920557 z_k1_s0041
                                 - 0.178000189892377 z_k1_s0042
                                 - 0.233000162525888 z_k1_s0043
                                 - 0.148000160619507 z_k1_s0044
                                 - 0.682000730296637 z_k1_s0045
                                 - 0.314999578064386 z_k1_s0046
                                 - 0.419999437419181 z_k1_s0047
                                 - 0.256999686158997 z_k1_s0050
                                 - 0.228999669188455 z_k1_s0051
                                 - 0.482000605206655 z_k1_s0052
                                 - 0.794998864309242 z_k1_s0053
                                 - 0.350998803918857 z_k1_s0054
                                 - 0.235999600381216 z_k1_s0055
                                 - 0.293000125406121 z_k1_s0056
                                 - 0.278999386422103 z_k1_s0057
                                 - 0.397999966710314 z_k1_s0060
                                 - 0.830005057330944 z_k1_s0061
                                 - 0.188000912394511 z_k1_s0062
                                 - 0.47400314247676 z_k1_s0063
                                 - 0.376001824789022 z_k1_s0064
                                 - 0.66900314384495 z_k1_s0065
                                 - 0.299001292036379 z_k1_s0066
                                 - 0.284001183684547 z_k1_s0067
                                 - 0.698999526713477 z_k1_s0070
                                 - 0.676999467640926 z_k1_s0071
                                 - 0.586000398386661 z_k1_s0072
                                 - 1.26299986602759 z_k1_s0073
                                 - 1.49100165332531 z_k1_s0074
                                 - 0.431999984878804 z_k1_s0075
                                 - 0.539000154677083 z_k1_s0076
                                 - 0.775999733467405 z_k1_s0077
                                 - 0.188000639245666 z_k1_s0100
                                 - 0.141000365933886 z_k1_s0101
                                 - 0.472002609513608 z_k1_s0102
                                 - 0.179000514462327 z_k1_s0103
                                 - 0.533002244596807 z_k1_s0104
                                 - 0.119000363575583 z_k1_s0105
                                 - 0.418001406812125 z_k1_s0106
                                 - 0.220000704585215 z_k1_s0107
                                 - 0.180999858644694 z_k1_s0110
                                 - 0.228000168530229 z_k1_s0111
                                 - 0.527000764738618 z_k1_s0112
                                 - 0.186999983142822 z_k1_s0113
                                 - 0.158000046751671 z_k1_s0114
                                 - 0.172999882747133 z_k1_s0115
                                 - 0.291999215310945 z_k1_s0116
                                 - 0.154999889452636 z_k1_s0117
                                 - 0.420999133536243 z_k1_s0120
                                 - 0.344999298254391 z_k1_s0121
                                 - 0.527998960757971 z_k1_s0122
                                 - 0.920997392891534 z_k1_s0123
                                 - 0.312999709235197 z_k1_s0124
                                 - 0.621997035759197 z_k1_s0125
                                 - 1.07900063752203 z_k1_s0126
                                 - 0.763997739734276 z_k1_s0127
                                 - 0.201999117147051 z_k1_s0130
                                 - 0.147999894006349 z_k1_s0131
                                 - 0.101999873931442 z_k1_s0132
                                 - 0.219999964793131 z_k1_s0133
                                 - 0.198000270113594 z_k1_s0134
                                 - 0.276000422249218 z_k1_s0135
                                 - 0.232999990149068 z_k1_s0136
                                 - 0.129999951759598 z_k1_s0137
                                 - 0.295000500363672 z_k1_s0140
                                 - 0.220000324488081 z_k1_s0141
                                 - 0.367000985036766 z_k1_s0142
                                 - 0.72699981939625 z_k1_s0143
                                 - 0.683999938686612 z_k1_s0144
                                 - 0.731001561069837 z_k1_s0145
                                 - 0.38199944104925 z_k1_s0146
                                 - 0.183000185084035 z_k1_s0147
                                 - 0.248999093223216 z_k1_s0150
                                 - 0.139999829482689 z_k1_s0151
                                 - 0.536998742444747 z_k1_s0152
                                 - 0.178999902681351 z_k1_s0153
                                 - 0.595999274083447 z_k1_s0154
                                 - 0.181999778327495 z_k1_s0155
                                 - 0.196999880757697 z_k1_s0156
                                 - 0.334000800193548 z_k1_s0157
                                 - 0.326999905411482 z_k1_s0160
                                 - 0.365999894130283 z_k1_s0161
                                 - 0.152999955742987 z_k1_s0162
                                 - 0.250999927395358 z_k1_s0163
                                 - 0.087999974544986 z_k1_s0164
                                 - 0.141999958924864 z_k1_s0165
                                 - 0.10499996962754 z_k1_s0166
                                 - 0.167999951404064 z_k1_s0167
                                 - 0.133999874903632 z_k1_s0170
                                 - 0.106999856918614 z_k1_s0171
                                 - 0.0969999327018599 z_k1_s0172
                                 - 0.370999637039176 z_k1_s0173
                                 - 0.128999838595327 z_k1_s0174
                                 - 0.0949999478585091 z_k1_s0175
                                 - 0.123999950686878 z_k1_s0176
                                 - 0.426999806252417 z_k1_s0177
                                 - 0.28000019648848 z_k1_s1000
                                 - 0.191000043636726 z_k1_s1001
                                 - 0.355999443206264 z_k1_s1002
                                 - 0.227999687154602 z_k1_s1003
                                 - 0.214999559821132 z_k1_s1004
                                 - 0.190000063795709 z_k1_s1005
                                 - 0.247000472334672 z_k1_s1006
                                 - 0.240001002847803 z_k1_s1007
                                 - 0.65200090054958 z_k1_s1010
                                 - 0.510000950405287 z_k1_s1011
                                 - 0.467999899219202 z_k1_s1012
                                 - 0.40199989415496 z_k1_s1013
                                 - 0.571001987861386 z_k1_s1014
                                 - 0.457999959203393 z_k1_s1015
                                 - 0.546999525584067 z_k1_s1016
                                 - 1.40400120125725 z_k1_s1017
                                 - 0.0840000566366124 z_k1_s1020
                                 - 0.22399993223082 z_k1_s1021
                                 - 0.278999725117723 z_k1_s1022
                                 - 0.366000680466959 z_k1_s1023
                                 - 0.137000125582354 z_k1_s1024
                                 - 0.249999953668345 z_k1_s1025
                                 - 0.145000170047772 z_k1_s1026
                                 - 0.156000128625153 z_k1_s1027
                                 - 0.575997293991365 z_k1_s1030
                                 - 0.22399960486249 z_k1_s1031
                                 - 0.783000249311497 z_k1_s1032
                                 - 0.933999976345149 z_k1_s1033
                                 - 0.909000119465283 z_k1_s1034
                                 - 0.717996678228756 z_k1_s1035
                                 - 0.241999797555626 z_k1_s1036
                                 - 0.542997680069697 z_k1_s1037
                                 - 0.543998758486761 z_k1_s1040
                                 - 0.520001762670163 z_k1_s1041
                                 - 0.40499969508791 z_k1_s1042
                                 - 0.389999817803135 z_k1_s1043
                                 - 0.563001812006382 z_k1_s1044
                                 - 1.38899786232706 z_k1_s1045
                                 - 0.458999955272862 z_k1_s1046
                                 - 0.744002136282056 z_k1_s1047
                                 - 0.448001294372362 z_k1_s1050
                                 - 0.244000010545586 z_k1_s1051
                                 - 0.352000820006835 z_k1_s1052
                                 - 0.665000757802599 z_k1_s1053
                                 - 0.342000310927042 z_k1_s1054
                                 - 0.247999938377474 z_k1_s1055
                                 - 0.247999938377474 z_k1_s1056
                                 - 0.210999985382451 z_k1_s1057
                                 - 0.240000551934973 z_k1_s1060
                                 - 0.479002810440967 z_k1_s1061
                                 - 0.297000711343353 z_k1_s1062
                                 - 0.482000280334723 z_k1_s1063
                                 - 0.650001022760142 z_k1_s1064
                                 - 0.595001010919096 z_k1_s1065
                                 - 0.272000647106359 z_k1_s1066
                                 - 0.222000518632371 z_k1_s1067
                                 - 0.236000162377251 z_k1_s1070
                                 - 0.446000246831522 z_k1_s1071
                                 - 0.315001012182323 z_k1_s1072
                                 - 0.179000468354289 z_k1_s1073
                                 - 0.266999778477233 z_k1_s1074
                                 - 0.0950000803722139 z_k1_s1075
                                 - 0.105000042227135 z_k1_s1076
                                 - 0.12800022014373 z_k1_s1077
                                 - 0.172000056826739 z_k1_s1100
                                 - 0.433000606976601 z_k1_s1101
                                 - 0.405999347788721 z_k1_s1102
                                 - 0.196999914074816 z_k1_s1103
                                 - 0.314999090525744 z_k1_s1104
                                 - 0.21000001768381 z_k1_s1105
                                 - 0.381001128540594 z_k1_s1106
                                 - 0.226000001202577 z_k1_s1107
                                 - 0.163000533807499 z_k1_s1110
                                 - 0.188999601199895 z_k1_s1111
                                 - 0.121000022296286 z_k1_s1112
                                 - 0.0740000561481509 z_k1_s1113
                                 - 0.086999975644429 z_k1_s1114
                                 - 0.262999914548099 z_k1_s1115
                                 - 0.319000082162942 z_k1_s1116
                                 - 0.0699999523185002 z_k1_s1117
                                 - 0.173999807173669 z_k1_s1120
                                 - 0.128999308336275 z_k1_s1121
                                 - 0.197000345023571 z_k1_s1122
                                 - 0.309999334551156 z_k1_s1123
                                 - 0.0789998714823446 z_k1_s1124
                                 - 0.155000303774854 z_k1_s1125
                                 - 0.128999308336275 z_k1_s1126
                                 - 0.099999892106703 z_k1_s1127
                                 - 0.604996437231693 z_k1_s1130
                                 - 0.417998772265983 z_k1_s1131
                                 - 1.07499880316702 z_k1_s1132
                                 - 0.36199927891846 z_k1_s1133
                                 - 0.312999265490558 z_k1_s1134
                                 - 1.54899445878727 z_k1_s1135
                                 - 0.384999061507645 z_k1_s1136
                                 - 0.512998872870908 z_k1_s1137
                                 - 0.895000244000196 z_k1_s1140
                                 - 0.279999366338656 z_k1_s1141
                                 - 0.329999471927343 z_k1_s1142
                                 - 0.704997775646576 z_k1_s1143
                                 - 0.212999523433326 z_k1_s1144
                                 - 0.728999296278735 z_k1_s1145
                                 - 0.453000948977516 z_k1_s1146
                                 - 0.222999774230686 z_k1_s1147
                                 - 0.647000126983886 z_k1_s1150
                                 - 1.38299815281036 z_k1_s1151
                                 - 0.725999262021319 z_k1_s1152
                                 - 0.455998815570651 z_k1_s1153
                                 - 0.903995059706503 z_k1_s1154
                                 - 0.40699893894092 z_k1_s1155
                                 - 0.735998025826477 z_k1_s1156
                                 - 0.674996665638329 z_k1_s1157
                                 - 0.357000451298197 z_k1_s1160
                                 - 1.32500325061426 z_k1_s1161
                                 - 0.670003557380847 z_k1_s1162
                                 - 1.39400120706848 z_k1_s1163
                                 - 0.425000481745406 z_k1_s1164
                                 - 0.730998287155024 z_k1_s1165
                                 - 0.476000796031165 z_k1_s1166
                                 - 0.4510007676932 z_k1_s1167
                                 - 0.259999126597318 z_k1_s1170
                                 - 0.146999548037677 z_k1_s1171
                                 - 0.308000283506409 z_k1_s1172
                                 - 0.0729999744675114 z_k1_s1173
                                 - 0.134999870809855 z_k1_s1174
                                 - 0.0919999734111053 z_k1_s1175
                                 - 0.126999949991429 z_k1_s1176
                                 - 0.21799964430049 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.01537 w_k1_ADM_s000
                                 + 1.37009 w_k1_AXP_s000
                                 + 1.18416 w_k1_DIS_s000 + 1.1365 w_k1_FDX_s000
                                 + 1.15852 w_k1_IBM_s000
                                 + 1.0292 w_k1_INTC_s000 + 1.05288 w_k1_KO_s000
                                 + 1.48266 w_k1_MAT_s000
                                 + 1.12406 w_k1_MSFT_s000
                                 + 1.01784 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.926245 w_k1_ADM_s000
                                 + 0.809782 w_k1_AXP_s000
                                 + 0.704644 w_k1_DIS_s000
                                 + 0.806243 w_k1_FDX_s000
                                 + 0.779718 w_k1_IBM_s000
                                 + 0.831027 w_k1_INTC_s000
                                 + 0.787615 w_k1_KO_s000
                                 + 0.455083 w_k1_MAT_s000
                                 + 0.794877 w_k1_MSFT_s000
                                 + 0.806396 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.684478 w_k1_ADM_s000
                                 + 0.785394 w_k1_AXP_s000
                                 + 0.579032 w_k1_DIS_s000
                                 + 0.632217 w_k1_FDX_s000
                                 + 0.835298 w_k1_IBM_s000
                                 + 0.784276 w_k1_INTC_s000
                                 + 0.730788 w_k1_KO_s000
                                 + 0.668816 w_k1_MAT_s000
                                 + 0.688131 w_k1_MSFT_s000
                                 + 0.787489 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.04003 w_k1_ADM_s000
                                 + 1.3576 w_k1_AXP_s000 + 1.05631 w_k1_DIS_s000
                                 + 0.935474 w_k1_FDX_s000
                                 + 1.58634 w_k1_IBM_s000
                                 + 0.974793 w_k1_INTC_s000
                                 + 1.25624 w_k1_KO_s000 + 1.534 w_k1_MAT_s000
                                 + 0.941865 w_k1_MSFT_s000
                                 + 1.04999 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.939849 w_k1_ADM_s000
                                 + 0.936184 w_k1_AXP_s000
                                 + 0.883014 w_k1_DIS_s000
                                 + 0.823004 w_k1_FDX_s000
                                 + 1.18792 w_k1_IBM_s000
                                 + 0.850219 w_k1_INTC_s000
                                 + 0.939504 w_k1_KO_s000
                                 + 0.639942 w_k1_MAT_s000
                                 + 0.749739 w_k1_MSFT_s000
                                 + 0.945173 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.05329 w_k1_ADM_s000
                                 + 0.927269 w_k1_AXP_s000
                                 + 1.13703 w_k1_DIS_s000
                                 + 1.07081 w_k1_FDX_s000
                                 + 0.705459 w_k1_IBM_s000
                                 + 1.00147 w_k1_INTC_s000
                                 + 0.952512 w_k1_KO_s000
                                 + 0.646948 w_k1_MAT_s000
                                 + 0.969296 w_k1_MSFT_s000
                                 + 0.887787 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.43964 w_k1_ADM_s000
                                 + 1.40937 w_k1_AXP_s000
                                 + 1.65997 w_k1_DIS_s000
                                 + 1.30145 w_k1_FDX_s000
                                 + 1.12538 w_k1_IBM_s000
                                 + 0.922783 w_k1_INTC_s000
                                 + 1.1938 w_k1_KO_s000 + 0.869804 w_k1_MAT_s000
                                 + 0.996351 w_k1_MSFT_s000
                                 + 1.15219 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.23174 w_k1_ADM_s000
                                 + 1.43148 w_k1_AXP_s000
                                 + 1.51177 w_k1_DIS_s000
                                 + 1.49316 w_k1_FDX_s000
                                 + 0.812246 w_k1_IBM_s000
                                 + 1.33588 w_k1_INTC_s000
                                 + 1.00984 w_k1_KO_s000
                                 + 0.619768 w_k1_MAT_s000
                                 + 1.30271 w_k1_MSFT_s000
                                 + 1.20193 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.02965 w_k1_ADM_s001
                                 + 1.02997 w_k1_AXP_s001
                                 + 0.908283 w_k1_DIS_s001
                                 + 0.956667 w_k1_FDX_s001
                                 + 0.885516 w_k1_IBM_s001
                                 + 0.93096 w_k1_INTC_s001
                                 + 0.906673 w_k1_KO_s001
                                 + 1.19948 w_k1_MAT_s001
                                 + 0.955201 w_k1_MSFT_s001
                                 + 1.05457 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.9255 w_k1_ADM_s001
                                 + 0.80826 w_k1_AXP_s001
                                 + 0.817303 w_k1_DIS_s001
                                 + 0.979901 w_k1_FDX_s001
                                 + 0.664142 w_k1_IBM_s001
                                 + 0.980031 w_k1_INTC_s001
                                 + 0.761163 w_k1_KO_s001
                                 + 0.722885 w_k1_MAT_s001
                                 + 0.920089 w_k1_MSFT_s001
                                 + 0.91142 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.65579 w_k1_ADM_s001
                                 + 1.6116 w_k1_AXP_s001 + 2.05231 w_k1_DIS_s001
                                 + 1.42774 w_k1_FDX_s001
                                 + 1.64926 w_k1_IBM_s001
                                 + 1.38453 w_k1_INTC_s001
                                 + 1.65379 w_k1_KO_s001 + 1.47601 w_k1_MAT_s001
                                 + 1.48215 w_k1_MSFT_s001
                                 + 1.35121 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.00897 w_k1_ADM_s001
                                 + 1.12394 w_k1_AXP_s001
                                 + 1.37449 w_k1_DIS_s001
                                 + 1.21681 w_k1_FDX_s001
                                 + 1.01094 w_k1_IBM_s001
                                 + 1.27785 w_k1_INTC_s001
                                 + 1.12227 w_k1_KO_s001 + 1.44695 w_k1_MAT_s001
                                 + 1.25195 w_k1_MSFT_s001
                                 + 1.23107 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.20703 w_k1_ADM_s001
                                 + 1.12633 w_k1_AXP_s001
                                 + 1.09755 w_k1_DIS_s001
                                 + 1.19361 w_k1_FDX_s001
                                 + 0.682362 w_k1_IBM_s001
                                 + 1.35292 w_k1_INTC_s001
                                 + 0.858016 w_k1_KO_s001
                                 + 0.525616 w_k1_MAT_s001
                                 + 0.91453 w_k1_MSFT_s001
                                 + 1.06271 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.774734 w_k1_ADM_s001
                                 + 0.674864 w_k1_AXP_s001
                                 + 0.654044 w_k1_DIS_s001
                                 + 0.746807 w_k1_FDX_s001
                                 + 0.880195 w_k1_IBM_s001
                                 + 0.916682 w_k1_INTC_s001
                                 + 0.816136 w_k1_KO_s001
                                 + 0.957156 w_k1_MAT_s001
                                 + 0.690822 w_k1_MSFT_s001
                                 + 0.877742 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.837958 w_k1_ADM_s001
                                 + 0.954026 w_k1_AXP_s001
                                 + 1.10571 w_k1_DIS_s001
                                 + 0.959591 w_k1_FDX_s001
                                 + 1.37659 w_k1_IBM_s001
                                 + 1.0335 w_k1_INTC_s001 + 1.22842 w_k1_KO_s001
                                 + 1.87003 w_k1_MAT_s001
                                 + 0.961867 w_k1_MSFT_s001
                                 + 1.10875 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.934503 w_k1_ADM_s001
                                 + 0.680871 w_k1_AXP_s001
                                 + 0.960978 w_k1_DIS_s001
                                 + 0.842528 w_k1_FDX_s001
                                 + 0.90473 w_k1_IBM_s001
                                 + 1.15261 w_k1_INTC_s001
                                 + 0.99682 w_k1_KO_s001
                                 + 0.758024 w_k1_MAT_s001
                                 + 1.12479 w_k1_MSFT_s001
                                 + 0.991294 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.05099 w_k1_ADM_s002
                                 + 0.764991 w_k1_AXP_s002
                                 + 0.691919 w_k1_DIS_s002
                                 + 0.816636 w_k1_FDX_s002
                                 + 0.754346 w_k1_IBM_s002
                                 + 0.813463 w_k1_INTC_s002
                                 + 0.810732 w_k1_KO_s002
                                 + 0.790977 w_k1_MAT_s002
                                 + 0.762631 w_k1_MSFT_s002
                                 + 0.861559 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.41093 w_k1_ADM_s002
                                 + 1.33877 w_k1_AXP_s002 + 1.2907 w_k1_DIS_s002
                                 + 1.32411 w_k1_FDX_s002
                                 + 1.21032 w_k1_IBM_s002
                                 + 1.34067 w_k1_INTC_s002
                                 + 1.02316 w_k1_KO_s002 + 1.09658 w_k1_MAT_s002
                                 + 1.14446 w_k1_MSFT_s002
                                 + 1.06957 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.26025 w_k1_ADM_s002
                                 + 1.00905 w_k1_AXP_s002
                                 + 1.13324 w_k1_DIS_s002
                                 + 1.07273 w_k1_FDX_s002
                                 + 0.956208 w_k1_IBM_s002
                                 + 1.01919 w_k1_INTC_s002
                                 + 0.996465 w_k1_KO_s002
                                 + 0.613323 w_k1_MAT_s002
                                 + 1.01472 w_k1_MSFT_s002
                                 + 0.98948 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.799196 w_k1_ADM_s002
                                 + 0.827691 w_k1_AXP_s002
                                 + 0.953963 w_k1_DIS_s002
                                 + 0.975367 w_k1_FDX_s002
                                 + 0.959659 w_k1_IBM_s002
                                 + 1.01998 w_k1_INTC_s002
                                 + 1.03941 w_k1_KO_s002 + 1.1043 w_k1_MAT_s002
                                 + 0.985765 w_k1_MSFT_s002
                                 + 1.04173 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.83151 w_k1_ADM_s002
                                 + 0.949635 w_k1_AXP_s002
                                 + 0.909561 w_k1_DIS_s002
                                 + 0.903343 w_k1_FDX_s002
                                 + 1.26398 w_k1_IBM_s002
                                 + 0.900213 w_k1_INTC_s002
                                 + 1.00689 w_k1_KO_s002 + 1.26554 w_k1_MAT_s002
                                 + 0.91397 w_k1_MSFT_s002
                                 + 0.987782 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.62642 w_k1_ADM_s002
                                 + 1.12686 w_k1_AXP_s002
                                 + 1.28232 w_k1_DIS_s002
                                 + 1.31312 w_k1_FDX_s002
                                 + 0.799537 w_k1_IBM_s002
                                 + 1.10986 w_k1_INTC_s002
                                 + 0.989964 w_k1_KO_s002
                                 + 1.02399 w_k1_MAT_s002
                                 + 1.10838 w_k1_MSFT_s002
                                 + 1.05157 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.11119 w_k1_ADM_s002
                                 + 1.29707 w_k1_AXP_s002
                                 + 1.63245 w_k1_DIS_s002
                                 + 1.29847 w_k1_FDX_s002
                                 + 1.17453 w_k1_IBM_s002
                                 + 1.5399 w_k1_INTC_s002 + 1.30466 w_k1_KO_s002
                                 + 1.2696 w_k1_MAT_s002 + 1.34 w_k1_MSFT_s002
                                 + 1.10659 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.915138 w_k1_ADM_s002
                                 + 0.77029 w_k1_AXP_s002
                                 + 0.694634 w_k1_DIS_s002
                                 + 0.736496 w_k1_FDX_s002
                                 + 1.24276 w_k1_IBM_s002
                                 + 0.99418 w_k1_INTC_s002
                                 + 0.941957 w_k1_KO_s002
                                 + 1.01948 w_k1_MAT_s002
                                 + 0.824074 w_k1_MSFT_s002
                                 + 0.920038 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.01753 w_k1_ADM_s003
                                 + 1.28803 w_k1_AXP_s003
                                 + 1.21296 w_k1_DIS_s003
                                 + 1.13664 w_k1_FDX_s003
                                 + 1.32184 w_k1_IBM_s003
                                 + 1.04443 w_k1_INTC_s003
                                 + 1.36709 w_k1_KO_s003 + 1.60494 w_k1_MAT_s003
                                 + 1.09701 w_k1_MSFT_s003
                                 + 1.12347 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.967942 w_k1_ADM_s003
                                 + 1.04671 w_k1_AXP_s003
                                 + 0.820505 w_k1_DIS_s003
                                 + 0.857885 w_k1_FDX_s003
                                 + 1.21084 w_k1_IBM_s003
                                 + 0.94466 w_k1_INTC_s003
                                 + 1.15182 w_k1_KO_s003 + 1.34822 w_k1_MAT_s003
                                 + 0.925197 w_k1_MSFT_s003
                                 + 0.961158 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.33138 w_k1_ADM_s003
                                 + 1.11319 w_k1_AXP_s003
                                 + 1.18186 w_k1_DIS_s003 + 1.2692 w_k1_FDX_s003
                                 + 0.848854 w_k1_IBM_s003
                                 + 1.02624 w_k1_INTC_s003 + 1.0899 w_k1_KO_s003
                                 + 1.06948 w_k1_MAT_s003
                                 + 1.24164 w_k1_MSFT_s003
                                 + 1.05484 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.1531 w_k1_ADM_s003
                                 + 0.690565 w_k1_AXP_s003
                                 + 0.781317 w_k1_DIS_s003
                                 + 0.8919 w_k1_FDX_s003
                                 + 0.759888 w_k1_IBM_s003
                                 + 0.904732 w_k1_INTC_s003
                                 + 0.846505 w_k1_KO_s003
                                 + 0.56124 w_k1_MAT_s003
                                 + 0.871088 w_k1_MSFT_s003
                                 + 0.911363 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.892143 w_k1_ADM_s003
                                 + 0.751602 w_k1_AXP_s003
                                 + 0.737204 w_k1_DIS_s003
                                 + 0.91587 w_k1_FDX_s003 + 0.7227 w_k1_IBM_s003
                                 + 0.907327 w_k1_INTC_s003
                                 + 0.777884 w_k1_KO_s003
                                 + 0.704858 w_k1_MAT_s003
                                 + 0.994735 w_k1_MSFT_s003
                                 + 0.854822 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.13698 w_k1_ADM_s003
                                 + 1.187 w_k1_AXP_s003 + 0.720223 w_k1_DIS_s003
                                 + 0.988831 w_k1_FDX_s003
                                 + 0.846818 w_k1_IBM_s003
                                 + 0.941037 w_k1_INTC_s003
                                 + 0.745066 w_k1_KO_s003
                                 + 1.01719 w_k1_MAT_s003
                                 + 0.934521 w_k1_MSFT_s003
                                 + 0.945399 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.845456 w_k1_ADM_s003
                                 + 0.751049 w_k1_AXP_s003
                                 + 0.792004 w_k1_DIS_s003
                                 + 0.854297 w_k1_FDX_s003
                                 + 1.01647 w_k1_IBM_s003
                                 + 1.02728 w_k1_INTC_s003
                                 + 1.00561 w_k1_KO_s003
                                 + 0.905916 w_k1_MAT_s003
                                 + 0.967356 w_k1_MSFT_s003
                                 + 0.95712 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.21854 w_k1_ADM_s003
                                 + 1.40312 w_k1_AXP_s003
                                 + 1.63988 w_k1_DIS_s003
                                 + 1.29576 w_k1_FDX_s003
                                 + 1.26828 w_k1_IBM_s003
                                 + 1.44321 w_k1_INTC_s003
                                 + 1.50863 w_k1_KO_s003 + 1.70894 w_k1_MAT_s003
                                 + 1.58162 w_k1_MSFT_s003
                                 + 1.29609 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.27374 w_k1_ADM_s004
                                 + 1.28302 w_k1_AXP_s004
                                 + 1.11783 w_k1_DIS_s004
                                 + 1.22403 w_k1_FDX_s004
                                 + 0.803231 w_k1_IBM_s004
                                 + 0.871787 w_k1_INTC_s004
                                 + 1.06617 w_k1_KO_s004 + 1.03318 w_k1_MAT_s004
                                 + 1.03152 w_k1_MSFT_s004
                                 + 0.999857 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.03597 w_k1_ADM_s004
                                 + 1.10767 w_k1_AXP_s004
                                 + 1.21105 w_k1_DIS_s004
                                 + 1.14472 w_k1_FDX_s004
                                 + 0.988162 w_k1_IBM_s004
                                 + 1.00633 w_k1_INTC_s004
                                 + 1.23343 w_k1_KO_s004 + 1.24503 w_k1_MAT_s004
                                 + 1.08569 w_k1_MSFT_s004
                                 + 1.01913 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.1232 w_k1_ADM_s004
                                 + 1.24324 w_k1_AXP_s004
                                 + 1.41498 w_k1_DIS_s004
                                 + 1.21236 w_k1_FDX_s004
                                 + 1.19932 w_k1_IBM_s004
                                 + 0.936214 w_k1_INTC_s004
                                 + 1.36304 w_k1_KO_s004 + 1.38681 w_k1_MAT_s004
                                 + 1.16861 w_k1_MSFT_s004
                                 + 1.11789 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.10938 w_k1_ADM_s004
                                 + 1.24377 w_k1_AXP_s004
                                 + 1.21038 w_k1_DIS_s004
                                 + 1.09104 w_k1_FDX_s004
                                 + 1.60231 w_k1_IBM_s004
                                 + 1.22817 w_k1_INTC_s004
                                 + 1.60197 w_k1_KO_s004 + 1.56877 w_k1_MAT_s004
                                 + 1.02509 w_k1_MSFT_s004
                                 + 1.13185 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.45447 w_k1_ADM_s004
                                 + 1.56062 w_k1_AXP_s004 + 1.8553 w_k1_DIS_s004
                                 + 1.56698 w_k1_FDX_s004
                                 + 1.27663 w_k1_IBM_s004
                                 + 1.33548 w_k1_INTC_s004
                                 + 1.76319 w_k1_KO_s004 + 1.68723 w_k1_MAT_s004
                                 + 1.44105 w_k1_MSFT_s004
                                 + 1.35952 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.924992 w_k1_ADM_s004
                                 + 0.725792 w_k1_AXP_s004
                                 + 0.844446 w_k1_DIS_s004
                                 + 0.871298 w_k1_FDX_s004
                                 + 0.979994 w_k1_IBM_s004
                                 + 0.827259 w_k1_INTC_s004
                                 + 1.03223 w_k1_KO_s004
                                 + 0.948296 w_k1_MAT_s004
                                 + 0.855146 w_k1_MSFT_s004
                                 + 0.85479 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.921642 w_k1_ADM_s004
                                 + 1.29342 w_k1_AXP_s004
                                 + 1.05152 w_k1_DIS_s004
                                 + 1.12442 w_k1_FDX_s004
                                 + 1.20207 w_k1_IBM_s004
                                 + 1.21617 w_k1_INTC_s004
                                 + 1.37426 w_k1_KO_s004 + 1.44403 w_k1_MAT_s004
                                 + 0.943624 w_k1_MSFT_s004
                                 + 1.12148 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.796111 w_k1_ADM_s004
                                 + 0.825637 w_k1_AXP_s004
                                 + 0.850505 w_k1_DIS_s004
                                 + 0.900957 w_k1_FDX_s004
                                 + 0.644759 w_k1_IBM_s004
                                 + 0.923411 w_k1_INTC_s004
                                 + 0.893718 w_k1_KO_s004
                                 + 0.62239 w_k1_MAT_s004
                                 + 0.947086 w_k1_MSFT_s004
                                 + 0.90146 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.688538 w_k1_ADM_s005
                                 + 0.604456 w_k1_AXP_s005
                                 + 0.513716 w_k1_DIS_s005
                                 + 0.712826 w_k1_FDX_s005
                                 + 0.635095 w_k1_IBM_s005
                                 + 0.669576 w_k1_INTC_s005
                                 + 0.639498 w_k1_KO_s005
                                 + 0.638513 w_k1_MAT_s005
                                 + 0.671422 w_k1_MSFT_s005
                                 + 0.727445 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.07375 w_k1_ADM_s005
                                 + 0.884731 w_k1_AXP_s005
                                 + 0.822587 w_k1_DIS_s005
                                 + 0.85118 w_k1_FDX_s005
                                 + 1.27266 w_k1_IBM_s005
                                 + 1.0441 w_k1_INTC_s005 + 1.08016 w_k1_KO_s005
                                 + 0.704209 w_k1_MAT_s005
                                 + 0.985116 w_k1_MSFT_s005
                                 + 0.931304 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.3462 w_k1_ADM_s005
                                 + 1.39903 w_k1_AXP_s005
                                 + 0.998319 w_k1_DIS_s005
                                 + 1.23199 w_k1_FDX_s005
                                 + 0.892137 w_k1_IBM_s005
                                 + 1.25224 w_k1_INTC_s005
                                 + 0.893351 w_k1_KO_s005
                                 + 0.461409 w_k1_MAT_s005
                                 + 1.07052 w_k1_MSFT_s005
                                 + 0.948317 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.02493 w_k1_ADM_s005
                                 + 0.926645 w_k1_AXP_s005
                                 + 0.898203 w_k1_DIS_s005
                                 + 0.93578 w_k1_FDX_s005
                                 + 0.878056 w_k1_IBM_s005
                                 + 1.12137 w_k1_INTC_s005
                                 + 0.853016 w_k1_KO_s005
                                 + 0.595584 w_k1_MAT_s005
                                 + 0.988074 w_k1_MSFT_s005
                                 + 0.869927 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.889865 w_k1_ADM_s005
                                 + 0.664807 w_k1_AXP_s005
                                 + 0.662964 w_k1_DIS_s005
                                 + 0.708395 w_k1_FDX_s005
                                 + 0.993822 w_k1_IBM_s005
                                 + 0.851082 w_k1_INTC_s005
                                 + 0.914351 w_k1_KO_s005
                                 + 0.463893 w_k1_MAT_s005
                                 + 0.774006 w_k1_MSFT_s005
                                 + 0.734448 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.25124 w_k1_ADM_s005
                                 + 1.66637 w_k1_AXP_s005
                                 + 1.84939 w_k1_DIS_s005
                                 + 1.45636 w_k1_FDX_s005
                                 + 1.60958 w_k1_IBM_s005
                                 + 1.51255 w_k1_INTC_s005
                                 + 1.47802 w_k1_KO_s005 + 1.24016 w_k1_MAT_s005
                                 + 1.21927 w_k1_MSFT_s005
                                 + 1.27606 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.15852 w_k1_ADM_s005
                                 + 1.33172 w_k1_AXP_s005
                                 + 1.09277 w_k1_DIS_s005
                                 + 1.05183 w_k1_FDX_s005
                                 + 1.41733 w_k1_IBM_s005
                                 + 1.02228 w_k1_INTC_s005
                                 + 1.21224 w_k1_KO_s005
                                 + 0.801467 w_k1_MAT_s005
                                 + 1.0929 w_k1_MSFT_s005 + 1.102 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.58065 w_k1_ADM_s005
                                 + 1.42897 w_k1_AXP_s005 + 1.4327 w_k1_DIS_s005
                                 + 1.12993 w_k1_FDX_s005
                                 + 1.50867 w_k1_IBM_s005
                                 + 0.985668 w_k1_INTC_s005
                                 + 1.42066 w_k1_KO_s005 + 1.05369 w_k1_MAT_s005
                                 + 1.11247 w_k1_MSFT_s005
                                 + 1.13963 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.70846 w_k1_ADM_s006
                                 + 0.628587 w_k1_AXP_s006
                                 + 0.606433 w_k1_DIS_s006
                                 + 0.712596 w_k1_FDX_s006
                                 + 1.05505 w_k1_IBM_s006
                                 + 0.859743 w_k1_INTC_s006
                                 + 0.83902 w_k1_KO_s006 + 1.00174 w_k1_MAT_s006
                                 + 0.733647 w_k1_MSFT_s006
                                 + 0.8061 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.82621 w_k1_ADM_s006
                                 + 0.851855 w_k1_AXP_s006
                                 + 0.780684 w_k1_DIS_s006
                                 + 0.8471 w_k1_FDX_s006
                                 + 0.837898 w_k1_IBM_s006
                                 + 0.884381 w_k1_INTC_s006
                                 + 0.792918 w_k1_KO_s006
                                 + 1.01346 w_k1_MAT_s006
                                 + 0.907483 w_k1_MSFT_s006
                                 + 0.829216 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.41584 w_k1_ADM_s006
                                 + 1.65821 w_k1_AXP_s006
                                 + 1.85324 w_k1_DIS_s006
                                 + 1.57318 w_k1_FDX_s006
                                 + 1.05373 w_k1_IBM_s006
                                 + 1.85276 w_k1_INTC_s006
                                 + 1.41163 w_k1_KO_s006 + 1.41067 w_k1_MAT_s006
                                 + 1.6628 w_k1_MSFT_s006 + 1.21585 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.10946 w_k1_ADM_s006
                                 + 0.940215 w_k1_AXP_s006
                                 + 1.01875 w_k1_DIS_s006
                                 + 1.14542 w_k1_FDX_s006
                                 + 0.803138 w_k1_IBM_s006
                                 + 1.02321 w_k1_INTC_s006
                                 + 0.886083 w_k1_KO_s006
                                 + 0.773835 w_k1_MAT_s006
                                 + 1.04866 w_k1_MSFT_s006
                                 + 0.974948 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.923086 w_k1_ADM_s006
                                 + 0.886043 w_k1_AXP_s006
                                 + 1.2657 w_k1_DIS_s006 + 1.0008 w_k1_FDX_s006
                                 + 1.20449 w_k1_IBM_s006
                                 + 1.25432 w_k1_INTC_s006
                                 + 1.25144 w_k1_KO_s006
                                 + 0.997468 w_k1_MAT_s006
                                 + 1.2452 w_k1_MSFT_s006 + 1.00732 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.0057 w_k1_ADM_s006
                                 + 1.18766 w_k1_AXP_s006
                                 + 0.92438 w_k1_DIS_s006
                                 + 1.05482 w_k1_FDX_s006
                                 + 1.14691 w_k1_IBM_s006
                                 + 1.08885 w_k1_INTC_s006
                                 + 0.953649 w_k1_KO_s006
                                 + 1.40774 w_k1_MAT_s006
                                 + 1.06259 w_k1_MSFT_s006
                                 + 1.07396 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.992124 w_k1_ADM_s006
                                 + 1.34874 w_k1_AXP_s006
                                 + 1.33735 w_k1_DIS_s006
                                 + 1.17947 w_k1_FDX_s006
                                 + 1.66493 w_k1_IBM_s006
                                 + 1.4072 w_k1_INTC_s006 + 1.3766 w_k1_KO_s006
                                 + 1.73951 w_k1_MAT_s006
                                 + 1.14532 w_k1_MSFT_s006 + 1.1752 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.19456 w_k1_ADM_s006
                                 + 1.25386 w_k1_AXP_s006
                                 + 1.41229 w_k1_DIS_s006
                                 + 1.34837 w_k1_FDX_s006
                                 + 0.840565 w_k1_IBM_s006
                                 + 1.4024 w_k1_INTC_s006 + 1.13593 w_k1_KO_s006
                                 + 1.08086 w_k1_MAT_s006
                                 + 1.19546 w_k1_MSFT_s006
                                 + 1.06156 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.30385 w_k1_ADM_s007
                                 + 1.43394 w_k1_AXP_s007
                                 + 1.31603 w_k1_DIS_s007
                                 + 1.20701 w_k1_FDX_s007
                                 + 1.25676 w_k1_IBM_s007
                                 + 1.21127 w_k1_INTC_s007
                                 + 1.20832 w_k1_KO_s007 + 1.08896 w_k1_MAT_s007
                                 + 1.24471 w_k1_MSFT_s007 + 1.1242 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.16162 w_k1_ADM_s007
                                 + 1.31868 w_k1_AXP_s007 + 1.0896 w_k1_DIS_s007
                                 + 1.00503 w_k1_FDX_s007
                                 + 1.53902 w_k1_IBM_s007 + 1.009 w_k1_INTC_s007
                                 + 1.17295 w_k1_KO_s007 + 1.35921 w_k1_MAT_s007
                                 + 0.906829 w_k1_MSFT_s007
                                 + 1.09213 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.23296 w_k1_ADM_s007
                                 + 1.38001 w_k1_AXP_s007
                                 + 1.61719 w_k1_DIS_s007
                                 + 1.40128 w_k1_FDX_s007
                                 + 1.01758 w_k1_IBM_s007
                                 + 1.05125 w_k1_INTC_s007
                                 + 1.30094 w_k1_KO_s007
                                 + 0.924156 w_k1_MAT_s007
                                 + 1.15763 w_k1_MSFT_s007
                                 + 1.20787 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.852049 w_k1_ADM_s007
                                 + 0.825953 w_k1_AXP_s007
                                 + 0.672686 w_k1_DIS_s007
                                 + 0.76506 w_k1_FDX_s007
                                 + 0.849402 w_k1_IBM_s007
                                 + 0.845818 w_k1_INTC_s007
                                 + 0.768317 w_k1_KO_s007
                                 + 0.827826 w_k1_MAT_s007
                                 + 0.812034 w_k1_MSFT_s007
                                 + 0.89527 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.05045 w_k1_ADM_s007
                                 + 0.932443 w_k1_AXP_s007
                                 + 0.921806 w_k1_DIS_s007
                                 + 0.984663 w_k1_FDX_s007
                                 + 0.869618 w_k1_IBM_s007
                                 + 1.04832 w_k1_INTC_s007
                                 + 1.00354 w_k1_KO_s007
                                 + 0.865722 w_k1_MAT_s007
                                 + 0.963718 w_k1_MSFT_s007
                                 + 1.02336 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.18574 w_k1_ADM_s007
                                 + 1.32185 w_k1_AXP_s007
                                 + 1.04356 w_k1_DIS_s007
                                 + 1.26474 w_k1_FDX_s007
                                 + 0.810494 w_k1_IBM_s007
                                 + 1.12493 w_k1_INTC_s007
                                 + 1.03866 w_k1_KO_s007
                                 + 0.992059 w_k1_MAT_s007
                                 + 1.14611 w_k1_MSFT_s007
                                 + 1.06637 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.994169 w_k1_ADM_s007
                                 + 1.09744 w_k1_AXP_s007
                                 + 1.13752 w_k1_DIS_s007
                                 + 1.09753 w_k1_FDX_s007
                                 + 1.26138 w_k1_IBM_s007
                                 + 1.15703 w_k1_INTC_s007
                                 + 1.16514 w_k1_KO_s007 + 1.15944 w_k1_MAT_s007
                                 + 1.08209 w_k1_MSFT_s007 + 1.1319 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.857216 w_k1_ADM_s007
                                 + 0.843726 w_k1_AXP_s007
                                 + 1.00919 w_k1_DIS_s007
                                 + 0.832366 w_k1_FDX_s007
                                 + 1.39057 w_k1_IBM_s007
                                 + 0.79718 w_k1_INTC_s007
                                 + 1.13636 w_k1_KO_s007 + 1.17836 w_k1_MAT_s007
                                 + 0.87458 w_k1_MSFT_s007
                                 + 0.908839 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.994291 w_k1_ADM_s010
                                 + 0.988816 w_k1_AXP_s010
                                 + 0.926418 w_k1_DIS_s010
                                 + 0.776475 w_k1_FDX_s010
                                 + 1.48947 w_k1_IBM_s010
                                 + 0.903685 w_k1_INTC_s010
                                 + 1.06955 w_k1_KO_s010 + 1.13412 w_k1_MAT_s010
                                 + 0.821465 w_k1_MSFT_s010
                                 + 0.872295 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.1205 w_k1_ADM_s010
                                 + 1.41894 w_k1_AXP_s010
                                 + 1.35745 w_k1_DIS_s010
                                 + 1.06935 w_k1_FDX_s010
                                 + 1.59253 w_k1_IBM_s010
                                 + 1.19329 w_k1_INTC_s010
                                 + 1.31561 w_k1_KO_s010 + 2.30063 w_k1_MAT_s010
                                 + 1.20133 w_k1_MSFT_s010
                                 + 1.22948 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.863358 w_k1_ADM_s010
                                 + 0.770743 w_k1_AXP_s010
                                 + 0.618776 w_k1_DIS_s010
                                 + 0.756622 w_k1_FDX_s010
                                 + 0.78167 w_k1_IBM_s010
                                 + 0.721683 w_k1_INTC_s010
                                 + 0.707366 w_k1_KO_s010
                                 + 0.797402 w_k1_MAT_s010
                                 + 0.813063 w_k1_MSFT_s010
                                 + 0.855537 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.48999 w_k1_ADM_s010
                                 + 1.40563 w_k1_AXP_s010
                                 + 1.55031 w_k1_DIS_s010
                                 + 1.34271 w_k1_FDX_s010
                                 + 1.02194 w_k1_IBM_s010
                                 + 1.4701 w_k1_INTC_s010 + 1.22629 w_k1_KO_s010
                                 + 1.11254 w_k1_MAT_s010
                                 + 1.31481 w_k1_MSFT_s010
                                 + 1.20763 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.01543 w_k1_ADM_s010
                                 + 0.846903 w_k1_AXP_s010
                                 + 0.989273 w_k1_DIS_s010
                                 + 0.888525 w_k1_FDX_s010
                                 + 0.975888 w_k1_IBM_s010
                                 + 1.07739 w_k1_INTC_s010
                                 + 1.05689 w_k1_KO_s010 + 1.04433 w_k1_MAT_s010
                                 + 1.05739 w_k1_MSFT_s010
                                 + 0.997853 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.24635 w_k1_ADM_s010
                                 + 1.20228 w_k1_AXP_s010
                                 + 0.763622 w_k1_DIS_s010
                                 + 0.999041 w_k1_FDX_s010
                                 + 0.818247 w_k1_IBM_s010
                                 + 0.907297 w_k1_INTC_s010
                                 + 0.782821 w_k1_KO_s010
                                 + 0.769039 w_k1_MAT_s010
                                 + 0.812325 w_k1_MSFT_s010
                                 + 0.908642 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.36634 w_k1_ADM_s010
                                 + 1.24984 w_k1_AXP_s010 + 1.3586 w_k1_DIS_s010
                                 + 1.10736 w_k1_FDX_s010
                                 + 1.24553 w_k1_IBM_s010
                                 + 1.03097 w_k1_INTC_s010
                                 + 1.27539 w_k1_KO_s010 + 1.23054 w_k1_MAT_s010
                                 + 1.26138 w_k1_MSFT_s010
                                 + 1.09258 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.1148 w_k1_ADM_s010
                                 + 1.2407 w_k1_AXP_s010 + 1.05967 w_k1_DIS_s010
                                 + 1.10658 w_k1_FDX_s010
                                 + 0.953031 w_k1_IBM_s010
                                 + 1.04327 w_k1_INTC_s010
                                 + 0.909022 w_k1_KO_s010
                                 + 1.34723 w_k1_MAT_s010
                                 + 1.0591 w_k1_MSFT_s010
                                 + 0.959089 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.849899 w_k1_ADM_s011
                                 + 0.815295 w_k1_AXP_s011
                                 + 1.14563 w_k1_DIS_s011
                                 + 1.04844 w_k1_FDX_s011
                                 + 1.24418 w_k1_IBM_s011
                                 + 1.07003 w_k1_INTC_s011
                                 + 1.17855 w_k1_KO_s011 + 1.00606 w_k1_MAT_s011
                                 + 0.979097 w_k1_MSFT_s011
                                 + 1.04064 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.988928 w_k1_ADM_s011
                                 + 1.21685 w_k1_AXP_s011
                                 + 1.21405 w_k1_DIS_s011
                                 + 1.08287 w_k1_FDX_s011
                                 + 1.68609 w_k1_IBM_s011
                                 + 1.16241 w_k1_INTC_s011
                                 + 1.47472 w_k1_KO_s011 + 1.96377 w_k1_MAT_s011
                                 + 1.1955 w_k1_MSFT_s011 + 1.25189 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.693184 w_k1_ADM_s011
                                 + 0.730879 w_k1_AXP_s011
                                 + 0.830039 w_k1_DIS_s011
                                 + 0.895696 w_k1_FDX_s011
                                 + 0.866991 w_k1_IBM_s011
                                 + 0.941309 w_k1_INTC_s011
                                 + 0.932904 w_k1_KO_s011
                                 + 0.760215 w_k1_MAT_s011
                                 + 0.927209 w_k1_MSFT_s011
                                 + 0.92001 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.13276 w_k1_ADM_s011
                                 + 1.12665 w_k1_AXP_s011
                                 + 0.895848 w_k1_DIS_s011
                                 + 0.942119 w_k1_FDX_s011
                                 + 1.13799 w_k1_IBM_s011
                                 + 0.896249 w_k1_INTC_s011
                                 + 0.979682 w_k1_KO_s011
                                 + 1.35737 w_k1_MAT_s011
                                 + 0.939385 w_k1_MSFT_s011
                                 + 1.06835 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.07223 w_k1_ADM_s011
                                 + 1.40197 w_k1_AXP_s011 + 1.1341 w_k1_DIS_s011
                                 + 1.2431 w_k1_FDX_s011 + 1.01175 w_k1_IBM_s011
                                 + 1.13027 w_k1_INTC_s011
                                 + 1.14082 w_k1_KO_s011 + 1.3975 w_k1_MAT_s011
                                 + 1.11237 w_k1_MSFT_s011
                                 + 1.22928 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.653354 w_k1_ADM_s011
                                 + 0.881603 w_k1_AXP_s011
                                 + 1.35959 w_k1_DIS_s011
                                 + 1.17751 w_k1_FDX_s011
                                 + 0.938296 w_k1_IBM_s011
                                 + 1.24459 w_k1_INTC_s011
                                 + 1.26781 w_k1_KO_s011
                                 + 0.888541 w_k1_MAT_s011
                                 + 1.16897 w_k1_MSFT_s011
                                 + 1.07348 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.790523 w_k1_ADM_s011
                                 + 0.860926 w_k1_AXP_s011
                                 + 0.879566 w_k1_DIS_s011
                                 + 1.07944 w_k1_FDX_s011
                                 + 0.631922 w_k1_IBM_s011
                                 + 1.02849 w_k1_INTC_s011
                                 + 0.743898 w_k1_KO_s011
                                 + 0.723222 w_k1_MAT_s011
                                 + 1.04129 w_k1_MSFT_s011
                                 + 0.964698 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.45171 w_k1_ADM_s011
                                 + 1.53038 w_k1_AXP_s011
                                 + 1.47885 w_k1_DIS_s011
                                 + 1.43919 w_k1_FDX_s011
                                 + 1.06086 w_k1_IBM_s011
                                 + 1.24239 w_k1_INTC_s011
                                 + 1.24754 w_k1_KO_s011 + 1.1334 w_k1_MAT_s011
                                 + 1.2704 w_k1_MSFT_s011 + 1.33204 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.07211 w_k1_ADM_s012
                                 + 0.862743 w_k1_AXP_s012
                                 + 1.01013 w_k1_DIS_s012
                                 + 1.09866 w_k1_FDX_s012
                                 + 0.700979 w_k1_IBM_s012
                                 + 1.01911 w_k1_INTC_s012
                                 + 0.804153 w_k1_KO_s012
                                 + 0.809996 w_k1_MAT_s012
                                 + 1.09208 w_k1_MSFT_s012
                                 + 0.999977 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.964741 w_k1_ADM_s012
                                 + 0.67579 w_k1_AXP_s012
                                 + 0.927046 w_k1_DIS_s012
                                 + 0.854897 w_k1_FDX_s012
                                 + 0.883982 w_k1_IBM_s012
                                 + 0.892533 w_k1_INTC_s012
                                 + 0.785731 w_k1_KO_s012
                                 + 0.583051 w_k1_MAT_s012
                                 + 1.0423 w_k1_MSFT_s012
                                 + 0.855421 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.781002 w_k1_ADM_s012
                                 + 0.52864 w_k1_AXP_s012
                                 + 0.594499 w_k1_DIS_s012
                                 + 0.674016 w_k1_FDX_s012
                                 + 0.663477 w_k1_IBM_s012
                                 + 0.689365 w_k1_INTC_s012
                                 + 0.637041 w_k1_KO_s012
                                 + 0.464724 w_k1_MAT_s012
                                 + 0.667923 w_k1_MSFT_s012
                                 + 0.789951 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.20853 w_k1_ADM_s012
                                 + 1.01675 w_k1_AXP_s012
                                 + 1.09717 w_k1_DIS_s012
                                 + 1.06086 w_k1_FDX_s012
                                 + 0.961966 w_k1_IBM_s012
                                 + 1.10005 w_k1_INTC_s012
                                 + 0.929063 w_k1_KO_s012
                                 + 0.977585 w_k1_MAT_s012
                                 + 1.03078 w_k1_MSFT_s012
                                 + 1.04975 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.751414 w_k1_ADM_s012
                                 + 0.755385 w_k1_AXP_s012
                                 + 0.546887 w_k1_DIS_s012
                                 + 0.681633 w_k1_FDX_s012
                                 + 0.888522 w_k1_IBM_s012
                                 + 0.904907 w_k1_INTC_s012
                                 + 0.702234 w_k1_KO_s012
                                 + 0.640984 w_k1_MAT_s012
                                 + 0.799283 w_k1_MSFT_s012
                                 + 0.898178 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.03593 w_k1_ADM_s012
                                 + 1.08749 w_k1_AXP_s012
                                 + 1.21485 w_k1_DIS_s012
                                 + 1.04284 w_k1_FDX_s012
                                 + 1.54244 w_k1_IBM_s012
                                 + 0.968337 w_k1_INTC_s012
                                 + 1.09409 w_k1_KO_s012 + 1.02826 w_k1_MAT_s012
                                 + 0.954726 w_k1_MSFT_s012
                                 + 1.18605 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.56094 w_k1_ADM_s012
                                 + 1.28138 w_k1_AXP_s012 + 1.5142 w_k1_DIS_s012
                                 + 1.28828 w_k1_FDX_s012
                                 + 1.01303 w_k1_IBM_s012
                                 + 1.18706 w_k1_INTC_s012
                                 + 1.10703 w_k1_KO_s012
                                 + 0.995756 w_k1_MAT_s012
                                 + 1.24067 w_k1_MSFT_s012 + 1.1641 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.93278 w_k1_ADM_s012
                                 + 0.85438 w_k1_AXP_s012 + 0.8203 w_k1_DIS_s012
                                 + 0.823908 w_k1_FDX_s012
                                 + 1.17971 w_k1_IBM_s012
                                 + 0.838482 w_k1_INTC_s012
                                 + 0.975284 w_k1_KO_s012
                                 + 0.880399 w_k1_MAT_s012
                                 + 0.831754 w_k1_MSFT_s012
                                 + 1.01962 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.878015 w_k1_ADM_s013
                                 + 0.89989 w_k1_AXP_s013
                                 + 0.954117 w_k1_DIS_s013
                                 + 0.953417 w_k1_FDX_s013
                                 + 0.862469 w_k1_IBM_s013
                                 + 0.928667 w_k1_INTC_s013
                                 + 0.909461 w_k1_KO_s013
                                 + 0.815405 w_k1_MAT_s013
                                 + 0.907123 w_k1_MSFT_s013
                                 + 0.881204 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.09028 w_k1_ADM_s013
                                 + 1.2099 w_k1_AXP_s013 + 1.60139 w_k1_DIS_s013
                                 + 1.32006 w_k1_FDX_s013
                                 + 0.993619 w_k1_IBM_s013
                                 + 1.15221 w_k1_INTC_s013
                                 + 1.29437 w_k1_KO_s013 + 1.13884 w_k1_MAT_s013
                                 + 1.27451 w_k1_MSFT_s013 + 1.1594 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.08236 w_k1_ADM_s013
                                 + 1.17228 w_k1_AXP_s013
                                 + 0.918028 w_k1_DIS_s013
                                 + 1.14773 w_k1_FDX_s013
                                 + 0.667543 w_k1_IBM_s013
                                 + 0.908697 w_k1_INTC_s013
                                 + 0.802903 w_k1_KO_s013
                                 + 0.747383 w_k1_MAT_s013
                                 + 0.990749 w_k1_MSFT_s013
                                 + 0.828007 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.639058 w_k1_ADM_s013
                                 + 0.882514 w_k1_AXP_s013
                                 + 0.719862 w_k1_DIS_s013
                                 + 0.804858 w_k1_FDX_s013
                                 + 0.791528 w_k1_IBM_s013
                                 + 0.93045 w_k1_INTC_s013
                                 + 0.63541 w_k1_KO_s013
                                 + 0.657303 w_k1_MAT_s013
                                 + 0.812925 w_k1_MSFT_s013
                                 + 0.780647 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.33606 w_k1_ADM_s013
                                 + 1.22561 w_k1_AXP_s013
                                 + 1.39854 w_k1_DIS_s013
                                 + 1.12289 w_k1_FDX_s013
                                 + 1.48629 w_k1_IBM_s013
                                 + 1.13047 w_k1_INTC_s013
                                 + 1.43585 w_k1_KO_s013 + 1.28041 w_k1_MAT_s013
                                 + 1.12321 w_k1_MSFT_s013
                                 + 1.17583 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.0632 w_k1_ADM_s013
                                 + 1.27332 w_k1_AXP_s013
                                 + 1.20467 w_k1_DIS_s013
                                 + 1.09839 w_k1_FDX_s013
                                 + 1.10644 w_k1_IBM_s013
                                 + 1.03646 w_k1_INTC_s013
                                 + 1.05504 w_k1_KO_s013
                                 + 0.918469 w_k1_MAT_s013
                                 + 0.89395 w_k1_MSFT_s013
                                 + 1.04025 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.804584 w_k1_ADM_s013
                                 + 0.812092 w_k1_AXP_s013
                                 + 0.846595 w_k1_DIS_s013
                                 + 0.839451 w_k1_FDX_s013
                                 + 1.1107 w_k1_IBM_s013
                                 + 0.747988 w_k1_INTC_s013
                                 + 0.97354 w_k1_KO_s013
                                 + 0.690342 w_k1_MAT_s013
                                 + 0.732003 w_k1_MSFT_s013
                                 + 0.950698 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.26054 w_k1_ADM_s013
                                 + 0.942589 w_k1_AXP_s013
                                 + 1.22989 w_k1_DIS_s013
                                 + 1.10221 w_k1_FDX_s013
                                 + 0.837889 w_k1_IBM_s013
                                 + 0.953292 w_k1_INTC_s013
                                 + 0.982319 w_k1_KO_s013
                                 + 0.820141 w_k1_MAT_s013
                                 + 1.07192 w_k1_MSFT_s013
                                 + 0.997438 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.22883 w_k1_ADM_s014
                                 + 1.29584 w_k1_AXP_s014 + 1.3865 w_k1_DIS_s014
                                 + 1.06183 w_k1_FDX_s014
                                 + 1.57634 w_k1_IBM_s014
                                 + 1.07849 w_k1_INTC_s014
                                 + 1.36995 w_k1_KO_s014 + 1.25372 w_k1_MAT_s014
                                 + 1.17085 w_k1_MSFT_s014
                                 + 1.07695 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.54985 w_k1_ADM_s014
                                 + 1.45187 w_k1_AXP_s014
                                 + 1.71344 w_k1_DIS_s014
                                 + 1.38176 w_k1_FDX_s014
                                 + 1.25718 w_k1_IBM_s014
                                 + 1.58322 w_k1_INTC_s014
                                 + 1.46606 w_k1_KO_s014
                                 + 0.915596 w_k1_MAT_s014
                                 + 1.59378 w_k1_MSFT_s014
                                 + 1.35844 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.980221 w_k1_ADM_s014
                                 + 1.50921 w_k1_AXP_s014
                                 + 1.10869 w_k1_DIS_s014
                                 + 1.17204 w_k1_FDX_s014
                                 + 1.35136 w_k1_IBM_s014
                                 + 1.06167 w_k1_INTC_s014 + 1.0236 w_k1_KO_s014
                                 + 1.28345 w_k1_MAT_s014
                                 + 1.05686 w_k1_MSFT_s014
                                 + 1.16669 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.793952 w_k1_ADM_s014
                                 + 0.751859 w_k1_AXP_s014
                                 + 0.669637 w_k1_DIS_s014
                                 + 0.79037 w_k1_FDX_s014
                                 + 0.820911 w_k1_IBM_s014
                                 + 0.710939 w_k1_INTC_s014
                                 + 0.738836 w_k1_KO_s014
                                 + 0.664402 w_k1_MAT_s014
                                 + 0.801447 w_k1_MSFT_s014
                                 + 0.858617 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.786232 w_k1_ADM_s014
                                 + 0.967706 w_k1_AXP_s014
                                 + 0.841852 w_k1_DIS_s014
                                 + 0.848602 w_k1_FDX_s014
                                 + 1.20757 w_k1_IBM_s014
                                 + 1.11395 w_k1_INTC_s014
                                 + 1.16707 w_k1_KO_s014 + 1.03467 w_k1_MAT_s014
                                 + 1.03914 w_k1_MSFT_s014
                                 + 0.996139 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.01706 w_k1_ADM_s014
                                 + 1.2501 w_k1_AXP_s014 + 1.09979 w_k1_DIS_s014
                                 + 1.15767 w_k1_FDX_s014
                                 + 1.00521 w_k1_IBM_s014
                                 + 1.15549 w_k1_INTC_s014
                                 + 1.10083 w_k1_KO_s014 + 1.21854 w_k1_MAT_s014
                                 + 1.19691 w_k1_MSFT_s014
                                 + 1.07488 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.977189 w_k1_ADM_s014
                                 + 0.889023 w_k1_AXP_s014
                                 + 0.908873 w_k1_DIS_s014
                                 + 1.04962 w_k1_FDX_s014
                                 + 0.678108 w_k1_IBM_s014
                                 + 0.955395 w_k1_INTC_s014
                                 + 0.886989 w_k1_KO_s014
                                 + 0.506358 w_k1_MAT_s014
                                 + 0.969062 w_k1_MSFT_s014
                                 + 0.984187 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.1097 w_k1_ADM_s014
                                 + 0.893055 w_k1_AXP_s014
                                 + 1.40495 w_k1_DIS_s014
                                 + 1.24985 w_k1_FDX_s014
                                 + 0.831562 w_k1_IBM_s014
                                 + 1.09438 w_k1_INTC_s014
                                 + 1.11565 w_k1_KO_s014
                                 + 0.795953 w_k1_MAT_s014
                                 + 1.26357 w_k1_MSFT_s014
                                 + 0.999506 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.882922 w_k1_ADM_s015
                                 + 1.12474 w_k1_AXP_s015
                                 + 1.02226 w_k1_DIS_s015
                                 + 0.958291 w_k1_FDX_s015
                                 + 1.5152 w_k1_IBM_s015
                                 + 1.08145 w_k1_INTC_s015
                                 + 1.22883 w_k1_KO_s015 + 1.57719 w_k1_MAT_s015
                                 + 1.02419 w_k1_MSFT_s015
                                 + 1.14268 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.43384 w_k1_ADM_s015
                                 + 1.6962 w_k1_AXP_s015 + 1.94452 w_k1_DIS_s015
                                 + 1.44536 w_k1_FDX_s015
                                 + 1.21577 w_k1_IBM_s015
                                 + 1.45412 w_k1_INTC_s015 + 1.4117 w_k1_KO_s015
                                 + 1.63428 w_k1_MAT_s015
                                 + 1.57217 w_k1_MSFT_s015 + 1.3149 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.03016 w_k1_ADM_s015
                                 + 0.932095 w_k1_AXP_s015
                                 + 0.86386 w_k1_DIS_s015
                                 + 0.850072 w_k1_FDX_s015
                                 + 1.04867 w_k1_IBM_s015
                                 + 0.856997 w_k1_INTC_s015
                                 + 1.01031 w_k1_KO_s015
                                 + 0.981782 w_k1_MAT_s015
                                 + 0.839004 w_k1_MSFT_s015
                                 + 0.961478 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.49548 w_k1_ADM_s015
                                 + 1.31957 w_k1_AXP_s015
                                 + 1.43513 w_k1_DIS_s015
                                 + 1.19496 w_k1_FDX_s015
                                 + 1.24645 w_k1_IBM_s015
                                 + 1.10939 w_k1_INTC_s015
                                 + 1.26924 w_k1_KO_s015 + 1.76677 w_k1_MAT_s015
                                 + 1.17993 w_k1_MSFT_s015 + 1.1352 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.918163 w_k1_ADM_s015
                                 + 0.939094 w_k1_AXP_s015
                                 + 0.86831 w_k1_DIS_s015
                                 + 0.987147 w_k1_FDX_s015
                                 + 0.699556 w_k1_IBM_s015
                                 + 0.947094 w_k1_INTC_s015
                                 + 0.859184 w_k1_KO_s015
                                 + 0.798099 w_k1_MAT_s015
                                 + 0.988193 w_k1_MSFT_s015
                                 + 1.02887 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.41992 w_k1_ADM_s015
                                 + 1.39447 w_k1_AXP_s015
                                 + 1.15922 w_k1_DIS_s015
                                 + 1.16985 w_k1_FDX_s015
                                 + 1.09373 w_k1_IBM_s015
                                 + 1.11679 w_k1_INTC_s015
                                 + 1.00343 w_k1_KO_s015 + 1.25764 w_k1_MAT_s015
                                 + 1.02851 w_k1_MSFT_s015
                                 + 1.08172 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.53545 w_k1_ADM_s015
                                 + 1.20021 w_k1_AXP_s015
                                 + 1.33451 w_k1_DIS_s015
                                 + 1.35947 w_k1_FDX_s015
                                 + 0.77038 w_k1_IBM_s015
                                 + 1.1382 w_k1_INTC_s015
                                 + 0.948496 w_k1_KO_s015
                                 + 0.850126 w_k1_MAT_s015
                                 + 1.16365 w_k1_MSFT_s015
                                 + 1.06047 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.2829 w_k1_ADM_s015
                                 + 0.962106 w_k1_AXP_s015
                                 + 0.813673 w_k1_DIS_s015
                                 + 0.897619 w_k1_FDX_s015
                                 + 0.910851 w_k1_IBM_s015
                                 + 0.913149 w_k1_INTC_s015
                                 + 0.916875 w_k1_KO_s015
                                 + 0.910445 w_k1_MAT_s015
                                 + 0.892091 w_k1_MSFT_s015
                                 + 1.00337 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.919149 w_k1_ADM_s016
                                 + 0.837834 w_k1_AXP_s016
                                 + 0.872833 w_k1_DIS_s016
                                 + 0.913616 w_k1_FDX_s016
                                 + 0.833563 w_k1_IBM_s016
                                 + 0.604624 w_k1_INTC_s016
                                 + 0.894203 w_k1_KO_s016
                                 + 0.775441 w_k1_MAT_s016
                                 + 0.752108 w_k1_MSFT_s016
                                 + 0.997207 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.07393 w_k1_ADM_s016
                                 + 1.06179 w_k1_AXP_s016
                                 + 0.944273 w_k1_DIS_s016
                                 + 0.971046 w_k1_FDX_s016
                                 + 1.08355 w_k1_IBM_s016
                                 + 0.922822 w_k1_INTC_s016
                                 + 1.06063 w_k1_KO_s016 + 1.20861 w_k1_MAT_s016
                                 + 0.946973 w_k1_MSFT_s016
                                 + 0.949535 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.40411 w_k1_ADM_s016
                                 + 1.70507 w_k1_AXP_s016
                                 + 1.20986 w_k1_DIS_s016
                                 + 1.31702 w_k1_FDX_s016
                                 + 1.22827 w_k1_IBM_s016
                                 + 0.965393 w_k1_INTC_s016
                                 + 1.15315 w_k1_KO_s016 + 2.10228 w_k1_MAT_s016
                                 + 1.10543 w_k1_MSFT_s016
                                 + 1.12881 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.928837 w_k1_ADM_s016
                                 + 1.2894 w_k1_AXP_s016
                                 + 0.916813 w_k1_DIS_s016
                                 + 1.00629 w_k1_FDX_s016
                                 + 1.46234 w_k1_IBM_s016
                                 + 0.96503 w_k1_INTC_s016
                                 + 1.21222 w_k1_KO_s016 + 1.88408 w_k1_MAT_s016
                                 + 0.909816 w_k1_MSFT_s016
                                 + 1.04349 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.854802 w_k1_ADM_s016
                                 + 0.962982 w_k1_AXP_s016
                                 + 0.606564 w_k1_DIS_s016
                                 + 0.80603 w_k1_FDX_s016 + 0.7779 w_k1_IBM_s016
                                 + 0.737494 w_k1_INTC_s016
                                 + 0.734602 w_k1_KO_s016
                                 + 1.17438 w_k1_MAT_s016
                                 + 0.706442 w_k1_MSFT_s016
                                 + 0.80404 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.11963 w_k1_ADM_s016
                                 + 1.23794 w_k1_AXP_s016 + 1.2556 w_k1_DIS_s016
                                 + 1.28945 w_k1_FDX_s016
                                 + 0.887593 w_k1_IBM_s016
                                 + 0.832794 w_k1_INTC_s016
                                 + 1.06832 w_k1_KO_s016 + 1.34778 w_k1_MAT_s016
                                 + 1.00695 w_k1_MSFT_s016
                                 + 1.09073 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.5701 w_k1_ADM_s016
                                 + 1.32188 w_k1_AXP_s016
                                 + 1.66509 w_k1_DIS_s016 + 1.5031 w_k1_FDX_s016
                                 + 1.20256 w_k1_IBM_s016
                                 + 0.956797 w_k1_INTC_s016
                                 + 1.1394 w_k1_KO_s016 + 1.26554 w_k1_MAT_s016
                                 + 1.30419 w_k1_MSFT_s016
                                 + 1.20737 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.94573 w_k1_ADM_s016
                                 + 1.09241 w_k1_AXP_s016
                                 + 0.934073 w_k1_DIS_s016
                                 + 1.18105 w_k1_FDX_s016
                                 + 0.618492 w_k1_IBM_s016
                                 + 0.953192 w_k1_INTC_s016
                                 + 0.819251 w_k1_KO_s016
                                 + 0.777194 w_k1_MAT_s016
                                 + 0.900413 w_k1_MSFT_s016
                                 + 1.05802 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.770246 w_k1_ADM_s017
                                 + 1.33973 w_k1_AXP_s017
                                 + 1.29451 w_k1_DIS_s017
                                 + 1.09399 w_k1_FDX_s017
                                 + 1.30211 w_k1_IBM_s017
                                 + 0.881867 w_k1_INTC_s017
                                 + 1.14917 w_k1_KO_s017 + 2.13584 w_k1_MAT_s017
                                 + 1.01094 w_k1_MSFT_s017
                                 + 1.04625 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.04498 w_k1_ADM_s017
                                 + 1.3696 w_k1_AXP_s017 + 1.06459 w_k1_DIS_s017
                                 + 1.04924 w_k1_FDX_s017
                                 + 1.50994 w_k1_IBM_s017
                                 + 0.882629 w_k1_INTC_s017 + 1.173 w_k1_KO_s017
                                 + 2.02503 w_k1_MAT_s017
                                 + 0.997767 w_k1_MSFT_s017
                                 + 1.03143 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.859643 w_k1_ADM_s017
                                 + 1.15713 w_k1_AXP_s017
                                 + 1.21925 w_k1_DIS_s017
                                 + 1.22781 w_k1_FDX_s017
                                 + 0.79617 w_k1_IBM_s017
                                 + 1.1985 w_k1_INTC_s017 + 1.23382 w_k1_KO_s017
                                 + 1.30929 w_k1_MAT_s017
                                 + 1.23734 w_k1_MSFT_s017
                                 + 1.07104 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.892671 w_k1_ADM_s017
                                 + 0.891928 w_k1_AXP_s017
                                 + 0.897894 w_k1_DIS_s017
                                 + 0.916791 w_k1_FDX_s017
                                 + 1.05626 w_k1_IBM_s017
                                 + 0.913714 w_k1_INTC_s017
                                 + 0.999841 w_k1_KO_s017
                                 + 1.42677 w_k1_MAT_s017
                                 + 0.85856 w_k1_MSFT_s017
                                 + 1.02466 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.04803 w_k1_ADM_s017
                                 + 1.60617 w_k1_AXP_s017
                                 + 1.61124 w_k1_DIS_s017
                                 + 1.10148 w_k1_FDX_s017
                                 + 1.89717 w_k1_IBM_s017
                                 + 1.45994 w_k1_INTC_s017 + 1.7527 w_k1_KO_s017
                                 + 2.26981 w_k1_MAT_s017
                                 + 1.18773 w_k1_MSFT_s017
                                 + 1.19475 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.3989 w_k1_ADM_s017
                                 + 1.54476 w_k1_AXP_s017 + 1.9714 w_k1_DIS_s017
                                 + 1.52289 w_k1_FDX_s017
                                 + 1.24376 w_k1_IBM_s017
                                 + 1.46608 w_k1_INTC_s017
                                 + 1.64365 w_k1_KO_s017 + 1.67147 w_k1_MAT_s017
                                 + 1.46328 w_k1_MSFT_s017
                                 + 1.38596 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.941017 w_k1_ADM_s017
                                 + 1.35236 w_k1_AXP_s017
                                 + 1.22904 w_k1_DIS_s017
                                 + 1.07092 w_k1_FDX_s017
                                 + 1.01422 w_k1_IBM_s017
                                 + 1.19819 w_k1_INTC_s017
                                 + 0.983689 w_k1_KO_s017
                                 + 1.46308 w_k1_MAT_s017
                                 + 1.17254 w_k1_MSFT_s017
                                 + 1.14186 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.776245 w_k1_ADM_s017
                                 + 0.746483 w_k1_AXP_s017
                                 + 0.731209 w_k1_DIS_s017
                                 + 0.898543 w_k1_FDX_s017
                                 + 0.692085 w_k1_IBM_s017
                                 + 0.910533 w_k1_INTC_s017
                                 + 0.754016 w_k1_KO_s017
                                 + 0.815032 w_k1_MAT_s017
                                 + 0.862122 w_k1_MSFT_s017
                                 + 0.941082 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.757845 w_k1_ADM_s100
                                 + 0.987732 w_k1_AXP_s100
                                 + 0.831555 w_k1_DIS_s100
                                 + 0.901758 w_k1_FDX_s100
                                 + 0.760805 w_k1_IBM_s100
                                 + 1.28717 w_k1_INTC_s100
                                 + 0.808533 w_k1_KO_s100
                                 + 0.763424 w_k1_MAT_s100
                                 + 0.92784 w_k1_MSFT_s100
                                 + 0.951543 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.913247 w_k1_ADM_s100
                                 + 1.42024 w_k1_AXP_s100
                                 + 1.46188 w_k1_DIS_s100
                                 + 1.02497 w_k1_FDX_s100
                                 + 1.73662 w_k1_IBM_s100
                                 + 1.46615 w_k1_INTC_s100
                                 + 1.56989 w_k1_KO_s100 + 1.64748 w_k1_MAT_s100
                                 + 1.09535 w_k1_MSFT_s100
                                 + 1.22984 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.35242 w_k1_ADM_s100
                                 + 1.11708 w_k1_AXP_s100
                                 + 1.01763 w_k1_DIS_s100
                                 + 1.06969 w_k1_FDX_s100
                                 + 0.921732 w_k1_IBM_s100
                                 + 1.47579 w_k1_INTC_s100
                                 + 1.00942 w_k1_KO_s100
                                 + 0.689433 w_k1_MAT_s100
                                 + 1.0607 w_k1_MSFT_s100 + 1.02133 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.833136 w_k1_ADM_s100
                                 + 0.739434 w_k1_AXP_s100
                                 + 0.617645 w_k1_DIS_s100
                                 + 0.778818 w_k1_FDX_s100
                                 + 0.698442 w_k1_IBM_s100
                                 + 1.23166 w_k1_INTC_s100
                                 + 0.759645 w_k1_KO_s100
                                 + 0.571169 w_k1_MAT_s100
                                 + 0.761211 w_k1_MSFT_s100
                                 + 0.88886 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.07219 w_k1_ADM_s100
                                 + 1.10517 w_k1_AXP_s100
                                 + 1.16419 w_k1_DIS_s100
                                 + 1.05612 w_k1_FDX_s100
                                 + 1.29199 w_k1_IBM_s100
                                 + 1.39103 w_k1_INTC_s100
                                 + 1.19421 w_k1_KO_s100 + 1.10721 w_k1_MAT_s100
                                 + 1.11112 w_k1_MSFT_s100
                                 + 1.06749 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.946455 w_k1_ADM_s100
                                 + 0.956688 w_k1_AXP_s100
                                 + 0.989871 w_k1_DIS_s100
                                 + 1.13489 w_k1_FDX_s100
                                 + 0.680598 w_k1_IBM_s100
                                 + 1.45312 w_k1_INTC_s100
                                 + 0.730137 w_k1_KO_s100
                                 + 0.540671 w_k1_MAT_s100
                                 + 1.07099 w_k1_MSFT_s100
                                 + 0.906049 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.941945 w_k1_ADM_s100
                                 + 1.37562 w_k1_AXP_s100
                                 + 1.43793 w_k1_DIS_s100
                                 + 1.32853 w_k1_FDX_s100
                                 + 1.07562 w_k1_IBM_s100
                                 + 1.53307 w_k1_INTC_s100
                                 + 1.27581 w_k1_KO_s100 + 1.19683 w_k1_MAT_s100
                                 + 1.26793 w_k1_MSFT_s100
                                 + 1.11797 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.63019 w_k1_ADM_s100
                                 + 0.866918 w_k1_AXP_s100
                                 + 0.816417 w_k1_DIS_s100
                                 + 0.807336 w_k1_FDX_s100
                                 + 1.17988 w_k1_IBM_s100
                                 + 1.22737 w_k1_INTC_s100
                                 + 0.921049 w_k1_KO_s100
                                 + 1.01652 w_k1_MAT_s100
                                 + 0.814483 w_k1_MSFT_s100
                                 + 0.944472 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.768067 w_k1_ADM_s101
                                 + 0.856449 w_k1_AXP_s101
                                 + 0.858687 w_k1_DIS_s101
                                 + 0.761933 w_k1_FDX_s101
                                 + 1.24036 w_k1_IBM_s101
                                 + 0.829583 w_k1_INTC_s101
                                 + 0.968532 w_k1_KO_s101
                                 + 1.34313 w_k1_MAT_s101
                                 + 0.8101 w_k1_MSFT_s101
                                 + 0.944713 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.966457 w_k1_ADM_s101
                                 + 1.30385 w_k1_AXP_s101
                                 + 1.48761 w_k1_DIS_s101
                                 + 1.16787 w_k1_FDX_s101
                                 + 1.31864 w_k1_IBM_s101
                                 + 1.29563 w_k1_INTC_s101
                                 + 1.24616 w_k1_KO_s101 + 1.30076 w_k1_MAT_s101
                                 + 1.27661 w_k1_MSFT_s101
                                 + 1.03662 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.930183 w_k1_ADM_s101
                                 + 1.02725 w_k1_AXP_s101
                                 + 1.15311 w_k1_DIS_s101
                                 + 1.00299 w_k1_FDX_s101 + 1.0639 w_k1_IBM_s101
                                 + 0.904121 w_k1_INTC_s101
                                 + 0.906873 w_k1_KO_s101
                                 + 0.969921 w_k1_MAT_s101
                                 + 1.08422 w_k1_MSFT_s101
                                 + 0.957362 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.877956 w_k1_ADM_s101
                                 + 0.902949 w_k1_AXP_s101
                                 + 0.946022 w_k1_DIS_s101
                                 + 1.02174 w_k1_FDX_s101
                                 + 0.622067 w_k1_IBM_s101
                                 + 0.9326 w_k1_INTC_s101
                                 + 0.794568 w_k1_KO_s101
                                 + 0.719362 w_k1_MAT_s101
                                 + 0.96504 w_k1_MSFT_s101
                                 + 0.91392 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.906562 w_k1_ADM_s101
                                 + 1.36343 w_k1_AXP_s101
                                 + 1.15695 w_k1_DIS_s101
                                 + 1.06886 w_k1_FDX_s101
                                 + 0.945542 w_k1_IBM_s101
                                 + 0.817284 w_k1_INTC_s101
                                 + 0.961795 w_k1_KO_s101
                                 + 1.57636 w_k1_MAT_s101
                                 + 0.911099 w_k1_MSFT_s101
                                 + 1.05773 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.0775 w_k1_ADM_s101
                                 + 1.6042 w_k1_AXP_s101 + 1.48786 w_k1_DIS_s101
                                 + 1.15806 w_k1_FDX_s101
                                 + 1.72613 w_k1_IBM_s101
                                 + 1.03965 w_k1_INTC_s101
                                 + 1.43793 w_k1_KO_s101 + 1.86209 w_k1_MAT_s101
                                 + 1.13653 w_k1_MSFT_s101
                                 + 1.19101 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04062 w_k1_ADM_s101
                                 + 1.36201 w_k1_AXP_s101
                                 + 1.62986 w_k1_DIS_s101
                                 + 1.29987 w_k1_FDX_s101
                                 + 0.959334 w_k1_IBM_s101
                                 + 1.39133 w_k1_INTC_s101
                                 + 1.33932 w_k1_KO_s101 + 0.91504 w_k1_MAT_s101
                                 + 1.34883 w_k1_MSFT_s101
                                 + 1.14349 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.766113 w_k1_ADM_s101
                                 + 0.758692 w_k1_AXP_s101
                                 + 0.784115 w_k1_DIS_s101
                                 + 0.778294 w_k1_FDX_s101
                                 + 0.796191 w_k1_IBM_s101
                                 + 0.733831 w_k1_INTC_s101
                                 + 0.867103 w_k1_KO_s101
                                 + 0.844673 w_k1_MAT_s101
                                 + 0.862673 w_k1_MSFT_s101
                                 + 0.810028 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.26863 w_k1_ADM_s102
                                 + 1.34275 w_k1_AXP_s102
                                 + 1.64478 w_k1_DIS_s102
                                 + 1.52739 w_k1_FDX_s102
                                 + 0.861336 w_k1_IBM_s102
                                 + 1.27376 w_k1_INTC_s102
                                 + 1.26913 w_k1_KO_s102 + 1.45102 w_k1_MAT_s102
                                 + 1.38428 w_k1_MSFT_s102 + 1.1033 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.971218 w_k1_ADM_s102
                                 + 1.07323 w_k1_AXP_s102
                                 + 1.11289 w_k1_DIS_s102 + 1.0321 w_k1_FDX_s102
                                 + 0.991125 w_k1_IBM_s102
                                 + 1.00931 w_k1_INTC_s102
                                 + 1.00289 w_k1_KO_s102 + 1.12334 w_k1_MAT_s102
                                 + 1.11136 w_k1_MSFT_s102
                                 + 1.04549 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.29418 w_k1_ADM_s102
                                 + 1.24342 w_k1_AXP_s102
                                 + 1.48816 w_k1_DIS_s102
                                 + 1.14958 w_k1_FDX_s102
                                 + 1.20806 w_k1_IBM_s102
                                 + 1.30676 w_k1_INTC_s102
                                 + 1.21416 w_k1_KO_s102 + 1.04307 w_k1_MAT_s102
                                 + 1.23515 w_k1_MSFT_s102
                                 + 1.19838 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.865873 w_k1_ADM_s102
                                 + 0.724086 w_k1_AXP_s102
                                 + 0.583846 w_k1_DIS_s102
                                 + 0.712047 w_k1_FDX_s102
                                 + 0.934914 w_k1_IBM_s102
                                 + 0.670608 w_k1_INTC_s102
                                 + 0.738149 w_k1_KO_s102
                                 + 0.735445 w_k1_MAT_s102
                                 + 0.715577 w_k1_MSFT_s102
                                 + 0.83166 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.940587 w_k1_ADM_s102
                                 + 0.945796 w_k1_AXP_s102
                                 + 0.898916 w_k1_DIS_s102
                                 + 0.989705 w_k1_FDX_s102
                                 + 0.543715 w_k1_IBM_s102
                                 + 1.07067 w_k1_INTC_s102
                                 + 0.751855 w_k1_KO_s102 + 0.4032 w_k1_MAT_s102
                                 + 0.95898 w_k1_MSFT_s102
                                 + 0.971838 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.925116 w_k1_ADM_s102
                                 + 1.09727 w_k1_AXP_s102
                                 + 0.954731 w_k1_DIS_s102
                                 + 0.856203 w_k1_FDX_s102
                                 + 1.66833 w_k1_IBM_s102
                                 + 1.12183 w_k1_INTC_s102
                                 + 1.19928 w_k1_KO_s102 + 1.80138 w_k1_MAT_s102
                                 + 1.05071 w_k1_MSFT_s102
                                 + 1.10717 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.02499 w_k1_ADM_s102
                                 + 0.801187 w_k1_AXP_s102
                                 + 0.966708 w_k1_DIS_s102
                                 + 0.943262 w_k1_FDX_s102
                                 + 0.840734 w_k1_IBM_s102
                                 + 0.954446 w_k1_INTC_s102
                                 + 0.955509 w_k1_KO_s102
                                 + 0.619682 w_k1_MAT_s102
                                 + 0.957611 w_k1_MSFT_s102
                                 + 0.966737 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.984146 w_k1_ADM_s102
                                 + 0.814687 w_k1_AXP_s102
                                 + 1.19277 w_k1_DIS_s102
                                 + 1.11392 w_k1_FDX_s102
                                 + 0.837384 w_k1_IBM_s102
                                 + 1.17973 w_k1_INTC_s102
                                 + 1.17383 w_k1_KO_s102
                                 + 0.666456 w_k1_MAT_s102
                                 + 1.05187 w_k1_MSFT_s102
                                 + 0.985105 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.27381 w_k1_ADM_s103
                                 + 1.53722 w_k1_AXP_s103
                                 + 1.10763 w_k1_DIS_s103
                                 + 1.26669 w_k1_FDX_s103
                                 + 1.04238 w_k1_IBM_s103
                                 + 0.960321 w_k1_INTC_s103
                                 + 1.01074 w_k1_KO_s103 + 1.53079 w_k1_MAT_s103
                                 + 1.00035 w_k1_MSFT_s103
                                 + 1.01628 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.855396 w_k1_ADM_s103
                                 + 0.719913 w_k1_AXP_s103
                                 + 1.15766 w_k1_DIS_s103
                                 + 1.16212 w_k1_FDX_s103
                                 + 0.799827 w_k1_IBM_s103
                                 + 1.16185 w_k1_INTC_s103
                                 + 1.02651 w_k1_KO_s103
                                 + 0.587745 w_k1_MAT_s103
                                 + 1.23845 w_k1_MSFT_s103
                                 + 0.90586 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.680953 w_k1_ADM_s103
                                 + 0.891839 w_k1_AXP_s103
                                 + 0.74511 w_k1_DIS_s103
                                 + 0.762727 w_k1_FDX_s103
                                 + 1.12823 w_k1_IBM_s103
                                 + 1.03167 w_k1_INTC_s103
                                 + 0.97327 w_k1_KO_s103
                                 + 0.784088 w_k1_MAT_s103
                                 + 0.827981 w_k1_MSFT_s103
                                 + 0.872139 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.15833 w_k1_ADM_s103
                                 + 1.19469 w_k1_AXP_s103
                                 + 1.39742 w_k1_DIS_s103
                                 + 1.21185 w_k1_FDX_s103 + 1.374 w_k1_IBM_s103
                                 + 1.0048 w_k1_INTC_s103 + 1.39936 w_k1_KO_s103
                                 + 1.31394 w_k1_MAT_s103
                                 + 1.28736 w_k1_MSFT_s103
                                 + 1.04854 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.847606 w_k1_ADM_s103
                                 + 1.0397 w_k1_AXP_s103 + 1.09218 w_k1_DIS_s103
                                 + 1.018 w_k1_FDX_s103 + 1.06166 w_k1_IBM_s103
                                 + 1.01761 w_k1_INTC_s103
                                 + 1.27237 w_k1_KO_s103 + 1.05853 w_k1_MAT_s103
                                 + 0.926876 w_k1_MSFT_s103
                                 + 0.950477 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.724228 w_k1_ADM_s103
                                 + 0.730644 w_k1_AXP_s103
                                 + 0.595822 w_k1_DIS_s103
                                 + 0.777266 w_k1_FDX_s103
                                 + 0.781957 w_k1_IBM_s103
                                 + 0.824155 w_k1_INTC_s103
                                 + 0.631504 w_k1_KO_s103
                                 + 0.694056 w_k1_MAT_s103
                                 + 0.721556 w_k1_MSFT_s103
                                 + 0.729483 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.770713 w_k1_ADM_s103
                                 + 1.09991 w_k1_AXP_s103
                                 + 0.788639 w_k1_DIS_s103
                                 + 0.963388 w_k1_FDX_s103
                                 + 0.660967 w_k1_IBM_s103
                                 + 0.862943 w_k1_INTC_s103
                                 + 0.742581 w_k1_KO_s103 + 1.1247 w_k1_MAT_s103
                                 + 0.855182 w_k1_MSFT_s103
                                 + 0.905149 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.953014 w_k1_ADM_s103
                                 + 0.913714 w_k1_AXP_s103
                                 + 0.840137 w_k1_DIS_s103
                                 + 0.951971 w_k1_FDX_s103
                                 + 0.860831 w_k1_IBM_s103
                                 + 1.12837 w_k1_INTC_s103
                                 + 0.919674 w_k1_KO_s103
                                 + 0.777482 w_k1_MAT_s103
                                 + 1.04069 w_k1_MSFT_s103
                                 + 0.993598 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.742897 w_k1_ADM_s104
                                 + 0.716082 w_k1_AXP_s104
                                 + 0.592524 w_k1_DIS_s104
                                 + 0.728904 w_k1_FDX_s104
                                 + 0.825827 w_k1_IBM_s104
                                 + 0.886515 w_k1_INTC_s104
                                 + 0.73246 w_k1_KO_s104
                                 + 0.750264 w_k1_MAT_s104
                                 + 0.843576 w_k1_MSFT_s104
                                 + 0.831993 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.890575 w_k1_ADM_s104
                                 + 1.12941 w_k1_AXP_s104
                                 + 0.891046 w_k1_DIS_s104
                                 + 0.963666 w_k1_FDX_s104
                                 + 1.07989 w_k1_IBM_s104
                                 + 1.03838 w_k1_INTC_s104
                                 + 0.870713 w_k1_KO_s104
                                 + 1.06217 w_k1_MAT_s104
                                 + 0.901181 w_k1_MSFT_s104
                                 + 1.01047 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.08728 w_k1_ADM_s104
                                 + 1.40859 w_k1_AXP_s104
                                 + 1.28238 w_k1_DIS_s104
                                 + 1.10069 w_k1_FDX_s104
                                 + 1.67226 w_k1_IBM_s104
                                 + 1.31666 w_k1_INTC_s104
                                 + 1.26229 w_k1_KO_s104 + 1.68463 w_k1_MAT_s104
                                 + 1.15918 w_k1_MSFT_s104
                                 + 1.28765 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.03081 w_k1_ADM_s104
                                 + 0.975208 w_k1_AXP_s104
                                 + 1.1635 w_k1_DIS_s104
                                 + 0.984183 w_k1_FDX_s104
                                 + 1.31941 w_k1_IBM_s104
                                 + 0.95337 w_k1_INTC_s104
                                 + 1.19976 w_k1_KO_s104 + 1.46762 w_k1_MAT_s104
                                 + 1.11378 w_k1_MSFT_s104
                                 + 1.24045 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.913758 w_k1_ADM_s104
                                 + 0.800239 w_k1_AXP_s104
                                 + 0.854353 w_k1_DIS_s104
                                 + 0.813314 w_k1_FDX_s104
                                 + 1.30741 w_k1_IBM_s104
                                 + 0.887932 w_k1_INTC_s104
                                 + 1.04826 w_k1_KO_s104 + 1.1379 w_k1_MAT_s104
                                 + 0.921011 w_k1_MSFT_s104
                                 + 1.06616 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.04914 w_k1_ADM_s104
                                 + 0.970291 w_k1_AXP_s104
                                 + 0.939043 w_k1_DIS_s104
                                 + 0.934312 w_k1_FDX_s104
                                 + 0.904455 w_k1_IBM_s104
                                 + 0.981672 w_k1_INTC_s104
                                 + 0.724354 w_k1_KO_s104
                                 + 0.824036 w_k1_MAT_s104
                                 + 0.974139 w_k1_MSFT_s104
                                 + 0.989725 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.4339 w_k1_ADM_s104
                                 + 1.44502 w_k1_AXP_s104
                                 + 1.63724 w_k1_DIS_s104
                                 + 1.44716 w_k1_FDX_s104
                                 + 0.953977 w_k1_IBM_s104
                                 + 1.27425 w_k1_INTC_s104
                                 + 1.11085 w_k1_KO_s104
                                 + 0.868272 w_k1_MAT_s104
                                 + 1.23879 w_k1_MSFT_s104
                                 + 1.25101 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.17416 w_k1_ADM_s104
                                 + 1.23019 w_k1_AXP_s104
                                 + 1.29298 w_k1_DIS_s104
                                 + 1.15398 w_k1_FDX_s104
                                 + 1.08071 w_k1_IBM_s104
                                 + 0.88725 w_k1_INTC_s104 + 1.0904 w_k1_KO_s104
                                 + 1.42224 w_k1_MAT_s104
                                 + 1.11105 w_k1_MSFT_s104
                                 + 1.22029 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.24331 w_k1_ADM_s105
                                 + 1.19927 w_k1_AXP_s105
                                 + 1.09085 w_k1_DIS_s105
                                 + 1.03273 w_k1_FDX_s105 + 1.4871 w_k1_IBM_s105
                                 + 0.811784 w_k1_INTC_s105
                                 + 1.36646 w_k1_KO_s105 + 1.61313 w_k1_MAT_s105
                                 + 1.08611 w_k1_MSFT_s105
                                 + 1.14668 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.70536 w_k1_ADM_s105
                                 + 1.39693 w_k1_AXP_s105
                                 + 1.67401 w_k1_DIS_s105
                                 + 1.40225 w_k1_FDX_s105
                                 + 1.28049 w_k1_IBM_s105
                                 + 0.957898 w_k1_INTC_s105
                                 + 1.49806 w_k1_KO_s105 + 1.46231 w_k1_MAT_s105
                                 + 1.41456 w_k1_MSFT_s105
                                 + 1.23869 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.01302 w_k1_ADM_s105
                                 + 0.85338 w_k1_AXP_s105
                                 + 0.711004 w_k1_DIS_s105
                                 + 0.867008 w_k1_FDX_s105
                                 + 0.675539 w_k1_IBM_s105
                                 + 0.684456 w_k1_INTC_s105
                                 + 0.82187 w_k1_KO_s105
                                 + 0.828007 w_k1_MAT_s105
                                 + 0.850109 w_k1_MSFT_s105
                                 + 0.892113 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.17233 w_k1_ADM_s105
                                 + 1.12007 w_k1_AXP_s105
                                 + 1.01134 w_k1_DIS_s105
                                 + 1.05548 w_k1_FDX_s105
                                 + 0.910412 w_k1_IBM_s105
                                 + 0.965747 w_k1_INTC_s105
                                 + 0.864856 w_k1_KO_s105
                                 + 0.82814 w_k1_MAT_s105
                                 + 1.03121 w_k1_MSFT_s105
                                 + 1.00646 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.02416 w_k1_ADM_s105
                                 + 0.943788 w_k1_AXP_s105
                                 + 0.831882 w_k1_DIS_s105
                                 + 0.858422 w_k1_FDX_s105
                                 + 0.943952 w_k1_IBM_s105
                                 + 0.741098 w_k1_INTC_s105
                                 + 1.00037 w_k1_KO_s105 + 1.08157 w_k1_MAT_s105
                                 + 0.804106 w_k1_MSFT_s105
                                 + 0.928321 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.36716 w_k1_ADM_s105
                                 + 1.58322 w_k1_AXP_s105
                                 + 1.31092 w_k1_DIS_s105
                                 + 1.42434 w_k1_FDX_s105
                                 + 0.744186 w_k1_IBM_s105
                                 + 1.44505 w_k1_INTC_s105
                                 + 1.01819 w_k1_KO_s105 + 1.10328 w_k1_MAT_s105
                                 + 1.43628 w_k1_MSFT_s105
                                 + 1.11145 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.999868 w_k1_ADM_s105
                                 + 0.691123 w_k1_AXP_s105
                                 + 1.01446 w_k1_DIS_s105
                                 + 0.82529 w_k1_FDX_s105
                                 + 1.15617 w_k1_IBM_s105
                                 + 0.92085 w_k1_INTC_s105
                                 + 1.17224 w_k1_KO_s105 + 1.12829 w_k1_MAT_s105
                                 + 1.12919 w_k1_MSFT_s105
                                 + 0.942887 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.879874 w_k1_ADM_s105
                                 + 0.712783 w_k1_AXP_s105
                                 + 0.545733 w_k1_DIS_s105
                                 + 0.652992 w_k1_FDX_s105
                                 + 0.857222 w_k1_IBM_s105
                                 + 0.658102 w_k1_INTC_s105
                                 + 0.681874 w_k1_KO_s105
                                 + 0.93039 w_k1_MAT_s105
                                 + 0.764539 w_k1_MSFT_s105
                                 + 0.836147 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.01051 w_k1_ADM_s106
                                 + 1.01162 w_k1_AXP_s106
                                 + 1.35415 w_k1_DIS_s106
                                 + 1.00262 w_k1_FDX_s106
                                 + 1.07821 w_k1_IBM_s106
                                 + 1.14506 w_k1_INTC_s106 + 1.3192 w_k1_KO_s106
                                 + 0.876242 w_k1_MAT_s106
                                 + 1.23973 w_k1_MSFT_s106
                                 + 1.07339 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.02901 w_k1_ADM_s106
                                 + 1.15809 w_k1_AXP_s106
                                 + 1.00564 w_k1_DIS_s106
                                 + 1.00137 w_k1_FDX_s106
                                 + 0.854558 w_k1_IBM_s106
                                 + 0.950592 w_k1_INTC_s106
                                 + 0.948028 w_k1_KO_s106
                                 + 0.76703 w_k1_MAT_s106
                                 + 1.01779 w_k1_MSFT_s106
                                 + 0.999145 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.23119 w_k1_ADM_s106
                                 + 1.37494 w_k1_AXP_s106
                                 + 1.61118 w_k1_DIS_s106
                                 + 1.19772 w_k1_FDX_s106
                                 + 1.45608 w_k1_IBM_s106
                                 + 1.39946 w_k1_INTC_s106 + 1.5873 w_k1_KO_s106
                                 + 1.21991 w_k1_MAT_s106
                                 + 1.24858 w_k1_MSFT_s106
                                 + 1.27133 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.03747 w_k1_ADM_s106
                                 + 1.05965 w_k1_AXP_s106
                                 + 1.38468 w_k1_DIS_s106
                                 + 1.21387 w_k1_FDX_s106
                                 + 0.855438 w_k1_IBM_s106
                                 + 1.22736 w_k1_INTC_s106
                                 + 1.18556 w_k1_KO_s106
                                 + 0.764355 w_k1_MAT_s106
                                 + 1.27394 w_k1_MSFT_s106
                                 + 1.20591 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.842358 w_k1_ADM_s106
                                 + 0.702448 w_k1_AXP_s106
                                 + 0.752176 w_k1_DIS_s106
                                 + 0.852187 w_k1_FDX_s106
                                 + 0.577867 w_k1_IBM_s106
                                 + 0.913838 w_k1_INTC_s106
                                 + 0.785402 w_k1_KO_s106
                                 + 0.42421 w_k1_MAT_s106
                                 + 0.909497 w_k1_MSFT_s106
                                 + 0.885128 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.721349 w_k1_ADM_s106
                                 + 0.866892 w_k1_AXP_s106
                                 + 0.942522 w_k1_DIS_s106
                                 + 0.788781 w_k1_FDX_s106
                                 + 1.11377 w_k1_IBM_s106
                                 + 0.92236 w_k1_INTC_s106
                                 + 1.03377 w_k1_KO_s106 + 1.0329 w_k1_MAT_s106
                                 + 1.03957 w_k1_MSFT_s106
                                 + 0.983253 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.949063 w_k1_ADM_s106
                                 + 1.15781 w_k1_AXP_s106
                                 + 1.12644 w_k1_DIS_s106
                                 + 0.956436 w_k1_FDX_s106
                                 + 1.31924 w_k1_IBM_s106
                                 + 1.13502 w_k1_INTC_s106
                                 + 1.14709 w_k1_KO_s106 + 0.94704 w_k1_MAT_s106
                                 + 1.01602 w_k1_MSFT_s106
                                 + 1.06591 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.07031 w_k1_ADM_s106
                                 + 1.29313 w_k1_AXP_s106
                                 + 1.03682 w_k1_DIS_s106
                                 + 0.988925 w_k1_FDX_s106
                                 + 1.16998 w_k1_IBM_s106
                                 + 1.0962 w_k1_INTC_s106
                                 + 0.922223 w_k1_KO_s106
                                 + 0.922682 w_k1_MAT_s106
                                 + 0.998812 w_k1_MSFT_s106
                                 + 1.07894 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.958331 w_k1_ADM_s107
                                 + 1.10333 w_k1_AXP_s107 + 1.1322 w_k1_DIS_s107
                                 + 1.05795 w_k1_FDX_s107 + 1.1419 w_k1_IBM_s107
                                 + 1.10356 w_k1_INTC_s107
                                 + 0.975354 w_k1_KO_s107
                                 + 1.04001 w_k1_MAT_s107
                                 + 1.00723 w_k1_MSFT_s107
                                 + 1.10654 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.754501 w_k1_ADM_s107
                                 + 0.711517 w_k1_AXP_s107
                                 + 0.63134 w_k1_DIS_s107 + 0.811 w_k1_FDX_s107
                                 + 0.7882 w_k1_IBM_s107
                                 + 0.840373 w_k1_INTC_s107
                                 + 0.683484 w_k1_KO_s107
                                 + 0.835502 w_k1_MAT_s107
                                 + 0.728426 w_k1_MSFT_s107
                                 + 0.92361 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.08065 w_k1_ADM_s107
                                 + 1.27833 w_k1_AXP_s107
                                 + 0.913927 w_k1_DIS_s107
                                 + 1.00841 w_k1_FDX_s107
                                 + 1.24266 w_k1_IBM_s107
                                 + 0.946825 w_k1_INTC_s107
                                 + 0.84356 w_k1_KO_s107 + 1.28378 w_k1_MAT_s107
                                 + 0.818723 w_k1_MSFT_s107
                                 + 1.0823 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.636651 w_k1_ADM_s107
                                 + 0.853436 w_k1_AXP_s107
                                 + 1.0791 w_k1_DIS_s107
                                 + 0.940542 w_k1_FDX_s107
                                 + 1.28769 w_k1_IBM_s107
                                 + 1.21074 w_k1_INTC_s107
                                 + 1.10667 w_k1_KO_s107 + 1.03048 w_k1_MAT_s107
                                 + 1.04588 w_k1_MSFT_s107
                                 + 1.11545 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.00479 w_k1_ADM_s107
                                 + 1.13692 w_k1_AXP_s107
                                 + 0.978083 w_k1_DIS_s107
                                 + 1.16449 w_k1_FDX_s107
                                 + 0.774931 w_k1_IBM_s107
                                 + 1.26125 w_k1_INTC_s107
                                 + 0.965183 w_k1_KO_s107
                                 + 0.995022 w_k1_MAT_s107
                                 + 1.06015 w_k1_MSFT_s107
                                 + 1.02331 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.47734 w_k1_ADM_s107
                                 + 1.68251 w_k1_AXP_s107
                                 + 1.58792 w_k1_DIS_s107
                                 + 1.42733 w_k1_FDX_s107
                                 + 1.26476 w_k1_IBM_s107
                                 + 1.5649 w_k1_INTC_s107 + 1.33108 w_k1_KO_s107
                                 + 1.19598 w_k1_MAT_s107
                                 + 1.3501 w_k1_MSFT_s107 + 1.30232 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.13627 w_k1_ADM_s107
                                 + 1.31724 w_k1_AXP_s107
                                 + 1.30248 w_k1_DIS_s107 + 1.1861 w_k1_FDX_s107
                                 + 1.79026 w_k1_IBM_s107
                                 + 1.40503 w_k1_INTC_s107
                                 + 1.30006 w_k1_KO_s107 + 1.51006 w_k1_MAT_s107
                                 + 1.14752 w_k1_MSFT_s107
                                 + 1.24891 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.881355 w_k1_ADM_s107
                                 + 1.16725 w_k1_AXP_s107
                                 + 0.838891 w_k1_DIS_s107
                                 + 0.949207 w_k1_FDX_s107
                                 + 0.786503 w_k1_IBM_s107
                                 + 0.939478 w_k1_INTC_s107
                                 + 0.826046 w_k1_KO_s107 + 1.0606 w_k1_MAT_s107
                                 + 0.866038 w_k1_MSFT_s107
                                 + 0.905595 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.9656 w_k1_ADM_s110
                                 + 1.05925 w_k1_AXP_s110
                                 + 1.11089 w_k1_DIS_s110
                                 + 1.03497 w_k1_FDX_s110
                                 + 1.32621 w_k1_IBM_s110
                                 + 0.820147 w_k1_INTC_s110
                                 + 1.21328 w_k1_KO_s110 + 1.60219 w_k1_MAT_s110
                                 + 1.05225 w_k1_MSFT_s110
                                 + 1.08477 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.01827 w_k1_ADM_s110
                                 + 1.01553 w_k1_AXP_s110
                                 + 0.93095 w_k1_DIS_s110
                                 + 0.975477 w_k1_FDX_s110
                                 + 0.983441 w_k1_IBM_s110
                                 + 0.772372 w_k1_INTC_s110
                                 + 0.990906 w_k1_KO_s110
                                 + 1.22126 w_k1_MAT_s110
                                 + 0.893949 w_k1_MSFT_s110
                                 + 1.0202 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.11653 w_k1_ADM_s110
                                 + 0.926647 w_k1_AXP_s110
                                 + 0.905745 w_k1_DIS_s110
                                 + 1.01256 w_k1_FDX_s110
                                 + 0.79448 w_k1_IBM_s110
                                 + 0.967564 w_k1_INTC_s110
                                 + 0.932149 w_k1_KO_s110
                                 + 0.917667 w_k1_MAT_s110
                                 + 1.04581 w_k1_MSFT_s110
                                 + 0.98987 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.22115 w_k1_ADM_s110
                                 + 1.30767 w_k1_AXP_s110
                                 + 0.955317 w_k1_DIS_s110
                                 + 0.96943 w_k1_FDX_s110
                                 + 1.37374 w_k1_IBM_s110
                                 + 0.933797 w_k1_INTC_s110
                                 + 1.07024 w_k1_KO_s110 + 1.7645 w_k1_MAT_s110
                                 + 0.860983 w_k1_MSFT_s110
                                 + 1.07718 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.20277 w_k1_ADM_s110
                                 + 1.11114 w_k1_AXP_s110
                                 + 1.33255 w_k1_DIS_s110 + 1.2769 w_k1_FDX_s110
                                 + 0.798983 w_k1_IBM_s110
                                 + 1.00139 w_k1_INTC_s110
                                 + 1.05959 w_k1_KO_s110 + 1.08127 w_k1_MAT_s110
                                 + 1.23635 w_k1_MSFT_s110 + 1.0997 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.783784 w_k1_ADM_s110
                                 + 0.723059 w_k1_AXP_s110
                                 + 0.618609 w_k1_DIS_s110
                                 + 0.767384 w_k1_FDX_s110
                                 + 0.731213 w_k1_IBM_s110
                                 + 0.76634 w_k1_INTC_s110
                                 + 0.720003 w_k1_KO_s110
                                 + 0.627172 w_k1_MAT_s110
                                 + 0.723128 w_k1_MSFT_s110
                                 + 0.84394 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.42757 w_k1_ADM_s110
                                 + 1.36633 w_k1_AXP_s110
                                 + 1.62607 w_k1_DIS_s110
                                 + 1.28219 w_k1_FDX_s110
                                 + 1.46463 w_k1_IBM_s110
                                 + 1.24677 w_k1_INTC_s110 + 1.412 w_k1_KO_s110
                                 + 1.59369 w_k1_MAT_s110
                                 + 1.31477 w_k1_MSFT_s110 + 1.2688 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.756464 w_k1_ADM_s110
                                 + 0.741409 w_k1_AXP_s110
                                 + 0.710655 w_k1_DIS_s110
                                 + 0.755592 w_k1_FDX_s110
                                 + 1.09066 w_k1_IBM_s110
                                 + 0.75211 w_k1_INTC_s110
                                 + 0.869279 w_k1_KO_s110
                                 + 1.22704 w_k1_MAT_s110
                                 + 0.775199 w_k1_MSFT_s110
                                 + 0.816867 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.852565 w_k1_ADM_s111
                                 + 1.00386 w_k1_AXP_s111
                                 + 1.12854 w_k1_DIS_s111
                                 + 1.08504 w_k1_FDX_s111
                                 + 0.725483 w_k1_IBM_s111
                                 + 1.05989 w_k1_INTC_s111
                                 + 0.85316 w_k1_KO_s111
                                 + 0.749306 w_k1_MAT_s111
                                 + 1.25195 w_k1_MSFT_s111
                                 + 1.09078 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.799179 w_k1_ADM_s111
                                 + 1.08856 w_k1_AXP_s111
                                 + 1.02629 w_k1_DIS_s111
                                 + 1.01316 w_k1_FDX_s111
                                 + 1.23523 w_k1_IBM_s111
                                 + 0.775675 w_k1_INTC_s111
                                 + 1.02153 w_k1_KO_s111 + 1.53799 w_k1_MAT_s111
                                 + 0.854383 w_k1_MSFT_s111
                                 + 1.08108 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.13214 w_k1_ADM_s111
                                 + 1.40075 w_k1_AXP_s111 + 1.687 w_k1_DIS_s111
                                 + 1.38709 w_k1_FDX_s111 + 1.1359 w_k1_IBM_s111
                                 + 1.19054 w_k1_INTC_s111
                                 + 1.23192 w_k1_KO_s111 + 1.44447 w_k1_MAT_s111
                                 + 1.43732 w_k1_MSFT_s111
                                 + 1.19847 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.951108 w_k1_ADM_s111
                                 + 1.12477 w_k1_AXP_s111
                                 + 0.789889 w_k1_DIS_s111
                                 + 0.899883 w_k1_FDX_s111
                                 + 0.885714 w_k1_IBM_s111
                                 + 0.852192 w_k1_INTC_s111
                                 + 0.758732 w_k1_KO_s111
                                 + 1.21977 w_k1_MAT_s111
                                 + 0.795742 w_k1_MSFT_s111
                                 + 0.888278 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.936792 w_k1_ADM_s111
                                 + 1.02219 w_k1_AXP_s111
                                 + 1.56297 w_k1_DIS_s111
                                 + 1.08008 w_k1_FDX_s111
                                 + 1.64269 w_k1_IBM_s111
                                 + 1.21004 w_k1_INTC_s111
                                 + 1.43625 w_k1_KO_s111 + 1.55516 w_k1_MAT_s111
                                 + 1.20232 w_k1_MSFT_s111
                                 + 1.20726 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.0368 w_k1_ADM_s111
                                 + 1.10115 w_k1_AXP_s111
                                 + 1.24063 w_k1_DIS_s111
                                 + 1.05139 w_k1_FDX_s111
                                 + 0.963918 w_k1_IBM_s111
                                 + 0.988088 w_k1_INTC_s111
                                 + 0.954282 w_k1_KO_s111
                                 + 0.971189 w_k1_MAT_s111
                                 + 1.1074 w_k1_MSFT_s111 + 1.15043 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.788597 w_k1_ADM_s111
                                 + 0.775907 w_k1_AXP_s111
                                 + 0.672668 w_k1_DIS_s111
                                 + 0.775563 w_k1_FDX_s111
                                 + 0.723089 w_k1_IBM_s111
                                 + 0.711006 w_k1_INTC_s111
                                 + 0.651996 w_k1_KO_s111
                                 + 0.662846 w_k1_MAT_s111
                                 + 0.746167 w_k1_MSFT_s111
                                 + 0.861362 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.943521 w_k1_ADM_s111
                                 + 1.17553 w_k1_AXP_s111
                                 + 0.98519 w_k1_DIS_s111
                                 + 0.944561 w_k1_FDX_s111
                                 + 1.61922 w_k1_IBM_s111
                                 + 0.956122 w_k1_INTC_s111
                                 + 1.11869 w_k1_KO_s111 + 1.72127 w_k1_MAT_s111
                                 + 1.08542 w_k1_MSFT_s111
                                 + 0.999697 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.06556 w_k1_ADM_s112
                                 + 1.2706 w_k1_AXP_s112 + 1.14586 w_k1_DIS_s112
                                 + 1.13085 w_k1_FDX_s112
                                 + 0.962504 w_k1_IBM_s112
                                 + 1.03261 w_k1_INTC_s112
                                 + 0.928506 w_k1_KO_s112
                                 + 1.12259 w_k1_MAT_s112
                                 + 1.17884 w_k1_MSFT_s112 + 1.2348 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.977262 w_k1_ADM_s112
                                 + 0.886602 w_k1_AXP_s112
                                 + 0.80981 w_k1_DIS_s112
                                 + 0.914172 w_k1_FDX_s112
                                 + 0.741769 w_k1_IBM_s112
                                 + 0.793347 w_k1_INTC_s112
                                 + 0.804316 w_k1_KO_s112
                                 + 0.840699 w_k1_MAT_s112
                                 + 0.958927 w_k1_MSFT_s112
                                 + 1.07288 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.825345 w_k1_ADM_s112
                                 + 1.00303 w_k1_AXP_s112
                                 + 0.858525 w_k1_DIS_s112
                                 + 0.865593 w_k1_FDX_s112
                                 + 1.35375 w_k1_IBM_s112
                                 + 0.8961 w_k1_INTC_s112
                                 + 0.890603 w_k1_KO_s112
                                 + 1.28825 w_k1_MAT_s112
                                 + 0.881176 w_k1_MSFT_s112
                                 + 1.00992 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.829408 w_k1_ADM_s112
                                 + 1.37778 w_k1_AXP_s112
                                 + 1.20701 w_k1_DIS_s112
                                 + 1.13578 w_k1_FDX_s112 + 1.2812 w_k1_IBM_s112
                                 + 1.20859 w_k1_INTC_s112 + 1.14 w_k1_KO_s112
                                 + 1.29879 w_k1_MAT_s112
                                 + 1.13698 w_k1_MSFT_s112
                                 + 1.18473 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.993835 w_k1_ADM_s112
                                 + 0.834247 w_k1_AXP_s112
                                 + 0.706146 w_k1_DIS_s112
                                 + 0.876099 w_k1_FDX_s112
                                 + 0.557619 w_k1_IBM_s112
                                 + 0.681727 w_k1_INTC_s112
                                 + 0.627815 w_k1_KO_s112
                                 + 0.390261 w_k1_MAT_s112
                                 + 0.812866 w_k1_MSFT_s112
                                 + 0.89892 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.836814 w_k1_ADM_s112
                                 + 1.04042 w_k1_AXP_s112
                                 + 0.940039 w_k1_DIS_s112
                                 + 0.909716 w_k1_FDX_s112
                                 + 0.810382 w_k1_IBM_s112
                                 + 0.862863 w_k1_INTC_s112
                                 + 0.915184 w_k1_KO_s112
                                 + 0.682293 w_k1_MAT_s112
                                 + 0.993159 w_k1_MSFT_s112
                                 + 0.916855 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.882486 w_k1_ADM_s112
                                 + 1.06508 w_k1_AXP_s112
                                 + 0.74501 w_k1_DIS_s112
                                 + 0.840686 w_k1_FDX_s112
                                 + 0.977747 w_k1_IBM_s112
                                 + 0.919521 w_k1_INTC_s112
                                 + 0.70065 w_k1_KO_s112
                                 + 0.822699 w_k1_MAT_s112
                                 + 0.762618 w_k1_MSFT_s112
                                 + 0.913029 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.13739 w_k1_ADM_s112
                                 + 1.51035 w_k1_AXP_s112
                                 + 1.63195 w_k1_DIS_s112
                                 + 1.43438 w_k1_FDX_s112
                                 + 1.05007 w_k1_IBM_s112
                                 + 1.61564 w_k1_INTC_s112
                                 + 1.25255 w_k1_KO_s112 + 0.67318 w_k1_MAT_s112
                                 + 1.3746 w_k1_MSFT_s112 + 1.25781 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.0889 w_k1_ADM_s113
                                 + 1.18582 w_k1_AXP_s113
                                 + 1.42905 w_k1_DIS_s113
                                 + 1.17696 w_k1_FDX_s113
                                 + 1.15448 w_k1_IBM_s113
                                 + 1.08145 w_k1_INTC_s113
                                 + 1.14358 w_k1_KO_s113 + 1.05269 w_k1_MAT_s113
                                 + 1.18431 w_k1_MSFT_s113
                                 + 1.16272 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.10808 w_k1_ADM_s113
                                 + 1.10384 w_k1_AXP_s113
                                 + 1.09882 w_k1_DIS_s113
                                 + 0.953269 w_k1_FDX_s113
                                 + 1.4302 w_k1_IBM_s113
                                 + 0.988406 w_k1_INTC_s113
                                 + 1.04294 w_k1_KO_s113 + 1.43238 w_k1_MAT_s113
                                 + 0.881106 w_k1_MSFT_s113
                                 + 1.00042 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.09761 w_k1_ADM_s113
                                 + 1.1175 w_k1_AXP_s113 + 1.09434 w_k1_DIS_s113
                                 + 1.05499 w_k1_FDX_s113
                                 + 0.975014 w_k1_IBM_s113
                                 + 1.07585 w_k1_INTC_s113
                                 + 0.906948 w_k1_KO_s113
                                 + 0.773406 w_k1_MAT_s113
                                 + 1.04764 w_k1_MSFT_s113
                                 + 1.03675 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.13453 w_k1_ADM_s113
                                 + 1.64345 w_k1_AXP_s113 + 0.9395 w_k1_DIS_s113
                                 + 1.19003 w_k1_FDX_s113
                                 + 1.06714 w_k1_IBM_s113
                                 + 1.25293 w_k1_INTC_s113
                                 + 0.985109 w_k1_KO_s113
                                 + 1.43385 w_k1_MAT_s113
                                 + 1.03219 w_k1_MSFT_s113
                                 + 1.17053 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.888887 w_k1_ADM_s113
                                 + 0.683158 w_k1_AXP_s113
                                 + 0.980938 w_k1_DIS_s113
                                 + 0.839379 w_k1_FDX_s113
                                 + 1.04307 w_k1_IBM_s113
                                 + 1.00447 w_k1_INTC_s113
                                 + 1.07038 w_k1_KO_s113
                                 + 0.600696 w_k1_MAT_s113
                                 + 0.895484 w_k1_MSFT_s113
                                 + 0.937807 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.982571 w_k1_ADM_s113
                                 + 0.815622 w_k1_AXP_s113
                                 + 0.784521 w_k1_DIS_s113
                                 + 0.858035 w_k1_FDX_s113
                                 + 0.727789 w_k1_IBM_s113
                                 + 0.695513 w_k1_INTC_s113
                                 + 0.797331 w_k1_KO_s113
                                 + 0.731357 w_k1_MAT_s113
                                 + 0.889969 w_k1_MSFT_s113
                                 + 0.935085 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.863227 w_k1_ADM_s113
                                 + 0.995313 w_k1_AXP_s113
                                 + 0.77267 w_k1_DIS_s113
                                 + 0.791092 w_k1_FDX_s113
                                 + 1.12362 w_k1_IBM_s113
                                 + 0.935266 w_k1_INTC_s113
                                 + 0.903882 w_k1_KO_s113
                                 + 1.07844 w_k1_MAT_s113
                                 + 0.855139 w_k1_MSFT_s113
                                 + 0.930152 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.21909 w_k1_ADM_s113
                                 + 1.50652 w_k1_AXP_s113
                                 + 1.66665 w_k1_DIS_s113
                                 + 1.44678 w_k1_FDX_s113
                                 + 1.24039 w_k1_IBM_s113
                                 + 1.36516 w_k1_INTC_s113
                                 + 1.30214 w_k1_KO_s113 + 1.46049 w_k1_MAT_s113
                                 + 1.37848 w_k1_MSFT_s113
                                 + 1.30815 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.10173 w_k1_ADM_s114
                                 + 0.835525 w_k1_AXP_s114
                                 + 0.663733 w_k1_DIS_s114
                                 + 0.752025 w_k1_FDX_s114
                                 + 0.828543 w_k1_IBM_s114
                                 + 0.877726 w_k1_INTC_s114
                                 + 0.89923 w_k1_KO_s114 + 0.70552 w_k1_MAT_s114
                                 + 0.915903 w_k1_MSFT_s114
                                 + 0.828116 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.953408 w_k1_ADM_s114
                                 + 0.626214 w_k1_AXP_s114
                                 + 0.674655 w_k1_DIS_s114
                                 + 0.764964 w_k1_FDX_s114
                                 + 0.601382 w_k1_IBM_s114
                                 + 0.920068 w_k1_INTC_s114
                                 + 0.774546 w_k1_KO_s114
                                 + 0.509457 w_k1_MAT_s114
                                 + 0.970726 w_k1_MSFT_s114
                                 + 0.849382 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.51267 w_k1_ADM_s114
                                 + 1.49159 w_k1_AXP_s114
                                 + 1.56936 w_k1_DIS_s114
                                 + 1.36536 w_k1_FDX_s114
                                 + 1.05242 w_k1_IBM_s114
                                 + 1.28788 w_k1_INTC_s114
                                 + 1.27348 w_k1_KO_s114 + 1.26142 w_k1_MAT_s114
                                 + 1.31379 w_k1_MSFT_s114
                                 + 1.26994 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.12053 w_k1_ADM_s114
                                 + 0.889105 w_k1_AXP_s114
                                 + 0.89043 w_k1_DIS_s114
                                 + 0.876517 w_k1_FDX_s114
                                 + 1.12324 w_k1_IBM_s114
                                 + 0.891687 w_k1_INTC_s114
                                 + 1.0049 w_k1_KO_s114 + 0.962948 w_k1_MAT_s114
                                 + 1.01036 w_k1_MSFT_s114 + 1.0594 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.74821 w_k1_ADM_s114
                                 + 1.6312 w_k1_AXP_s114 + 1.74926 w_k1_DIS_s114
                                 + 1.56714 w_k1_FDX_s114
                                 + 1.85948 w_k1_IBM_s114
                                 + 1.41644 w_k1_INTC_s114 + 1.7039 w_k1_KO_s114
                                 + 2.12324 w_k1_MAT_s114
                                 + 1.57054 w_k1_MSFT_s114
                                 + 1.23827 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.17952 w_k1_ADM_s114
                                 + 1.14417 w_k1_AXP_s114
                                 + 1.06501 w_k1_DIS_s114 + 1.141 w_k1_FDX_s114
                                 + 0.880508 w_k1_IBM_s114
                                 + 1.12628 w_k1_INTC_s114
                                 + 0.942784 w_k1_KO_s114
                                 + 0.914347 w_k1_MAT_s114
                                 + 1.22971 w_k1_MSFT_s114
                                 + 0.996917 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.69941 w_k1_ADM_s114
                                 + 1.42398 w_k1_AXP_s114
                                 + 1.40581 w_k1_DIS_s114
                                 + 1.22032 w_k1_FDX_s114
                                 + 1.39703 w_k1_IBM_s114
                                 + 1.09728 w_k1_INTC_s114
                                 + 1.37587 w_k1_KO_s114 + 1.3823 w_k1_MAT_s114
                                 + 1.27971 w_k1_MSFT_s114
                                 + 1.23814 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.10473 w_k1_ADM_s114
                                 + 0.904342 w_k1_AXP_s114
                                 + 0.987317 w_k1_DIS_s114
                                 + 0.780023 w_k1_FDX_s114
                                 + 1.53134 w_k1_IBM_s114
                                 + 0.861236 w_k1_INTC_s114
                                 + 1.2379 w_k1_KO_s114 + 1.36586 w_k1_MAT_s114
                                 + 0.93262 w_k1_MSFT_s114
                                 + 1.01507 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.986466 w_k1_ADM_s115
                                 + 0.947164 w_k1_AXP_s115
                                 + 0.669607 w_k1_DIS_s115
                                 + 0.846108 w_k1_FDX_s115
                                 + 0.832084 w_k1_IBM_s115
                                 + 0.900193 w_k1_INTC_s115
                                 + 0.803201 w_k1_KO_s115
                                 + 0.817674 w_k1_MAT_s115
                                 + 0.696832 w_k1_MSFT_s115
                                 + 0.892027 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.07093 w_k1_ADM_s115
                                 + 0.898226 w_k1_AXP_s115
                                 + 0.877682 w_k1_DIS_s115
                                 + 0.945379 w_k1_FDX_s115
                                 + 1.08599 w_k1_IBM_s115
                                 + 0.916724 w_k1_INTC_s115
                                 + 0.951095 w_k1_KO_s115
                                 + 1.07033 w_k1_MAT_s115
                                 + 0.959555 w_k1_MSFT_s115
                                 + 0.981996 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.54678 w_k1_ADM_s115
                                 + 0.938316 w_k1_AXP_s115
                                 + 1.11708 w_k1_DIS_s115
                                 + 1.16129 w_k1_FDX_s115
                                 + 0.948274 w_k1_IBM_s115
                                 + 1.13614 w_k1_INTC_s115
                                 + 1.02364 w_k1_KO_s115
                                 + 0.740992 w_k1_MAT_s115
                                 + 1.09027 w_k1_MSFT_s115
                                 + 1.19328 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.13539 w_k1_ADM_s115
                                 + 1.19449 w_k1_AXP_s115
                                 + 1.04781 w_k1_DIS_s115
                                 + 0.970099 w_k1_FDX_s115
                                 + 1.69233 w_k1_IBM_s115
                                 + 0.914055 w_k1_INTC_s115
                                 + 1.16767 w_k1_KO_s115 + 1.57436 w_k1_MAT_s115
                                 + 0.85793 w_k1_MSFT_s115
                                 + 1.00247 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.52739 w_k1_ADM_s115
                                 + 1.24779 w_k1_AXP_s115
                                 + 1.38843 w_k1_DIS_s115
                                 + 1.28275 w_k1_FDX_s115
                                 + 1.11258 w_k1_IBM_s115
                                 + 1.18141 w_k1_INTC_s115
                                 + 1.19198 w_k1_KO_s115
                                 + 0.987602 w_k1_MAT_s115
                                 + 1.23039 w_k1_MSFT_s115
                                 + 0.999627 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.961863 w_k1_ADM_s115
                                 + 0.590215 w_k1_AXP_s115
                                 + 0.56722 w_k1_DIS_s115
                                 + 0.743019 w_k1_FDX_s115
                                 + 0.748436 w_k1_IBM_s115
                                 + 0.750612 w_k1_INTC_s115
                                 + 0.721555 w_k1_KO_s115
                                 + 0.573004 w_k1_MAT_s115
                                 + 0.746944 w_k1_MSFT_s115
                                 + 0.772197 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.14895 w_k1_ADM_s115
                                 + 1.33908 w_k1_AXP_s115
                                 + 1.43245 w_k1_DIS_s115
                                 + 1.17258 w_k1_FDX_s115
                                 + 1.36351 w_k1_IBM_s115
                                 + 1.48969 w_k1_INTC_s115
                                 + 1.27107 w_k1_KO_s115 + 1.21522 w_k1_MAT_s115
                                 + 1.14239 w_k1_MSFT_s115
                                 + 1.13369 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.25503 w_k1_ADM_s115
                                 + 0.959428 w_k1_AXP_s115
                                 + 1.1494 w_k1_DIS_s115 + 1.14702 w_k1_FDX_s115
                                 + 0.699376 w_k1_IBM_s115
                                 + 1.21918 w_k1_INTC_s115
                                 + 0.908252 w_k1_KO_s115
                                 + 0.576428 w_k1_MAT_s115
                                 + 1.01919 w_k1_MSFT_s115
                                 + 1.07191 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.55855 w_k1_ADM_s116
                                 + 1.52961 w_k1_AXP_s116
                                 + 1.78246 w_k1_DIS_s116
                                 + 1.46437 w_k1_FDX_s116
                                 + 1.64456 w_k1_IBM_s116
                                 + 1.36961 w_k1_INTC_s116
                                 + 1.64493 w_k1_KO_s116 + 1.8772 w_k1_MAT_s116
                                 + 1.28026 w_k1_MSFT_s116 + 1.2867 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.05799 w_k1_ADM_s116
                                 + 1.12423 w_k1_AXP_s116
                                 + 1.08895 w_k1_DIS_s116
                                 + 1.14405 w_k1_FDX_s116
                                 + 0.928988 w_k1_IBM_s116
                                 + 1.03021 w_k1_INTC_s116
                                 + 0.927408 w_k1_KO_s116
                                 + 0.778206 w_k1_MAT_s116
                                 + 1.04816 w_k1_MSFT_s116
                                 + 0.912215 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.965456 w_k1_ADM_s116
                                 + 1.39649 w_k1_AXP_s116
                                 + 1.61811 w_k1_DIS_s116 + 1.2836 w_k1_FDX_s116
                                 + 1.11755 w_k1_IBM_s116
                                 + 1.05795 w_k1_INTC_s116
                                 + 1.18318 w_k1_KO_s116 + 1.26516 w_k1_MAT_s116
                                 + 1.21803 w_k1_MSFT_s116 + 1.1194 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.853772 w_k1_ADM_s116
                                 + 0.786676 w_k1_AXP_s116
                                 + 0.60416 w_k1_DIS_s116
                                 + 0.768269 w_k1_FDX_s116
                                 + 0.917326 w_k1_IBM_s116
                                 + 0.824239 w_k1_INTC_s116
                                 + 0.832884 w_k1_KO_s116
                                 + 0.956324 w_k1_MAT_s116
                                 + 0.804157 w_k1_MSFT_s116
                                 + 0.735629 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.09736 w_k1_ADM_s116
                                 + 1.23954 w_k1_AXP_s116
                                 + 1.18775 w_k1_DIS_s116
                                 + 1.18695 w_k1_FDX_s116
                                 + 1.47305 w_k1_IBM_s116
                                 + 0.914337 w_k1_INTC_s116
                                 + 1.22703 w_k1_KO_s116 + 1.78236 w_k1_MAT_s116
                                 + 1.12815 w_k1_MSFT_s116
                                 + 1.00929 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.853957 w_k1_ADM_s116
                                 + 0.782962 w_k1_AXP_s116
                                 + 1.02788 w_k1_DIS_s116
                                 + 0.928121 w_k1_FDX_s116
                                 + 0.755523 w_k1_IBM_s116
                                 + 0.970988 w_k1_INTC_s116
                                 + 1.01081 w_k1_KO_s116
                                 + 0.544977 w_k1_MAT_s116
                                 + 1.03172 w_k1_MSFT_s116
                                 + 0.857335 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.03007 w_k1_ADM_s116
                                 + 0.934407 w_k1_AXP_s116
                                 + 1.24618 w_k1_DIS_s116
                                 + 1.07827 w_k1_FDX_s116
                                 + 1.39318 w_k1_IBM_s116
                                 + 0.848781 w_k1_INTC_s116
                                 + 1.24397 w_k1_KO_s116 + 1.14781 w_k1_MAT_s116
                                 + 0.959062 w_k1_MSFT_s116
                                 + 0.93785 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.03253 w_k1_ADM_s116
                                 + 1.00027 w_k1_AXP_s116
                                 + 0.876113 w_k1_DIS_s116
                                 + 1.04127 w_k1_FDX_s116
                                 + 1.00834 w_k1_IBM_s116
                                 + 0.744597 w_k1_INTC_s116
                                 + 0.887177 w_k1_KO_s116
                                 + 1.19672 w_k1_MAT_s116
                                 + 0.906821 w_k1_MSFT_s116
                                 + 0.950636 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.05505 w_k1_ADM_s117
                                 + 1.05707 w_k1_AXP_s117
                                 + 1.05015 w_k1_DIS_s117
                                 + 1.12703 w_k1_FDX_s117
                                 + 0.763043 w_k1_IBM_s117
                                 + 0.879593 w_k1_INTC_s117
                                 + 0.993579 w_k1_KO_s117
                                 + 0.785246 w_k1_MAT_s117
                                 + 0.979609 w_k1_MSFT_s117
                                 + 0.881358 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.647333 w_k1_ADM_s117
                                 + 0.934852 w_k1_AXP_s117
                                 + 0.779759 w_k1_DIS_s117
                                 + 0.761729 w_k1_FDX_s117
                                 + 1.04321 w_k1_IBM_s117
                                 + 0.714313 w_k1_INTC_s117
                                 + 0.985576 w_k1_KO_s117 + 1.0841 w_k1_MAT_s117
                                 + 0.837402 w_k1_MSFT_s117
                                 + 0.986776 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.15781 w_k1_ADM_s117
                                 + 1.08393 w_k1_AXP_s117
                                 + 1.10782 w_k1_DIS_s117
                                 + 0.948761 w_k1_FDX_s117
                                 + 1.3464 w_k1_IBM_s117
                                 + 0.740266 w_k1_INTC_s117
                                 + 1.31067 w_k1_KO_s117 + 1.31556 w_k1_MAT_s117
                                 + 0.901954 w_k1_MSFT_s117
                                 + 1.04336 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.27359 w_k1_ADM_s117
                                 + 1.27999 w_k1_AXP_s117
                                 + 1.47516 w_k1_DIS_s117
                                 + 1.31183 w_k1_FDX_s117
                                 + 1.09584 w_k1_IBM_s117
                                 + 0.995817 w_k1_INTC_s117
                                 + 1.42203 w_k1_KO_s117
                                 + 0.896656 w_k1_MAT_s117
                                 + 1.39856 w_k1_MSFT_s117
                                 + 1.27997 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.710668 w_k1_ADM_s117
                                 + 0.873204 w_k1_AXP_s117
                                 + 0.595546 w_k1_DIS_s117
                                 + 0.83991 w_k1_FDX_s117
                                 + 0.596538 w_k1_IBM_s117
                                 + 0.669398 w_k1_INTC_s117
                                 + 0.728173 w_k1_KO_s117
                                 + 0.570154 w_k1_MAT_s117
                                 + 0.748888 w_k1_MSFT_s117
                                 + 0.782147 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.60189 w_k1_ADM_s117
                                 + 1.97617 w_k1_AXP_s117 + 2.0009 w_k1_DIS_s117
                                 + 1.38985 w_k1_FDX_s117
                                 + 1.94142 w_k1_IBM_s117
                                 + 1.11564 w_k1_INTC_s117
                                 + 2.00447 w_k1_KO_s117 + 2.29814 w_k1_MAT_s117
                                 + 1.22249 w_k1_MSFT_s117
                                 + 1.31764 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.27027 w_k1_ADM_s117
                                 + 1.51883 w_k1_AXP_s117
                                 + 1.22092 w_k1_DIS_s117
                                 + 1.22789 w_k1_FDX_s117
                                 + 1.23951 w_k1_IBM_s117
                                 + 1.05404 w_k1_INTC_s117
                                 + 1.30486 w_k1_KO_s117 + 1.36097 w_k1_MAT_s117
                                 + 1.02208 w_k1_MSFT_s117
                                 + 1.05059 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.7705 w_k1_ADM_s117
                                 + 0.723403 w_k1_AXP_s117
                                 + 0.732436 w_k1_DIS_s117
                                 + 0.818872 w_k1_FDX_s117
                                 + 0.653844 w_k1_IBM_s117
                                 + 0.734161 w_k1_INTC_s117
                                 + 0.870565 w_k1_KO_s117
                                 + 0.600412 w_k1_MAT_s117
                                 + 0.828093 w_k1_MSFT_s117
                                 + 0.82803 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0183045739218604 w_k1_ADM_s000
                                 + 0.0509801288892126 w_k1_ADM_s001
                                 + 0.0284761556639187 w_k1_ADM_s002
                                 + 0.0208425345446853 w_k1_ADM_s003
                                 + 0.0244598582035081 w_k1_ADM_s004
                                 + 0.0327461042148916 w_k1_ADM_s005
                                 + 0.0341559356713448 w_k1_ADM_s006
                                 + 0.067759784596551 w_k1_ADM_s007
                                 + 0.0252506585914195 w_k1_ADM_s010
                                 + 0.0169473148925942 w_k1_ADM_s011
                                 + 0.0558604839220615 w_k1_ADM_s012
                                 + 0.0149903499341679 w_k1_ADM_s013
                                 + 0.0349803404561513 w_k1_ADM_s014
                                 + 0.0277809223208184 w_k1_ADM_s015
                                 + 0.0169953970438842 w_k1_ADM_s016
                                 + 0.0134582516237311 w_k1_ADM_s017
                                 + 0.018522964604195 w_k1_ADM_s100
                                 + 0.0443792401230972 w_k1_ADM_s101
                                 + 0.0166439385062525 w_k1_ADM_s102
                                 + 0.0453485007501656 w_k1_ADM_s103
                                 + 0.0521303760735819 w_k1_ADM_s104
                                 + 0.0323223579408336 w_k1_ADM_s105
                                 + 0.0307363884734314 w_k1_ADM_s106
                                 + 0.0165778873299827 w_k1_ADM_s107
                                 + 0.0255920443574931 w_k1_ADM_s110
                                 + 0.0116917219843071 w_k1_ADM_s111
                                 + 0.0116698075680758 w_k1_ADM_s112
                                 + 0.0547053657159503 w_k1_ADM_s113
                                 + 0.0479057974352595 w_k1_ADM_s114
                                 + 0.0722503170447564 w_k1_ADM_s115
                                 + 0.0584186699113041 w_k1_ADM_s116
                                 + 0.0139165446068036 w_k1_ADM_s117
                                 + 0.0196459364194943 w_k1_AXP_s000
                                 + 0.050567239583519 w_k1_AXP_s001
                                 + 0.0254463590259659 w_k1_AXP_s002
                                 + 0.0200787200745302 w_k1_AXP_s003
                                 + 0.025287335947804 w_k1_AXP_s004
                                 + 0.0318444813408336 w_k1_AXP_s005
                                 + 0.0360170620618896 w_k1_AXP_s006
                                 + 0.0695449869295892 w_k1_AXP_s007
                                 + 0.0239123300928697 w_k1_AXP_s010
                                 + 0.0188349459620427 w_k1_AXP_s011
                                 + 0.0476011166862991 w_k1_AXP_s012
                                 + 0.015804294396711 w_k1_AXP_s013
                                 + 0.0388093964887477 w_k1_AXP_s014
                                 + 0.0262554166923636 w_k1_AXP_s015
                                 + 0.0182995361466468 w_k1_AXP_s016
                                 + 0.0160960561272495 w_k1_AXP_s017
                                 + 0.0208132345844086 w_k1_AXP_s100
                                 + 0.0539057663544993 w_k1_AXP_s101
                                 + 0.0161227947820563 w_k1_AXP_s102
                                 + 0.0509285534764272 w_k1_AXP_s103
                                 + 0.0530444860785064 w_k1_AXP_s104
                                 + 0.0296057322148209 w_k1_AXP_s105
                                 + 0.0329102161288194 w_k1_AXP_s106
                                 + 0.0188427799035339 w_k1_AXP_s107
                                 + 0.0244572732094895 w_k1_AXP_s110
                                 + 0.0133042524784382 w_k1_AXP_s111
                                 + 0.0147576358119543 w_k1_AXP_s112
                                 + 0.0560831906099669 w_k1_AXP_s113
                                 + 0.0407045186230903 w_k1_AXP_s114
                                 + 0.0608234668747656 w_k1_AXP_s115
                                 + 0.0606301041793464 w_k1_AXP_s116
                                 + 0.0148983302823198 w_k1_AXP_s117
                                 + 0.0189015149161762 w_k1_DIS_s000
                                 + 0.0559544186849036 w_k1_DIS_s001
                                 + 0.0264415723083153 w_k1_DIS_s002
                                 + 0.0192989963056579 w_k1_DIS_s003
                                 + 0.0260604537214684 w_k1_DIS_s004
                                 + 0.0290471676871931 w_k1_DIS_s005
                                 + 0.0361591353962715 w_k1_DIS_s006
                                 + 0.0667632608045941 w_k1_DIS_s007
                                 + 0.0235431918030223 w_k1_DIS_s010
                                 + 0.0198956815326112 w_k1_DIS_s011
                                 + 0.0525681832005542 w_k1_DIS_s012
                                 + 0.0164828769060476 w_k1_DIS_s013
                                 + 0.0366376186115732 w_k1_DIS_s014
                                 + 0.0251070980212549 w_k1_DIS_s015
                                 + 0.0160968003238137 w_k1_DIS_s016
                                 + 0.0159852026965371 w_k1_DIS_s017
                                 + 0.0200463946596911 w_k1_DIS_s100
                                 + 0.0557299493662841 w_k1_DIS_s101
                                 + 0.0170441257041471 w_k1_DIS_s102
                                 + 0.0485355171559589 w_k1_DIS_s103
                                 + 0.0525761008366178 w_k1_DIS_s104
                                 + 0.0279632418142527 w_k1_DIS_s105
                                 + 0.0353892258673469 w_k1_DIS_s106
                                 + 0.0168596367477359 w_k1_DIS_s107
                                 + 0.024799067793303 w_k1_DIS_s110
                                 + 0.0138630822731502 w_k1_DIS_s111
                                 + 0.013079387393749 w_k1_DIS_s112
                                 + 0.0551510643398839 w_k1_DIS_s113
                                 + 0.0393457112914806 w_k1_DIS_s114
                                 + 0.0625198111182027 w_k1_DIS_s115
                                 + 0.0625002368885153 w_k1_DIS_s116
                                 + 0.0141576676606007 w_k1_DIS_s117
                                 + 0.0178142529002549 w_k1_FDX_s000
                                 + 0.0523630531325367 w_k1_FDX_s001
                                 + 0.0265374251187291 w_k1_FDX_s002
                                 + 0.0201293309271532 w_k1_FDX_s003
                                 + 0.025647408237304 w_k1_FDX_s004
                                 + 0.0293166223387679 w_k1_FDX_s005
                                 + 0.0364298467556174 w_k1_FDX_s006
                                 + 0.0656351196520987 w_k1_FDX_s007
                                 + 0.0219302835599347 w_k1_FDX_s010
                                 + 0.0202325962231279 w_k1_FDX_s011
                                 + 0.0497191592551998 w_k1_FDX_s012
                                 + 0.0154646312798756 w_k1_FDX_s013
                                 + 0.0367834603923896 w_k1_FDX_s014
                                 + 0.0248022129053334 w_k1_FDX_s015
                                 + 0.0171851206090044 w_k1_FDX_s016
                                 + 0.0152132659553985 w_k1_FDX_s017
                                 + 0.019712445313304 w_k1_FDX_s100
                                 + 0.049170010046329 w_k1_FDX_s101
                                 + 0.0160101860038483 w_k1_FDX_s102
                                 + 0.0495250200879616 w_k1_FDX_s103
                                 + 0.0500570578172872 w_k1_FDX_s104
                                 + 0.0280116591305628 w_k1_FDX_s105
                                 + 0.031640387577334 w_k1_FDX_s106
                                 + 0.0178994449297323 w_k1_FDX_s107
                                 + 0.024251259803789 w_k1_FDX_s110
                                 + 0.0128678454814311 w_k1_FDX_s111
                                 + 0.0129941212401189 w_k1_FDX_s112
                                 + 0.0531400752542006 w_k1_FDX_s113
                                 + 0.0384810194923158 w_k1_FDX_s114
                                 + 0.0623960792012235 w_k1_FDX_s115
                                 + 0.0613541533013288 w_k1_FDX_s116
                                 + 0.0136869252572165 w_k1_FDX_s117
                                 + 0.0183145757068218 w_k1_IBM_s000
                                 + 0.0531819626671027 w_k1_IBM_s001
                                 + 0.0264488265966667 w_k1_IBM_s002
                                 + 0.0197844168074526 w_k1_IBM_s003
                                 + 0.0250785735363444 w_k1_IBM_s004
                                 + 0.0314760847354283 w_k1_IBM_s005
                                 + 0.0365087608942644 w_k1_IBM_s006
                                 + 0.0699518849945519 w_k1_IBM_s007
                                 + 0.0240426965911405 w_k1_IBM_s010
                                 + 0.0195046979526944 w_k1_IBM_s011
                                 + 0.0506822409069009 w_k1_IBM_s012
                                 + 0.0153088126452501 w_k1_IBM_s013
                                 + 0.0380635445860555 w_k1_IBM_s014
                                 + 0.0240572313540475 w_k1_IBM_s015
                                 + 0.0164879543606679 w_k1_IBM_s016
                                 + 0.0158932160393695 w_k1_IBM_s017
                                 + 0.0198804265755608 w_k1_IBM_s100
                                 + 0.0520228164567777 w_k1_IBM_s101
                                 + 0.0171770180437734 w_k1_IBM_s102
                                 + 0.0510016404152028 w_k1_IBM_s103
                                 + 0.0543691422556908 w_k1_IBM_s104
                                 + 0.0279687112808882 w_k1_IBM_s105
                                 + 0.0316976354779722 w_k1_IBM_s106
                                 + 0.0185867089059277 w_k1_IBM_s107
                                 + 0.0245687227456436 w_k1_IBM_s110
                                 + 0.0129513401954297 w_k1_IBM_s111
                                 + 0.0132782042179392 w_k1_IBM_s112
                                 + 0.0525345351395932 w_k1_IBM_s113
                                 + 0.0406141496807724 w_k1_IBM_s114
                                 + 0.0628642393499605 w_k1_IBM_s115
                                 + 0.0614178669044663 w_k1_IBM_s116
                                 + 0.0140552839455293 w_k1_IBM_s117
                                 + 0.0169246433780063 w_k1_INTC_s000
                                 + 0.0563635757922018 w_k1_INTC_s001
                                 + 0.0268024923818576 w_k1_INTC_s002
                                 + 0.0197955125016086 w_k1_INTC_s003
                                 + 0.0241775991537466 w_k1_INTC_s004
                                 + 0.0313646658580325 w_k1_INTC_s005
                                 + 0.0392865227404627 w_k1_INTC_s006
                                 + 0.065053379717549 w_k1_INTC_s007
                                 + 0.0228461960652555 w_k1_INTC_s010
                                 + 0.020091592879164 w_k1_INTC_s011
                                 + 0.0492107009236589 w_k1_INTC_s012
                                 + 0.0146360869232448 w_k1_INTC_s013
                                 + 0.0374478928779302 w_k1_INTC_s014
                                 + 0.024285842646297 w_k1_INTC_s015
                                 + 0.0136914832395796 w_k1_INTC_s016
                                 + 0.0153283568950986 w_k1_INTC_s017
                                 + 0.0269504385525051 w_k1_INTC_s100
                                 + 0.0473388193430891 w_k1_INTC_s101
                                 + 0.0169268218538815 w_k1_INTC_s102
                                 + 0.0489795976918834 w_k1_INTC_s103
                                 + 0.0503572773740224 w_k1_INTC_s104
                                 + 0.0245961727797139 w_k1_INTC_s105
                                 + 0.0343226366122144 w_k1_INTC_s106
                                 + 0.0190342146156786 w_k1_INTC_s107
                                 + 0.0217365714787359 w_k1_INTC_s110
                                 + 0.0118536250075538 w_k1_INTC_s111
                                 + 0.0130099103126315 w_k1_INTC_s112
                                 + 0.0512966933923736 w_k1_INTC_s113
                                 + 0.0390870319987065 w_k1_INTC_s114
                                 + 0.0638473580781683 w_k1_INTC_s115
                                 + 0.0555002189196575 w_k1_INTC_s116
                                 + 0.0112131121136802 w_k1_INTC_s117
                                 + 0.0176011328321014 w_k1_KO_s000
                                 + 0.0537802691758694 w_k1_KO_s001
                                 + 0.0255718328270767 w_k1_KO_s002
                                 + 0.0210769157861864 w_k1_KO_s003
                                 + 0.029183307450135 w_k1_KO_s004
                                 + 0.0294174751892369 w_k1_KO_s005
                                 + 0.0352019828251536 w_k1_KO_s006
                                 + 0.0682624386084749 w_k1_KO_s007
                                 + 0.0232954373451679 w_k1_KO_s010
                                 + 0.0203456221989365 w_k1_KO_s011
                                 + 0.0464156656866337 w_k1_KO_s012
                                 + 0.0152698953064773 w_k1_KO_s013
                                 + 0.0378544317366062 w_k1_KO_s014
                                 + 0.0246113201141271 w_k1_KO_s015
                                 + 0.0163491124208295 w_k1_KO_s016
                                 + 0.0159630732827775 w_k1_KO_s017
                                 + 0.0199044929487697 w_k1_KO_s100
                                 + 0.0516863904154317 w_k1_KO_s101
                                 + 0.0166465839877047 w_k1_KO_s102
                                 + 0.0517028175524327 w_k1_KO_s103
                                 + 0.0474780133867973 w_k1_KO_s104
                                 + 0.0287136007393005 w_k1_KO_s105
                                 + 0.0345593579087792 w_k1_KO_s106
                                 + 0.0162523121614935 w_k1_KO_s107
                                 + 0.0244643738009211 w_k1_KO_s110
                                 + 0.0119956767980877 w_k1_KO_s111
                                 + 0.0120125374487938 w_k1_KO_s112
                                 + 0.0504546575607619 w_k1_KO_s113
                                 + 0.0409992404758047 w_k1_KO_s114
                                 + 0.0603044003159672 w_k1_KO_s115
                                 + 0.0602247241544339 w_k1_KO_s116
                                 + 0.0154891824675446 w_k1_KO_s117
                                 + 0.0157623337246432 w_k1_MAT_s000
                                 + 0.0646191212505425 w_k1_MAT_s001
                                 + 0.0251048005959291 w_k1_MAT_s002
                                 + 0.0220931677589558 w_k1_MAT_s003
                                 + 0.0275112227371408 w_k1_MAT_s004
                                 + 0.020055612370396 w_k1_MAT_s005
                                 + 0.0401579041169351 w_k1_MAT_s006
                                 + 0.0652717467387414 w_k1_MAT_s007
                                 + 0.0257203116913158 w_k1_MAT_s010
                                 + 0.0204567834757974 w_k1_MAT_s011
                                 + 0.0427513350787838 w_k1_MAT_s012
                                 + 0.0132858622535892 w_k1_MAT_s013
                                 + 0.034628838707438 w_k1_MAT_s014
                                 + 0.0264110586360666 w_k1_MAT_s015
                                 + 0.0204865328640326 w_k1_MAT_s016
                                 + 0.0214024744752528 w_k1_MAT_s017
                                 + 0.0178445232117629 w_k1_MAT_s100
                                 + 0.0572160737409226 w_k1_MAT_s101
                                 + 0.0163310804405003 w_k1_MAT_s102
                                 + 0.0500942846739913 w_k1_MAT_s103
                                 + 0.0545702953756594 w_k1_MAT_s104
                                 + 0.0304129962912421 w_k1_MAT_s105
                                 + 0.0266118640478303 w_k1_MAT_s106
                                 + 0.0188052727610483 w_k1_MAT_s107
                                 + 0.0288137557191875 w_k1_MAT_s110
                                 + 0.0140051897785062 w_k1_MAT_s111
                                 + 0.0127022148369776 w_k1_MAT_s112
                                 + 0.0507138343027715 w_k1_MAT_s113
                                 + 0.0391880945235722 w_k1_MAT_s114
                                 + 0.0567465138306504 w_k1_MAT_s115
                                 + 0.0612402646241798 w_k1_MAT_s116
                                 + 0.0142630646187782 w_k1_MAT_s117
                                 + 0.016602024059822 w_k1_MSFT_s000
                                 + 0.0526761892506568 w_k1_MSFT_s001
                                 + 0.0252998605484576 w_k1_MSFT_s002
                                 + 0.0208523305789357 w_k1_MSFT_s003
                                 + 0.0242200376187072 w_k1_MSFT_s004
                                 + 0.0288967617458399 w_k1_MSFT_s005
                                 + 0.0371592075122638 w_k1_MSFT_s006
                                 + 0.0638189008381034 w_k1_MSFT_s007
                                 + 0.0236346301335111 w_k1_MSFT_s010
                                 + 0.0199304935872632 w_k1_MSFT_s011
                                 + 0.0493951983036217 w_k1_MSFT_s012
                                 + 0.0143079783547419 w_k1_MSFT_s013
                                 + 0.0385368339927065 w_k1_MSFT_s014
                                 + 0.0244022628303412 w_k1_MSFT_s015
                                 + 0.0148338874091259 w_k1_MSFT_s016
                                 + 0.0148652343602043 w_k1_MSFT_s017
                                 + 0.019712062842977 w_k1_MSFT_s100
                                 + 0.0506439240357656 w_k1_MSFT_s101
                                 + 0.0166873578737703 w_k1_MSFT_s102
                                 + 0.0483696607508957 w_k1_MSFT_s103
                                 + 0.0509819752739475 w_k1_MSFT_s104
                                 + 0.0288928445904881 w_k1_MSFT_s105
                                 + 0.0347774280059544 w_k1_MSFT_s106
                                 + 0.0165035832966155 w_k1_MSFT_s107
                                 + 0.0237970912139558 w_k1_MSFT_s110
                                 + 0.0130820186793881 w_k1_MSFT_s111
                                 + 0.0130886717430182 w_k1_MSFT_s112
                                 + 0.0527989762115547 w_k1_MSFT_s113
                                 + 0.0425605352221361 w_k1_MSFT_s114
                                 + 0.0590568199758931 w_k1_MSFT_s115
                                 + 0.0588209015552026 w_k1_MSFT_s116
                                 + 0.012815890510151 w_k1_MSFT_s117
                                 + 0.0171866233727398 w_k1_PG_s000
                                 + 0.0556931689617 w_k1_PG_s001
                                 + 0.0256520907829098 w_k1_PG_s002
                                 + 0.0197355887454768 w_k1_PG_s003
                                 + 0.0243878366309269 w_k1_PG_s004
                                 + 0.0274868401812034 w_k1_PG_s005
                                 + 0.0344989796133474 w_k1_PG_s006
                                 + 0.0666558126044977 w_k1_PG_s007
                                 + 0.022650180852872 w_k1_PG_s010
                                 + 0.0202650955279995 w_k1_PG_s011
                                 + 0.0515392047013676 w_k1_PG_s012
                                 + 0.0147689604263797 w_k1_PG_s013
                                 + 0.0369491545259397 w_k1_PG_s014
                                 + 0.0254224264581922 w_k1_PG_s015
                                 + 0.0163840033507369 w_k1_PG_s016
                                 + 0.0156382062780376 w_k1_PG_s017
                                 + 0.0197205702702069 w_k1_PG_s100
                                 + 0.0487228921134601 w_k1_PG_s101
                                 + 0.0166939068098834 w_k1_PG_s102
                                 + 0.0459682556268242 w_k1_PG_s103
                                 + 0.0544041029367875 w_k1_PG_s104
                                 + 0.0280266419235462 w_k1_PG_s105
                                 + 0.0338486641148334 w_k1_PG_s106
                                 + 0.0185766158564654 w_k1_PG_s107
                                 + 0.0243407675237009 w_k1_PG_s110
                                 + 0.0134521680312337 w_k1_PG_s111
                                 + 0.0137616418166696 w_k1_PG_s112
                                 + 0.0543101332265674 w_k1_PG_s113
                                 + 0.0392268630386478 w_k1_PG_s114
                                 + 0.0603455627533633 w_k1_PG_s115
                                 + 0.0537432408212468 w_k1_PG_s116
                                 + 0.0132974860120377 w_k1_PG_s117 >= 1.11
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
