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
 w_k1_ADM_s0_return_constraint:  1.17884 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.974983 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.00869 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.871907 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.98135 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.964512 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.35142 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.07393 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.2489 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.873858 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.07411 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.08691 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.00882 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.01549 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.758431 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.966424 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.923173 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.03206 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.951491 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.2057 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.04423 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.17692 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.731009 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.875809 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.05242 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.9264 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.880094 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.15948 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.38133 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.800377 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.880821 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.25258 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.880109 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.21341 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.800551 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.07522 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.38353 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.25696 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.3011 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.945489 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.16164 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.764661 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.884372 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.1512 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.19426 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.57648 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.32737 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.816182 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.884274 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.75042 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.02388 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.11339 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.617408 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.824458 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.962727 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.1201 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.899982 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.1907 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.17854 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.60421 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.623594 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.809311 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.13641 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.824849 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.866058 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.4553 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.68703 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.803878 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.629707 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.19174 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.759253 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.54586 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.596697 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.900956 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.51973 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.16455 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.37946 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.923376 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.19262 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.708825 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.868731 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.18368 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.16401 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.59581 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.63271 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.823795 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.0637 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.678729 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.13815 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.903643 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.651057 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.837898 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.865793 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.29847 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.839665 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.11465 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.01923 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.8424 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.641819 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.670604 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.22015 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.03777 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.904014 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.63934 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 2.01936 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.686807 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.647801 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.19627 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.733034 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.57488 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.526006 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.893223 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.80976 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.18528 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.25128 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.934615 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.08254 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.810393 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.973919 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0593 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.13363 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.23614 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.40929 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.726079 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.07028 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.836645 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.01264 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.989468 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.745568 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.916447 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.832303 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.20673 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.943121 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.17707 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.886966 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.34698 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.741301 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.841275 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.18965 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.928793 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.952075 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.33757 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.64218 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.703251 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.763804 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.10257 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.803033 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.23093 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.707584 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.991093 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.61418 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.28578 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.11326 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.01544 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.17618 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.774296 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.700362 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.23671 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.952526 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.62788 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.969838 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.36861 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.707606 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.783654 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.82732 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.17517 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.717082 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.02188 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.02706 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.08972 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.692688 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.874303 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.57683 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.65442 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.67804 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.776042 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.13347 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.30299 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.900292 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.33759 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.20169 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.21889 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.783702 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.22947 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.10848 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.8086 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.608304 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.808464 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.05305 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.74407 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15952 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.924537 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.14402 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.933416 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.01802 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.173 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.912011 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.17238 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.03266 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.927621 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.979812 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.8105 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.933775 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.919413 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.734539 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.748197 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.80341 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.939034 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.873074 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.03908 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.95041 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.28391 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.716543 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.841023 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.05185 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.984036 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.887738 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.19813 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.37771 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.856971 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.746209 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.01949 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.801875 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.15189 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.683891 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.843905 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.32547 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.28093 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.16366 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.972851 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.21638 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.790298 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.812125 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.02293 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.958839 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.37242 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.16503 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.05841 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.813142 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.830669 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.33098 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.02469 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.715936 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.875524 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.05193 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.28563 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.770208 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.984534 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.18291 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.62874 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.725001 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.757694 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.09332 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.17596 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.914457 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.36098 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.43604 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.854925 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.878759 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.27886 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.07909 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.74484 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.710744 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.982672 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.56219 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.13736 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.25213 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.04991 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.26549 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.648039 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.730543 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.10898 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.871241 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.57992 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.13778 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.05393 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.679406 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.77411 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.49355 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.19224 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.683865 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.925406 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.25226 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.26711 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.733508 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.06625 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.5956 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.71315 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.623097 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.796799 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.21998 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.977987 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.868432 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.1935 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.47596 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.958907 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.734995 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.23432 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.13282 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 2.32048 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.643003 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.89134 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.31821 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.897341 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1849 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.912267 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.13097 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.844657 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.954848 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.16197 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.00775 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.17733 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.3749 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.891624 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.04288 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.807482 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.02364 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.893827 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.763587 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.908632 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.90872 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.2402 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.911818 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.01989 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.965661 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.36289 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.776656 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.748904 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.2185 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.0506 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.991294 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.22624 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.58562 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.819036 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.770508 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.01878 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.796936 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.21868 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.725096 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.940629 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.38069 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.22827 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16379 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.985794 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.09571 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.801419 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.898369 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.05753 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.976441 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.14533 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.1329 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.847349 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.914912 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.840026 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.04102 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.03407 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.812464 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.994961 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.01235 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.25309 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.905432 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.03929 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.10215 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.40673 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.79087 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.884184 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.10513 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.06615 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.884058 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.26496 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.31805 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.859453 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.845057 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.01201 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.882592 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.37234 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.83401 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.951057 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.24039 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.04986 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.212000295693903 z_k1_s0000
                                 - 0.322000449119985 z_k1_s0001
                                 - 0.237000330563467 z_k1_s0002
                                 - 0.575000801999972 z_k1_s0003
                                 - 0.161000224559992 z_k1_s0004
                                 - 0.314000437961724 z_k1_s0005
                                 - 0.279000389144334 z_k1_s0006
                                 - 0.400000557913024 z_k1_s0007
                                 - 0.0719999845607839 z_k1_s0010
                                 - 0.170999857356855 z_k1_s0011
                                 - 0.289999898564266 z_k1_s0012
                                 - 0.137999946858168 z_k1_s0013
                                 - 0.11100005862288 z_k1_s0014
                                 - 0.116999939586272 z_k1_s0015
                                 - 0.32799970986578 z_k1_s0016
                                 - 0.186000266265377 z_k1_s0017
                                 - 0.125999863485918 z_k1_s0020
                                 - 0.25399927626587 z_k1_s0021
                                 - 0.243999539798392 z_k1_s0022
                                 - 0.141999839833344 z_k1_s0023
                                 - 0.351998683644464 z_k1_s0024
                                 - 0.528998994112097 z_k1_s0025
                                 - 0.174999766174946 z_k1_s0026
                                 - 0.16799975164798 z_k1_s0027
                                 - 0.101999968070426 z_k1_s0030
                                 - 0.0979999039892644 z_k1_s0031
                                 - 0.0790000161035458 z_k1_s0032
                                 - 0.0900000138268265 z_k1_s0033
                                 - 0.271999359521403 z_k1_s0034
                                 - 0.075999908542703 z_k1_s0035
                                 - 0.318999576975312 z_k1_s0036
                                 - 0.154000206125816 z_k1_s0037
                                 - 0.466000236725332 z_k1_s0040
                                 - 0.399998521432138 z_k1_s0041
                                 - 0.356998241648879 z_k1_s0042
                                 - 0.174999623576131 z_k1_s0043
                                 - 0.24100061767047 z_k1_s0044
                                 - 0.419999769701647 z_k1_s0045
                                 - 0.151999870863526 z_k1_s0046
                                 - 0.192999785420215 z_k1_s0047
                                 - 1.00999917895768 z_k1_s0050
                                 - 0.567000689772306 z_k1_s0051
                                 - 0.64100035345015 z_k1_s0052
                                 - 0.269000143794989 z_k1_s0053
                                 - 0.619999380924414 z_k1_s0054
                                 - 1.26600300978814 z_k1_s0055
                                 - 0.386000083204968 z_k1_s0056
                                 - 1.05200112400915 z_k1_s0057
                                 - 0.138999429468138 z_k1_s0060
                                 - 0.165000464994531 z_k1_s0061
                                 - 0.157000053140258 z_k1_s0062
                                 - 0.0710000495067403 z_k1_s0063
                                 - 0.188000588575586 z_k1_s0064
                                 - 0.0729999706703119 z_k1_s0065
                                 - 0.137999529486351 z_k1_s0066
                                 - 0.280999770881614 z_k1_s0067
                                 - 0.269000313997926 z_k1_s0070
                                 - 0.0979999177985214 z_k1_s0071
                                 - 0.159000136448593 z_k1_s0072
                                 - 0.0719999426050358 z_k1_s0073
                                 - 0.303000315458634 z_k1_s0074
                                 - 0.176999791574887 z_k1_s0075
                                 - 0.142999937014163 z_k1_s0076
                                 - 0.247999492184044 z_k1_s0077
                                 - 0.182000528924907 z_k1_s0100
                                 - 0.0720001471844323 z_k1_s0101
                                 - 0.0900002781057622 z_k1_s0102
                                 - 0.11199997795297 z_k1_s0103
                                 - 0.0580001122902224 z_k1_s0104
                                 - 0.0370000975989962 z_k1_s0105
                                 - 0.102000114386057 z_k1_s0106
                                 - 0.100000141672675 z_k1_s0107
                                 - 0.106999239147678 z_k1_s0110
                                 - 0.0979995444800493 z_k1_s0111
                                 - 0.110999288554744 z_k1_s0112
                                 - 0.113999742108317 z_k1_s0113
                                 - 0.0499997845917914 z_k1_s0114
                                 - 0.0939994950729824 z_k1_s0115
                                 - 0.140999659107746 z_k1_s0116
                                 - 0.117999791515384 z_k1_s0117
                                 - 0.0609999940515935 z_k1_s0120
                                 - 0.0449999551528277 z_k1_s0121
                                 - 0.0589999891892477 z_k1_s0122
                                 - 0.0339999592599165 z_k1_s0123
                                 - 0.0920002236679028 z_k1_s0124
                                 - 0.0830000475373963 z_k1_s0125
                                 - 0.123000144784311 z_k1_s0126
                                 - 0.11999967474094 z_k1_s0127
                                 - 0.0830000168606275 z_k1_s0130
                                 - 0.205000018125646 z_k1_s0131
                                 - 0.0850000360813657 z_k1_s0132
                                 - 0.159999585659036 z_k1_s0133
                                 - 0.0769999591984128 z_k1_s0134
                                 - 0.0599999910221401 z_k1_s0135
                                 - 0.0950000345850557 z_k1_s0136
                                 - 0.211000466187864 z_k1_s0137
                                 - 0.150999688706407 z_k1_s0140
                                 - 0.0659999987471969 z_k1_s0141
                                 - 0.0530000470757885 z_k1_s0142
                                 - 0.0749999950581777 z_k1_s0143
                                 - 0.101000053201024 z_k1_s0144
                                 - 0.212000188303154 z_k1_s0145
                                 - 0.0559999942461056 z_k1_s0146
                                 - 0.060000027006548 z_k1_s0147
                                 - 0.122000292064982 z_k1_s0150
                                 - 0.122999613507155 z_k1_s0151
                                 - 0.246000431016048 z_k1_s0152
                                 - 0.200000340630919 z_k1_s0153
                                 - 0.150000556473624 z_k1_s0154
                                 - 0.120999766621072 z_k1_s0155
                                 - 0.0730000701502074 z_k1_s0156
                                 - 0.168999703892284 z_k1_s0157
                                 - 0.0800001294688004 z_k1_s0160
                                 - 0.171000689660031 z_k1_s0161
                                 - 0.051000036656308 z_k1_s0162
                                 - 0.139000500232354 z_k1_s0163
                                 - 0.17600046835157 z_k1_s0164
                                 - 0.189000581150354 z_k1_s0165
                                 - 0.171000689660031 z_k1_s0166
                                 - 0.170000504521462 z_k1_s0167
                                 - 0.0460000139351666 z_k1_s0170
                                 - 0.0949998495400377 z_k1_s0171
                                 - 0.0529998241644522 z_k1_s0172
                                 - 0.0300000111968475 z_k1_s0173
                                 - 0.0829999808612837 z_k1_s0174
                                 - 0.0630000380633782 z_k1_s0175
                                 - 0.0669998326229807 z_k1_s0176
                                 - 0.0480000082149571 z_k1_s0177
                                 - 0.619999242635346 z_k1_s1000
                                 - 0.278999440736321 z_k1_s1001
                                 - 0.669997983608215 z_k1_s1002
                                 - 0.991997040619875 z_k1_s1003
                                 - 0.505999142302851 z_k1_s1004
                                 - 0.299999436223929 z_k1_s1005
                                 - 0.529998013690824 z_k1_s1006
                                 - 0.471999648921964 z_k1_s1007
                                 - 0.255999840729782 z_k1_s1010
                                 - 0.232000014192599 z_k1_s1011
                                 - 0.338999185212547 z_k1_s1012
                                 - 0.144000027222991 z_k1_s1013
                                 - 0.406000158337035 z_k1_s1014
                                 - 0.148999868709916 z_k1_s1015
                                 - 0.676001210630039 z_k1_s1016
                                 - 0.468000221974706 z_k1_s1017
                                 - 0.653000251209625 z_k1_s1020
                                 - 0.455000080556419 z_k1_s1021
                                 - 0.648998928751174 z_k1_s1022
                                 - 0.403999991221474 z_k1_s1023
                                 - 0.493000064903974 z_k1_s1024
                                 - 1.06599902496004 z_k1_s1025
                                 - 1.08199832479016 z_k1_s1026
                                 - 1.18800341992071 z_k1_s1027
                                 - 0.35300056616194 z_k1_s1030
                                 - 0.382001554827214 z_k1_s1031
                                 - 0.291999583521672 z_k1_s1032
                                 - 0.204000033890625 z_k1_s1033
                                 - 0.389000197883652 z_k1_s1034
                                 - 0.281000145918012 z_k1_s1035
                                 - 0.577000523986906 z_k1_s1036
                                 - 0.414001548802209 z_k1_s1037
                                 - 0.604998022864316 z_k1_s1040
                                 - 1.2409990489501 z_k1_s1041
                                 - 0.299000187068601 z_k1_s1042
                                 - 0.376000209069543 z_k1_s1043
                                 - 1.17600110219628 z_k1_s1044
                                 - 0.534999863590926 z_k1_s1045
                                 - 1.17099848167646 z_k1_s1046
                                 - 0.617001904111619 z_k1_s1047
                                 - 0.505000050166743 z_k1_s1050
                                 - 0.245999097514808 z_k1_s1051
                                 - 0.229000141028098 z_k1_s1052
                                 - 0.171999939842924 z_k1_s1053
                                 - 0.138999900356777 z_k1_s1054
                                 - 0.302999542500003 z_k1_s1055
                                 - 0.150999981206291 z_k1_s1056
                                 - 0.692999122608933 z_k1_s1057
                                 - 0.254000061241749 z_k1_s1060
                                 - 0.281999823331338 z_k1_s1061
                                 - 0.287000474418915 z_k1_s1062
                                 - 0.203999376367243 z_k1_s1063
                                 - 0.228000559587192 z_k1_s1064
                                 - 0.106999950871114 z_k1_s1065
                                 - 0.450000337869291 z_k1_s1066
                                 - 0.130000032873337 z_k1_s1067
                                 - 0.452000836559737 z_k1_s1070
                                 - 0.266999466769182 z_k1_s1071
                                 - 0.278999338794992 z_k1_s1072
                                 - 0.248999273031315 z_k1_s1073
                                 - 0.653999196593072 z_k1_s1074
                                 - 0.636997577960464 z_k1_s1075
                                 - 0.829997576771051 z_k1_s1076
                                 - 0.488997099246664 z_k1_s1077
                                 - 0.245999939830298 z_k1_s1100
                                 - 0.397998053179634 z_k1_s1101
                                 - 0.497998344330071 z_k1_s1102
                                 - 0.284000050467464 z_k1_s1103
                                 - 0.594000176634042 z_k1_s1104
                                 - 0.787001514954161 z_k1_s1105
                                 - 0.306999729232177 z_k1_s1106
                                 - 0.76799913043625 z_k1_s1107
                                 - 2.08299964779094 z_k1_s1110
                                 - 0.979003272651952 z_k1_s1111
                                 - 1.77600068975506 z_k1_s1112
                                 - 1.62900677644328 z_k1_s1113
                                 - 0.635001331808411 z_k1_s1114
                                 - 0.906001263934516 z_k1_s1115
                                 - 0.473000559506722 z_k1_s1116
                                 - 0.893001381338969 z_k1_s1117
                                 - 1.00999465061599 z_k1_s1120
                                 - 0.629998866344511 z_k1_s1121
                                 - 1.42699859878961 z_k1_s1122
                                 - 0.312999447473999 z_k1_s1123
                                 - 0.72899872030716 z_k1_s1124
                                 - 1.1429998804386 z_k1_s1125
                                 - 0.678998703251445 z_k1_s1126
                                 - 1.56100026276794 z_k1_s1127
                                 - 0.565000293833516 z_k1_s1130
                                 - 0.261000043624875 z_k1_s1131
                                 - 0.842001283942938 z_k1_s1132
                                 - 0.340999764548245 z_k1_s1133
                                 - 0.498000402272709 z_k1_s1134
                                 - 0.458000927310975 z_k1_s1135
                                 - 0.518000139753576 z_k1_s1136
                                 - 0.372000128643491 z_k1_s1137
                                 - 0.397999717367109 z_k1_s1140
                                 - 0.547998386686993 z_k1_s1141
                                 - 0.402999851327682 z_k1_s1142
                                 - 0.28199998378012 z_k1_s1143
                                 - 0.299999807638514 z_k1_s1144
                                 - 0.42100132118524 z_k1_s1145
                                 - 0.407999985288254 z_k1_s1146
                                 - 1.35499885683409 z_k1_s1147
                                 - 1.5869998091894 z_k1_s1150
                                 - 0.508998848641017 z_k1_s1151
                                 - 0.838998297743247 z_k1_s1152
                                 - 1.76799796679312 z_k1_s1153
                                 - 1.44499901585839 z_k1_s1154
                                 - 0.840997936706407 z_k1_s1155
                                 - 1.15899779023323 z_k1_s1156
                                 - 0.660998717686039 z_k1_s1157
                                 - 0.358999575266407 z_k1_s1160
                                 - 0.179999924032182 z_k1_s1161
                                 - 0.700999114567553 z_k1_s1162
                                 - 0.317001347352228 z_k1_s1163
                                 - 0.293001127401271 z_k1_s1164
                                 - 0.365999759252103 z_k1_s1165
                                 - 0.361001271262317 z_k1_s1166
                                 - 0.299000463389011 z_k1_s1167
                                 - 0.557000947188645 z_k1_s1170
                                 - 0.258001101140985 z_k1_s1171
                                 - 0.561001503913748 z_k1_s1172
                                 - 0.367001420645361 z_k1_s1173
                                 - 0.865002045867786 z_k1_s1174
                                 - 1.00500296717804 z_k1_s1175
                                 - 0.382001187855953 z_k1_s1176
                                 - 0.646001731779467 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.25058 w_k1_ADM_s000
                                 + 1.37896 w_k1_AXP_s000 + 1.5065 w_k1_DIS_s000
                                 + 1.30967 w_k1_FDX_s000
                                 + 1.64649 w_k1_IBM_s000
                                 + 1.36631 w_k1_INTC_s000
                                 + 1.80122 w_k1_KO_s000 + 1.92713 w_k1_MAT_s000
                                 + 1.46299 w_k1_MSFT_s000
                                 + 1.49191 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.743477 w_k1_ADM_s000
                                 + 0.820594 w_k1_AXP_s000
                                 + 0.955578 w_k1_DIS_s000
                                 + 1.00667 w_k1_FDX_s000
                                 + 0.902562 w_k1_IBM_s000
                                 + 1.25201 w_k1_INTC_s000
                                 + 1.02966 w_k1_KO_s000
                                 + 0.981112 w_k1_MAT_s000
                                 + 1.21037 w_k1_MSFT_s000
                                 + 0.953179 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.979021 w_k1_ADM_s000
                                 + 1.09232 w_k1_AXP_s000
                                 + 1.31247 w_k1_DIS_s000
                                 + 1.18617 w_k1_FDX_s000
                                 + 1.09115 w_k1_IBM_s000
                                 + 1.27939 w_k1_INTC_s000
                                 + 1.35608 w_k1_KO_s000 + 1.46492 w_k1_MAT_s000
                                 + 1.28371 w_k1_MSFT_s000
                                 + 1.28326 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.03576 w_k1_ADM_s000
                                 + 1.02894 w_k1_AXP_s000
                                 + 1.03891 w_k1_DIS_s000
                                 + 0.943834 w_k1_FDX_s000 + 1.181 w_k1_IBM_s000
                                 + 1.00333 w_k1_INTC_s000
                                 + 1.21139 w_k1_KO_s000 + 1.40412 w_k1_MAT_s000
                                 + 1.05513 w_k1_MSFT_s000
                                 + 1.11995 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.25777 w_k1_ADM_s000
                                 + 1.4592 w_k1_AXP_s000 + 1.41545 w_k1_DIS_s000
                                 + 1.45405 w_k1_FDX_s000
                                 + 0.934532 w_k1_IBM_s000
                                 + 1.62594 w_k1_INTC_s000
                                 + 1.18484 w_k1_KO_s000 + 1.17387 w_k1_MAT_s000
                                 + 1.41335 w_k1_MSFT_s000
                                 + 1.19884 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.656541 w_k1_ADM_s000
                                 + 0.674452 w_k1_AXP_s000
                                 + 0.660364 w_k1_DIS_s000
                                 + 0.761486 w_k1_FDX_s000
                                 + 0.876532 w_k1_IBM_s000
                                 + 1.11769 w_k1_INTC_s000
                                 + 0.826679 w_k1_KO_s000
                                 + 0.604527 w_k1_MAT_s000
                                 + 0.859432 w_k1_MSFT_s000
                                 + 0.935633 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.14673 w_k1_ADM_s000
                                 + 1.06856 w_k1_AXP_s000
                                 + 0.876399 w_k1_DIS_s000
                                 + 1.13147 w_k1_FDX_s000
                                 + 0.690919 w_k1_IBM_s000
                                 + 1.10312 w_k1_INTC_s000
                                 + 0.778433 w_k1_KO_s000
                                 + 1.03624 w_k1_MAT_s000
                                 + 1.08497 w_k1_MSFT_s000
                                 + 0.962926 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.934915 w_k1_ADM_s000
                                 + 0.93355 w_k1_AXP_s000
                                 + 0.735943 w_k1_DIS_s000
                                 + 0.846481 w_k1_FDX_s000
                                 + 0.973916 w_k1_IBM_s000
                                 + 0.951834 w_k1_INTC_s000
                                 + 0.954538 w_k1_KO_s000
                                 + 1.10317 w_k1_MAT_s000
                                 + 0.866495 w_k1_MSFT_s000
                                 + 0.955782 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.31574 w_k1_ADM_s001
                                 + 1.12174 w_k1_AXP_s001
                                 + 1.27123 w_k1_DIS_s001
                                 + 1.08554 w_k1_FDX_s001 + 1.8844 w_k1_IBM_s001
                                 + 1.36323 w_k1_INTC_s001 + 1.6629 w_k1_KO_s001
                                 + 1.69384 w_k1_MAT_s001
                                 + 1.28749 w_k1_MSFT_s001
                                 + 1.36619 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.05807 w_k1_ADM_s001
                                 + 0.80199 w_k1_AXP_s001 + 1.231 w_k1_DIS_s001
                                 + 1.05226 w_k1_FDX_s001
                                 + 0.902534 w_k1_IBM_s001
                                 + 1.05977 w_k1_INTC_s001
                                 + 1.25426 w_k1_KO_s001 + 1.07717 w_k1_MAT_s001
                                 + 1.13767 w_k1_MSFT_s001
                                 + 1.12182 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.10365 w_k1_ADM_s001
                                 + 0.826422 w_k1_AXP_s001
                                 + 0.809927 w_k1_DIS_s001
                                 + 0.889392 w_k1_FDX_s001
                                 + 1.07932 w_k1_IBM_s001
                                 + 0.765608 w_k1_INTC_s001
                                 + 0.849409 w_k1_KO_s001
                                 + 1.05433 w_k1_MAT_s001
                                 + 0.865281 w_k1_MSFT_s001
                                 + 0.922085 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.23088 w_k1_ADM_s001
                                 + 0.96602 w_k1_AXP_s001
                                 + 0.997916 w_k1_DIS_s001
                                 + 1.13428 w_k1_FDX_s001
                                 + 0.74885 w_k1_IBM_s001
                                 + 1.04258 w_k1_INTC_s001
                                 + 1.04421 w_k1_KO_s001
                                 + 0.943239 w_k1_MAT_s001
                                 + 1.19318 w_k1_MSFT_s001 + 1.0902 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.907688 w_k1_ADM_s001
                                 + 0.69566 w_k1_AXP_s001 + 0.6265 w_k1_DIS_s001
                                 + 0.805636 w_k1_FDX_s001
                                 + 0.770774 w_k1_IBM_s001
                                 + 0.910215 w_k1_INTC_s001
                                 + 0.850361 w_k1_KO_s001
                                 + 0.737737 w_k1_MAT_s001
                                 + 0.693068 w_k1_MSFT_s001
                                 + 0.745832 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.26542 w_k1_ADM_s001
                                 + 0.619561 w_k1_AXP_s001
                                 + 0.681016 w_k1_DIS_s001
                                 + 0.827761 w_k1_FDX_s001
                                 + 0.653501 w_k1_IBM_s001
                                 + 0.783709 w_k1_INTC_s001
                                 + 0.722572 w_k1_KO_s001
                                 + 0.373074 w_k1_MAT_s001
                                 + 0.817903 w_k1_MSFT_s001
                                 + 0.793583 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.40447 w_k1_ADM_s001
                                 + 1.23691 w_k1_AXP_s001
                                 + 1.57309 w_k1_DIS_s001
                                 + 1.37929 w_k1_FDX_s001 + 1.3379 w_k1_IBM_s001
                                 + 1.1772 w_k1_INTC_s001 + 1.24355 w_k1_KO_s001
                                 + 1.43993 w_k1_MAT_s001
                                 + 1.47853 w_k1_MSFT_s001
                                 + 1.25647 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.50182 w_k1_ADM_s001
                                 + 1.37425 w_k1_AXP_s001
                                 + 1.02438 w_k1_DIS_s001
                                 + 1.07786 w_k1_FDX_s001
                                 + 1.07657 w_k1_IBM_s001
                                 + 0.92776 w_k1_INTC_s001
                                 + 0.96737 w_k1_KO_s001 + 1.26868 w_k1_MAT_s001
                                 + 0.862329 w_k1_MSFT_s001
                                 + 1.02099 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.13798 w_k1_ADM_s002
                                 + 1.05299 w_k1_AXP_s002
                                 + 1.46963 w_k1_DIS_s002
                                 + 1.13829 w_k1_FDX_s002
                                 + 1.16142 w_k1_IBM_s002
                                 + 1.2146 w_k1_INTC_s002 + 1.34259 w_k1_KO_s002
                                 + 0.802998 w_k1_MAT_s002
                                 + 1.1752 w_k1_MSFT_s002 + 1.02827 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.42608 w_k1_ADM_s002
                                 + 1.0722 w_k1_AXP_s002 + 1.06234 w_k1_DIS_s002
                                 + 1.2129 w_k1_FDX_s002
                                 + 0.929709 w_k1_IBM_s002
                                 + 1.25666 w_k1_INTC_s002
                                 + 0.93478 w_k1_KO_s002 + 0.78597 w_k1_MAT_s002
                                 + 0.860053 w_k1_MSFT_s002
                                 + 1.06151 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.4838 w_k1_ADM_s002
                                 + 1.18363 w_k1_AXP_s002
                                 + 1.12277 w_k1_DIS_s002 + 1.0083 w_k1_FDX_s002
                                 + 1.6169 w_k1_IBM_s002
                                 + 0.912683 w_k1_INTC_s002
                                 + 1.30275 w_k1_KO_s002 + 1.41645 w_k1_MAT_s002
                                 + 0.885639 w_k1_MSFT_s002
                                 + 1.14301 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.04701 w_k1_ADM_s002
                                 + 1.16907 w_k1_AXP_s002
                                 + 0.793037 w_k1_DIS_s002
                                 + 0.855999 w_k1_FDX_s002
                                 + 1.26109 w_k1_IBM_s002
                                 + 1.09502 w_k1_INTC_s002
                                 + 1.01213 w_k1_KO_s002 + 1.0577 w_k1_MAT_s002
                                 + 0.821555 w_k1_MSFT_s002
                                 + 0.897891 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.54272 w_k1_ADM_s002
                                 + 1.62336 w_k1_AXP_s002
                                 + 1.52536 w_k1_DIS_s002
                                 + 1.44633 w_k1_FDX_s002 + 1.4367 w_k1_IBM_s002
                                 + 1.58191 w_k1_INTC_s002
                                 + 1.30268 w_k1_KO_s002 + 1.44619 w_k1_MAT_s002
                                 + 1.2466 w_k1_MSFT_s002 + 1.23758 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.962927 w_k1_ADM_s002
                                 + 0.853887 w_k1_AXP_s002
                                 + 0.806307 w_k1_DIS_s002
                                 + 0.872068 w_k1_FDX_s002
                                 + 0.958308 w_k1_IBM_s002
                                 + 0.996409 w_k1_INTC_s002
                                 + 1.02493 w_k1_KO_s002
                                 + 0.918873 w_k1_MAT_s002
                                 + 0.913119 w_k1_MSFT_s002
                                 + 0.898848 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.993097 w_k1_ADM_s002
                                 + 0.857732 w_k1_AXP_s002
                                 + 1.12932 w_k1_DIS_s002
                                 + 0.973821 w_k1_FDX_s002
                                 + 1.02925 w_k1_IBM_s002
                                 + 1.06146 w_k1_INTC_s002 + 1.054 w_k1_KO_s002
                                 + 0.618031 w_k1_MAT_s002
                                 + 0.900662 w_k1_MSFT_s002
                                 + 0.953561 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.908544 w_k1_ADM_s002
                                 + 0.702249 w_k1_AXP_s002
                                 + 0.642878 w_k1_DIS_s002
                                 + 0.895754 w_k1_FDX_s002
                                 + 0.692202 w_k1_IBM_s002
                                 + 1.08306 w_k1_INTC_s002 + 0.6893 w_k1_KO_s002
                                 + 0.540568 w_k1_MAT_s002
                                 + 0.930808 w_k1_MSFT_s002
                                 + 0.859506 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.21877 w_k1_ADM_s003
                                 + 1.22088 w_k1_AXP_s003
                                 + 1.19617 w_k1_DIS_s003
                                 + 1.06496 w_k1_FDX_s003
                                 + 1.70356 w_k1_IBM_s003
                                 + 0.907085 w_k1_INTC_s003
                                 + 1.11587 w_k1_KO_s003 + 1.77211 w_k1_MAT_s003
                                 + 0.844383 w_k1_MSFT_s003
                                 + 1.09114 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.02406 w_k1_ADM_s003
                                 + 1.37095 w_k1_AXP_s003
                                 + 1.50225 w_k1_DIS_s003
                                 + 1.34678 w_k1_FDX_s003
                                 + 1.18089 w_k1_IBM_s003
                                 + 1.22358 w_k1_INTC_s003
                                 + 1.23645 w_k1_KO_s003 + 1.45807 w_k1_MAT_s003
                                 + 1.26279 w_k1_MSFT_s003
                                 + 1.22186 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.16027 w_k1_ADM_s003
                                 + 0.99982 w_k1_AXP_s003
                                 + 1.45164 w_k1_DIS_s003
                                 + 1.46616 w_k1_FDX_s003
                                 + 0.921413 w_k1_IBM_s003
                                 + 1.19604 w_k1_INTC_s003
                                 + 1.02266 w_k1_KO_s003 + 1.09643 w_k1_MAT_s003
                                 + 1.19397 w_k1_MSFT_s003
                                 + 1.20449 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.969496 w_k1_ADM_s003
                                 + 1.31326 w_k1_AXP_s003
                                 + 1.16536 w_k1_DIS_s003
                                 + 1.07744 w_k1_FDX_s003
                                 + 1.24962 w_k1_IBM_s003
                                 + 0.884091 w_k1_INTC_s003
                                 + 1.17044 w_k1_KO_s003 + 1.76968 w_k1_MAT_s003
                                 + 0.923659 w_k1_MSFT_s003
                                 + 1.20663 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.955449 w_k1_ADM_s003
                                 + 0.938196 w_k1_AXP_s003
                                 + 1.05125 w_k1_DIS_s003
                                 + 1.10106 w_k1_FDX_s003
                                 + 0.891216 w_k1_IBM_s003
                                 + 0.768732 w_k1_INTC_s003
                                 + 0.905228 w_k1_KO_s003
                                 + 1.14646 w_k1_MAT_s003
                                 + 0.976142 w_k1_MSFT_s003
                                 + 1.14049 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.08852 w_k1_ADM_s003
                                 + 0.803588 w_k1_AXP_s003
                                 + 1.0397 w_k1_DIS_s003
                                 + 0.993866 w_k1_FDX_s003
                                 + 1.16042 w_k1_IBM_s003
                                 + 0.787234 w_k1_INTC_s003
                                 + 1.03751 w_k1_KO_s003
                                 + 0.773934 w_k1_MAT_s003
                                 + 0.86319 w_k1_MSFT_s003
                                 + 1.02778 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.04959 w_k1_ADM_s003
                                 + 0.747112 w_k1_AXP_s003
                                 + 0.690356 w_k1_DIS_s003
                                 + 0.910452 w_k1_FDX_s003
                                 + 0.730852 w_k1_IBM_s003
                                 + 0.850745 w_k1_INTC_s003
                                 + 0.723533 w_k1_KO_s003
                                 + 0.758618 w_k1_MAT_s003
                                 + 0.784157 w_k1_MSFT_s003
                                 + 0.961672 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.893685 w_k1_ADM_s003
                                 + 0.881531 w_k1_AXP_s003
                                 + 0.607835 w_k1_DIS_s003
                                 + 0.888959 w_k1_FDX_s003
                                 + 1.01561 w_k1_IBM_s003
                                 + 0.784326 w_k1_INTC_s003
                                 + 0.867171 w_k1_KO_s003
                                 + 1.16727 w_k1_MAT_s003
                                 + 0.757002 w_k1_MSFT_s003
                                 + 0.970996 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.927328 w_k1_ADM_s004
                                 + 0.867184 w_k1_AXP_s004
                                 + 0.770332 w_k1_DIS_s004
                                 + 0.860749 w_k1_FDX_s004
                                 + 1.03636 w_k1_IBM_s004
                                 + 0.859936 w_k1_INTC_s004
                                 + 0.931172 w_k1_KO_s004
                                 + 0.975525 w_k1_MAT_s004
                                 + 0.8871 w_k1_MSFT_s004
                                 + 0.914882 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.944828 w_k1_ADM_s004
                                 + 1.03176 w_k1_AXP_s004
                                 + 1.15493 w_k1_DIS_s004
                                 + 1.12938 w_k1_FDX_s004
                                 + 1.02237 w_k1_IBM_s004
                                 + 1.19404 w_k1_INTC_s004
                                 + 1.12836 w_k1_KO_s004
                                 + 0.946214 w_k1_MAT_s004
                                 + 1.23167 w_k1_MSFT_s004
                                 + 1.06122 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.01771 w_k1_ADM_s004
                                 + 0.561732 w_k1_AXP_s004
                                 + 0.708588 w_k1_DIS_s004
                                 + 0.787138 w_k1_FDX_s004
                                 + 0.72713 w_k1_IBM_s004
                                 + 0.739987 w_k1_INTC_s004
                                 + 0.798004 w_k1_KO_s004
                                 + 0.529311 w_k1_MAT_s004
                                 + 0.854473 w_k1_MSFT_s004
                                 + 0.811318 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.34029 w_k1_ADM_s004
                                 + 1.18933 w_k1_AXP_s004 + 1.1219 w_k1_DIS_s004
                                 + 1.29804 w_k1_FDX_s004
                                 + 0.699682 w_k1_IBM_s004
                                 + 1.11513 w_k1_INTC_s004
                                 + 1.02122 w_k1_KO_s004
                                 + 0.715687 w_k1_MAT_s004
                                 + 1.23347 w_k1_MSFT_s004
                                 + 1.05827 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.20313 w_k1_ADM_s004
                                 + 1.27029 w_k1_AXP_s004
                                 + 1.54957 w_k1_DIS_s004
                                 + 1.33049 w_k1_FDX_s004
                                 + 0.848753 w_k1_IBM_s004
                                 + 1.36835 w_k1_INTC_s004
                                 + 1.06985 w_k1_KO_s004
                                 + 0.643854 w_k1_MAT_s004
                                 + 1.25703 w_k1_MSFT_s004
                                 + 1.19628 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.17375 w_k1_ADM_s004
                                 + 1.07724 w_k1_AXP_s004
                                 + 0.791565 w_k1_DIS_s004
                                 + 1.03955 w_k1_FDX_s004
                                 + 0.735964 w_k1_IBM_s004
                                 + 0.953318 w_k1_INTC_s004
                                 + 0.846127 w_k1_KO_s004
                                 + 0.603611 w_k1_MAT_s004
                                 + 0.916826 w_k1_MSFT_s004
                                 + 0.874201 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.15478 w_k1_ADM_s004
                                 + 0.997964 w_k1_AXP_s004
                                 + 1.27376 w_k1_DIS_s004
                                 + 1.04098 w_k1_FDX_s004
                                 + 1.54452 w_k1_IBM_s004
                                 + 0.943152 w_k1_INTC_s004
                                 + 1.42871 w_k1_KO_s004 + 1.3633 w_k1_MAT_s004
                                 + 1.08114 w_k1_MSFT_s004
                                 + 1.19863 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.0831 w_k1_ADM_s004
                                 + 1.23008 w_k1_AXP_s004
                                 + 1.46518 w_k1_DIS_s004
                                 + 1.21155 w_k1_FDX_s004 + 1.5241 w_k1_IBM_s004
                                 + 1.01854 w_k1_INTC_s004
                                 + 1.24871 w_k1_KO_s004 + 1.63942 w_k1_MAT_s004
                                 + 1.24565 w_k1_MSFT_s004
                                 + 1.17447 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.906876 w_k1_ADM_s005
                                 + 0.898601 w_k1_AXP_s005
                                 + 1.00391 w_k1_DIS_s005
                                 + 0.95556 w_k1_FDX_s005
                                 + 0.753114 w_k1_IBM_s005
                                 + 0.957681 w_k1_INTC_s005
                                 + 0.921774 w_k1_KO_s005
                                 + 0.650486 w_k1_MAT_s005
                                 + 0.993651 w_k1_MSFT_s005
                                 + 0.992574 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.766917 w_k1_ADM_s005
                                 + 1.06196 w_k1_AXP_s005
                                 + 0.916841 w_k1_DIS_s005
                                 + 0.848903 w_k1_FDX_s005
                                 + 1.54014 w_k1_IBM_s005
                                 + 1.06437 w_k1_INTC_s005
                                 + 1.05152 w_k1_KO_s005 + 1.3729 w_k1_MAT_s005
                                 + 0.902116 w_k1_MSFT_s005
                                 + 1.03121 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.984705 w_k1_ADM_s005
                                 + 1.44803 w_k1_AXP_s005
                                 + 1.41392 w_k1_DIS_s005
                                 + 1.41087 w_k1_FDX_s005
                                 + 0.968344 w_k1_IBM_s005
                                 + 1.29397 w_k1_INTC_s005
                                 + 1.12381 w_k1_KO_s005 + 1.05521 w_k1_MAT_s005
                                 + 1.16867 w_k1_MSFT_s005
                                 + 1.25928 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.5353 w_k1_ADM_s005
                                 + 1.70262 w_k1_AXP_s005
                                 + 1.83943 w_k1_DIS_s005
                                 + 1.47397 w_k1_FDX_s005
                                 + 1.50902 w_k1_IBM_s005
                                 + 1.11486 w_k1_INTC_s005
                                 + 1.20556 w_k1_KO_s005 + 1.35106 w_k1_MAT_s005
                                 + 1.31812 w_k1_MSFT_s005
                                 + 1.38944 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.07414 w_k1_ADM_s005
                                 + 1.09847 w_k1_AXP_s005
                                 + 1.07595 w_k1_DIS_s005
                                 + 1.13713 w_k1_FDX_s005
                                 + 0.675573 w_k1_IBM_s005
                                 + 0.964238 w_k1_INTC_s005
                                 + 0.844023 w_k1_KO_s005
                                 + 0.743699 w_k1_MAT_s005
                                 + 1.16687 w_k1_MSFT_s005
                                 + 1.06481 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.631985 w_k1_ADM_s005
                                 + 0.772394 w_k1_AXP_s005
                                 + 0.813692 w_k1_DIS_s005
                                 + 0.805305 w_k1_FDX_s005
                                 + 0.947784 w_k1_IBM_s005
                                 + 0.783289 w_k1_INTC_s005
                                 + 0.839944 w_k1_KO_s005
                                 + 0.723424 w_k1_MAT_s005
                                 + 0.820828 w_k1_MSFT_s005
                                 + 0.83527 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.822022 w_k1_ADM_s005
                                 + 0.779465 w_k1_AXP_s005
                                 + 0.688704 w_k1_DIS_s005
                                 + 0.778896 w_k1_FDX_s005
                                 + 0.48432 w_k1_IBM_s005
                                 + 0.906948 w_k1_INTC_s005
                                 + 0.469945 w_k1_KO_s005
                                 + 0.357388 w_k1_MAT_s005
                                 + 0.786762 w_k1_MSFT_s005
                                 + 0.778942 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.938626 w_k1_ADM_s005
                                 + 1.21984 w_k1_AXP_s005
                                 + 1.26871 w_k1_DIS_s005
                                 + 1.11828 w_k1_FDX_s005
                                 + 1.32393 w_k1_IBM_s005
                                 + 1.11902 w_k1_INTC_s005
                                 + 0.982092 w_k1_KO_s005
                                 + 1.22901 w_k1_MAT_s005
                                 + 1.07018 w_k1_MSFT_s005
                                 + 1.12412 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.15796 w_k1_ADM_s006
                                 + 0.84189 w_k1_AXP_s006
                                 + 0.979819 w_k1_DIS_s006
                                 + 0.867233 w_k1_FDX_s006
                                 + 0.997051 w_k1_IBM_s006
                                 + 0.801619 w_k1_INTC_s006
                                 + 0.931899 w_k1_KO_s006
                                 + 0.803703 w_k1_MAT_s006
                                 + 1.01636 w_k1_MSFT_s006
                                 + 1.04595 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.30227 w_k1_ADM_s006
                                 + 1.77272 w_k1_AXP_s006
                                 + 1.08926 w_k1_DIS_s006
                                 + 1.03699 w_k1_FDX_s006
                                 + 1.51564 w_k1_IBM_s006
                                 + 0.945939 w_k1_INTC_s006
                                 + 1.16389 w_k1_KO_s006 + 2.13829 w_k1_MAT_s006
                                 + 0.942833 w_k1_MSFT_s006
                                 + 1.21253 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.932239 w_k1_ADM_s006
                                 + 1.26616 w_k1_AXP_s006
                                 + 0.907064 w_k1_DIS_s006
                                 + 0.984448 w_k1_FDX_s006
                                 + 1.0539 w_k1_IBM_s006
                                 + 1.11236 w_k1_INTC_s006
                                 + 0.899777 w_k1_KO_s006
                                 + 0.983688 w_k1_MAT_s006
                                 + 0.933409 w_k1_MSFT_s006
                                 + 1.09542 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.42794 w_k1_ADM_s006
                                 + 1.82296 w_k1_AXP_s006
                                 + 1.84507 w_k1_DIS_s006 + 1.3329 w_k1_FDX_s006
                                 + 1.27567 w_k1_IBM_s006
                                 + 1.4771 w_k1_INTC_s006 + 1.50435 w_k1_KO_s006
                                 + 1.15996 w_k1_MAT_s006
                                 + 1.35585 w_k1_MSFT_s006
                                 + 1.31857 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.10894 w_k1_ADM_s006
                                 + 1.23259 w_k1_AXP_s006
                                 + 1.03407 w_k1_DIS_s006
                                 + 1.09186 w_k1_FDX_s006
                                 + 0.881329 w_k1_IBM_s006
                                 + 0.932344 w_k1_INTC_s006
                                 + 0.852208 w_k1_KO_s006
                                 + 1.26364 w_k1_MAT_s006
                                 + 1.00451 w_k1_MSFT_s006
                                 + 1.06855 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.0198 w_k1_ADM_s006
                                 + 0.903859 w_k1_AXP_s006
                                 + 0.548287 w_k1_DIS_s006
                                 + 0.699781 w_k1_FDX_s006
                                 + 0.854711 w_k1_IBM_s006
                                 + 0.791847 w_k1_INTC_s006
                                 + 0.779282 w_k1_KO_s006
                                 + 0.837899 w_k1_MAT_s006
                                 + 0.70362 w_k1_MSFT_s006
                                 + 0.891867 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.17935 w_k1_ADM_s006
                                 + 1.23679 w_k1_AXP_s006
                                 + 1.25127 w_k1_DIS_s006
                                 + 1.03078 w_k1_FDX_s006
                                 + 1.25678 w_k1_IBM_s006
                                 + 1.26733 w_k1_INTC_s006 + 1.0854 w_k1_KO_s006
                                 + 1.21713 w_k1_MAT_s006
                                 + 1.01314 w_k1_MSFT_s006
                                 + 1.15536 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.992198 w_k1_ADM_s006
                                 + 0.795328 w_k1_AXP_s006
                                 + 0.864227 w_k1_DIS_s006
                                 + 0.884186 w_k1_FDX_s006
                                 + 0.665394 w_k1_IBM_s006
                                 + 1.12895 w_k1_INTC_s006
                                 + 0.902456 w_k1_KO_s006
                                 + 0.649726 w_k1_MAT_s006
                                 + 0.993431 w_k1_MSFT_s006
                                 + 0.91262 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.01988 w_k1_ADM_s007
                                 + 1.0601 w_k1_AXP_s007 + 1.21616 w_k1_DIS_s007
                                 + 1.03842 w_k1_FDX_s007
                                 + 1.32702 w_k1_IBM_s007
                                 + 1.14766 w_k1_INTC_s007 + 1.2352 w_k1_KO_s007
                                 + 1.41587 w_k1_MAT_s007
                                 + 1.10807 w_k1_MSFT_s007
                                 + 1.10733 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.20617 w_k1_ADM_s007
                                 + 1.0978 w_k1_AXP_s007 + 1.34256 w_k1_DIS_s007
                                 + 1.30617 w_k1_FDX_s007
                                 + 0.954286 w_k1_IBM_s007
                                 + 1.10687 w_k1_INTC_s007
                                 + 1.03161 w_k1_KO_s007 + 1.07602 w_k1_MAT_s007
                                 + 1.33035 w_k1_MSFT_s007 + 1.0106 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.17785 w_k1_ADM_s007
                                 + 1.17383 w_k1_AXP_s007
                                 + 0.994864 w_k1_DIS_s007
                                 + 1.07697 w_k1_FDX_s007
                                 + 0.763136 w_k1_IBM_s007
                                 + 1.09522 w_k1_INTC_s007
                                 + 0.830356 w_k1_KO_s007
                                 + 0.842156 w_k1_MAT_s007
                                 + 1.12064 w_k1_MSFT_s007
                                 + 0.938435 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.14085 w_k1_ADM_s007
                                 + 1.15909 w_k1_AXP_s007
                                 + 1.50839 w_k1_DIS_s007
                                 + 1.19723 w_k1_FDX_s007
                                 + 1.04005 w_k1_IBM_s007
                                 + 1.48661 w_k1_INTC_s007
                                 + 1.08091 w_k1_KO_s007 + 1.24081 w_k1_MAT_s007
                                 + 1.49463 w_k1_MSFT_s007
                                 + 1.11593 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.01057 w_k1_ADM_s007
                                 + 0.877556 w_k1_AXP_s007
                                 + 0.963721 w_k1_DIS_s007
                                 + 0.864914 w_k1_FDX_s007
                                 + 1.00972 w_k1_IBM_s007
                                 + 0.929101 w_k1_INTC_s007
                                 + 0.975285 w_k1_KO_s007
                                 + 0.874054 w_k1_MAT_s007
                                 + 0.97479 w_k1_MSFT_s007
                                 + 0.842629 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.09345 w_k1_ADM_s007
                                 + 1.11827 w_k1_AXP_s007
                                 + 0.924672 w_k1_DIS_s007
                                 + 0.983648 w_k1_FDX_s007
                                 + 0.827067 w_k1_IBM_s007
                                 + 1.08711 w_k1_INTC_s007
                                 + 0.808517 w_k1_KO_s007
                                 + 0.87566 w_k1_MAT_s007
                                 + 1.05249 w_k1_MSFT_s007
                                 + 0.919642 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.2295 w_k1_ADM_s007
                                 + 1.57533 w_k1_AXP_s007
                                 + 1.44547 w_k1_DIS_s007
                                 + 1.34852 w_k1_FDX_s007
                                 + 1.30632 w_k1_IBM_s007
                                 + 1.16011 w_k1_INTC_s007
                                 + 1.23269 w_k1_KO_s007 + 1.63124 w_k1_MAT_s007
                                 + 1.21734 w_k1_MSFT_s007
                                 + 1.10578 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.976656 w_k1_ADM_s007
                                 + 0.77728 w_k1_AXP_s007
                                 + 0.634363 w_k1_DIS_s007
                                 + 0.808316 w_k1_FDX_s007
                                 + 0.800135 w_k1_IBM_s007
                                 + 0.859036 w_k1_INTC_s007
                                 + 0.746276 w_k1_KO_s007
                                 + 0.845128 w_k1_MAT_s007
                                 + 0.863087 w_k1_MSFT_s007
                                 + 0.821119 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.08296 w_k1_ADM_s010
                                 + 0.945248 w_k1_AXP_s010
                                 + 0.776035 w_k1_DIS_s010
                                 + 0.909214 w_k1_FDX_s010
                                 + 0.81406 w_k1_IBM_s010
                                 + 0.984917 w_k1_INTC_s010
                                 + 0.978714 w_k1_KO_s010
                                 + 0.752902 w_k1_MAT_s010
                                 + 1.04112 w_k1_MSFT_s010
                                 + 1.02022 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.937612 w_k1_ADM_s010
                                 + 0.842876 w_k1_AXP_s010
                                 + 0.58124 w_k1_DIS_s010
                                 + 0.756065 w_k1_FDX_s010
                                 + 0.693358 w_k1_IBM_s010
                                 + 0.760362 w_k1_INTC_s010
                                 + 0.737149 w_k1_KO_s010
                                 + 0.777172 w_k1_MAT_s010
                                 + 0.879827 w_k1_MSFT_s010
                                 + 0.888111 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.26322 w_k1_ADM_s010
                                 + 1.04738 w_k1_AXP_s010
                                 + 0.982387 w_k1_DIS_s010
                                 + 1.25894 w_k1_FDX_s010
                                 + 0.648489 w_k1_IBM_s010
                                 + 1.07177 w_k1_INTC_s010
                                 + 0.915382 w_k1_KO_s010
                                 + 0.680489 w_k1_MAT_s010
                                 + 1.18092 w_k1_MSFT_s010
                                 + 1.02202 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.03364 w_k1_ADM_s010
                                 + 1.30502 w_k1_AXP_s010 + 1.0755 w_k1_DIS_s010
                                 + 1.09072 w_k1_FDX_s010
                                 + 1.36257 w_k1_IBM_s010
                                 + 1.19205 w_k1_INTC_s010
                                 + 1.35137 w_k1_KO_s010 + 1.7192 w_k1_MAT_s010
                                 + 1.0912 w_k1_MSFT_s010 + 1.14165 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.86421 w_k1_ADM_s010
                                 + 0.765544 w_k1_AXP_s010
                                 + 0.596288 w_k1_DIS_s010
                                 + 0.751624 w_k1_FDX_s010
                                 + 1.21501 w_k1_IBM_s010
                                 + 0.776082 w_k1_INTC_s010
                                 + 0.949929 w_k1_KO_s010 + 1.0654 w_k1_MAT_s010
                                 + 0.822153 w_k1_MSFT_s010
                                 + 0.902372 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.33345 w_k1_ADM_s010
                                 + 1.61382 w_k1_AXP_s010
                                 + 1.03635 w_k1_DIS_s010
                                 + 1.39492 w_k1_FDX_s010
                                 + 0.788287 w_k1_IBM_s010
                                 + 1.02691 w_k1_INTC_s010
                                 + 0.974028 w_k1_KO_s010
                                 + 0.868447 w_k1_MAT_s010
                                 + 1.04164 w_k1_MSFT_s010
                                 + 1.09122 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.760825 w_k1_ADM_s010
                                 + 1.025 w_k1_AXP_s010 + 1.02074 w_k1_DIS_s010
                                 + 0.916103 w_k1_FDX_s010
                                 + 1.20685 w_k1_IBM_s010
                                 + 1.09637 w_k1_INTC_s010
                                 + 1.23362 w_k1_KO_s010
                                 + 0.925211 w_k1_MAT_s010
                                 + 0.946218 w_k1_MSFT_s010
                                 + 1.11878 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.32173 w_k1_ADM_s010
                                 + 1.31511 w_k1_AXP_s010
                                 + 1.49258 w_k1_DIS_s010
                                 + 1.32388 w_k1_FDX_s010 + 1.1209 w_k1_IBM_s010
                                 + 1.3664 w_k1_INTC_s010 + 1.38281 w_k1_KO_s010
                                 + 1.37992 w_k1_MAT_s010
                                 + 1.40004 w_k1_MSFT_s010
                                 + 1.20208 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.06218 w_k1_ADM_s011
                                 + 1.29462 w_k1_AXP_s011
                                 + 1.11615 w_k1_DIS_s011
                                 + 1.14611 w_k1_FDX_s011
                                 + 0.851389 w_k1_IBM_s011
                                 + 1.2694 w_k1_INTC_s011 + 1.20162 w_k1_KO_s011
                                 + 1.07207 w_k1_MAT_s011
                                 + 1.14815 w_k1_MSFT_s011
                                 + 1.02371 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.805513 w_k1_ADM_s011
                                 + 0.975568 w_k1_AXP_s011
                                 + 0.82478 w_k1_DIS_s011
                                 + 0.886033 w_k1_FDX_s011
                                 + 0.9669 w_k1_IBM_s011
                                 + 1.01746 w_k1_INTC_s011
                                 + 0.94318 w_k1_KO_s011 + 1.15929 w_k1_MAT_s011
                                 + 0.915569 w_k1_MSFT_s011
                                 + 0.937936 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.866376 w_k1_ADM_s011
                                 + 1.19456 w_k1_AXP_s011
                                 + 1.04464 w_k1_DIS_s011
                                 + 0.995462 w_k1_FDX_s011
                                 + 1.34347 w_k1_IBM_s011
                                 + 1.1963 w_k1_INTC_s011 + 1.12371 w_k1_KO_s011
                                 + 1.40269 w_k1_MAT_s011
                                 + 0.90248 w_k1_MSFT_s011
                                 + 1.02169 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.796777 w_k1_ADM_s011
                                 + 0.79829 w_k1_AXP_s011
                                 + 0.820489 w_k1_DIS_s011
                                 + 0.890008 w_k1_FDX_s011
                                 + 0.73669 w_k1_IBM_s011
                                 + 0.840979 w_k1_INTC_s011
                                 + 0.768523 w_k1_KO_s011
                                 + 0.829108 w_k1_MAT_s011
                                 + 0.848218 w_k1_MSFT_s011
                                 + 0.86885 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.45051 w_k1_ADM_s011
                                 + 1.8445 w_k1_AXP_s011 + 1.62932 w_k1_DIS_s011
                                 + 1.51135 w_k1_FDX_s011
                                 + 1.28175 w_k1_IBM_s011
                                 + 1.27658 w_k1_INTC_s011
                                 + 1.51532 w_k1_KO_s011 + 2.01573 w_k1_MAT_s011
                                 + 1.40672 w_k1_MSFT_s011
                                 + 1.36258 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.916785 w_k1_ADM_s011
                                 + 1.02829 w_k1_AXP_s011 + 1.1905 w_k1_DIS_s011
                                 + 1.08198 w_k1_FDX_s011
                                 + 1.00435 w_k1_IBM_s011
                                 + 1.14334 w_k1_INTC_s011
                                 + 1.08829 w_k1_KO_s011 + 1.2292 w_k1_MAT_s011
                                 + 1.04152 w_k1_MSFT_s011
                                 + 1.00452 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.722064 w_k1_ADM_s011
                                 + 0.860253 w_k1_AXP_s011
                                 + 0.681995 w_k1_DIS_s011
                                 + 0.761926 w_k1_FDX_s011
                                 + 1.05577 w_k1_IBM_s011
                                 + 0.87532 w_k1_INTC_s011
                                 + 0.878386 w_k1_KO_s011
                                 + 1.06673 w_k1_MAT_s011
                                 + 0.801725 w_k1_MSFT_s011
                                 + 0.867176 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.40923 w_k1_ADM_s011
                                 + 1.42535 w_k1_AXP_s011
                                 + 1.23734 w_k1_DIS_s011
                                 + 1.20092 w_k1_FDX_s011
                                 + 1.32316 w_k1_IBM_s011
                                 + 1.4027 w_k1_INTC_s011 + 1.24083 w_k1_KO_s011
                                 + 1.58373 w_k1_MAT_s011
                                 + 1.15827 w_k1_MSFT_s011
                                 + 1.18702 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.03005 w_k1_ADM_s012
                                 + 0.813287 w_k1_AXP_s012
                                 + 1.25524 w_k1_DIS_s012
                                 + 1.06933 w_k1_FDX_s012
                                 + 0.743004 w_k1_IBM_s012
                                 + 1.06149 w_k1_INTC_s012
                                 + 0.857098 w_k1_KO_s012
                                 + 0.549656 w_k1_MAT_s012
                                 + 1.22841 w_k1_MSFT_s012
                                 + 0.910286 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.860957 w_k1_ADM_s012
                                 + 1.06596 w_k1_AXP_s012
                                 + 0.810173 w_k1_DIS_s012
                                 + 1.03485 w_k1_FDX_s012
                                 + 0.760007 w_k1_IBM_s012
                                 + 0.857243 w_k1_INTC_s012
                                 + 0.71503 w_k1_KO_s012
                                 + 0.867364 w_k1_MAT_s012
                                 + 0.903091 w_k1_MSFT_s012
                                 + 0.90588 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.2062 w_k1_ADM_s012
                                 + 1.14222 w_k1_AXP_s012
                                 + 1.41414 w_k1_DIS_s012 + 1.2914 w_k1_FDX_s012
                                 + 0.879469 w_k1_IBM_s012
                                 + 1.34977 w_k1_INTC_s012
                                 + 0.969255 w_k1_KO_s012
                                 + 0.907797 w_k1_MAT_s012
                                 + 1.17363 w_k1_MSFT_s012 + 1.0147 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.12297 w_k1_ADM_s012
                                 + 1.56916 w_k1_AXP_s012
                                 + 1.68089 w_k1_DIS_s012
                                 + 1.29997 w_k1_FDX_s012
                                 + 1.68159 w_k1_IBM_s012
                                 + 1.4172 w_k1_INTC_s012 + 1.36688 w_k1_KO_s012
                                 + 1.1592 w_k1_MAT_s012
                                 + 1.23857 w_k1_MSFT_s012
                                 + 1.13079 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.03087 w_k1_ADM_s012
                                 + 1.45144 w_k1_AXP_s012
                                 + 1.38731 w_k1_DIS_s012
                                 + 1.33323 w_k1_FDX_s012
                                 + 1.21666 w_k1_IBM_s012
                                 + 1.14658 w_k1_INTC_s012
                                 + 1.05391 w_k1_KO_s012 + 1.10866 w_k1_MAT_s012
                                 + 1.02472 w_k1_MSFT_s012
                                 + 1.01049 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.822678 w_k1_ADM_s012
                                 + 1.00151 w_k1_AXP_s012
                                 + 1.03479 w_k1_DIS_s012 + 1.0637 w_k1_FDX_s012
                                 + 1.07862 w_k1_IBM_s012
                                 + 1.11273 w_k1_INTC_s012
                                 + 0.862662 w_k1_KO_s012
                                 + 1.16673 w_k1_MAT_s012
                                 + 0.985962 w_k1_MSFT_s012
                                 + 0.968937 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.635285 w_k1_ADM_s012
                                 + 0.677304 w_k1_AXP_s012
                                 + 0.681623 w_k1_DIS_s012
                                 + 0.762896 w_k1_FDX_s012
                                 + 0.842708 w_k1_IBM_s012
                                 + 0.744293 w_k1_INTC_s012
                                 + 0.733478 w_k1_KO_s012
                                 + 0.87963 w_k1_MAT_s012
                                 + 0.770843 w_k1_MSFT_s012
                                 + 0.788109 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.865152 w_k1_ADM_s012
                                 + 0.812003 w_k1_AXP_s012
                                 + 0.976426 w_k1_DIS_s012
                                 + 0.90019 w_k1_FDX_s012
                                 + 1.20432 w_k1_IBM_s012
                                 + 0.848603 w_k1_INTC_s012
                                 + 0.880857 w_k1_KO_s012
                                 + 0.826331 w_k1_MAT_s012
                                 + 0.803005 w_k1_MSFT_s012
                                 + 0.926686 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.33387 w_k1_ADM_s013
                                 + 1.38986 w_k1_AXP_s013
                                 + 1.51839 w_k1_DIS_s013
                                 + 1.33776 w_k1_FDX_s013
                                 + 0.982504 w_k1_IBM_s013
                                 + 1.53683 w_k1_INTC_s013
                                 + 1.12348 w_k1_KO_s013
                                 + 0.993244 w_k1_MAT_s013
                                 + 1.25993 w_k1_MSFT_s013 + 1.1505 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.988001 w_k1_ADM_s013
                                 + 1.13379 w_k1_AXP_s013
                                 + 1.25632 w_k1_DIS_s013
                                 + 1.07261 w_k1_FDX_s013
                                 + 1.24136 w_k1_IBM_s013
                                 + 1.10836 w_k1_INTC_s013
                                 + 0.992846 w_k1_KO_s013
                                 + 1.32041 w_k1_MAT_s013
                                 + 0.945237 w_k1_MSFT_s013
                                 + 0.909859 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.42674 w_k1_ADM_s013
                                 + 1.58305 w_k1_AXP_s013
                                 + 1.38778 w_k1_DIS_s013
                                 + 1.28881 w_k1_FDX_s013
                                 + 1.68958 w_k1_IBM_s013
                                 + 1.52608 w_k1_INTC_s013
                                 + 1.21218 w_k1_KO_s013 + 1.3988 w_k1_MAT_s013
                                 + 1.13919 w_k1_MSFT_s013
                                 + 1.12513 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.917126 w_k1_ADM_s013
                                 + 0.912546 w_k1_AXP_s013
                                 + 0.927912 w_k1_DIS_s013
                                 + 0.964665 w_k1_FDX_s013
                                 + 0.747987 w_k1_IBM_s013
                                 + 1.29897 w_k1_INTC_s013
                                 + 0.758898 w_k1_KO_s013
                                 + 0.631612 w_k1_MAT_s013
                                 + 0.877781 w_k1_MSFT_s013
                                 + 0.78444 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.07013 w_k1_ADM_s013
                                 + 1.03105 w_k1_AXP_s013
                                 + 1.21182 w_k1_DIS_s013
                                 + 1.22247 w_k1_FDX_s013
                                 + 0.681206 w_k1_IBM_s013
                                 + 1.37035 w_k1_INTC_s013
                                 + 0.851306 w_k1_KO_s013
                                 + 1.13097 w_k1_MAT_s013
                                 + 1.12808 w_k1_MSFT_s013
                                 + 0.982606 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.32779 w_k1_ADM_s013
                                 + 1.39564 w_k1_AXP_s013
                                 + 1.55139 w_k1_DIS_s013
                                 + 1.47033 w_k1_FDX_s013
                                 + 0.860464 w_k1_IBM_s013
                                 + 1.61854 w_k1_INTC_s013 + 1.069 w_k1_KO_s013
                                 + 1.10308 w_k1_MAT_s013
                                 + 1.48898 w_k1_MSFT_s013
                                 + 1.16043 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.994403 w_k1_ADM_s013
                                 + 0.837985 w_k1_AXP_s013
                                 + 1.29809 w_k1_DIS_s013
                                 + 0.978837 w_k1_FDX_s013
                                 + 0.940088 w_k1_IBM_s013
                                 + 1.19445 w_k1_INTC_s013
                                 + 1.16006 w_k1_KO_s013
                                 + 0.788713 w_k1_MAT_s013
                                 + 1.04986 w_k1_MSFT_s013
                                 + 0.909614 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.943632 w_k1_ADM_s013
                                 + 0.668993 w_k1_AXP_s013
                                 + 0.761379 w_k1_DIS_s013
                                 + 0.74712 w_k1_FDX_s013
                                 + 0.945617 w_k1_IBM_s013
                                 + 0.990961 w_k1_INTC_s013
                                 + 0.944121 w_k1_KO_s013
                                 + 0.746744 w_k1_MAT_s013
                                 + 0.83651 w_k1_MSFT_s013
                                 + 0.85921 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.890564 w_k1_ADM_s014
                                 + 0.923387 w_k1_AXP_s014
                                 + 1.1434 w_k1_DIS_s014 + 1.04964 w_k1_FDX_s014
                                 + 0.885111 w_k1_IBM_s014
                                 + 0.992899 w_k1_INTC_s014
                                 + 1.07372 w_k1_KO_s014 + 1.07826 w_k1_MAT_s014
                                 + 1.02727 w_k1_MSFT_s014
                                 + 1.07529 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.903598 w_k1_ADM_s014
                                 + 0.958744 w_k1_AXP_s014
                                 + 1.03416 w_k1_DIS_s014
                                 + 0.986215 w_k1_FDX_s014
                                 + 1.48444 w_k1_IBM_s014
                                 + 1.11095 w_k1_INTC_s014
                                 + 1.20299 w_k1_KO_s014 + 1.27971 w_k1_MAT_s014
                                 + 0.982717 w_k1_MSFT_s014
                                 + 1.13792 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.08216 w_k1_ADM_s014
                                 + 0.885035 w_k1_AXP_s014
                                 + 1.20055 w_k1_DIS_s014
                                 + 1.30878 w_k1_FDX_s014
                                 + 0.602858 w_k1_IBM_s014
                                 + 1.20787 w_k1_INTC_s014
                                 + 0.931764 w_k1_KO_s014
                                 + 0.494476 w_k1_MAT_s014
                                 + 1.19578 w_k1_MSFT_s014
                                 + 1.05493 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.04467 w_k1_ADM_s014
                                 + 1.01209 w_k1_AXP_s014
                                 + 1.53696 w_k1_DIS_s014
                                 + 1.32841 w_k1_FDX_s014
                                 + 1.06051 w_k1_IBM_s014
                                 + 1.33747 w_k1_INTC_s014
                                 + 1.35809 w_k1_KO_s014 + 1.03894 w_k1_MAT_s014
                                 + 1.24867 w_k1_MSFT_s014
                                 + 1.28594 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.751596 w_k1_ADM_s014
                                 + 0.593414 w_k1_AXP_s014
                                 + 0.645637 w_k1_DIS_s014
                                 + 0.747678 w_k1_FDX_s014
                                 + 0.903516 w_k1_IBM_s014
                                 + 0.775008 w_k1_INTC_s014
                                 + 0.80482 w_k1_KO_s014
                                 + 0.971678 w_k1_MAT_s014
                                 + 0.843784 w_k1_MSFT_s014
                                 + 1.0191 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.773285 w_k1_ADM_s014
                                 + 0.790336 w_k1_AXP_s014
                                 + 0.755381 w_k1_DIS_s014
                                 + 0.889721 w_k1_FDX_s014
                                 + 0.91508 w_k1_IBM_s014
                                 + 0.839997 w_k1_INTC_s014
                                 + 0.898529 w_k1_KO_s014
                                 + 0.888555 w_k1_MAT_s014
                                 + 0.980678 w_k1_MSFT_s014
                                 + 1.01618 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.33299 w_k1_ADM_s014
                                 + 1.71344 w_k1_AXP_s014
                                 + 1.27421 w_k1_DIS_s014
                                 + 1.23831 w_k1_FDX_s014
                                 + 0.996386 w_k1_IBM_s014
                                 + 1.18533 w_k1_INTC_s014
                                 + 1.18898 w_k1_KO_s014 + 1.36017 w_k1_MAT_s014
                                 + 1.16895 w_k1_MSFT_s014 + 1.1527 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.970844 w_k1_ADM_s014
                                 + 1.36162 w_k1_AXP_s014
                                 + 1.32288 w_k1_DIS_s014
                                 + 1.21262 w_k1_FDX_s014
                                 + 1.34904 w_k1_IBM_s014
                                 + 1.2374 w_k1_INTC_s014 + 1.22525 w_k1_KO_s014
                                 + 1.08284 w_k1_MAT_s014
                                 + 1.06029 w_k1_MSFT_s014
                                 + 1.16603 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.822268 w_k1_ADM_s015
                                 + 0.706302 w_k1_AXP_s015
                                 + 0.590471 w_k1_DIS_s015
                                 + 0.803737 w_k1_FDX_s015
                                 + 0.660254 w_k1_IBM_s015
                                 + 0.916073 w_k1_INTC_s015
                                 + 0.659259 w_k1_KO_s015
                                 + 0.485139 w_k1_MAT_s015
                                 + 0.81164 w_k1_MSFT_s015
                                 + 0.866844 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.986639 w_k1_ADM_s015
                                 + 0.895757 w_k1_AXP_s015
                                 + 1.10211 w_k1_DIS_s015
                                 + 1.12437 w_k1_FDX_s015
                                 + 0.769752 w_k1_IBM_s015
                                 + 0.968932 w_k1_INTC_s015
                                 + 0.830074 w_k1_KO_s015
                                 + 0.434419 w_k1_MAT_s015
                                 + 1.03908 w_k1_MSFT_s015
                                 + 0.960275 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.811182 w_k1_ADM_s015
                                 + 0.914807 w_k1_AXP_s015
                                 + 0.835948 w_k1_DIS_s015
                                 + 0.942491 w_k1_FDX_s015
                                 + 0.80923 w_k1_IBM_s015
                                 + 1.01783 w_k1_INTC_s015
                                 + 0.754039 w_k1_KO_s015
                                 + 0.768307 w_k1_MAT_s015
                                 + 1.04457 w_k1_MSFT_s015
                                 + 0.964154 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.817364 w_k1_ADM_s015
                                 + 0.723661 w_k1_AXP_s015
                                 + 0.805813 w_k1_DIS_s015
                                 + 0.849678 w_k1_FDX_s015
                                 + 1.05756 w_k1_IBM_s015
                                 + 0.812586 w_k1_INTC_s015
                                 + 1.02783 w_k1_KO_s015 + 1.13386 w_k1_MAT_s015
                                 + 0.950028 w_k1_MSFT_s015
                                 + 1.02592 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.25585 w_k1_ADM_s015
                                 + 1.18644 w_k1_AXP_s015 + 1.1172 w_k1_DIS_s015
                                 + 1.0953 w_k1_FDX_s015 + 1.51357 w_k1_IBM_s015
                                 + 0.966759 w_k1_INTC_s015
                                 + 1.29021 w_k1_KO_s015 + 1.39664 w_k1_MAT_s015
                                 + 1.02653 w_k1_MSFT_s015
                                 + 1.28684 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.23876 w_k1_ADM_s015
                                 + 1.62128 w_k1_AXP_s015
                                 + 1.41939 w_k1_DIS_s015
                                 + 1.38703 w_k1_FDX_s015
                                 + 0.929455 w_k1_IBM_s015
                                 + 0.83995 w_k1_INTC_s015
                                 + 1.02608 w_k1_KO_s015 + 1.23124 w_k1_MAT_s015
                                 + 1.12621 w_k1_MSFT_s015
                                 + 1.29025 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.75754 w_k1_ADM_s015
                                 + 0.71734 w_k1_AXP_s015
                                 + 0.538491 w_k1_DIS_s015
                                 + 0.67456 w_k1_FDX_s015
                                 + 0.854259 w_k1_IBM_s015
                                 + 0.629763 w_k1_INTC_s015
                                 + 0.75388 w_k1_KO_s015 + 0.8931 w_k1_MAT_s015
                                 + 0.811365 w_k1_MSFT_s015
                                 + 0.925859 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.10589 w_k1_ADM_s015
                                 + 1.18716 w_k1_AXP_s015
                                 + 1.24946 w_k1_DIS_s015
                                 + 1.08692 w_k1_FDX_s015
                                 + 0.95204 w_k1_IBM_s015
                                 + 1.08557 w_k1_INTC_s015
                                 + 1.07995 w_k1_KO_s015
                                 + 0.659496 w_k1_MAT_s015
                                 + 1.09999 w_k1_MSFT_s015
                                 + 1.08287 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.910428 w_k1_ADM_s016
                                 + 0.935041 w_k1_AXP_s016
                                 + 1.04647 w_k1_DIS_s016
                                 + 1.06936 w_k1_FDX_s016
                                 + 0.709884 w_k1_IBM_s016
                                 + 1.07292 w_k1_INTC_s016
                                 + 0.918708 w_k1_KO_s016
                                 + 0.949839 w_k1_MAT_s016
                                 + 1.13242 w_k1_MSFT_s016
                                 + 1.01054 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.08608 w_k1_ADM_s016
                                 + 0.838846 w_k1_AXP_s016
                                 + 0.986672 w_k1_DIS_s016
                                 + 0.933052 w_k1_FDX_s016
                                 + 1.52068 w_k1_IBM_s016
                                 + 0.955959 w_k1_INTC_s016
                                 + 1.29223 w_k1_KO_s016 + 1.26875 w_k1_MAT_s016
                                 + 1.00909 w_k1_MSFT_s016
                                 + 1.07697 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.11574 w_k1_ADM_s016
                                 + 1.13401 w_k1_AXP_s016
                                 + 1.18806 w_k1_DIS_s016 + 1.2525 w_k1_FDX_s016
                                 + 0.809036 w_k1_IBM_s016
                                 + 1.23548 w_k1_INTC_s016
                                 + 0.967347 w_k1_KO_s016
                                 + 1.00859 w_k1_MAT_s016
                                 + 1.20907 w_k1_MSFT_s016
                                 + 1.02481 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.26984 w_k1_ADM_s016
                                 + 1.11222 w_k1_AXP_s016
                                 + 1.57443 w_k1_DIS_s016
                                 + 1.32373 w_k1_FDX_s016
                                 + 0.989033 w_k1_IBM_s016
                                 + 1.08534 w_k1_INTC_s016
                                 + 1.27807 w_k1_KO_s016 + 1.1075 w_k1_MAT_s016
                                 + 1.17741 w_k1_MSFT_s016
                                 + 1.21241 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.82926 w_k1_ADM_s016
                                 + 0.675426 w_k1_AXP_s016
                                 + 0.912907 w_k1_DIS_s016
                                 + 0.865692 w_k1_FDX_s016
                                 + 0.852733 w_k1_IBM_s016
                                 + 0.76538 w_k1_INTC_s016
                                 + 0.873534 w_k1_KO_s016
                                 + 0.736507 w_k1_MAT_s016
                                 + 0.89266 w_k1_MSFT_s016
                                 + 0.840451 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.991195 w_k1_ADM_s016
                                 + 0.877078 w_k1_AXP_s016
                                 + 1.06271 w_k1_DIS_s016
                                 + 1.02505 w_k1_FDX_s016
                                 + 1.00793 w_k1_IBM_s016
                                 + 0.963106 w_k1_INTC_s016
                                 + 1.05997 w_k1_KO_s016 + 1.17094 w_k1_MAT_s016
                                 + 1.01715 w_k1_MSFT_s016
                                 + 1.05729 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.690879 w_k1_ADM_s016
                                 + 0.652838 w_k1_AXP_s016
                                 + 0.566879 w_k1_DIS_s016
                                 + 0.747845 w_k1_FDX_s016
                                 + 0.707828 w_k1_IBM_s016
                                 + 0.700115 w_k1_INTC_s016
                                 + 0.700076 w_k1_KO_s016
                                 + 0.757579 w_k1_MAT_s016
                                 + 0.73469 w_k1_MSFT_s016
                                 + 0.834036 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.04593 w_k1_ADM_s016
                                 + 1.21185 w_k1_AXP_s016 + 1.4604 w_k1_DIS_s016
                                 + 1.26197 w_k1_FDX_s016
                                 + 1.28338 w_k1_IBM_s016
                                 + 1.25402 w_k1_INTC_s016
                                 + 1.40407 w_k1_KO_s016 + 1.22436 w_k1_MAT_s016
                                 + 1.16526 w_k1_MSFT_s016
                                 + 1.13387 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.30422 w_k1_ADM_s017
                                 + 1.42786 w_k1_AXP_s017
                                 + 1.44991 w_k1_DIS_s017
                                 + 1.41711 w_k1_FDX_s017
                                 + 0.925945 w_k1_IBM_s017
                                 + 1.21555 w_k1_INTC_s017
                                 + 1.03656 w_k1_KO_s017 + 1.09038 w_k1_MAT_s017
                                 + 1.18976 w_k1_MSFT_s017 + 1.0095 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.26588 w_k1_ADM_s017
                                 + 0.85959 w_k1_AXP_s017
                                 + 1.03402 w_k1_DIS_s017
                                 + 0.969349 w_k1_FDX_s017
                                 + 1.43707 w_k1_IBM_s017
                                 + 0.740751 w_k1_INTC_s017
                                 + 1.10207 w_k1_KO_s017 + 1.76718 w_k1_MAT_s017
                                 + 0.98063 w_k1_MSFT_s017
                                 + 0.915737 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.4141 w_k1_ADM_s017
                                 + 1.2126 w_k1_AXP_s017 + 1.21252 w_k1_DIS_s017
                                 + 1.22343 w_k1_FDX_s017
                                 + 0.99168 w_k1_IBM_s017
                                 + 1.10487 w_k1_INTC_s017
                                 + 1.01409 w_k1_KO_s017 + 1.15293 w_k1_MAT_s017
                                 + 1.04231 w_k1_MSFT_s017
                                 + 0.957056 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.794489 w_k1_ADM_s017
                                 + 0.692173 w_k1_AXP_s017
                                 + 0.653332 w_k1_DIS_s017
                                 + 0.88337 w_k1_FDX_s017
                                 + 0.489974 w_k1_IBM_s017
                                 + 0.889378 w_k1_INTC_s017
                                 + 0.627937 w_k1_KO_s017
                                 + 0.533528 w_k1_MAT_s017
                                 + 0.845754 w_k1_MSFT_s017
                                 + 0.771913 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.41135 w_k1_ADM_s017
                                 + 1.12171 w_k1_AXP_s017
                                 + 1.03108 w_k1_DIS_s017
                                 + 1.13532 w_k1_FDX_s017
                                 + 0.809692 w_k1_IBM_s017
                                 + 0.869944 w_k1_INTC_s017
                                 + 0.938818 w_k1_KO_s017
                                 + 1.25317 w_k1_MAT_s017
                                 + 0.967272 w_k1_MSFT_s017
                                 + 0.927762 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.20481 w_k1_ADM_s017
                                 + 0.976853 w_k1_AXP_s017
                                 + 0.838413 w_k1_DIS_s017
                                 + 1.10289 w_k1_FDX_s017
                                 + 0.599716 w_k1_IBM_s017
                                 + 0.873612 w_k1_INTC_s017
                                 + 0.775125 w_k1_KO_s017
                                 + 0.988065 w_k1_MAT_s017
                                 + 0.983021 w_k1_MSFT_s017
                                 + 0.829366 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.913251 w_k1_ADM_s017
                                 + 0.772852 w_k1_AXP_s017
                                 + 0.761086 w_k1_DIS_s017
                                 + 0.854538 w_k1_FDX_s017
                                 + 0.898533 w_k1_IBM_s017
                                 + 0.709794 w_k1_INTC_s017
                                 + 0.828276 w_k1_KO_s017
                                 + 0.871852 w_k1_MAT_s017
                                 + 0.733587 w_k1_MSFT_s017
                                 + 0.861209 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.977172 w_k1_ADM_s017
                                 + 0.928029 w_k1_AXP_s017
                                 + 0.63379 w_k1_DIS_s017
                                 + 0.757385 w_k1_FDX_s017
                                 + 1.13624 w_k1_IBM_s017
                                 + 0.616864 w_k1_INTC_s017
                                 + 0.873856 w_k1_KO_s017 + 1.6267 w_k1_MAT_s017
                                 + 0.784263 w_k1_MSFT_s017
                                 + 0.827656 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.27641 w_k1_ADM_s100
                                 + 1.53653 w_k1_AXP_s100
                                 + 1.43232 w_k1_DIS_s100
                                 + 1.36283 w_k1_FDX_s100
                                 + 1.25189 w_k1_IBM_s100
                                 + 1.37834 w_k1_INTC_s100
                                 + 1.22423 w_k1_KO_s100 + 1.11442 w_k1_MAT_s100
                                 + 1.31622 w_k1_MSFT_s100
                                 + 1.17045 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.850997 w_k1_ADM_s100
                                 + 0.861341 w_k1_AXP_s100
                                 + 0.951746 w_k1_DIS_s100
                                 + 0.906508 w_k1_FDX_s100
                                 + 0.984163 w_k1_IBM_s100
                                 + 0.923336 w_k1_INTC_s100
                                 + 0.941666 w_k1_KO_s100
                                 + 0.704535 w_k1_MAT_s100
                                 + 0.881474 w_k1_MSFT_s100
                                 + 0.920748 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.852104 w_k1_ADM_s100
                                 + 0.739318 w_k1_AXP_s100
                                 + 0.699712 w_k1_DIS_s100
                                 + 0.860446 w_k1_FDX_s100
                                 + 0.687241 w_k1_IBM_s100
                                 + 0.941844 w_k1_INTC_s100
                                 + 0.707312 w_k1_KO_s100
                                 + 0.653132 w_k1_MAT_s100
                                 + 0.881182 w_k1_MSFT_s100
                                 + 0.875395 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.07537 w_k1_ADM_s100
                                 + 1.12985 w_k1_AXP_s100
                                 + 1.13886 w_k1_DIS_s100
                                 + 1.16683 w_k1_FDX_s100
                                 + 0.854196 w_k1_IBM_s100
                                 + 1.27513 w_k1_INTC_s100
                                 + 0.943353 w_k1_KO_s100
                                 + 0.76873 w_k1_MAT_s100
                                 + 1.13538 w_k1_MSFT_s100
                                 + 0.938403 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.925996 w_k1_ADM_s100
                                 + 1.17848 w_k1_AXP_s100
                                 + 0.964022 w_k1_DIS_s100
                                 + 1.03158 w_k1_FDX_s100 + 1.1902 w_k1_IBM_s100
                                 + 0.98969 w_k1_INTC_s100
                                 + 0.897305 w_k1_KO_s100 + 0.9917 w_k1_MAT_s100
                                 + 1.00554 w_k1_MSFT_s100
                                 + 0.923749 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.932891 w_k1_ADM_s100
                                 + 0.884967 w_k1_AXP_s100
                                 + 0.570777 w_k1_DIS_s100
                                 + 0.685492 w_k1_FDX_s100
                                 + 1.17805 w_k1_IBM_s100
                                 + 0.831444 w_k1_INTC_s100
                                 + 0.859782 w_k1_KO_s100
                                 + 0.902121 w_k1_MAT_s100
                                 + 0.752459 w_k1_MSFT_s100
                                 + 0.801343 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.692275 w_k1_ADM_s100
                                 + 0.619721 w_k1_AXP_s100
                                 + 0.509261 w_k1_DIS_s100
                                 + 0.637027 w_k1_FDX_s100
                                 + 0.900527 w_k1_IBM_s100
                                 + 0.730525 w_k1_INTC_s100
                                 + 0.710305 w_k1_KO_s100
                                 + 0.592428 w_k1_MAT_s100
                                 + 0.726033 w_k1_MSFT_s100
                                 + 0.73184 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.758946 w_k1_ADM_s100
                                 + 0.73957 w_k1_AXP_s100
                                 + 1.00908 w_k1_DIS_s100
                                 + 0.859349 w_k1_FDX_s100
                                 + 1.4676 w_k1_IBM_s100
                                 + 1.13821 w_k1_INTC_s100
                                 + 1.14313 w_k1_KO_s100 + 1.09325 w_k1_MAT_s100
                                 + 1.00958 w_k1_MSFT_s100
                                 + 0.941238 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.928391 w_k1_ADM_s101
                                 + 0.789059 w_k1_AXP_s101
                                 + 0.947517 w_k1_DIS_s101
                                 + 0.921256 w_k1_FDX_s101
                                 + 1.12907 w_k1_IBM_s101
                                 + 0.912852 w_k1_INTC_s101
                                 + 1.03663 w_k1_KO_s101
                                 + 0.691781 w_k1_MAT_s101
                                 + 1.02247 w_k1_MSFT_s101
                                 + 0.974342 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.889604 w_k1_ADM_s101
                                 + 0.784721 w_k1_AXP_s101
                                 + 0.809603 w_k1_DIS_s101
                                 + 0.749639 w_k1_FDX_s101
                                 + 1.40497 w_k1_IBM_s101
                                 + 0.733096 w_k1_INTC_s101
                                 + 1.19376 w_k1_KO_s101 + 1.30536 w_k1_MAT_s101
                                 + 0.904435 w_k1_MSFT_s101
                                 + 0.848578 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.855754 w_k1_ADM_s101
                                 + 0.743022 w_k1_AXP_s101
                                 + 0.624956 w_k1_DIS_s101
                                 + 0.742024 w_k1_FDX_s101
                                 + 0.955822 w_k1_IBM_s101
                                 + 0.785706 w_k1_INTC_s101
                                 + 0.849753 w_k1_KO_s101
                                 + 0.740042 w_k1_MAT_s101
                                 + 0.826621 w_k1_MSFT_s101
                                 + 0.870404 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.795663 w_k1_ADM_s101
                                 + 0.764827 w_k1_AXP_s101
                                 + 0.994919 w_k1_DIS_s101
                                 + 0.934346 w_k1_FDX_s101
                                 + 0.913381 w_k1_IBM_s101
                                 + 0.932815 w_k1_INTC_s101
                                 + 0.987001 w_k1_KO_s101
                                 + 0.673274 w_k1_MAT_s101
                                 + 1.12928 w_k1_MSFT_s101
                                 + 0.992035 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.0005 w_k1_ADM_s101
                                 + 1.31806 w_k1_AXP_s101 + 1.0697 w_k1_DIS_s101
                                 + 1.01496 w_k1_FDX_s101
                                 + 1.42699 w_k1_IBM_s101
                                 + 0.953802 w_k1_INTC_s101
                                 + 1.19611 w_k1_KO_s101 + 1.23732 w_k1_MAT_s101
                                 + 1.02288 w_k1_MSFT_s101
                                 + 1.10159 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.06204 w_k1_ADM_s101
                                 + 0.706611 w_k1_AXP_s101
                                 + 0.831098 w_k1_DIS_s101
                                 + 0.859634 w_k1_FDX_s101
                                 + 0.753562 w_k1_IBM_s101
                                 + 0.831963 w_k1_INTC_s101
                                 + 0.897692 w_k1_KO_s101
                                 + 0.703726 w_k1_MAT_s101
                                 + 1.03418 w_k1_MSFT_s101
                                 + 0.912667 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.28325 w_k1_ADM_s101
                                 + 1.58404 w_k1_AXP_s101
                                 + 1.52787 w_k1_DIS_s101
                                 + 1.37426 w_k1_FDX_s101
                                 + 1.04967 w_k1_IBM_s101
                                 + 1.08574 w_k1_INTC_s101 + 1.2439 w_k1_KO_s101
                                 + 1.18056 w_k1_MAT_s101
                                 + 1.48591 w_k1_MSFT_s101 + 1.2081 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.15614 w_k1_ADM_s101
                                 + 0.974228 w_k1_AXP_s101
                                 + 1.06744 w_k1_DIS_s101
                                 + 1.04902 w_k1_FDX_s101
                                 + 0.809222 w_k1_IBM_s101
                                 + 0.944168 w_k1_INTC_s101
                                 + 0.877319 w_k1_KO_s101
                                 + 1.03758 w_k1_MAT_s101
                                 + 1.20159 w_k1_MSFT_s101
                                 + 1.01814 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.980428 w_k1_ADM_s102
                                 + 0.944298 w_k1_AXP_s102
                                 + 0.997339 w_k1_DIS_s102
                                 + 0.960538 w_k1_FDX_s102
                                 + 0.706616 w_k1_IBM_s102
                                 + 1.16078 w_k1_INTC_s102
                                 + 0.850677 w_k1_KO_s102
                                 + 0.581318 w_k1_MAT_s102
                                 + 0.928013 w_k1_MSFT_s102
                                 + 0.944238 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.04649 w_k1_ADM_s102
                                 + 0.823984 w_k1_AXP_s102
                                 + 0.724338 w_k1_DIS_s102
                                 + 0.946033 w_k1_FDX_s102
                                 + 0.557867 w_k1_IBM_s102
                                 + 0.906737 w_k1_INTC_s102
                                 + 0.729374 w_k1_KO_s102
                                 + 0.658609 w_k1_MAT_s102
                                 + 0.873852 w_k1_MSFT_s102
                                 + 0.928099 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.2528 w_k1_ADM_s102
                                 + 1.4314 w_k1_AXP_s102 + 1.72197 w_k1_DIS_s102
                                 + 1.26257 w_k1_FDX_s102
                                 + 1.28745 w_k1_IBM_s102
                                 + 1.25524 w_k1_INTC_s102 + 1.5191 w_k1_KO_s102
                                 + 1.18676 w_k1_MAT_s102
                                 + 1.3354 w_k1_MSFT_s102 + 1.21764 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.877719 w_k1_ADM_s102
                                 + 0.842308 w_k1_AXP_s102
                                 + 0.824497 w_k1_DIS_s102
                                 + 0.694861 w_k1_FDX_s102
                                 + 1.0736 w_k1_IBM_s102
                                 + 0.737246 w_k1_INTC_s102
                                 + 1.01238 w_k1_KO_s102 + 1.04958 w_k1_MAT_s102
                                 + 0.731736 w_k1_MSFT_s102
                                 + 0.868506 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.832305 w_k1_ADM_s102
                                 + 0.732973 w_k1_AXP_s102
                                 + 0.767908 w_k1_DIS_s102
                                 + 0.750378 w_k1_FDX_s102
                                 + 0.711733 w_k1_IBM_s102
                                 + 0.869126 w_k1_INTC_s102
                                 + 0.843653 w_k1_KO_s102
                                 + 0.640631 w_k1_MAT_s102
                                 + 0.768147 w_k1_MSFT_s102
                                 + 0.772665 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.15508 w_k1_ADM_s102
                                 + 1.21067 w_k1_AXP_s102
                                 + 1.03819 w_k1_DIS_s102
                                 + 0.954899 w_k1_FDX_s102
                                 + 1.41154 w_k1_IBM_s102
                                 + 1.06223 w_k1_INTC_s102
                                 + 1.17186 w_k1_KO_s102 + 1.08091 w_k1_MAT_s102
                                 + 0.949305 w_k1_MSFT_s102
                                 + 1.05934 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.15636 w_k1_ADM_s102
                                 + 1.04392 w_k1_AXP_s102
                                 + 0.902515 w_k1_DIS_s102
                                 + 0.946476 w_k1_FDX_s102
                                 + 0.907492 w_k1_IBM_s102
                                 + 1.04074 w_k1_INTC_s102
                                 + 0.988938 w_k1_KO_s102
                                 + 0.962051 w_k1_MAT_s102
                                 + 0.915641 w_k1_MSFT_s102
                                 + 1.00348 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.11251 w_k1_ADM_s102
                                 + 1.30417 w_k1_AXP_s102
                                 + 1.25241 w_k1_DIS_s102 + 1.2229 w_k1_FDX_s102
                                 + 0.765245 w_k1_IBM_s102
                                 + 1.24105 w_k1_INTC_s102
                                 + 1.13109 w_k1_KO_s102
                                 + 0.585034 w_k1_MAT_s102
                                 + 1.20064 w_k1_MSFT_s102
                                 + 1.21358 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.61051 w_k1_ADM_s103
                                 + 0.970181 w_k1_AXP_s103
                                 + 0.826455 w_k1_DIS_s103
                                 + 0.862378 w_k1_FDX_s103
                                 + 0.964179 w_k1_IBM_s103
                                 + 0.85349 w_k1_INTC_s103
                                 + 0.874496 w_k1_KO_s103
                                 + 1.12278 w_k1_MAT_s103
                                 + 0.87332 w_k1_MSFT_s103
                                 + 0.84071 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.862073 w_k1_ADM_s103
                                 + 1.07133 w_k1_AXP_s103
                                 + 0.885195 w_k1_DIS_s103
                                 + 0.875469 w_k1_FDX_s103
                                 + 1.16724 w_k1_IBM_s103
                                 + 1.06101 w_k1_INTC_s103
                                 + 0.988065 w_k1_KO_s103
                                 + 1.57245 w_k1_MAT_s103
                                 + 0.830834 w_k1_MSFT_s103
                                 + 0.953614 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.18918 w_k1_ADM_s103
                                 + 1.59612 w_k1_AXP_s103
                                 + 1.57108 w_k1_DIS_s103
                                 + 1.28177 w_k1_FDX_s103
                                 + 1.33277 w_k1_IBM_s103
                                 + 1.39025 w_k1_INTC_s103
                                 + 1.34808 w_k1_KO_s103 + 1.20137 w_k1_MAT_s103
                                 + 1.21439 w_k1_MSFT_s103
                                 + 1.26274 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.03391 w_k1_ADM_s103
                                 + 1.19564 w_k1_AXP_s103
                                 + 0.938992 w_k1_DIS_s103
                                 + 0.951125 w_k1_FDX_s103
                                 + 1.63593 w_k1_IBM_s103
                                 + 1.08297 w_k1_INTC_s103
                                 + 1.17368 w_k1_KO_s103 + 1.63263 w_k1_MAT_s103
                                 + 0.92271 w_k1_MSFT_s103
                                 + 0.967973 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.979718 w_k1_ADM_s103
                                 + 1.31492 w_k1_AXP_s103
                                 + 1.15729 w_k1_DIS_s103
                                 + 1.15583 w_k1_FDX_s103
                                 + 0.744197 w_k1_IBM_s103
                                 + 1.10212 w_k1_INTC_s103
                                 + 0.946397 w_k1_KO_s103
                                 + 0.868358 w_k1_MAT_s103
                                 + 1.0333 w_k1_MSFT_s103 + 0.93537 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.28222 w_k1_ADM_s103
                                 + 1.41954 w_k1_AXP_s103
                                 + 0.898468 w_k1_DIS_s103
                                 + 1.05593 w_k1_FDX_s103
                                 + 0.880386 w_k1_IBM_s103
                                 + 1.00887 w_k1_INTC_s103
                                 + 0.881277 w_k1_KO_s103
                                 + 0.980978 w_k1_MAT_s103
                                 + 0.945201 w_k1_MSFT_s103
                                 + 1.01339 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.01564 w_k1_ADM_s103
                                 + 1.14463 w_k1_AXP_s103
                                 + 1.22212 w_k1_DIS_s103
                                 + 1.01029 w_k1_FDX_s103
                                 + 1.01741 w_k1_IBM_s103
                                 + 1.04779 w_k1_INTC_s103
                                 + 1.14559 w_k1_KO_s103 + 1.22003 w_k1_MAT_s103
                                 + 1.1302 w_k1_MSFT_s103
                                 + 0.990495 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.687736 w_k1_ADM_s103
                                 + 0.938342 w_k1_AXP_s103
                                 + 0.720773 w_k1_DIS_s103
                                 + 0.821206 w_k1_FDX_s103
                                 + 0.692288 w_k1_IBM_s103
                                 + 0.761664 w_k1_INTC_s103
                                 + 0.774717 w_k1_KO_s103
                                 + 0.920294 w_k1_MAT_s103
                                 + 0.838726 w_k1_MSFT_s103
                                 + 0.832102 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.39803 w_k1_ADM_s104
                                 + 1.71526 w_k1_AXP_s104
                                 + 1.34017 w_k1_DIS_s104
                                 + 1.18568 w_k1_FDX_s104
                                 + 1.47881 w_k1_IBM_s104
                                 + 1.31325 w_k1_INTC_s104
                                 + 1.31332 w_k1_KO_s104 + 1.54006 w_k1_MAT_s104
                                 + 1.09984 w_k1_MSFT_s104
                                 + 1.10215 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.01213 w_k1_ADM_s104
                                 + 0.995923 w_k1_AXP_s104
                                 + 0.937608 w_k1_DIS_s104
                                 + 0.968182 w_k1_FDX_s104
                                 + 0.855331 w_k1_IBM_s104
                                 + 0.970553 w_k1_INTC_s104
                                 + 0.916483 w_k1_KO_s104
                                 + 0.660977 w_k1_MAT_s104
                                 + 0.865715 w_k1_MSFT_s104
                                 + 1.06996 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.09812 w_k1_ADM_s104
                                 + 0.772342 w_k1_AXP_s104
                                 + 1.04489 w_k1_DIS_s104
                                 + 0.947922 w_k1_FDX_s104
                                 + 0.78209 w_k1_IBM_s104
                                 + 0.950486 w_k1_INTC_s104
                                 + 1.05384 w_k1_KO_s104
                                 + 0.717971 w_k1_MAT_s104
                                 + 1.10381 w_k1_MSFT_s104
                                 + 1.03481 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.749092 w_k1_ADM_s104
                                 + 0.917554 w_k1_AXP_s104
                                 + 1.05582 w_k1_DIS_s104
                                 + 0.890646 w_k1_FDX_s104
                                 + 1.43463 w_k1_IBM_s104
                                 + 0.995873 w_k1_INTC_s104
                                 + 1.08988 w_k1_KO_s104 + 1.23627 w_k1_MAT_s104
                                 + 1.02437 w_k1_MSFT_s104 + 1.0568 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.838778 w_k1_ADM_s104
                                 + 0.902935 w_k1_AXP_s104
                                 + 0.621579 w_k1_DIS_s104
                                 + 0.794785 w_k1_FDX_s104
                                 + 0.90646 w_k1_IBM_s104
                                 + 0.927801 w_k1_INTC_s104
                                 + 0.798343 w_k1_KO_s104
                                 + 0.849746 w_k1_MAT_s104
                                 + 0.808483 w_k1_MSFT_s104
                                 + 0.904203 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.774975 w_k1_ADM_s104
                                 + 1.08155 w_k1_AXP_s104
                                 + 0.78608 w_k1_DIS_s104
                                 + 0.890758 w_k1_FDX_s104
                                 + 1.29825 w_k1_IBM_s104
                                 + 1.01747 w_k1_INTC_s104
                                 + 1.02854 w_k1_KO_s104 + 1.35778 w_k1_MAT_s104
                                 + 0.946579 w_k1_MSFT_s104
                                 + 1.08986 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.43833 w_k1_ADM_s104
                                 + 1.43134 w_k1_AXP_s104
                                 + 1.39303 w_k1_DIS_s104
                                 + 1.32903 w_k1_FDX_s104
                                 + 1.03661 w_k1_IBM_s104
                                 + 1.32438 w_k1_INTC_s104
                                 + 1.31084 w_k1_KO_s104
                                 + 0.763425 w_k1_MAT_s104
                                 + 1.2156 w_k1_MSFT_s104 + 1.25978 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.04798 w_k1_ADM_s104
                                 + 1.22509 w_k1_AXP_s104
                                 + 1.02434 w_k1_DIS_s104
                                 + 1.14304 w_k1_FDX_s104
                                 + 0.788835 w_k1_IBM_s104
                                 + 1.11733 w_k1_INTC_s104
                                 + 0.771064 w_k1_KO_s104
                                 + 0.776217 w_k1_MAT_s104
                                 + 1.04111 w_k1_MSFT_s104
                                 + 1.01674 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.17363 w_k1_ADM_s105
                                 + 1.10643 w_k1_AXP_s105
                                 + 1.10081 w_k1_DIS_s105
                                 + 1.10246 w_k1_FDX_s105
                                 + 1.01352 w_k1_IBM_s105
                                 + 1.07893 w_k1_INTC_s105
                                 + 1.16591 w_k1_KO_s105 + 1.02097 w_k1_MAT_s105
                                 + 1.1955 w_k1_MSFT_s105 + 1.14177 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.40403 w_k1_ADM_s105
                                 + 1.51596 w_k1_AXP_s105
                                 + 1.23415 w_k1_DIS_s105
                                 + 1.11731 w_k1_FDX_s105
                                 + 1.55211 w_k1_IBM_s105
                                 + 1.09116 w_k1_INTC_s105
                                 + 1.36056 w_k1_KO_s105 + 2.10846 w_k1_MAT_s105
                                 + 1.10384 w_k1_MSFT_s105
                                 + 1.11007 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.20484 w_k1_ADM_s105
                                 + 0.823652 w_k1_AXP_s105
                                 + 1.12358 w_k1_DIS_s105
                                 + 0.928258 w_k1_FDX_s105
                                 + 1.53844 w_k1_IBM_s105
                                 + 0.830225 w_k1_INTC_s105
                                 + 1.1523 w_k1_KO_s105 + 1.51636 w_k1_MAT_s105
                                 + 1.09083 w_k1_MSFT_s105
                                 + 0.982173 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.40291 w_k1_ADM_s105
                                 + 1.50669 w_k1_AXP_s105
                                 + 1.36974 w_k1_DIS_s105
                                 + 1.47434 w_k1_FDX_s105
                                 + 0.870557 w_k1_IBM_s105
                                 + 1.2585 w_k1_INTC_s105
                                 + 0.984548 w_k1_KO_s105
                                 + 0.847074 w_k1_MAT_s105
                                 + 1.26365 w_k1_MSFT_s105
                                 + 0.984625 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.742025 w_k1_ADM_s105
                                 + 0.972124 w_k1_AXP_s105
                                 + 0.69467 w_k1_DIS_s105
                                 + 0.759408 w_k1_FDX_s105
                                 + 1.2756 w_k1_IBM_s105
                                 + 1.01018 w_k1_INTC_s105
                                 + 0.894061 w_k1_KO_s105
                                 + 1.35665 w_k1_MAT_s105
                                 + 0.803809 w_k1_MSFT_s105
                                 + 0.819484 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.01766 w_k1_ADM_s105
                                 + 0.72495 w_k1_AXP_s105
                                 + 0.644688 w_k1_DIS_s105
                                 + 0.791123 w_k1_FDX_s105
                                 + 0.733411 w_k1_IBM_s105
                                 + 0.914123 w_k1_INTC_s105
                                 + 0.682657 w_k1_KO_s105
                                 + 0.487543 w_k1_MAT_s105
                                 + 0.909655 w_k1_MSFT_s105
                                 + 0.772199 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.845463 w_k1_ADM_s105
                                 + 0.844438 w_k1_AXP_s105
                                 + 0.781524 w_k1_DIS_s105
                                 + 0.83325 w_k1_FDX_s105
                                 + 1.07281 w_k1_IBM_s105
                                 + 1.06198 w_k1_INTC_s105
                                 + 1.01214 w_k1_KO_s105 + 0.79009 w_k1_MAT_s105
                                 + 0.912852 w_k1_MSFT_s105
                                 + 0.835024 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.03719 w_k1_ADM_s105
                                 + 0.804793 w_k1_AXP_s105
                                 + 0.977712 w_k1_DIS_s105
                                 + 0.98161 w_k1_FDX_s105
                                 + 0.748399 w_k1_IBM_s105
                                 + 1.01018 w_k1_INTC_s105
                                 + 0.883213 w_k1_KO_s105
                                 + 0.70905 w_k1_MAT_s105
                                 + 1.16203 w_k1_MSFT_s105
                                 + 0.987061 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.922053 w_k1_ADM_s106
                                 + 0.723479 w_k1_AXP_s106
                                 + 0.871142 w_k1_DIS_s106
                                 + 0.907385 w_k1_FDX_s106
                                 + 0.754595 w_k1_IBM_s106
                                 + 0.967194 w_k1_INTC_s106
                                 + 0.788252 w_k1_KO_s106
                                 + 0.548198 w_k1_MAT_s106
                                 + 0.957124 w_k1_MSFT_s106
                                 + 0.954078 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.14891 w_k1_ADM_s106
                                 + 1.04996 w_k1_AXP_s106
                                 + 1.08727 w_k1_DIS_s106
                                 + 1.07931 w_k1_FDX_s106
                                 + 1.01309 w_k1_IBM_s106
                                 + 1.00282 w_k1_INTC_s106
                                 + 1.00367 w_k1_KO_s106
                                 + 0.785141 w_k1_MAT_s106
                                 + 0.948137 w_k1_MSFT_s106
                                 + 1.03574 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.26952 w_k1_ADM_s106
                                 + 1.65872 w_k1_AXP_s106 + 1.2147 w_k1_DIS_s106
                                 + 1.20516 w_k1_FDX_s106
                                 + 1.22735 w_k1_IBM_s106
                                 + 1.02044 w_k1_INTC_s106
                                 + 1.17082 w_k1_KO_s106 + 1.38611 w_k1_MAT_s106
                                 + 1.0487 w_k1_MSFT_s106 + 1.16831 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.02647 w_k1_ADM_s106
                                 + 1.09055 w_k1_AXP_s106
                                 + 1.31965 w_k1_DIS_s106
                                 + 0.999731 w_k1_FDX_s106
                                 + 1.71987 w_k1_IBM_s106
                                 + 1.10787 w_k1_INTC_s106
                                 + 1.48018 w_k1_KO_s106 + 1.92952 w_k1_MAT_s106
                                 + 1.10421 w_k1_MSFT_s106
                                 + 1.13133 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.05898 w_k1_ADM_s106
                                 + 1.10668 w_k1_AXP_s106
                                 + 1.37739 w_k1_DIS_s106
                                 + 1.32576 w_k1_FDX_s106
                                 + 0.687449 w_k1_IBM_s106
                                 + 1.2685 w_k1_INTC_s106 + 1.03003 w_k1_KO_s106
                                 + 0.663998 w_k1_MAT_s106
                                 + 1.20163 w_k1_MSFT_s106
                                 + 1.11783 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.04079 w_k1_ADM_s106
                                 + 0.824601 w_k1_AXP_s106
                                 + 1.0739 w_k1_DIS_s106
                                 + 0.963652 w_k1_FDX_s106
                                 + 0.665865 w_k1_IBM_s106
                                 + 0.941089 w_k1_INTC_s106
                                 + 0.889932 w_k1_KO_s106
                                 + 0.458907 w_k1_MAT_s106
                                 + 1.03478 w_k1_MSFT_s106
                                 + 0.980285 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.812916 w_k1_ADM_s106
                                 + 0.899418 w_k1_AXP_s106
                                 + 0.648294 w_k1_DIS_s106
                                 + 0.763577 w_k1_FDX_s106
                                 + 0.887814 w_k1_IBM_s106
                                 + 0.915439 w_k1_INTC_s106
                                 + 0.69247 w_k1_KO_s106
                                 + 0.679105 w_k1_MAT_s106
                                 + 0.696858 w_k1_MSFT_s106
                                 + 0.883718 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.8773 w_k1_ADM_s106
                                 + 1.02574 w_k1_AXP_s106
                                 + 1.17181 w_k1_DIS_s106
                                 + 0.95744 w_k1_FDX_s106
                                 + 1.24428 w_k1_IBM_s106
                                 + 1.0601 w_k1_INTC_s106 + 1.23443 w_k1_KO_s106
                                 + 1.08177 w_k1_MAT_s106
                                 + 1.12772 w_k1_MSFT_s106 + 1.0896 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.727384 w_k1_ADM_s107
                                 + 0.81452 w_k1_AXP_s107
                                 + 0.764621 w_k1_DIS_s107
                                 + 0.802961 w_k1_FDX_s107
                                 + 1.08122 w_k1_IBM_s107
                                 + 0.779513 w_k1_INTC_s107
                                 + 1.01443 w_k1_KO_s107 + 1.08713 w_k1_MAT_s107
                                 + 0.875949 w_k1_MSFT_s107
                                 + 0.940716 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.01833 w_k1_ADM_s107
                                 + 1.29223 w_k1_AXP_s107
                                 + 0.95523 w_k1_DIS_s107
                                 + 1.08197 w_k1_FDX_s107
                                 + 0.837607 w_k1_IBM_s107
                                 + 1.12251 w_k1_INTC_s107
                                 + 1.01343 w_k1_KO_s107
                                 + 0.928026 w_k1_MAT_s107
                                 + 0.96592 w_k1_MSFT_s107
                                 + 1.06577 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.743634 w_k1_ADM_s107
                                 + 0.728328 w_k1_AXP_s107
                                 + 1.12445 w_k1_DIS_s107
                                 + 0.996897 w_k1_FDX_s107
                                 + 0.900916 w_k1_IBM_s107
                                 + 0.846768 w_k1_INTC_s107
                                 + 1.10072 w_k1_KO_s107 + 1.01696 w_k1_MAT_s107
                                 + 1.0414 w_k1_MSFT_s107 + 1.01631 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.921397 w_k1_ADM_s107
                                 + 0.959293 w_k1_AXP_s107
                                 + 0.895359 w_k1_DIS_s107
                                 + 0.923628 w_k1_FDX_s107
                                 + 1.21122 w_k1_IBM_s107
                                 + 0.987835 w_k1_INTC_s107
                                 + 1.16871 w_k1_KO_s107 + 1.27075 w_k1_MAT_s107
                                 + 0.943868 w_k1_MSFT_s107
                                 + 1.03732 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.09265 w_k1_ADM_s107
                                 + 1.56369 w_k1_AXP_s107
                                 + 1.45201 w_k1_DIS_s107
                                 + 1.22663 w_k1_FDX_s107
                                 + 1.54608 w_k1_IBM_s107
                                 + 1.08095 w_k1_INTC_s107
                                 + 1.39172 w_k1_KO_s107 + 1.70346 w_k1_MAT_s107
                                 + 1.12907 w_k1_MSFT_s107
                                 + 1.14781 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.720804 w_k1_ADM_s107
                                 + 0.650834 w_k1_AXP_s107
                                 + 0.648013 w_k1_DIS_s107
                                 + 0.7737 w_k1_FDX_s107
                                 + 0.782382 w_k1_IBM_s107
                                 + 0.698981 w_k1_INTC_s107
                                 + 0.719709 w_k1_KO_s107
                                 + 0.632282 w_k1_MAT_s107
                                 + 0.712287 w_k1_MSFT_s107
                                 + 0.784241 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.22625 w_k1_ADM_s107
                                 + 1.25439 w_k1_AXP_s107
                                 + 1.34621 w_k1_DIS_s107
                                 + 1.30787 w_k1_FDX_s107
                                 + 1.16562 w_k1_IBM_s107
                                 + 1.26821 w_k1_INTC_s107
                                 + 1.36301 w_k1_KO_s107 + 1.12086 w_k1_MAT_s107
                                 + 1.25382 w_k1_MSFT_s107
                                 + 1.14607 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.04071 w_k1_ADM_s107
                                 + 1.10302 w_k1_AXP_s107
                                 + 1.02996 w_k1_DIS_s107 + 1.1781 w_k1_FDX_s107
                                 + 0.735782 w_k1_IBM_s107
                                 + 0.950719 w_k1_INTC_s107
                                 + 0.96159 w_k1_KO_s107
                                 + 0.745974 w_k1_MAT_s107
                                 + 1.03491 w_k1_MSFT_s107 + 1.0324 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.995705 w_k1_ADM_s110
                                 + 1.15138 w_k1_AXP_s110
                                 + 1.16072 w_k1_DIS_s110
                                 + 0.980669 w_k1_FDX_s110
                                 + 1.4186 w_k1_IBM_s110
                                 + 1.17763 w_k1_INTC_s110
                                 + 1.27491 w_k1_KO_s110 + 1.56703 w_k1_MAT_s110
                                 + 1.20277 w_k1_MSFT_s110
                                 + 1.15364 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.788327 w_k1_ADM_s110
                                 + 0.903206 w_k1_AXP_s110
                                 + 0.593627 w_k1_DIS_s110
                                 + 0.709244 w_k1_FDX_s110
                                 + 1.26436 w_k1_IBM_s110
                                 + 0.767523 w_k1_INTC_s110
                                 + 0.820527 w_k1_KO_s110
                                 + 1.10724 w_k1_MAT_s110
                                 + 0.750734 w_k1_MSFT_s110
                                 + 0.85264 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.12731 w_k1_ADM_s110
                                 + 1.73392 w_k1_AXP_s110
                                 + 1.39996 w_k1_DIS_s110
                                 + 1.26173 w_k1_FDX_s110
                                 + 1.31815 w_k1_IBM_s110
                                 + 1.1565 w_k1_INTC_s110 + 1.36015 w_k1_KO_s110
                                 + 1.45794 w_k1_MAT_s110
                                 + 1.04984 w_k1_MSFT_s110
                                 + 1.09588 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.17092 w_k1_ADM_s110
                                 + 1.17902 w_k1_AXP_s110
                                 + 1.21213 w_k1_DIS_s110
                                 + 1.26121 w_k1_FDX_s110
                                 + 0.739256 w_k1_IBM_s110
                                 + 1.4604 w_k1_INTC_s110
                                 + 0.919568 w_k1_KO_s110
                                 + 0.605361 w_k1_MAT_s110
                                 + 1.32488 w_k1_MSFT_s110
                                 + 0.992801 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.998905 w_k1_ADM_s110
                                 + 0.996048 w_k1_AXP_s110
                                 + 0.843587 w_k1_DIS_s110
                                 + 0.908713 w_k1_FDX_s110
                                 + 1.13548 w_k1_IBM_s110
                                 + 0.913621 w_k1_INTC_s110
                                 + 0.988542 w_k1_KO_s110
                                 + 0.928853 w_k1_MAT_s110
                                 + 0.939448 w_k1_MSFT_s110
                                 + 0.975719 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.997439 w_k1_ADM_s110
                                 + 1.24901 w_k1_AXP_s110
                                 + 1.05631 w_k1_DIS_s110
                                 + 1.10195 w_k1_FDX_s110
                                 + 0.979235 w_k1_IBM_s110
                                 + 1.16351 w_k1_INTC_s110
                                 + 0.920943 w_k1_KO_s110 + 1.1951 w_k1_MAT_s110
                                 + 1.01033 w_k1_MSFT_s110
                                 + 1.02196 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.02718 w_k1_ADM_s110
                                 + 0.806668 w_k1_AXP_s110
                                 + 0.873687 w_k1_DIS_s110
                                 + 1.01922 w_k1_FDX_s110
                                 + 0.73925 w_k1_IBM_s110
                                 + 1.07159 w_k1_INTC_s110 + 0.8953 w_k1_KO_s110
                                 + 0.879946 w_k1_MAT_s110
                                 + 1.0566 w_k1_MSFT_s110
                                 + 0.982113 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.811336 w_k1_ADM_s110
                                 + 0.832657 w_k1_AXP_s110
                                 + 0.70949 w_k1_DIS_s110
                                 + 0.86006 w_k1_FDX_s110
                                 + 0.792312 w_k1_IBM_s110
                                 + 0.893095 w_k1_INTC_s110
                                 + 0.787007 w_k1_KO_s110
                                 + 0.787556 w_k1_MAT_s110
                                 + 0.78464 w_k1_MSFT_s110
                                 + 0.766647 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.13629 w_k1_ADM_s111
                                 + 1.18242 w_k1_AXP_s111
                                 + 1.14564 w_k1_DIS_s111
                                 + 1.25011 w_k1_FDX_s111
                                 + 0.940245 w_k1_IBM_s111
                                 + 1.12639 w_k1_INTC_s111
                                 + 1.03036 w_k1_KO_s111 + 1.06211 w_k1_MAT_s111
                                 + 1.11483 w_k1_MSFT_s111
                                 + 1.11735 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.719529 w_k1_ADM_s111
                                 + 0.744609 w_k1_AXP_s111
                                 + 0.702321 w_k1_DIS_s111
                                 + 0.820762 w_k1_FDX_s111
                                 + 0.666096 w_k1_IBM_s111
                                 + 0.846561 w_k1_INTC_s111
                                 + 0.734218 w_k1_KO_s111
                                 + 0.651946 w_k1_MAT_s111
                                 + 0.812955 w_k1_MSFT_s111
                                 + 0.881797 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.19988 w_k1_ADM_s111
                                 + 1.24538 w_k1_AXP_s111
                                 + 1.42367 w_k1_DIS_s111
                                 + 1.23849 w_k1_FDX_s111
                                 + 1.39504 w_k1_IBM_s111
                                 + 1.44213 w_k1_INTC_s111
                                 + 1.50492 w_k1_KO_s111 + 1.38731 w_k1_MAT_s111
                                 + 1.34234 w_k1_MSFT_s111
                                 + 1.20402 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.73372 w_k1_ADM_s111
                                 + 0.776681 w_k1_AXP_s111
                                 + 0.799666 w_k1_DIS_s111
                                 + 0.830012 w_k1_FDX_s111
                                 + 0.908398 w_k1_IBM_s111
                                 + 0.963528 w_k1_INTC_s111
                                 + 0.943583 w_k1_KO_s111
                                 + 1.15539 w_k1_MAT_s111
                                 + 0.882491 w_k1_MSFT_s111
                                 + 0.986118 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.12863 w_k1_ADM_s111
                                 + 1.02179 w_k1_AXP_s111
                                 + 0.952587 w_k1_DIS_s111
                                 + 1.08647 w_k1_FDX_s111
                                 + 0.643318 w_k1_IBM_s111
                                 + 0.986386 w_k1_INTC_s111
                                 + 0.854714 w_k1_KO_s111
                                 + 0.692932 w_k1_MAT_s111
                                 + 0.943536 w_k1_MSFT_s111
                                 + 1.01149 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.936809 w_k1_ADM_s111
                                 + 0.802439 w_k1_AXP_s111
                                 + 0.964653 w_k1_DIS_s111
                                 + 0.964428 w_k1_FDX_s111
                                 + 1.02007 w_k1_IBM_s111
                                 + 1.01177 w_k1_INTC_s111
                                 + 0.995604 w_k1_KO_s111
                                 + 0.770564 w_k1_MAT_s111
                                 + 0.98185 w_k1_MSFT_s111
                                 + 1.04569 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.793583 w_k1_ADM_s111
                                 + 0.828119 w_k1_AXP_s111
                                 + 0.704797 w_k1_DIS_s111
                                 + 0.754599 w_k1_FDX_s111
                                 + 1.21916 w_k1_IBM_s111
                                 + 0.762564 w_k1_INTC_s111
                                 + 0.944628 w_k1_KO_s111
                                 + 1.26544 w_k1_MAT_s111
                                 + 0.785772 w_k1_MSFT_s111
                                 + 1.00414 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.675799 w_k1_ADM_s111
                                 + 0.933037 w_k1_AXP_s111
                                 + 0.899185 w_k1_DIS_s111
                                 + 0.837931 w_k1_FDX_s111
                                 + 1.37111 w_k1_IBM_s111
                                 + 0.819787 w_k1_INTC_s111
                                 + 1.11244 w_k1_KO_s111 + 1.8343 w_k1_MAT_s111
                                 + 0.922219 w_k1_MSFT_s111
                                 + 1.05938 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.741919 w_k1_ADM_s112
                                 + 0.709647 w_k1_AXP_s112
                                 + 0.685411 w_k1_DIS_s112
                                 + 0.716473 w_k1_FDX_s112
                                 + 0.769765 w_k1_IBM_s112
                                 + 0.812096 w_k1_INTC_s112
                                 + 0.727751 w_k1_KO_s112
                                 + 0.710576 w_k1_MAT_s112
                                 + 0.758091 w_k1_MSFT_s112
                                 + 0.735289 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.988321 w_k1_ADM_s112
                                 + 1.00316 w_k1_AXP_s112
                                 + 0.881679 w_k1_DIS_s112
                                 + 0.901131 w_k1_FDX_s112
                                 + 1.04274 w_k1_IBM_s112
                                 + 0.853847 w_k1_INTC_s112
                                 + 1.02064 w_k1_KO_s112 + 1.28931 w_k1_MAT_s112
                                 + 0.937967 w_k1_MSFT_s112
                                 + 1.03922 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.33858 w_k1_ADM_s112
                                 + 1.17133 w_k1_AXP_s112
                                 + 1.14297 w_k1_DIS_s112
                                 + 1.03071 w_k1_FDX_s112
                                 + 1.03227 w_k1_IBM_s112
                                 + 0.951422 w_k1_INTC_s112
                                 + 1.02185 w_k1_KO_s112 + 1.12481 w_k1_MAT_s112
                                 + 0.990095 w_k1_MSFT_s112
                                 + 1.09614 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.714948 w_k1_ADM_s112
                                 + 0.68295 w_k1_AXP_s112
                                 + 0.637105 w_k1_DIS_s112
                                 + 0.684509 w_k1_FDX_s112
                                 + 1.05403 w_k1_IBM_s112
                                 + 0.835147 w_k1_INTC_s112
                                 + 0.7938 w_k1_KO_s112 + 0.746805 w_k1_MAT_s112
                                 + 0.742714 w_k1_MSFT_s112
                                 + 0.940386 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.02623 w_k1_ADM_s112
                                 + 1.25907 w_k1_AXP_s112
                                 + 1.02934 w_k1_DIS_s112
                                 + 1.17223 w_k1_FDX_s112
                                 + 0.878967 w_k1_IBM_s112
                                 + 1.23961 w_k1_INTC_s112
                                 + 0.913479 w_k1_KO_s112
                                 + 1.28764 w_k1_MAT_s112
                                 + 1.11001 w_k1_MSFT_s112
                                 + 1.12807 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.06679 w_k1_ADM_s112
                                 + 0.947857 w_k1_AXP_s112
                                 + 1.13514 w_k1_DIS_s112
                                 + 1.06087 w_k1_FDX_s112
                                 + 0.69532 w_k1_IBM_s112
                                 + 1.15693 w_k1_INTC_s112
                                 + 0.993281 w_k1_KO_s112
                                 + 0.539968 w_k1_MAT_s112
                                 + 1.12195 w_k1_MSFT_s112
                                 + 1.03985 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.19676 w_k1_ADM_s112
                                 + 1.38261 w_k1_AXP_s112
                                 + 1.74283 w_k1_DIS_s112 + 1.3619 w_k1_FDX_s112
                                 + 1.20343 w_k1_IBM_s112
                                 + 1.11752 w_k1_INTC_s112
                                 + 1.28838 w_k1_KO_s112 + 1.55638 w_k1_MAT_s112
                                 + 1.36983 w_k1_MSFT_s112
                                 + 1.38628 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.12425 w_k1_ADM_s112
                                 + 1.16042 w_k1_AXP_s112
                                 + 1.06283 w_k1_DIS_s112
                                 + 0.956327 w_k1_FDX_s112
                                 + 1.39593 w_k1_IBM_s112
                                 + 0.950614 w_k1_INTC_s112
                                 + 1.17978 w_k1_KO_s112 + 1.22889 w_k1_MAT_s112
                                 + 1.00011 w_k1_MSFT_s112
                                 + 1.08567 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.23043 w_k1_ADM_s113
                                 + 1.38589 w_k1_AXP_s113
                                 + 1.19891 w_k1_DIS_s113
                                 + 1.23478 w_k1_FDX_s113
                                 + 0.821999 w_k1_IBM_s113
                                 + 1.18224 w_k1_INTC_s113
                                 + 1.02223 w_k1_KO_s113
                                 + 0.785384 w_k1_MAT_s113
                                 + 1.05698 w_k1_MSFT_s113
                                 + 0.987046 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.809046 w_k1_ADM_s113
                                 + 0.914189 w_k1_AXP_s113
                                 + 0.765354 w_k1_DIS_s113
                                 + 1.04639 w_k1_FDX_s113
                                 + 0.575661 w_k1_IBM_s113
                                 + 0.983471 w_k1_INTC_s113
                                 + 0.80246 w_k1_KO_s113
                                 + 0.725464 w_k1_MAT_s113
                                 + 0.965291 w_k1_MSFT_s113
                                 + 0.818318 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.980054 w_k1_ADM_s113
                                 + 0.940774 w_k1_AXP_s113
                                 + 1.04497 w_k1_DIS_s113
                                 + 0.991627 w_k1_FDX_s113
                                 + 1.05894 w_k1_IBM_s113
                                 + 0.855096 w_k1_INTC_s113
                                 + 1.05607 w_k1_KO_s113
                                 + 0.970928 w_k1_MAT_s113
                                 + 0.957411 w_k1_MSFT_s113
                                 + 0.888249 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.773596 w_k1_ADM_s113
                                 + 0.640505 w_k1_AXP_s113
                                 + 0.608783 w_k1_DIS_s113
                                 + 0.620814 w_k1_FDX_s113
                                 + 1.17072 w_k1_IBM_s113
                                 + 0.655794 w_k1_INTC_s113
                                 + 0.813758 w_k1_KO_s113
                                 + 0.983136 w_k1_MAT_s113
                                 + 0.770639 w_k1_MSFT_s113
                                 + 0.820654 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.41105 w_k1_ADM_s113
                                 + 1.02346 w_k1_AXP_s113
                                 + 1.59913 w_k1_DIS_s113
                                 + 1.36567 w_k1_FDX_s113 + 1.195 w_k1_IBM_s113
                                 + 1.10436 w_k1_INTC_s113
                                 + 1.27677 w_k1_KO_s113
                                 + 0.931946 w_k1_MAT_s113
                                 + 1.29583 w_k1_MSFT_s113
                                 + 1.09724 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.80099 w_k1_ADM_s113
                                 + 0.749092 w_k1_AXP_s113
                                 + 0.619676 w_k1_DIS_s113
                                 + 0.750796 w_k1_FDX_s113
                                 + 0.801175 w_k1_IBM_s113
                                 + 0.826973 w_k1_INTC_s113
                                 + 0.754484 w_k1_KO_s113
                                 + 0.747605 w_k1_MAT_s113
                                 + 0.710878 w_k1_MSFT_s113
                                 + 0.715498 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.13623 w_k1_ADM_s113
                                 + 1.487 w_k1_AXP_s113 + 1.40027 w_k1_DIS_s113
                                 + 1.20063 w_k1_FDX_s113
                                 + 1.55218 w_k1_IBM_s113
                                 + 1.02944 w_k1_INTC_s113 + 1.2262 w_k1_KO_s113
                                 + 1.42297 w_k1_MAT_s113
                                 + 1.10167 w_k1_MSFT_s113
                                 + 1.13313 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.913585 w_k1_ADM_s113
                                 + 0.882854 w_k1_AXP_s113
                                 + 0.98815 w_k1_DIS_s113
                                 + 0.989015 w_k1_FDX_s113
                                 + 0.786584 w_k1_IBM_s113
                                 + 1.13443 w_k1_INTC_s113
                                 + 0.906927 w_k1_KO_s113
                                 + 0.623625 w_k1_MAT_s113
                                 + 1.01304 w_k1_MSFT_s113
                                 + 0.858448 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.765022 w_k1_ADM_s114
                                 + 0.679769 w_k1_AXP_s114
                                 + 0.588051 w_k1_DIS_s114
                                 + 0.67606 w_k1_FDX_s114
                                 + 0.867148 w_k1_IBM_s114
                                 + 0.679094 w_k1_INTC_s114
                                 + 0.760888 w_k1_KO_s114
                                 + 0.825139 w_k1_MAT_s114
                                 + 0.64988 w_k1_MSFT_s114
                                 + 0.703395 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.7866 w_k1_ADM_s114
                                 + 0.839892 w_k1_AXP_s114
                                 + 0.871512 w_k1_DIS_s114
                                 + 0.93582 w_k1_FDX_s114
                                 + 0.911664 w_k1_IBM_s114
                                 + 0.856698 w_k1_INTC_s114
                                 + 0.884382 w_k1_KO_s114
                                 + 0.68539 w_k1_MAT_s114
                                 + 0.876827 w_k1_MSFT_s114
                                 + 0.869649 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.911922 w_k1_ADM_s114
                                 + 0.933615 w_k1_AXP_s114
                                 + 0.783901 w_k1_DIS_s114
                                 + 0.78914 w_k1_FDX_s114
                                 + 1.58702 w_k1_IBM_s114
                                 + 0.909342 w_k1_INTC_s114
                                 + 1.02755 w_k1_KO_s114 + 1.26778 w_k1_MAT_s114
                                 + 0.753881 w_k1_MSFT_s114
                                 + 0.840598 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.10544 w_k1_ADM_s114
                                 + 1.0216 w_k1_AXP_s114 + 1.02986 w_k1_DIS_s114
                                 + 1.13719 w_k1_FDX_s114
                                 + 0.827604 w_k1_IBM_s114
                                 + 1.19614 w_k1_INTC_s114
                                 + 1.01513 w_k1_KO_s114
                                 + 0.768854 w_k1_MAT_s114
                                 + 1.00324 w_k1_MSFT_s114
                                 + 0.925157 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.697562 w_k1_ADM_s114
                                 + 0.643509 w_k1_AXP_s114
                                 + 0.793072 w_k1_DIS_s114
                                 + 0.731205 w_k1_FDX_s114
                                 + 1.20051 w_k1_IBM_s114
                                 + 0.881712 w_k1_INTC_s114
                                 + 1.00951 w_k1_KO_s114
                                 + 0.944595 w_k1_MAT_s114
                                 + 0.794087 w_k1_MSFT_s114
                                 + 0.81955 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.812847 w_k1_ADM_s114
                                 + 0.792069 w_k1_AXP_s114
                                 + 0.818211 w_k1_DIS_s114
                                 + 0.892936 w_k1_FDX_s114
                                 + 0.667048 w_k1_IBM_s114
                                 + 0.935455 w_k1_INTC_s114
                                 + 0.83735 w_k1_KO_s114
                                 + 0.419123 w_k1_MAT_s114
                                 + 0.804081 w_k1_MSFT_s114
                                 + 0.829083 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.07186 w_k1_ADM_s114
                                 + 1.157 w_k1_AXP_s114 + 1.5254 w_k1_DIS_s114
                                 + 1.38954 w_k1_FDX_s114
                                 + 0.772428 w_k1_IBM_s114
                                 + 1.21356 w_k1_INTC_s114
                                 + 1.15901 w_k1_KO_s114
                                 + 0.991078 w_k1_MAT_s114
                                 + 1.24949 w_k1_MSFT_s114
                                 + 1.03768 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.25093 w_k1_ADM_s114
                                 + 1.15333 w_k1_AXP_s114
                                 + 1.42639 w_k1_DIS_s114
                                 + 1.19918 w_k1_FDX_s114
                                 + 1.19434 w_k1_IBM_s114
                                 + 1.11663 w_k1_INTC_s114
                                 + 1.34189 w_k1_KO_s114 + 1.35828 w_k1_MAT_s114
                                 + 1.15201 w_k1_MSFT_s114
                                 + 1.07026 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.929323 w_k1_ADM_s115
                                 + 1.3236 w_k1_AXP_s115
                                 + 0.949216 w_k1_DIS_s115
                                 + 0.947084 w_k1_FDX_s115
                                 + 1.47678 w_k1_IBM_s115
                                 + 1.27792 w_k1_INTC_s115
                                 + 1.14181 w_k1_KO_s115 + 1.60893 w_k1_MAT_s115
                                 + 1.01338 w_k1_MSFT_s115
                                 + 1.13458 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.869227 w_k1_ADM_s115
                                 + 0.905276 w_k1_AXP_s115
                                 + 0.85072 w_k1_DIS_s115 + 1.0171 w_k1_FDX_s115
                                 + 0.71756 w_k1_IBM_s115
                                 + 1.08961 w_k1_INTC_s115
                                 + 0.83996 w_k1_KO_s115
                                 + 0.681417 w_k1_MAT_s115
                                 + 0.898865 w_k1_MSFT_s115
                                 + 0.837601 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.1556 w_k1_ADM_s115
                                 + 1.13436 w_k1_AXP_s115 + 1.3061 w_k1_DIS_s115
                                 + 1.21869 w_k1_FDX_s115
                                 + 0.909815 w_k1_IBM_s115
                                 + 1.16367 w_k1_INTC_s115
                                 + 1.16907 w_k1_KO_s115
                                 + 0.801149 w_k1_MAT_s115
                                 + 1.2027 w_k1_MSFT_s115 + 1.11946 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.798321 w_k1_ADM_s115
                                 + 0.834352 w_k1_AXP_s115
                                 + 0.634379 w_k1_DIS_s115
                                 + 0.772227 w_k1_FDX_s115
                                 + 0.812106 w_k1_IBM_s115
                                 + 1.064 w_k1_INTC_s115 + 0.769088 w_k1_KO_s115
                                 + 0.640984 w_k1_MAT_s115
                                 + 0.82188 w_k1_MSFT_s115
                                 + 0.79594 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.894584 w_k1_ADM_s115
                                 + 1.0541 w_k1_AXP_s115
                                 + 0.841715 w_k1_DIS_s115
                                 + 0.902401 w_k1_FDX_s115
                                 + 1.01482 w_k1_IBM_s115
                                 + 0.944312 w_k1_INTC_s115
                                 + 1.0132 w_k1_KO_s115 + 1.26945 w_k1_MAT_s115
                                 + 0.874453 w_k1_MSFT_s115
                                 + 0.921506 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.32279 w_k1_ADM_s115
                                 + 1.65805 w_k1_AXP_s115
                                 + 1.51425 w_k1_DIS_s115
                                 + 1.38084 w_k1_FDX_s115
                                 + 1.15952 w_k1_IBM_s115
                                 + 1.21907 w_k1_INTC_s115
                                 + 1.20523 w_k1_KO_s115 + 1.81446 w_k1_MAT_s115
                                 + 1.12546 w_k1_MSFT_s115
                                 + 1.26017 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.13712 w_k1_ADM_s115
                                 + 1.17854 w_k1_AXP_s115
                                 + 1.21494 w_k1_DIS_s115
                                 + 1.08244 w_k1_FDX_s115
                                 + 1.01391 w_k1_IBM_s115
                                 + 1.14315 w_k1_INTC_s115
                                 + 0.985023 w_k1_KO_s115
                                 + 0.990326 w_k1_MAT_s115
                                 + 1.0048 w_k1_MSFT_s115
                                 + 0.980107 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.88513 w_k1_ADM_s115
                                 + 0.843548 w_k1_AXP_s115
                                 + 0.90564 w_k1_DIS_s115
                                 + 0.923535 w_k1_FDX_s115
                                 + 0.778117 w_k1_IBM_s115
                                 + 0.859302 w_k1_INTC_s115
                                 + 0.812294 w_k1_KO_s115
                                 + 0.799202 w_k1_MAT_s115
                                 + 0.953268 w_k1_MSFT_s115
                                 + 0.96929 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.1192 w_k1_ADM_s116
                                 + 0.942996 w_k1_AXP_s116
                                 + 1.22381 w_k1_DIS_s116
                                 + 1.11964 w_k1_FDX_s116
                                 + 0.981092 w_k1_IBM_s116
                                 + 0.894686 w_k1_INTC_s116
                                 + 0.972513 w_k1_KO_s116
                                 + 0.817377 w_k1_MAT_s116
                                 + 1.09222 w_k1_MSFT_s116
                                 + 1.13244 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.28438 w_k1_ADM_s116
                                 + 1.381 w_k1_AXP_s116 + 1.49621 w_k1_DIS_s116
                                 + 1.13555 w_k1_FDX_s116
                                 + 1.70835 w_k1_IBM_s116
                                 + 1.17816 w_k1_INTC_s116
                                 + 1.44128 w_k1_KO_s116 + 1.82251 w_k1_MAT_s116
                                 + 1.22226 w_k1_MSFT_s116
                                 + 1.39229 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.02066 w_k1_ADM_s116
                                 + 0.864127 w_k1_AXP_s116
                                 + 0.889423 w_k1_DIS_s116
                                 + 0.942611 w_k1_FDX_s116
                                 + 0.817346 w_k1_IBM_s116
                                 + 1.05953 w_k1_INTC_s116
                                 + 0.834947 w_k1_KO_s116
                                 + 0.723192 w_k1_MAT_s116
                                 + 1.06221 w_k1_MSFT_s116
                                 + 1.01368 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00234 w_k1_ADM_s116
                                 + 0.712942 w_k1_AXP_s116
                                 + 0.630998 w_k1_DIS_s116
                                 + 0.722836 w_k1_FDX_s116
                                 + 0.665796 w_k1_IBM_s116
                                 + 0.859695 w_k1_INTC_s116
                                 + 0.795248 w_k1_KO_s116
                                 + 0.613335 w_k1_MAT_s116
                                 + 0.850942 w_k1_MSFT_s116
                                 + 0.923988 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.53775 w_k1_ADM_s116
                                 + 1.4582 w_k1_AXP_s116 + 1.67854 w_k1_DIS_s116
                                 + 1.3553 w_k1_FDX_s116 + 1.11321 w_k1_IBM_s116
                                 + 1.3154 w_k1_INTC_s116 + 1.30207 w_k1_KO_s116
                                 + 1.6 w_k1_MAT_s116 + 1.57098 w_k1_MSFT_s116
                                 + 1.23821 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.38844 w_k1_ADM_s116
                                 + 1.24456 w_k1_AXP_s116
                                 + 0.996796 w_k1_DIS_s116
                                 + 1.16907 w_k1_FDX_s116
                                 + 0.937041 w_k1_IBM_s116
                                 + 1.11762 w_k1_INTC_s116
                                 + 0.953102 w_k1_KO_s116
                                 + 1.11288 w_k1_MAT_s116
                                 + 1.12419 w_k1_MSFT_s116
                                 + 1.12006 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.828034 w_k1_ADM_s116
                                 + 0.850564 w_k1_AXP_s116
                                 + 0.777249 w_k1_DIS_s116
                                 + 0.771054 w_k1_FDX_s116
                                 + 0.927875 w_k1_IBM_s116
                                 + 0.867152 w_k1_INTC_s116
                                 + 0.732568 w_k1_KO_s116
                                 + 0.99153 w_k1_MAT_s116
                                 + 0.848815 w_k1_MSFT_s116
                                 + 0.853487 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.01723 w_k1_ADM_s116
                                 + 1.06962 w_k1_AXP_s116
                                 + 0.953142 w_k1_DIS_s116
                                 + 0.959623 w_k1_FDX_s116
                                 + 1.40826 w_k1_IBM_s116
                                 + 0.8284 w_k1_INTC_s116
                                 + 0.999141 w_k1_KO_s116
                                 + 1.32952 w_k1_MAT_s116
                                 + 0.932574 w_k1_MSFT_s116
                                 + 1.01142 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.35085 w_k1_ADM_s117
                                 + 1.03555 w_k1_AXP_s117
                                 + 1.29256 w_k1_DIS_s117
                                 + 1.19917 w_k1_FDX_s117
                                 + 0.894677 w_k1_IBM_s117
                                 + 0.857115 w_k1_INTC_s117
                                 + 1.01199 w_k1_KO_s117 + 0.9026 w_k1_MAT_s117
                                 + 1.08028 w_k1_MSFT_s117
                                 + 1.09687 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.24123 w_k1_ADM_s117
                                 + 1.37187 w_k1_AXP_s117
                                 + 1.39511 w_k1_DIS_s117
                                 + 1.21005 w_k1_FDX_s117
                                 + 1.65825 w_k1_IBM_s117
                                 + 1.00728 w_k1_INTC_s117
                                 + 1.28998 w_k1_KO_s117 + 1.55598 w_k1_MAT_s117
                                 + 1.10143 w_k1_MSFT_s117
                                 + 1.09798 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.9641 w_k1_ADM_s117
                                 + 0.804571 w_k1_AXP_s117
                                 + 1.00078 w_k1_DIS_s117
                                 + 0.911642 w_k1_FDX_s117
                                 + 0.900249 w_k1_IBM_s117
                                 + 0.898011 w_k1_INTC_s117
                                 + 0.81053 w_k1_KO_s117
                                 + 0.486222 w_k1_MAT_s117
                                 + 0.866405 w_k1_MSFT_s117
                                 + 0.90153 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.21814 w_k1_ADM_s117
                                 + 1.38623 w_k1_AXP_s117
                                 + 1.75216 w_k1_DIS_s117
                                 + 1.41081 w_k1_FDX_s117
                                 + 1.33001 w_k1_IBM_s117
                                 + 0.997039 w_k1_INTC_s117
                                 + 1.18157 w_k1_KO_s117 + 1.54612 w_k1_MAT_s117
                                 + 1.26536 w_k1_MSFT_s117 + 1.3368 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.10626 w_k1_ADM_s117
                                 + 1.12549 w_k1_AXP_s117
                                 + 0.902967 w_k1_DIS_s117
                                 + 1.0144 w_k1_FDX_s117
                                 + 0.960398 w_k1_IBM_s117
                                 + 0.967259 w_k1_INTC_s117
                                 + 0.923424 w_k1_KO_s117
                                 + 0.902588 w_k1_MAT_s117
                                 + 0.825014 w_k1_MSFT_s117
                                 + 0.884679 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.883663 w_k1_ADM_s117
                                 + 0.703187 w_k1_AXP_s117
                                 + 0.647822 w_k1_DIS_s117
                                 + 0.789837 w_k1_FDX_s117
                                 + 0.773871 w_k1_IBM_s117
                                 + 0.859073 w_k1_INTC_s117
                                 + 0.771445 w_k1_KO_s117
                                 + 0.846493 w_k1_MAT_s117
                                 + 0.812941 w_k1_MSFT_s117
                                 + 0.882569 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.947531 w_k1_ADM_s117
                                 + 1.12806 w_k1_AXP_s117
                                 + 0.91464 w_k1_DIS_s117
                                 + 0.944993 w_k1_FDX_s117
                                 + 1.42733 w_k1_IBM_s117
                                 + 1.06007 w_k1_INTC_s117
                                 + 1.07633 w_k1_KO_s117 + 1.45256 w_k1_MAT_s117
                                 + 0.927462 w_k1_MSFT_s117
                                 + 1.07201 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.03322 w_k1_ADM_s117
                                 + 1.09229 w_k1_AXP_s117
                                 + 1.21996 w_k1_DIS_s117
                                 + 1.02281 w_k1_FDX_s117
                                 + 1.17949 w_k1_IBM_s117
                                 + 0.777344 w_k1_INTC_s117
                                 + 1.00218 w_k1_KO_s117 + 1.05677 w_k1_MAT_s117
                                 + 0.847808 w_k1_MSFT_s117
                                 + 1.06291 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0243467444083672 w_k1_ADM_s000
                                 + 0.0175439499855921 w_k1_ADM_s001
                                 + 0.0239518011478937 w_k1_ADM_s002
                                 + 0.0121864366156529 w_k1_ADM_s003
                                 + 0.0257543007781065 w_k1_ADM_s004
                                 + 0.0516577587181475 w_k1_ADM_s005
                                 + 0.0134806021668095 w_k1_ADM_s006
                                 + 0.0157974411181618 w_k1_ADM_s007
                                 + 0.00803304784792988 w_k1_ADM_s010
                                 + 0.00806393989739305 w_k1_ADM_s011
                                 + 0.00556003480732812 w_k1_ADM_s012
                                 + 0.0103690665259145 w_k1_ADM_s013
                                 + 0.00702563029569008 w_k1_ADM_s014
                                 + 0.0116516077730463 w_k1_ADM_s015
                                 + 0.0112120195559626 w_k1_ADM_s016
                                 + 0.00580171304974898 w_k1_ADM_s017
                                 + 0.0414001931606915 w_k1_ADM_s100
                                 + 0.0282173080031918 w_k1_ADM_s101
                                 + 0.0649852293114992 w_k1_ADM_s102
                                 + 0.0271514442611251 w_k1_ADM_s103
                                 + 0.0644375786075993 w_k1_ADM_s104
                                 + 0.0271321085985441 w_k1_ADM_s105
                                 + 0.0196462003259365 w_k1_ADM_s106
                                 + 0.0373800364498807 w_k1_ADM_s107
                                 + 0.0376942090887709 w_k1_ADM_s110
                                 + 0.089418218098983 w_k1_ADM_s111
                                 + 0.0804091967682925 w_k1_ADM_s112
                                 + 0.0399333579973768 w_k1_ADM_s113
                                 + 0.0409857827418789 w_k1_ADM_s114
                                 + 0.0860637332611735 w_k1_ADM_s115
                                 + 0.0322801027126509 w_k1_ADM_s116
                                 + 0.0493500507383386 w_k1_ADM_s117
                                 + 0.0252535201831204 w_k1_AXP_s000
                                 + 0.0140190233051492 w_k1_AXP_s001
                                 + 0.0215103405251213 w_k1_AXP_s002
                                 + 0.011464075969218 w_k1_AXP_s003
                                 + 0.0237315686405312 w_k1_AXP_s004
                                 + 0.0613896154602882 w_k1_AXP_s005
                                 + 0.0142961204132455 w_k1_AXP_s006
                                 + 0.0154471552408834 w_k1_AXP_s007
                                 + 0.00813323973565213 w_k1_AXP_s010
                                 + 0.00936098464909308 w_k1_AXP_s011
                                 + 0.00615727766282359 w_k1_AXP_s012
                                 + 0.0101224729725784 w_k1_AXP_s013
                                 + 0.00730657987323275 w_k1_AXP_s014
                                 + 0.0119325889852451 w_k1_AXP_s015
                                 + 0.0103297473720587 w_k1_AXP_s016
                                 + 0.00483345324075849 w_k1_AXP_s017
                                 + 0.0434843766885293 w_k1_AXP_s100
                                 + 0.0291324261574659 w_k1_AXP_s101
                                 + 0.0659161599630833 w_k1_AXP_s102
                                 + 0.0342102338219912 w_k1_AXP_s103
                                 + 0.0692206078998857 w_k1_AXP_s104
                                 + 0.0241945463337605 w_k1_AXP_s105
                                 + 0.0205701775501498 w_k1_AXP_s106
                                 + 0.0417300065849952 w_k1_AXP_s107
                                 + 0.0430279682870572 w_k1_AXP_s110
                                 + 0.0926972228176635 w_k1_AXP_s111
                                 + 0.0798544575643413 w_k1_AXP_s112
                                 + 0.0398363292793267 w_k1_AXP_s113
                                 + 0.039564767067078 w_k1_AXP_s114
                                 + 0.0982929563134373 w_k1_AXP_s115
                                 + 0.0292850536031572 w_k1_AXP_s116
                                 + 0.0470765788749304 w_k1_AXP_s117
                                 + 0.0250964052339796 w_k1_DIS_s000
                                 + 0.0153034885152708 w_k1_DIS_s001
                                 + 0.0211066764511055 w_k1_DIS_s002
                                 + 0.0116757807592603 w_k1_DIS_s003
                                 + 0.0245253716039891 w_k1_DIS_s004
                                 + 0.0623267691170139 w_k1_DIS_s005
                                 + 0.0123928480982436 w_k1_DIS_s006
                                 + 0.0154520403579056 w_k1_DIS_s007
                                 + 0.00718262855829882 w_k1_DIS_s010
                                 + 0.0084528406873179 w_k1_DIS_s011
                                 + 0.00668142797016692 w_k1_DIS_s012
                                 + 0.0112036229944149 w_k1_DIS_s013
                                 + 0.00795887675676689 w_k1_DIS_s014
                                 + 0.0116419791252723 w_k1_DIS_s015
                                 + 0.0123860729299683 w_k1_DIS_s016
                                 + 0.00468605169734803 w_k1_DIS_s017
                                 + 0.0415734510327636 w_k1_DIS_s100
                                 + 0.028760560353457 w_k1_DIS_s101
                                 + 0.0638116523479946 w_k1_DIS_s102
                                 + 0.0298641475155603 w_k1_DIS_s103
                                 + 0.0609856771005707 w_k1_DIS_s104
                                 + 0.0243986696479133 w_k1_DIS_s105
                                 + 0.0201872811813364 w_k1_DIS_s106
                                 + 0.041207143540002 w_k1_DIS_s107
                                 + 0.0370873995653044 w_k1_DIS_s110
                                 + 0.0952025618216183 w_k1_DIS_s111
                                 + 0.0796845796170202 w_k1_DIS_s112
                                 + 0.0413771441974176 w_k1_DIS_s113
                                 + 0.0445547314793729 w_k1_DIS_s114
                                 + 0.086533156964962 w_k1_DIS_s115
                                 + 0.0295439589337003 w_k1_DIS_s116
                                 + 0.0485400252851229 w_k1_DIS_s117
                                 + 0.0255310936502745 w_k1_FDX_s000
                                 + 0.0151171193968674 w_k1_FDX_s001
                                 + 0.0211041081141326 w_k1_FDX_s002
                                 + 0.0125576148719096 w_k1_FDX_s003
                                 + 0.02510340349213 w_k1_FDX_s004
                                 + 0.0594893390020928 w_k1_FDX_s005
                                 + 0.0118790061225043 w_k1_FDX_s006
                                 + 0.0149425366534291 w_k1_FDX_s007
                                 + 0.00776417037148109 w_k1_FDX_s010
                                 + 0.00847832348645311 w_k1_FDX_s011
                                 + 0.00644992274733907 w_k1_FDX_s012
                                 + 0.010277962128713 w_k1_FDX_s013
                                 + 0.00798820770003426 w_k1_FDX_s014
                                 + 0.0120320030711493 w_k1_FDX_s015
                                 + 0.0118149294262816 w_k1_FDX_s016
                                 + 0.00505939813110939 w_k1_FDX_s017
                                 + 0.0430271906449563 w_k1_FDX_s100
                                 + 0.0275595131917521 w_k1_FDX_s101
                                 + 0.060225596961919 w_k1_FDX_s102
                                 + 0.0287640927791689 w_k1_FDX_s103
                                 + 0.0620993234757357 w_k1_FDX_s104
                                 + 0.0244910035764524 w_k1_FDX_s105
                                 + 0.0195812894517393 w_k1_FDX_s106
                                 + 0.041166172351222 w_k1_FDX_s107
                                 + 0.0389048299892884 w_k1_FDX_s110
                                 + 0.0962804381416597 w_k1_FDX_s111
                                 + 0.0746110155560489 w_k1_FDX_s112
                                 + 0.0403121722210494 w_k1_FDX_s113
                                 + 0.0420771882715821 w_k1_FDX_s114
                                 + 0.0873875262424813 w_k1_FDX_s115
                                 + 0.028865216228888 w_k1_FDX_s116
                                 + 0.0470230694927968 w_k1_FDX_s117
                                 + 0.0258538550104558 w_k1_IBM_s000
                                 + 0.0150744256207648 w_k1_IBM_s001
                                 + 0.0226515035877051 w_k1_IBM_s002
                                 + 0.0119489525458558 w_k1_IBM_s003
                                 + 0.0231649123302584 w_k1_IBM_s004
                                 + 0.0585901605406386 w_k1_IBM_s005
                                 + 0.0123320067294791 w_k1_IBM_s006
                                 + 0.0148428393121539 w_k1_IBM_s007
                                 + 0.00743879638153497 w_k1_IBM_s010
                                 + 0.00882451849999639 w_k1_IBM_s011
                                 + 0.0063821589117779 w_k1_IBM_s012
                                 + 0.0099223332893554 w_k1_IBM_s013
                                 + 0.0076510661383031 w_k1_IBM_s014
                                 + 0.0114856929372881 w_k1_IBM_s015
                                 + 0.0117536017485378 w_k1_IBM_s016
                                 + 0.00466100649066407 w_k1_IBM_s017
                                 + 0.0448420048786366 w_k1_IBM_s100
                                 + 0.0285047763958514 w_k1_IBM_s101
                                 + 0.0573114119656867 w_k1_IBM_s102
                                 + 0.0291967707016614 w_k1_IBM_s103
                                 + 0.0619055128509572 w_k1_IBM_s104
                                 + 0.0247584964924071 w_k1_IBM_s105
                                 + 0.0196972013678542 w_k1_IBM_s106
                                 + 0.0410206839068034 w_k1_IBM_s107
                                 + 0.0399915281797818 w_k1_IBM_s110
                                 + 0.0970177764985489 w_k1_IBM_s111
                                 + 0.0766903320263539 w_k1_IBM_s112
                                 + 0.0396420948638033 w_k1_IBM_s113
                                 + 0.0429212900441604 w_k1_IBM_s114
                                 + 0.0903904340267565 w_k1_IBM_s115
                                 + 0.0286889454546315 w_k1_IBM_s116
                                 + 0.0483500608890369 w_k1_IBM_s117
                                 + 0.0287417420884247 w_k1_INTC_s000
                                 + 0.0139668797421626 w_k1_INTC_s001
                                 + 0.0230205572049056 w_k1_INTC_s002
                                 + 0.0104758672947402 w_k1_INTC_s003
                                 + 0.0240776952146935 w_k1_INTC_s004
                                 + 0.0581685377139053 w_k1_INTC_s005
                                 + 0.0127223137431581 w_k1_INTC_s006
                                 + 0.0155124232170303 w_k1_INTC_s007
                                 + 0.00795449435007623 w_k1_INTC_s010
                                 + 0.00924435525597038 w_k1_INTC_s011
                                 + 0.00622370428007592 w_k1_INTC_s012
                                 + 0.0121751755668886 w_k1_INTC_s013
                                 + 0.00784555426779938 w_k1_INTC_s014
                                 + 0.0111992542424215 w_k1_INTC_s015
                                 + 0.0111281472810435 w_k1_INTC_s016
                                 + 0.00415934297035653 w_k1_INTC_s017
                                 + 0.0468276529273327 w_k1_INTC_s100
                                 + 0.0249148585702959 w_k1_INTC_s101
                                 + 0.0644431773202403 w_k1_INTC_s102
                                 + 0.0296559123059327 w_k1_INTC_s103
                                 + 0.0653329596800541 w_k1_INTC_s104
                                 + 0.0249767422959772 w_k1_INTC_s105
                                 + 0.0198701003202725 w_k1_INTC_s106
                                 + 0.0380396770417308 w_k1_INTC_s107
                                 + 0.0405910751911912 w_k1_INTC_s110
                                 + 0.0994165892333567 w_k1_INTC_s111
                                 + 0.0744596195823971 w_k1_INTC_s112
                                 + 0.0375225374569214 w_k1_INTC_s113
                                 + 0.0411003055030342 w_k1_INTC_s114
                                 + 0.0972280712680395 w_k1_INTC_s115
                                 + 0.0290371196183238 w_k1_INTC_s116
                                 + 0.0421415896428222 w_k1_INTC_s117
                                 + 0.0278068770144931 w_k1_KO_s000
                                 + 0.0149138262174095 w_k1_KO_s001
                                 + 0.0216917286515288 w_k1_KO_s002
                                 + 0.0111054379331718 w_k1_KO_s003
                                 + 0.0242023999799896 w_k1_KO_s004
                                 + 0.0537308803557982 w_k1_KO_s005
                                 + 0.0119012756908221 w_k1_KO_s006
                                 + 0.0144318872284644 w_k1_KO_s007
                                 + 0.00820185043689596 w_k1_KO_s010
                                 + 0.00881680262405147 w_k1_KO_s011
                                 + 0.0055260058609331 w_k1_KO_s012
                                 + 0.00960347245159659 w_k1_KO_s013
                                 + 0.00804642089664421 w_k1_KO_s014
                                 + 0.0112882102244809 w_k1_KO_s015
                                 + 0.0123393950825559 w_k1_KO_s016
                                 + 0.00449157196425496 w_k1_KO_s017
                                 + 0.0405943383166002 w_k1_KO_s100
                                 + 0.028433634295151 w_k1_KO_s101
                                 + 0.0636114395593827 w_k1_KO_s102
                                 + 0.0291673716173559 w_k1_KO_s103
                                 + 0.0615666461330938 w_k1_KO_s104
                                 + 0.0245271830353605 w_k1_KO_s105
                                 + 0.0192339028254172 w_k1_KO_s106
                                 + 0.0429736294032021 w_k1_KO_s107
                                 + 0.0376996094634459 w_k1_KO_s110
                                 + 0.0995986644097671 w_k1_KO_s111
                                 + 0.0760235446507718 w_k1_KO_s112
                                 + 0.0390764100499263 w_k1_KO_s113
                                 + 0.0443435599997918 w_k1_KO_s114
                                 + 0.0873641840410011 w_k1_KO_s115
                                 + 0.0273949852141529 w_k1_KO_s116
                                 + 0.0441747958385226 w_k1_KO_s117
                                 + 0.029882223399146 w_k1_MAT_s000
                                 + 0.015758848208674 w_k1_MAT_s001
                                 + 0.0199073026753958 w_k1_MAT_s002
                                 + 0.0136194903617771 w_k1_MAT_s003
                                 + 0.0207960195861445 w_k1_MAT_s004
                                 + 0.0528307247025368 w_k1_MAT_s005
                                 + 0.0135059727908615 w_k1_MAT_s006
                                 + 0.0157224932112841 w_k1_MAT_s007
                                 + 0.00773069388682653 w_k1_MAT_s010
                                 + 0.0103215487779171 w_k1_MAT_s011
                                 + 0.00571722801811669 w_k1_MAT_s012
                                 + 0.00958839579799896 w_k1_MAT_s013
                                 + 0.00779058827314407 w_k1_MAT_s014
                                 + 0.0106352422938985 w_k1_MAT_s015
                                 + 0.0118694781558346 w_k1_MAT_s016
                                 + 0.00597907032612853 w_k1_MAT_s017
                                 + 0.0369011385528884 w_k1_MAT_s100
                                 + 0.0271861933075719 w_k1_MAT_s101
                                 + 0.0507754432134049 w_k1_MAT_s102
                                 + 0.0337928392115898 w_k1_MAT_s103
                                 + 0.0553012755549757 w_k1_MAT_s104
                                 + 0.0247418661306193 w_k1_MAT_s105
                                 + 0.0179880961017702 w_k1_MAT_s106
                                 + 0.0415122689041209 w_k1_MAT_s107
                                 + 0.0409141277429608 w_k1_MAT_s110
                                 + 0.105713646424538 w_k1_MAT_s111
                                 + 0.0789974252983448 w_k1_MAT_s112
                                 + 0.0356146059880468 w_k1_MAT_s113
                                 + 0.0413638698180102 w_k1_MAT_s114
                                 + 0.0974200138712953 w_k1_MAT_s115
                                 + 0.0295446093291924 w_k1_MAT_s116
                                 + 0.0461341645680577 w_k1_MAT_s117
                                 + 0.027575515141794 w_k1_MSFT_s000
                                 + 0.0152080687151643 w_k1_MSFT_s001
                                 + 0.0193511580973118 w_k1_MSFT_s002
                                 + 0.0108516988497304 w_k1_MSFT_s003
                                 + 0.0251971217150113 w_k1_MSFT_s004
                                 + 0.0581092959752302 w_k1_MSFT_s005
                                 + 0.0119883146055533 w_k1_MSFT_s006
                                 + 0.0158401742487651 w_k1_MSFT_s007
                                 + 0.00804073859098394 w_k1_MSFT_s010
                                 + 0.00827404413860037 w_k1_MSFT_s011
                                 + 0.00594211045532933 w_k1_MSFT_s012
                                 + 0.00988065237664116 w_k1_MSFT_s013
                                 + 0.00799208182571553 w_k1_MSFT_s014
                                 + 0.0120917625389507 w_k1_MSFT_s015
                                 + 0.0116154982962152 w_k1_MSFT_s016
                                 + 0.00457512309790417 w_k1_MSFT_s017
                                 + 0.0437452702610141 w_k1_MSFT_s100
                                 + 0.030506247580013 w_k1_MSFT_s101
                                 + 0.0597363113899922 w_k1_MSFT_s102
                                 + 0.0283541195837367 w_k1_MSFT_s103
                                 + 0.0597798438948223 w_k1_MSFT_s104
                                 + 0.0267289997662338 w_k1_MSFT_s105
                                 + 0.0188160326410945 w_k1_MSFT_s106
                                 + 0.0391827367938249 w_k1_MSFT_s107
                                 + 0.0377389885691462 w_k1_MSFT_s110
                                 + 0.0962357408204661 w_k1_MSFT_s111
                                 + 0.0758478860250651 w_k1_MSFT_s112
                                 + 0.0383648688126498 w_k1_MSFT_s113
                                 + 0.0397338867763993 w_k1_MSFT_s114
                                 + 0.085326637058613 w_k1_MSFT_s115
                                 + 0.0308348678593799 w_k1_MSFT_s116
                                 + 0.0426895449860287 w_k1_MSFT_s117
                                 + 0.0270908736258254 w_k1_PG_s000
                                 + 0.0148571158350843 w_k1_PG_s001
                                 + 0.0202796490888629 w_k1_PG_s002
                                 + 0.0127942013480804 w_k1_PG_s003
                                 + 0.0238999011132537 w_k1_PG_s004
                                 + 0.0596903724636293 w_k1_PG_s005
                                 + 0.0129293344161461 w_k1_PG_s006
                                 + 0.0140632588413818 w_k1_PG_s007
                                 + 0.00796508343499793 w_k1_PG_s010
                                 + 0.00838801743104422 w_k1_PG_s011
                                 + 0.00576132765688665 w_k1_PG_s012
                                 + 0.009161522799628 w_k1_PG_s013
                                 + 0.00842700537575292 w_k1_PG_s014
                                 + 0.0126597447033968 w_k1_PG_s015
                                 + 0.0116892428496307 w_k1_PG_s016
                                 + 0.00433995761109184 w_k1_PG_s017
                                 + 0.0403992857618777 w_k1_PG_s100
                                 + 0.0276062060483361 w_k1_PG_s101
                                 + 0.0621767641450439 w_k1_PG_s102
                                 + 0.0279186871931848 w_k1_PG_s103
                                 + 0.0645033297701492 w_k1_PG_s104
                                 + 0.0240194847507021 w_k1_PG_s105
                                 + 0.0199958825294353 w_k1_PG_s106
                                 + 0.0395790854802635 w_k1_PG_s107
                                 + 0.037250003713322 w_k1_PG_s110
                                 + 0.09946135604106 w_k1_PG_s111
                                 + 0.0790279941074251 w_k1_PG_s112
                                 + 0.035794409066344 w_k1_PG_s113
                                 + 0.0382465677017335 w_k1_PG_s114
                                 + 0.0874137845147659 w_k1_PG_s115
                                 + 0.030439153536318 w_k1_PG_s116
                                 + 0.0463898926571047 w_k1_PG_s117 >= 1.13
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
