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
 w_k1_ADM_s0_return_constraint:  1.02808 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.01348 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.09734 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.85951 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.809285 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.901658 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.44633 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.928565 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.05617 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.04322 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.52556 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.795639 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.916479 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.21697 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.0455 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.00365 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.41723 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.26343 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.18513 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.7822 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.917615 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.959389 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.67323 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.23966 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.764787 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.04055 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.34914 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.821921 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.72985 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.1563 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.15689 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.31153 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.01676 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.770236 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.978085 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.09297 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.803528 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.36552 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.06035 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.04446 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.22351 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.766971 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.782794 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.07653 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.19672 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.772317 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.16829 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.82942 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.76386 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.659637 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.09617 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.32032 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.911378 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.04246 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.47261 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.34432 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.18322 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.645986 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.839458 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.839998 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.56565 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.745072 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.768175 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.941857 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.14823 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.527408 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.617111 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.998706 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.02377 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.53387 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.811729 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.61076 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.713651 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.881304 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.55859 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.28477 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.998978 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.19365 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.35737 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.697393 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.784599 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.03365 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.2822 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.935992 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.08401 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.764403 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 2.01287 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.612243 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.8199 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.21465 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.03581 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.12463 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.86519 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.68971 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.32403 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.638178 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.799764 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.845566 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.78493 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.856346 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.651823 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.806747 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.47273 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.585073 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.566655 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.0327 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.965465 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.66386 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.847894 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.500892 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.59609 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.970685 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.524718 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.51144 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.02635 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.0109 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.17963 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.812634 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.867726 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.05027 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.25216 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.859259 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.04719 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.960939 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.49062 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.701043 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.874215 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.10312 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.10585 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.930138 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.57744 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.19904 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.17375 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.748008 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.839328 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.824169 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.51269 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.02741 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.674944 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.923203 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.29063 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.700131 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.66427 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.04019 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.855271 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.2506 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.866591 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.642846 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.717409 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.05996 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.583967 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.26706 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.929143 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.34188 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.22688 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.864042 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.680544 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.11961 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.82961 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.23315 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.17194 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.661421 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.46333 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.961281 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.42281 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.73675 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.588738 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.53107 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.02659 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.75662 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.992938 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.576682 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.894747 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.18764 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.26544 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.58794 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.17548 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.85956 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.980738 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.683288 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.865816 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.15855 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.34946 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.79498 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.896971 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.550709 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.915011 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.622072 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.802201 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.0022 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.980361 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.95034 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.24335 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.876566 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.06448 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.13107 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.950968 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.910902 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.960815 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.897452 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.36381 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.651456 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.827475 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.98376 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.05721 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.942941 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.18071 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.05993 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.11591 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.820576 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.807867 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.891087 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.24358 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.88963 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.771229 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.916046 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.11467 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.654014 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.69546 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.949106 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.07531 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.45608 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.911399 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.704049 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.736206 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.975505 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.643565 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.06036 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.952889 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.17096 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.19539 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.846439 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.788673 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.05793 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.07557 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.13661 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.096 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.82345 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.74417 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.887756 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.14112 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.39985 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.856038 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.11338 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.27213 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.53429 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.15131 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.634585 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.85192 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.00229 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.47094 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.790114 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.996766 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.86457 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.19341 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.685657 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.790849 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.20285 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.11976 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.6628 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.960169 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.647651 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.873859 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.87204 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.72502 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.20626 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.983968 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.33986 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.08885 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.757723 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.647499 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.11365 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.11467 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.00049 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.41787 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.73306 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.91345 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.20387 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.94881 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 2.20532 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.545563 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.35277 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.68695 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 2.24377 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.00963 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.532417 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.695059 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.31516 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.29051 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.398012 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.934035 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.882531 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.18117 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.578113 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.804115 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.17894 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.35161 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.91296 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.820207 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.527855 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.906518 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.550264 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.708362 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.11388 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.01039 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.04613 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.19282 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.801154 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.943261 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.999841 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.10533 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.904318 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.05968 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.906321 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.50902 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.748667 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.796586 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.02424 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.10812 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.04331 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.54637 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.31341 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.21868 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.80787 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.829519 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.873754 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.29637 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.924617 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.717315 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.922652 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.17017 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.750944 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.737794 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.10555 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.02953 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.4591 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.914568 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.714622 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.750918 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.01893 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.711046 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.29189 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.963779 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.05733 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.12583 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.899573 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.838137 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.04838 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.12172 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.04725 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.12346 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.894975 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.3657 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.86533 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.06013 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.15888 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.956814 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.03132 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.30552 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.23218 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.00398 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.776212 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.914033 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.941448 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.34863 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.943533 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.899601 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.02051 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.21865 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.815991 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.884044 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.08688 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.988442 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.24461 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.931574 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.785154 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.833018 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.891572 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.760036 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.20334 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.481999405229879 z_k1_s0000
                                 - 0.561999390176665 z_k1_s0001
                                 - 1.71400118970842 z_k1_s0002
                                 - 0.387999590942241 z_k1_s0003
                                 - 1.13799860969418 z_k1_s0004
                                 - 1.09899668474701 z_k1_s0005
                                 - 0.396999673261173 z_k1_s0006
                                 - 0.940999570899319 z_k1_s0007
                                 - 0.321000421801421 z_k1_s0010
                                 - 0.237000109969242 z_k1_s0011
                                 - 0.477000510003962 z_k1_s0012
                                 - 0.431999283022274 z_k1_s0013
                                 - 0.285000065016167 z_k1_s0014
                                 - 0.496000680951733 z_k1_s0015
                                 - 0.421999357470841 z_k1_s0016
                                 - 0.454001618435813 z_k1_s0017
                                 - 0.405001355269555 z_k1_s0020
                                 - 0.563000551319075 z_k1_s0021
                                 - 0.351001140653499 z_k1_s0022
                                 - 1.53700666441259 z_k1_s0023
                                 - 0.620000580029795 z_k1_s0024
                                 - 0.613000844241669 z_k1_s0025
                                 - 0.317001043206481 z_k1_s0026
                                 - 0.681004091346078 z_k1_s0027
                                 - 1.85900599941668 z_k1_s0030
                                 - 1.49300061275647 z_k1_s0031
                                 - 1.45700061992179 z_k1_s0032
                                 - 1.00000162079938 z_k1_s0033
                                 - 1.60200514065113 z_k1_s0034
                                 - 0.810000930681969 z_k1_s0035
                                 - 1.57300423650526 z_k1_s0036
                                 - 1.12499704633019 z_k1_s0037
                                 - 0.469000587141906 z_k1_s0040
                                 - 0.504999122455474 z_k1_s0041
                                 - 0.30399990483758 z_k1_s0042
                                 - 0.38699920287181 z_k1_s0043
                                 - 0.381998619550431 z_k1_s0044
                                 - 0.252999934259895 z_k1_s0045
                                 - 0.534999475383547 z_k1_s0046
                                 - 0.311999894051726 z_k1_s0047
                                 - 0.619998029939073 z_k1_s0050
                                 - 0.711998994511833 z_k1_s0051
                                 - 0.564000407462727 z_k1_s0052
                                 - 0.482000594083621 z_k1_s0053
                                 - 0.744998094846569 z_k1_s0054
                                 - 0.709999222673524 z_k1_s0055
                                 - 1.09400125567104 z_k1_s0056
                                 - 0.803002938168072 z_k1_s0057
                                 - 0.247999581493093 z_k1_s0060
                                 - 0.350999281242162 z_k1_s0061
                                 - 0.547999503170486 z_k1_s0062
                                 - 0.422999937805154 z_k1_s0063
                                 - 0.192000142833651 z_k1_s0064
                                 - 0.195000190290455 z_k1_s0065
                                 - 0.236000115266325 z_k1_s0066
                                 - 0.219000087544586 z_k1_s0067
                                 - 0.164999803593751 z_k1_s0070
                                 - 0.477000341166744 z_k1_s0071
                                 - 0.339998638484773 z_k1_s0072
                                 - 0.413000034628833 z_k1_s0073
                                 - 0.157000004476342 z_k1_s0074
                                 - 0.261000502600447 z_k1_s0075
                                 - 0.329000230297401 z_k1_s0076
                                 - 0.249999702414774 z_k1_s0077
                                 - 0.237999846894007 z_k1_s0100
                                 - 0.308999886794313 z_k1_s0101
                                 - 0.864001010640303 z_k1_s0102
                                 - 0.261999874967349 z_k1_s0103
                                 - 0.200999933064261 z_k1_s0104
                                 - 0.503998863540315 z_k1_s0105
                                 - 0.292999868078752 z_k1_s0106
                                 - 0.781001129303312 z_k1_s0107
                                 - 0.462001483318061 z_k1_s0110
                                 - 0.161000064841194 z_k1_s0111
                                 - 0.686001233775413 z_k1_s0112
                                 - 0.524999748134381 z_k1_s0113
                                 - 0.423998639651518 z_k1_s0114
                                 - 0.677000466801768 z_k1_s0115
                                 - 0.301999885563012 z_k1_s0116
                                 - 0.314999848880628 z_k1_s0117
                                 - 0.231999931620937 z_k1_s0120
                                 - 0.280998952879262 z_k1_s0121
                                 - 0.291999007600512 z_k1_s0122
                                 - 0.168000102242743 z_k1_s0123
                                 - 0.398999539889033 z_k1_s0124
                                 - 0.298999042423126 z_k1_s0125
                                 - 0.570000390555735 z_k1_s0126
                                 - 0.204000036830476 z_k1_s0127
                                 - 0.435999970908491 z_k1_s0130
                                 - 0.292998620645875 z_k1_s0131
                                 - 0.194999876995847 z_k1_s0132
                                 - 0.288998693435655 z_k1_s0133
                                 - 0.288998693435655 z_k1_s0134
                                 - 0.134999914843279 z_k1_s0135
                                 - 0.414999035555754 z_k1_s0136
                                 - 0.582998613383166 z_k1_s0137
                                 - 1.86399614487665 z_k1_s0140
                                 - 1.41699400774593 z_k1_s0141
                                 - 1.19699775321358 z_k1_s0142
                                 - 1.79800418591306 z_k1_s0143
                                 - 1.87000275050559 z_k1_s0144
                                 - 0.838999446131585 z_k1_s0145
                                 - 2.01599439557126 z_k1_s0146
                                 - 0.52800023407247 z_k1_s0147
                                 - 0.351000457357751 z_k1_s0150
                                 - 0.556998901546094 z_k1_s0151
                                 - 0.318000446550641 z_k1_s0152
                                 - 0.517998850728566 z_k1_s0153
                                 - 0.362000321459992 z_k1_s0154
                                 - 0.605000895630979 z_k1_s0155
                                 - 1.10399982891008 z_k1_s0156
                                 - 0.370000374807217 z_k1_s0157
                                 - 0.336000612627209 z_k1_s0160
                                 - 0.437001310530432 z_k1_s0161
                                 - 1.36200272864841 z_k1_s0162
                                 - 0.695002111589657 z_k1_s0163
                                 - 0.760000461309215 z_k1_s0164
                                 - 1.06100458148634 z_k1_s0165
                                 - 0.972000684300971 z_k1_s0166
                                 - 0.350000762591156 z_k1_s0167
                                 - 0.594000043286593 z_k1_s0170
                                 - 1.01400062528788 z_k1_s0171
                                 - 0.361000435258173 z_k1_s0172
                                 - 0.262000428043741 z_k1_s0173
                                 - 0.862000154715672 z_k1_s0174
                                 - 0.318000505643913 z_k1_s0175
                                 - 0.709999684143461 z_k1_s0176
                                 - 0.428000463115443 z_k1_s0177
                                 - 0.0529997327090816 z_k1_s1000
                                 - 0.0419997422678611 z_k1_s1001
                                 - 0.102999136172632 z_k1_s1002
                                 - 0.0529997327090816 z_k1_s1003
                                 - 0.0999992749241754 z_k1_s1004
                                 - 0.0299998292769601 z_k1_s1005
                                 - 0.0519996229605712 z_k1_s1006
                                 - 0.0349997696233151 z_k1_s1007
                                 - 0.0569998030574499 z_k1_s1010
                                 - 0.0199999550657215 z_k1_s1011
                                 - 0.016999966725853 z_k1_s1012
                                 - 0.0149999662992911 z_k1_s1013
                                 - 0.0549998272308367 z_k1_s1014
                                 - 0.0279999075720716 z_k1_s1015
                                 - 0.0329998471386047 z_k1_s1016
                                 - 0.0209999675789767 z_k1_s1017
                                 - 0.0289999273238489 z_k1_s1020
                                 - 0.0569998442917371 z_k1_s1021
                                 - 0.0699999789192272 z_k1_s1022
                                 - 0.0209999563758668 z_k1_s1023
                                 - 0.0519997925119332 z_k1_s1024
                                 - 0.0249999231999072 z_k1_s1025
                                 - 0.022999939787887 z_k1_s1026
                                 - 0.0959998751751938 z_k1_s1027
                                 - 0.104999693238301 z_k1_s1030
                                 - 0.149998996628613 z_k1_s1031
                                 - 0.135999762793671 z_k1_s1032
                                 - 0.0569997769581815 z_k1_s1033
                                 - 0.185999428336542 z_k1_s1034
                                 - 0.0669997100667558 z_k1_s1035
                                 - 0.16499909409061 z_k1_s1036
                                 - 0.122999414594426 z_k1_s1037
                                 - 0.0919999572992066 z_k1_s1040
                                 - 0.099000241050387 z_k1_s1041
                                 - 0.0339999982192795 z_k1_s1042
                                 - 0.0759997687253265 z_k1_s1043
                                 - 0.114999785623922 z_k1_s1044
                                 - 0.0890002036917551 z_k1_s1045
                                 - 0.0850003175483713 z_k1_s1046
                                 - 0.0540000085365089 z_k1_s1047
                                 - 0.0550000225267118 z_k1_s1050
                                 - 0.0469998711050257 z_k1_s1051
                                 - 0.065999794232653 z_k1_s1052
                                 - 0.0739999456543391 z_k1_s1053
                                 - 0.0369998952273692 z_k1_s1054
                                 - 0.0239999498663558 z_k1_s1055
                                 - 0.0669996676607381 z_k1_s1056
                                 - 0.0179999720997419 z_k1_s1057
                                 - 0.0449999799570152 z_k1_s1060
                                 - 0.0269999057744577 z_k1_s1061
                                 - 0.0919996058687432 z_k1_s1062
                                 - 0.0219999414902436 z_k1_s1063
                                 - 0.0239999162439624 z_k1_s1064
                                 - 0.0219999414902436 z_k1_s1065
                                 - 0.0209999267134671 z_k1_s1066
                                 - 0.0209999267134671 z_k1_s1067
                                 - 0.102999807658507 z_k1_s1070
                                 - 0.0670002022143427 z_k1_s1071
                                 - 0.0779998602947841 z_k1_s1072
                                 - 0.125999538553411 z_k1_s1073
                                 - 0.0529999129310613 z_k1_s1074
                                 - 0.0800000767636027 z_k1_s1075
                                 - 0.0729996256223498 z_k1_s1076
                                 - 0.0329999550400831 z_k1_s1077
                                 - 0.358999334744174 z_k1_s1100
                                 - 0.473999996547476 z_k1_s1101
                                 - 0.295998597747814 z_k1_s1102
                                 - 0.223999890422081 z_k1_s1103
                                 - 0.133999923140895 z_k1_s1104
                                 - 0.444999697501508 z_k1_s1105
                                 - 0.39900044597956 z_k1_s1106
                                 - 0.201999785842305 z_k1_s1107
                                 - 0.534001646155195 z_k1_s1110
                                 - 0.296000783606232 z_k1_s1111
                                 - 0.305001539082979 z_k1_s1112
                                 - 0.182000457195648 z_k1_s1113
                                 - 0.178000616185271 z_k1_s1114
                                 - 0.268000682129883 z_k1_s1115
                                 - 0.124000333737762 z_k1_s1116
                                 - 0.137000323023033 z_k1_s1117
                                 - 0.609997878272491 z_k1_s1120
                                 - 0.595000874179916 z_k1_s1121
                                 - 1.09399697979913 z_k1_s1122
                                 - 0.589997744821203 z_k1_s1123
                                 - 0.280999604593604 z_k1_s1124
                                 - 0.405999406665522 z_k1_s1125
                                 - 0.297999511627473 z_k1_s1126
                                 - 0.253999630834092 z_k1_s1127
                                 - 0.149999596731165 z_k1_s1130
                                 - 0.0790000340118728 z_k1_s1131
                                 - 0.128999969988964 z_k1_s1132
                                 - 0.0790000340118728 z_k1_s1133
                                 - 0.156999824312078 z_k1_s1134
                                 - 0.279000622720669 z_k1_s1135
                                 - 0.0770000745888086 z_k1_s1136
                                 - 0.106000278223644 z_k1_s1137
                                 - 0.117999911385459 z_k1_s1140
                                 - 0.132000519838566 z_k1_s1141
                                 - 0.0900000762805729 z_k1_s1142
                                 - 0.0620000437754974 z_k1_s1143
                                 - 0.0970000844068418 z_k1_s1144
                                 - 0.0860000575369772 z_k1_s1145
                                 - 0.290000618659982 z_k1_s1146
                                 - 0.111999932620113 z_k1_s1147
                                 - 0.139999828640692 z_k1_s1150
                                 - 0.166999663892736 z_k1_s1151
                                 - 0.342000108194046 z_k1_s1152
                                 - 0.206999614932934 z_k1_s1153
                                 - 0.113000125088736 z_k1_s1154
                                 - 0.0830000301084991 z_k1_s1155
                                 - 0.108000131208712 z_k1_s1156
                                 - 0.156999676132687 z_k1_s1157
                                 - 0.280000433050275 z_k1_s1160
                                 - 0.113999961513295 z_k1_s1161
                                 - 0.31299944588819 z_k1_s1162
                                 - 0.125999980072592 z_k1_s1163
                                 - 0.0820000552218556 z_k1_s1164
                                 - 0.296999707542502 z_k1_s1165
                                 - 0.0690000709159553 z_k1_s1166
                                 - 0.151000126137811 z_k1_s1167
                                 - 0.375999772005687 z_k1_s1170
                                 - 1.04599728143817 z_k1_s1171
                                 - 0.886999335302842 z_k1_s1172
                                 - 0.738999335504905 z_k1_s1173
                                 - 0.738999335504905 z_k1_s1174
                                 - 0.333999961630499 z_k1_s1175
                                 - 0.332999765745479 z_k1_s1176
                                 - 0.221999999696905 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.925872 w_k1_ADM_s000
                                 + 1.48395 w_k1_AXP_s000
                                 + 1.75224 w_k1_DIS_s000
                                 + 1.52013 w_k1_FDX_s000
                                 + 1.26565 w_k1_IBM_s000
                                 + 1.64132 w_k1_INTC_s000
                                 + 1.35867 w_k1_KO_s000 + 1.23169 w_k1_MAT_s000
                                 + 1.5946 w_k1_MSFT_s000 + 1.27361 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.908135 w_k1_ADM_s000
                                 + 1.18357 w_k1_AXP_s000
                                 + 1.26242 w_k1_DIS_s000
                                 + 1.04405 w_k1_FDX_s000 + 1.676 w_k1_IBM_s000
                                 + 1.00953 w_k1_INTC_s000
                                 + 1.24251 w_k1_KO_s000 + 1.51459 w_k1_MAT_s000
                                 + 1.07273 w_k1_MSFT_s000
                                 + 1.06414 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.767647 w_k1_ADM_s000
                                 + 0.863025 w_k1_AXP_s000
                                 + 1.12249 w_k1_DIS_s000
                                 + 1.00187 w_k1_FDX_s000
                                 + 1.01165 w_k1_IBM_s000
                                 + 1.21039 w_k1_INTC_s000
                                 + 1.04134 w_k1_KO_s000
                                 + 0.666631 w_k1_MAT_s000
                                 + 1.0514 w_k1_MSFT_s000
                                 + 0.935322 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.811607 w_k1_ADM_s000
                                 + 0.839338 w_k1_AXP_s000
                                 + 0.59048 w_k1_DIS_s000
                                 + 0.831021 w_k1_FDX_s000
                                 + 0.722583 w_k1_IBM_s000
                                 + 0.917883 w_k1_INTC_s000
                                 + 0.703724 w_k1_KO_s000
                                 + 0.745892 w_k1_MAT_s000
                                 + 0.810562 w_k1_MSFT_s000
                                 + 0.913189 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.721499 w_k1_ADM_s000
                                 + 0.665598 w_k1_AXP_s000
                                 + 0.703014 w_k1_DIS_s000
                                 + 0.789812 w_k1_FDX_s000
                                 + 0.868661 w_k1_IBM_s000
                                 + 0.951843 w_k1_INTC_s000
                                 + 0.797658 w_k1_KO_s000
                                 + 0.695656 w_k1_MAT_s000
                                 + 0.889089 w_k1_MSFT_s000
                                 + 0.835486 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.969807 w_k1_ADM_s000
                                 + 1.07164 w_k1_AXP_s000
                                 + 1.17044 w_k1_DIS_s000
                                 + 1.26592 w_k1_FDX_s000
                                 + 0.922088 w_k1_IBM_s000
                                 + 1.02916 w_k1_INTC_s000
                                 + 0.995427 w_k1_KO_s000
                                 + 1.06938 w_k1_MAT_s000
                                 + 1.15826 w_k1_MSFT_s000
                                 + 1.07358 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.993805 w_k1_ADM_s000
                                 + 1.57339 w_k1_AXP_s000
                                 + 1.13125 w_k1_DIS_s000
                                 + 1.12007 w_k1_FDX_s000
                                 + 1.29899 w_k1_IBM_s000
                                 + 1.12181 w_k1_INTC_s000
                                 + 1.08764 w_k1_KO_s000 + 1.37697 w_k1_MAT_s000
                                 + 1.05655 w_k1_MSFT_s000 + 1.1067 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.963443 w_k1_ADM_s000
                                 + 1.01545 w_k1_AXP_s000
                                 + 0.826132 w_k1_DIS_s000
                                 + 0.888371 w_k1_FDX_s000
                                 + 1.23171 w_k1_IBM_s000
                                 + 1.0297 w_k1_INTC_s000
                                 + 0.976135 w_k1_KO_s000
                                 + 1.07806 w_k1_MAT_s000
                                 + 0.972817 w_k1_MSFT_s000
                                 + 0.886905 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.37942 w_k1_ADM_s001
                                 + 1.55666 w_k1_AXP_s001
                                 + 1.27257 w_k1_DIS_s001
                                 + 1.21285 w_k1_FDX_s001
                                 + 1.44778 w_k1_IBM_s001
                                 + 1.14741 w_k1_INTC_s001
                                 + 1.19816 w_k1_KO_s001 + 2.08732 w_k1_MAT_s001
                                 + 1.30812 w_k1_MSFT_s001
                                 + 1.15024 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.02068 w_k1_ADM_s001
                                 + 0.960102 w_k1_AXP_s001
                                 + 0.983628 w_k1_DIS_s001
                                 + 0.874348 w_k1_FDX_s001
                                 + 1.57068 w_k1_IBM_s001
                                 + 0.98541 w_k1_INTC_s001
                                 + 1.12862 w_k1_KO_s001 + 1.37335 w_k1_MAT_s001
                                 + 0.957586 w_k1_MSFT_s001
                                 + 0.949853 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.16327 w_k1_ADM_s001
                                 + 1.23543 w_k1_AXP_s001
                                 + 0.996725 w_k1_DIS_s001
                                 + 1.15294 w_k1_FDX_s001
                                 + 0.699966 w_k1_IBM_s001
                                 + 1.14361 w_k1_INTC_s001
                                 + 0.787635 w_k1_KO_s001
                                 + 1.01968 w_k1_MAT_s001
                                 + 1.06884 w_k1_MSFT_s001
                                 + 1.04535 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.04691 w_k1_ADM_s001
                                 + 0.84059 w_k1_AXP_s001
                                 + 0.831033 w_k1_DIS_s001
                                 + 0.864822 w_k1_FDX_s001
                                 + 1.00531 w_k1_IBM_s001
                                 + 0.920715 w_k1_INTC_s001
                                 + 0.978993 w_k1_KO_s001
                                 + 1.29821 w_k1_MAT_s001
                                 + 0.947743 w_k1_MSFT_s001
                                 + 0.886725 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.949415 w_k1_ADM_s001
                                 + 1.21992 w_k1_AXP_s001
                                 + 1.03771 w_k1_DIS_s001
                                 + 0.997906 w_k1_FDX_s001
                                 + 1.2596 w_k1_IBM_s001
                                 + 0.999895 w_k1_INTC_s001
                                 + 1.00113 w_k1_KO_s001 + 1.48235 w_k1_MAT_s001
                                 + 0.951164 w_k1_MSFT_s001
                                 + 0.964173 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.38495 w_k1_ADM_s001
                                 + 1.33517 w_k1_AXP_s001
                                 + 1.39254 w_k1_DIS_s001 + 1.1821 w_k1_FDX_s001
                                 + 1.02705 w_k1_IBM_s001
                                 + 1.16076 w_k1_INTC_s001
                                 + 1.06686 w_k1_KO_s001 + 1.30676 w_k1_MAT_s001
                                 + 1.21191 w_k1_MSFT_s001
                                 + 1.10686 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.920962 w_k1_ADM_s001
                                 + 0.954784 w_k1_AXP_s001
                                 + 0.824409 w_k1_DIS_s001
                                 + 0.900613 w_k1_FDX_s001
                                 + 0.745825 w_k1_IBM_s001
                                 + 0.883068 w_k1_INTC_s001
                                 + 0.809801 w_k1_KO_s001
                                 + 1.14779 w_k1_MAT_s001
                                 + 1.05372 w_k1_MSFT_s001
                                 + 0.900863 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.951017 w_k1_ADM_s001
                                 + 0.985492 w_k1_AXP_s001
                                 + 0.640587 w_k1_DIS_s001
                                 + 0.88162 w_k1_FDX_s001
                                 + 0.82539 w_k1_IBM_s001
                                 + 0.874621 w_k1_INTC_s001
                                 + 0.729318 w_k1_KO_s001
                                 + 0.722329 w_k1_MAT_s001
                                 + 0.795362 w_k1_MSFT_s001
                                 + 0.81874 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.942571 w_k1_ADM_s002
                                 + 1.22014 w_k1_AXP_s002
                                 + 1.76737 w_k1_DIS_s002
                                 + 1.29941 w_k1_FDX_s002
                                 + 1.50046 w_k1_IBM_s002
                                 + 1.36895 w_k1_INTC_s002
                                 + 1.65563 w_k1_KO_s002 + 1.68252 w_k1_MAT_s002
                                 + 1.2913 w_k1_MSFT_s002 + 1.22714 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.26428 w_k1_ADM_s002
                                 + 1.27811 w_k1_AXP_s002
                                 + 1.24905 w_k1_DIS_s002 + 1.2544 w_k1_FDX_s002
                                 + 0.90174 w_k1_IBM_s002
                                 + 1.11654 w_k1_INTC_s002
                                 + 1.12274 w_k1_KO_s002 + 1.04991 w_k1_MAT_s002
                                 + 1.11649 w_k1_MSFT_s002
                                 + 1.14135 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.19328 w_k1_ADM_s002
                                 + 1.36694 w_k1_AXP_s002
                                 + 1.30502 w_k1_DIS_s002
                                 + 1.21093 w_k1_FDX_s002
                                 + 1.29078 w_k1_IBM_s002
                                 + 1.15425 w_k1_INTC_s002
                                 + 1.19588 w_k1_KO_s002 + 1.29625 w_k1_MAT_s002
                                 + 1.09551 w_k1_MSFT_s002
                                 + 1.09622 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.911964 w_k1_ADM_s002
                                 + 0.752885 w_k1_AXP_s002
                                 + 0.732215 w_k1_DIS_s002
                                 + 0.875058 w_k1_FDX_s002
                                 + 0.800581 w_k1_IBM_s002
                                 + 0.786111 w_k1_INTC_s002
                                 + 0.832814 w_k1_KO_s002
                                 + 0.789596 w_k1_MAT_s002
                                 + 0.873952 w_k1_MSFT_s002
                                 + 0.85183 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.09927 w_k1_ADM_s002
                                 + 1.03965 w_k1_AXP_s002
                                 + 1.08352 w_k1_DIS_s002
                                 + 1.05873 w_k1_FDX_s002
                                 + 0.99089 w_k1_IBM_s002
                                 + 0.97913 w_k1_INTC_s002
                                 + 1.05549 w_k1_KO_s002 + 1.04563 w_k1_MAT_s002
                                 + 1.02924 w_k1_MSFT_s002
                                 + 1.00594 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.02902 w_k1_ADM_s002
                                 + 1.0081 w_k1_AXP_s002 + 1.41647 w_k1_DIS_s002
                                 + 1.14083 w_k1_FDX_s002 + 1.0529 w_k1_IBM_s002
                                 + 1.05583 w_k1_INTC_s002
                                 + 1.20164 w_k1_KO_s002
                                 + 0.975216 w_k1_MAT_s002
                                 + 1.16201 w_k1_MSFT_s002
                                 + 1.08987 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.994528 w_k1_ADM_s002
                                 + 1.1317 w_k1_AXP_s002 + 1.08113 w_k1_DIS_s002
                                 + 0.959331 w_k1_FDX_s002
                                 + 1.47378 w_k1_IBM_s002
                                 + 0.962389 w_k1_INTC_s002
                                 + 1.06254 w_k1_KO_s002 + 1.45983 w_k1_MAT_s002
                                 + 0.851942 w_k1_MSFT_s002
                                 + 0.992339 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.12441 w_k1_ADM_s002
                                 + 0.733472 w_k1_AXP_s002
                                 + 0.944958 w_k1_DIS_s002
                                 + 0.911311 w_k1_FDX_s002
                                 + 1.14506 w_k1_IBM_s002
                                 + 0.90169 w_k1_INTC_s002
                                 + 1.00591 w_k1_KO_s002 + 1.04547 w_k1_MAT_s002
                                 + 0.880413 w_k1_MSFT_s002
                                 + 0.932058 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.753982 w_k1_ADM_s003
                                 + 0.638781 w_k1_AXP_s003
                                 + 0.609787 w_k1_DIS_s003
                                 + 0.724831 w_k1_FDX_s003
                                 + 0.827884 w_k1_IBM_s003
                                 + 0.656693 w_k1_INTC_s003
                                 + 0.682765 w_k1_KO_s003
                                 + 0.616692 w_k1_MAT_s003
                                 + 0.705158 w_k1_MSFT_s003
                                 + 0.81087 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.02751 w_k1_ADM_s003
                                 + 1.36352 w_k1_AXP_s003
                                 + 1.39613 w_k1_DIS_s003
                                 + 1.13266 w_k1_FDX_s003
                                 + 1.59247 w_k1_IBM_s003
                                 + 1.30845 w_k1_INTC_s003
                                 + 1.34262 w_k1_KO_s003 + 1.50516 w_k1_MAT_s003
                                 + 1.26034 w_k1_MSFT_s003
                                 + 1.18649 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.967672 w_k1_ADM_s003
                                 + 0.816097 w_k1_AXP_s003
                                 + 0.854742 w_k1_DIS_s003
                                 + 0.975423 w_k1_FDX_s003
                                 + 0.856355 w_k1_IBM_s003
                                 + 0.768469 w_k1_INTC_s003
                                 + 0.868103 w_k1_KO_s003
                                 + 0.905923 w_k1_MAT_s003
                                 + 0.98697 w_k1_MSFT_s003 + 1.0438 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.886037 w_k1_ADM_s003
                                 + 0.989315 w_k1_AXP_s003
                                 + 1.16819 w_k1_DIS_s003
                                 + 0.96375 w_k1_FDX_s003
                                 + 1.18192 w_k1_IBM_s003
                                 + 0.914783 w_k1_INTC_s003
                                 + 1.20273 w_k1_KO_s003 + 1.0137 w_k1_MAT_s003
                                 + 0.986219 w_k1_MSFT_s003
                                 + 0.995259 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.943488 w_k1_ADM_s003
                                 + 1.29389 w_k1_AXP_s003
                                 + 0.953263 w_k1_DIS_s003
                                 + 0.975842 w_k1_FDX_s003
                                 + 1.0983 w_k1_IBM_s003
                                 + 0.846601 w_k1_INTC_s003
                                 + 0.987172 w_k1_KO_s003
                                 + 1.32438 w_k1_MAT_s003
                                 + 0.936119 w_k1_MSFT_s003
                                 + 1.10747 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.750047 w_k1_ADM_s003
                                 + 0.661944 w_k1_AXP_s003
                                 + 0.971279 w_k1_DIS_s003
                                 + 0.832471 w_k1_FDX_s003
                                 + 0.833052 w_k1_IBM_s003
                                 + 0.856193 w_k1_INTC_s003
                                 + 0.907174 w_k1_KO_s003
                                 + 0.823583 w_k1_MAT_s003
                                 + 0.959253 w_k1_MSFT_s003
                                 + 0.948988 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.27688 w_k1_ADM_s003
                                 + 1.36087 w_k1_AXP_s003
                                 + 1.48445 w_k1_DIS_s003 + 1.3581 w_k1_FDX_s003
                                 + 0.951673 w_k1_IBM_s003
                                 + 0.986906 w_k1_INTC_s003
                                 + 1.1581 w_k1_KO_s003 + 1.28112 w_k1_MAT_s003
                                 + 1.1892 w_k1_MSFT_s003 + 1.147 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.832541 w_k1_ADM_s003
                                 + 0.735139 w_k1_AXP_s003
                                 + 1.0508 w_k1_DIS_s003
                                 + 0.865192 w_k1_FDX_s003
                                 + 1.09923 w_k1_IBM_s003
                                 + 0.829751 w_k1_INTC_s003
                                 + 1.07203 w_k1_KO_s003 + 1.15093 w_k1_MAT_s003
                                 + 1.01483 w_k1_MSFT_s003
                                 + 1.01057 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.800045 w_k1_ADM_s004
                                 + 0.772433 w_k1_AXP_s004
                                 + 0.692334 w_k1_DIS_s004
                                 + 0.854772 w_k1_FDX_s004
                                 + 0.603693 w_k1_IBM_s004
                                 + 0.710622 w_k1_INTC_s004
                                 + 0.551055 w_k1_KO_s004
                                 + 0.50122 w_k1_MAT_s004
                                 + 0.752745 w_k1_MSFT_s004
                                 + 0.754226 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.849327 w_k1_ADM_s004
                                 + 0.869766 w_k1_AXP_s004
                                 + 0.711888 w_k1_DIS_s004
                                 + 0.819962 w_k1_FDX_s004
                                 + 1.08817 w_k1_IBM_s004
                                 + 0.744468 w_k1_INTC_s004
                                 + 0.806465 w_k1_KO_s004
                                 + 1.00798 w_k1_MAT_s004
                                 + 0.771318 w_k1_MSFT_s004
                                 + 0.883316 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.85838 w_k1_ADM_s004
                                 + 1.00085 w_k1_AXP_s004
                                 + 1.08047 w_k1_DIS_s004 + 1.046 w_k1_FDX_s004
                                 + 1.03887 w_k1_IBM_s004
                                 + 1.0144 w_k1_INTC_s004 + 1.02012 w_k1_KO_s004
                                 + 0.924535 w_k1_MAT_s004
                                 + 0.980472 w_k1_MSFT_s004
                                 + 1.1392 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.963995 w_k1_ADM_s004
                                 + 0.818848 w_k1_AXP_s004
                                 + 0.818093 w_k1_DIS_s004
                                 + 0.935143 w_k1_FDX_s004
                                 + 0.871169 w_k1_IBM_s004
                                 + 0.971183 w_k1_INTC_s004
                                 + 0.938923 w_k1_KO_s004
                                 + 0.860529 w_k1_MAT_s004
                                 + 0.927349 w_k1_MSFT_s004
                                 + 0.954031 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.942082 w_k1_ADM_s004
                                 + 1.16024 w_k1_AXP_s004 + 1.0805 w_k1_DIS_s004
                                 + 1.02839 w_k1_FDX_s004
                                 + 1.72734 w_k1_IBM_s004
                                 + 1.24277 w_k1_INTC_s004
                                 + 1.14648 w_k1_KO_s004 + 1.48784 w_k1_MAT_s004
                                 + 1.05418 w_k1_MSFT_s004
                                 + 1.12839 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.45135 w_k1_ADM_s004
                                 + 1.26877 w_k1_AXP_s004
                                 + 1.82624 w_k1_DIS_s004
                                 + 1.34117 w_k1_FDX_s004
                                 + 1.34488 w_k1_IBM_s004
                                 + 1.12349 w_k1_INTC_s004
                                 + 1.32557 w_k1_KO_s004 + 1.18769 w_k1_MAT_s004
                                 + 1.37407 w_k1_MSFT_s004
                                 + 1.14092 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.31421 w_k1_ADM_s004
                                 + 1.30962 w_k1_AXP_s004
                                 + 1.35089 w_k1_DIS_s004
                                 + 1.41942 w_k1_FDX_s004
                                 + 0.789421 w_k1_IBM_s004
                                 + 1.1455 w_k1_INTC_s004
                                 + 0.902976 w_k1_KO_s004
                                 + 0.878393 w_k1_MAT_s004
                                 + 1.24764 w_k1_MSFT_s004
                                 + 1.04178 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.846528 w_k1_ADM_s004
                                 + 0.896844 w_k1_AXP_s004
                                 + 0.985154 w_k1_DIS_s004
                                 + 1.10922 w_k1_FDX_s004
                                 + 0.688963 w_k1_IBM_s004
                                 + 1.11498 w_k1_INTC_s004
                                 + 0.872084 w_k1_KO_s004
                                 + 0.695662 w_k1_MAT_s004
                                 + 1.10153 w_k1_MSFT_s004
                                 + 1.02925 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.925557 w_k1_ADM_s005
                                 + 1.23287 w_k1_AXP_s005
                                 + 1.29751 w_k1_DIS_s005
                                 + 1.10029 w_k1_FDX_s005
                                 + 1.08466 w_k1_IBM_s005
                                 + 1.3354 w_k1_INTC_s005
                                 + 0.979055 w_k1_KO_s005
                                 + 1.01698 w_k1_MAT_s005
                                 + 1.08271 w_k1_MSFT_s005
                                 + 1.11263 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.31391 w_k1_ADM_s005
                                 + 1.34621 w_k1_AXP_s005
                                 + 1.15647 w_k1_DIS_s005
                                 + 1.24956 w_k1_FDX_s005
                                 + 0.900012 w_k1_IBM_s005
                                 + 1.30654 w_k1_INTC_s005
                                 + 1.03397 w_k1_KO_s005
                                 + 0.942807 w_k1_MAT_s005
                                 + 1.15018 w_k1_MSFT_s005
                                 + 1.08143 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.939584 w_k1_ADM_s005
                                 + 1.27315 w_k1_AXP_s005
                                 + 1.49548 w_k1_DIS_s005
                                 + 1.11581 w_k1_FDX_s005
                                 + 1.68224 w_k1_IBM_s005
                                 + 1.45772 w_k1_INTC_s005
                                 + 1.44115 w_k1_KO_s005 + 1.69311 w_k1_MAT_s005
                                 + 1.19641 w_k1_MSFT_s005
                                 + 1.21435 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.04018 w_k1_ADM_s005
                                 + 1.79807 w_k1_AXP_s005
                                 + 1.63348 w_k1_DIS_s005
                                 + 1.35542 w_k1_FDX_s005
                                 + 1.23136 w_k1_IBM_s005
                                 + 1.29224 w_k1_INTC_s005
                                 + 1.13144 w_k1_KO_s005 + 1.07886 w_k1_MAT_s005
                                 + 1.1293 w_k1_MSFT_s005 + 1.09461 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.636324 w_k1_ADM_s005
                                 + 0.732709 w_k1_AXP_s005
                                 + 0.570597 w_k1_DIS_s005
                                 + 0.758166 w_k1_FDX_s005
                                 + 0.749852 w_k1_IBM_s005
                                 + 0.817089 w_k1_INTC_s005
                                 + 0.704641 w_k1_KO_s005
                                 + 0.600372 w_k1_MAT_s005
                                 + 0.729568 w_k1_MSFT_s005
                                 + 0.870516 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.08604 w_k1_ADM_s005
                                 + 0.920236 w_k1_AXP_s005
                                 + 0.726951 w_k1_DIS_s005
                                 + 0.909405 w_k1_FDX_s005
                                 + 0.779606 w_k1_IBM_s005
                                 + 1.00852 w_k1_INTC_s005
                                 + 0.827038 w_k1_KO_s005
                                 + 0.608621 w_k1_MAT_s005
                                 + 0.842424 w_k1_MSFT_s005
                                 + 0.854983 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.01361 w_k1_ADM_s005
                                 + 0.902438 w_k1_AXP_s005
                                 + 1.07358 w_k1_DIS_s005
                                 + 0.96087 w_k1_FDX_s005
                                 + 0.911756 w_k1_IBM_s005
                                 + 1.05853 w_k1_INTC_s005
                                 + 0.950869 w_k1_KO_s005
                                 + 0.726023 w_k1_MAT_s005
                                 + 0.994938 w_k1_MSFT_s005
                                 + 0.953197 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.811131 w_k1_ADM_s005
                                 + 1.02167 w_k1_AXP_s005
                                 + 0.949925 w_k1_DIS_s005
                                 + 0.891766 w_k1_FDX_s005
                                 + 1.34696 w_k1_IBM_s005
                                 + 0.911476 w_k1_INTC_s005
                                 + 0.970304 w_k1_KO_s005
                                 + 0.968009 w_k1_MAT_s005
                                 + 0.910287 w_k1_MSFT_s005
                                 + 0.892458 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.45782 w_k1_ADM_s006
                                 + 1.24123 w_k1_AXP_s006
                                 + 1.40626 w_k1_DIS_s006
                                 + 1.29676 w_k1_FDX_s006
                                 + 0.98499 w_k1_IBM_s006
                                 + 1.54173 w_k1_INTC_s006
                                 + 1.22565 w_k1_KO_s006
                                 + 0.757367 w_k1_MAT_s006
                                 + 1.42386 w_k1_MSFT_s006
                                 + 1.10653 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.18915 w_k1_ADM_s006
                                 + 1.28225 w_k1_AXP_s006 + 1.3109 w_k1_DIS_s006
                                 + 1.03066 w_k1_FDX_s006
                                 + 1.44587 w_k1_IBM_s006
                                 + 1.10463 w_k1_INTC_s006 + 1.2086 w_k1_KO_s006
                                 + 1.60117 w_k1_MAT_s006
                                 + 1.14643 w_k1_MSFT_s006
                                 + 1.08434 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.998684 w_k1_ADM_s006
                                 + 0.99373 w_k1_AXP_s006
                                 + 0.959022 w_k1_DIS_s006
                                 + 1.01262 w_k1_FDX_s006
                                 + 0.757312 w_k1_IBM_s006
                                 + 0.972618 w_k1_INTC_s006
                                 + 0.795718 w_k1_KO_s006
                                 + 0.736089 w_k1_MAT_s006
                                 + 0.894668 w_k1_MSFT_s006
                                 + 0.929498 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.89811 w_k1_ADM_s006
                                 + 0.94722 w_k1_AXP_s006 + 0.8938 w_k1_DIS_s006
                                 + 0.908597 w_k1_FDX_s006
                                 + 1.1002 w_k1_IBM_s006
                                 + 0.916813 w_k1_INTC_s006
                                 + 0.975851 w_k1_KO_s006
                                 + 0.963129 w_k1_MAT_s006
                                 + 0.962428 w_k1_MSFT_s006
                                 + 0.969693 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.08785 w_k1_ADM_s006
                                 + 0.875347 w_k1_AXP_s006
                                 + 0.972706 w_k1_DIS_s006
                                 + 0.809384 w_k1_FDX_s006
                                 + 1.65561 w_k1_IBM_s006
                                 + 0.724428 w_k1_INTC_s006
                                 + 1.16289 w_k1_KO_s006 + 1.18408 w_k1_MAT_s006
                                 + 0.83409 w_k1_MSFT_s006
                                 + 1.01565 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.22968 w_k1_ADM_s006
                                 + 1.08432 w_k1_AXP_s006
                                 + 1.17323 w_k1_DIS_s006
                                 + 1.00153 w_k1_FDX_s006
                                 + 1.03311 w_k1_IBM_s006
                                 + 0.893661 w_k1_INTC_s006
                                 + 1.05904 w_k1_KO_s006 + 1.01315 w_k1_MAT_s006
                                 + 0.98158 w_k1_MSFT_s006
                                 + 1.11155 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.890048 w_k1_ADM_s006
                                 + 0.953169 w_k1_AXP_s006
                                 + 0.755383 w_k1_DIS_s006
                                 + 0.895655 w_k1_FDX_s006
                                 + 0.645013 w_k1_IBM_s006
                                 + 0.870333 w_k1_INTC_s006
                                 + 0.668137 w_k1_KO_s006
                                 + 0.585448 w_k1_MAT_s006
                                 + 0.887762 w_k1_MSFT_s006
                                 + 0.881254 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.710217 w_k1_ADM_s006
                                 + 0.711074 w_k1_AXP_s006
                                 + 0.629498 w_k1_DIS_s006
                                 + 0.63807 w_k1_FDX_s006
                                 + 0.921328 w_k1_IBM_s006
                                 + 0.835047 w_k1_INTC_s006
                                 + 0.742801 w_k1_KO_s006
                                 + 0.706269 w_k1_MAT_s006
                                 + 0.725792 w_k1_MSFT_s006
                                 + 0.800044 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.893004 w_k1_ADM_s007
                                 + 1.05364 w_k1_AXP_s007
                                 + 0.980972 w_k1_DIS_s007
                                 + 0.94936 w_k1_FDX_s007
                                 + 1.01951 w_k1_IBM_s007
                                 + 0.937779 w_k1_INTC_s007
                                 + 1.04314 w_k1_KO_s007 + 1.08281 w_k1_MAT_s007
                                 + 1.04743 w_k1_MSFT_s007
                                 + 1.04238 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.861731 w_k1_ADM_s007
                                 + 1.07994 w_k1_AXP_s007
                                 + 0.72212 w_k1_DIS_s007
                                 + 0.964552 w_k1_FDX_s007
                                 + 0.750136 w_k1_IBM_s007
                                 + 1.07819 w_k1_INTC_s007
                                 + 0.848426 w_k1_KO_s007
                                 + 0.912648 w_k1_MAT_s007
                                 + 0.882908 w_k1_MSFT_s007
                                 + 1.06734 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.997569 w_k1_ADM_s007
                                 + 1.29581 w_k1_AXP_s007 + 1.2288 w_k1_DIS_s007
                                 + 1.13393 w_k1_FDX_s007
                                 + 1.31609 w_k1_IBM_s007
                                 + 1.01229 w_k1_INTC_s007
                                 + 1.22615 w_k1_KO_s007 + 1.26804 w_k1_MAT_s007
                                 + 1.08742 w_k1_MSFT_s007
                                 + 1.15237 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.11465 w_k1_ADM_s007
                                 + 1.3759 w_k1_AXP_s007 + 1.34291 w_k1_DIS_s007
                                 + 1.30669 w_k1_FDX_s007
                                 + 0.961244 w_k1_IBM_s007
                                 + 1.34958 w_k1_INTC_s007
                                 + 1.12453 w_k1_KO_s007 + 1.03432 w_k1_MAT_s007
                                 + 1.19655 w_k1_MSFT_s007
                                 + 1.21114 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.839292 w_k1_ADM_s007
                                 + 0.8477 w_k1_AXP_s007
                                 + 0.978917 w_k1_DIS_s007
                                 + 0.961842 w_k1_FDX_s007
                                 + 0.684752 w_k1_IBM_s007
                                 + 0.824316 w_k1_INTC_s007
                                 + 0.854416 w_k1_KO_s007
                                 + 0.668045 w_k1_MAT_s007
                                 + 1.01009 w_k1_MSFT_s007
                                 + 1.04242 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.915015 w_k1_ADM_s007
                                 + 0.971968 w_k1_AXP_s007
                                 + 1.33127 w_k1_DIS_s007
                                 + 1.12241 w_k1_FDX_s007
                                 + 1.04597 w_k1_IBM_s007
                                 + 1.25912 w_k1_INTC_s007
                                 + 1.05157 w_k1_KO_s007
                                 + 0.985342 w_k1_MAT_s007
                                 + 1.06239 w_k1_MSFT_s007
                                 + 1.09019 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.796589 w_k1_ADM_s007
                                 + 0.710983 w_k1_AXP_s007
                                 + 0.77222 w_k1_DIS_s007
                                 + 0.795894 w_k1_FDX_s007
                                 + 1.04633 w_k1_IBM_s007
                                 + 0.927111 w_k1_INTC_s007
                                 + 0.953028 w_k1_KO_s007
                                 + 1.01349 w_k1_MAT_s007
                                 + 0.910255 w_k1_MSFT_s007
                                 + 0.956827 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.943011 w_k1_ADM_s007
                                 + 1.17847 w_k1_AXP_s007
                                 + 1.30529 w_k1_DIS_s007
                                 + 1.04825 w_k1_FDX_s007
                                 + 1.84537 w_k1_IBM_s007
                                 + 1.29233 w_k1_INTC_s007
                                 + 1.43924 w_k1_KO_s007 + 1.78375 w_k1_MAT_s007
                                 + 1.251 w_k1_MSFT_s007 + 1.09699 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.08104 w_k1_ADM_s010
                                 + 0.991178 w_k1_AXP_s010
                                 + 1.12418 w_k1_DIS_s010
                                 + 1.06929 w_k1_FDX_s010
                                 + 0.713441 w_k1_IBM_s010
                                 + 0.94402 w_k1_INTC_s010
                                 + 0.839599 w_k1_KO_s010
                                 + 0.840607 w_k1_MAT_s010
                                 + 1.05287 w_k1_MSFT_s010
                                 + 0.946395 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.863627 w_k1_ADM_s010
                                 + 0.766195 w_k1_AXP_s010
                                 + 0.761847 w_k1_DIS_s010
                                 + 0.768256 w_k1_FDX_s010
                                 + 1.2576 w_k1_IBM_s010
                                 + 0.815846 w_k1_INTC_s010
                                 + 0.842229 w_k1_KO_s010 + 1.2241 w_k1_MAT_s010
                                 + 0.724817 w_k1_MSFT_s010
                                 + 0.877805 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.04017 w_k1_ADM_s010
                                 + 0.871359 w_k1_AXP_s010
                                 + 0.682554 w_k1_DIS_s010
                                 + 0.86931 w_k1_FDX_s010
                                 + 0.833039 w_k1_IBM_s010
                                 + 0.821991 w_k1_INTC_s010
                                 + 0.70744 w_k1_KO_s010 + 0.94332 w_k1_MAT_s010
                                 + 0.712358 w_k1_MSFT_s010
                                 + 0.908525 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.25209 w_k1_ADM_s010
                                 + 1.43501 w_k1_AXP_s010
                                 + 1.14251 w_k1_DIS_s010
                                 + 1.09911 w_k1_FDX_s010
                                 + 1.34911 w_k1_IBM_s010
                                 + 0.989513 w_k1_INTC_s010
                                 + 1.0194 w_k1_KO_s010 + 2.10984 w_k1_MAT_s010
                                 + 0.906339 w_k1_MSFT_s010
                                 + 1.10878 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.951081 w_k1_ADM_s010
                                 + 0.9579 w_k1_AXP_s010 + 1.13728 w_k1_DIS_s010
                                 + 0.990071 w_k1_FDX_s010
                                 + 1.42875 w_k1_IBM_s010
                                 + 1.01843 w_k1_INTC_s010
                                 + 1.30642 w_k1_KO_s010 + 1.76505 w_k1_MAT_s010
                                 + 1.13215 w_k1_MSFT_s010
                                 + 1.13068 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.17409 w_k1_ADM_s010
                                 + 1.45484 w_k1_AXP_s010
                                 + 1.29045 w_k1_DIS_s010
                                 + 1.36099 w_k1_FDX_s010
                                 + 1.09339 w_k1_IBM_s010
                                 + 1.24134 w_k1_INTC_s010
                                 + 1.05244 w_k1_KO_s010
                                 + 0.990824 w_k1_MAT_s010
                                 + 1.11821 w_k1_MSFT_s010
                                 + 1.04009 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.0868 w_k1_ADM_s010
                                 + 1.03447 w_k1_AXP_s010
                                 + 1.55678 w_k1_DIS_s010
                                 + 1.19377 w_k1_FDX_s010 + 1.1099 w_k1_IBM_s010
                                 + 1.14389 w_k1_INTC_s010 + 1.3476 w_k1_KO_s010
                                 + 1.26728 w_k1_MAT_s010
                                 + 1.12513 w_k1_MSFT_s010
                                 + 1.14415 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.899002 w_k1_ADM_s010
                                 + 0.827676 w_k1_AXP_s010
                                 + 0.823219 w_k1_DIS_s010
                                 + 0.936418 w_k1_FDX_s010
                                 + 0.938514 w_k1_IBM_s010
                                 + 0.891087 w_k1_INTC_s010
                                 + 0.801467 w_k1_KO_s010
                                 + 1.22668 w_k1_MAT_s010
                                 + 0.907329 w_k1_MSFT_s010
                                 + 0.918633 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.807871 w_k1_ADM_s011
                                 + 0.717907 w_k1_AXP_s011
                                 + 0.73276 w_k1_DIS_s011
                                 + 0.845269 w_k1_FDX_s011
                                 + 0.719471 w_k1_IBM_s011
                                 + 1.04457 w_k1_INTC_s011
                                 + 0.831235 w_k1_KO_s011
                                 + 0.697057 w_k1_MAT_s011
                                 + 0.796054 w_k1_MSFT_s011
                                 + 0.869467 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.84444 w_k1_ADM_s011
                                 + 0.732353 w_k1_AXP_s011
                                 + 0.631645 w_k1_DIS_s011
                                 + 0.701263 w_k1_FDX_s011
                                 + 1.41452 w_k1_IBM_s011
                                 + 0.907789 w_k1_INTC_s011
                                 + 1.1662 w_k1_KO_s011 + 1.50279 w_k1_MAT_s011
                                 + 0.792827 w_k1_MSFT_s011
                                 + 0.9655 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.3164 w_k1_ADM_s011
                                 + 1.36933 w_k1_AXP_s011
                                 + 1.20936 w_k1_DIS_s011
                                 + 1.27739 w_k1_FDX_s011
                                 + 1.10252 w_k1_IBM_s011
                                 + 1.32253 w_k1_INTC_s011 + 1.308 w_k1_KO_s011
                                 + 1.77124 w_k1_MAT_s011
                                 + 1.27877 w_k1_MSFT_s011
                                 + 1.16924 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.20324 w_k1_ADM_s011
                                 + 1.15497 w_k1_AXP_s011
                                 + 0.96797 w_k1_DIS_s011
                                 + 1.09372 w_k1_FDX_s011
                                 + 0.937488 w_k1_IBM_s011
                                 + 1.12684 w_k1_INTC_s011 + 1.0496 w_k1_KO_s011
                                 + 0.747495 w_k1_MAT_s011
                                 + 1.03172 w_k1_MSFT_s011
                                 + 1.03077 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.19142 w_k1_ADM_s011
                                 + 1.23045 w_k1_AXP_s011
                                 + 1.28551 w_k1_DIS_s011
                                 + 1.23393 w_k1_FDX_s011
                                 + 1.46088 w_k1_IBM_s011
                                 + 1.09753 w_k1_INTC_s011
                                 + 1.44664 w_k1_KO_s011 + 1.44738 w_k1_MAT_s011
                                 + 1.06626 w_k1_MSFT_s011 + 1.2505 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.979691 w_k1_ADM_s011
                                 + 1.02344 w_k1_AXP_s011
                                 + 0.902829 w_k1_DIS_s011
                                 + 1.06494 w_k1_FDX_s011
                                 + 0.825289 w_k1_IBM_s011
                                 + 1.1097 w_k1_INTC_s011
                                 + 0.903327 w_k1_KO_s011
                                 + 1.04158 w_k1_MAT_s011
                                 + 1.08265 w_k1_MSFT_s011
                                 + 0.956752 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.903913 w_k1_ADM_s011
                                 + 0.781544 w_k1_AXP_s011
                                 + 0.994778 w_k1_DIS_s011
                                 + 0.90264 w_k1_FDX_s011 + 1.1047 w_k1_IBM_s011
                                 + 0.938685 w_k1_INTC_s011
                                 + 1.16271 w_k1_KO_s011 + 1.15242 w_k1_MAT_s011
                                 + 0.8308 w_k1_MSFT_s011 + 0.94745 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.79181 w_k1_ADM_s011
                                 + 0.679782 w_k1_AXP_s011
                                 + 0.620505 w_k1_DIS_s011
                                 + 0.713594 w_k1_FDX_s011
                                 + 0.974745 w_k1_IBM_s011
                                 + 0.855021 w_k1_INTC_s011
                                 + 0.949428 w_k1_KO_s011
                                 + 1.10293 w_k1_MAT_s011
                                 + 0.87272 w_k1_MSFT_s011
                                 + 0.871687 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.973037 w_k1_ADM_s012
                                 + 0.99268 w_k1_AXP_s012
                                 + 0.72502 w_k1_DIS_s012
                                 + 0.94814 w_k1_FDX_s012
                                 + 0.740465 w_k1_IBM_s012
                                 + 1.11263 w_k1_INTC_s012
                                 + 0.793837 w_k1_KO_s012
                                 + 0.440925 w_k1_MAT_s012
                                 + 0.860846 w_k1_MSFT_s012
                                 + 0.914254 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.841559 w_k1_ADM_s012
                                 + 0.662721 w_k1_AXP_s012
                                 + 0.90438 w_k1_DIS_s012
                                 + 0.873803 w_k1_FDX_s012
                                 + 0.735512 w_k1_IBM_s012
                                 + 1.19003 w_k1_INTC_s012
                                 + 0.881212 w_k1_KO_s012
                                 + 0.499931 w_k1_MAT_s012
                                 + 0.956666 w_k1_MSFT_s012
                                 + 0.854304 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.2187 w_k1_ADM_s012
                                 + 1.34656 w_k1_AXP_s012
                                 + 1.22735 w_k1_DIS_s012
                                 + 1.27157 w_k1_FDX_s012
                                 + 0.866308 w_k1_IBM_s012
                                 + 1.12145 w_k1_INTC_s012
                                 + 0.879827 w_k1_KO_s012
                                 + 0.575258 w_k1_MAT_s012
                                 + 1.0832 w_k1_MSFT_s012 + 1.07397 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.0324 w_k1_ADM_s012
                                 + 1.51571 w_k1_AXP_s012
                                 + 1.80162 w_k1_DIS_s012
                                 + 1.43272 w_k1_FDX_s012
                                 + 1.20217 w_k1_IBM_s012
                                 + 1.68286 w_k1_INTC_s012
                                 + 1.27321 w_k1_KO_s012 + 1.27864 w_k1_MAT_s012
                                 + 1.26966 w_k1_MSFT_s012
                                 + 1.10294 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.19137 w_k1_ADM_s012
                                 + 1.23134 w_k1_AXP_s012
                                 + 1.43384 w_k1_DIS_s012
                                 + 1.10556 w_k1_FDX_s012
                                 + 1.31246 w_k1_IBM_s012
                                 + 1.40936 w_k1_INTC_s012
                                 + 1.25091 w_k1_KO_s012 + 1.00827 w_k1_MAT_s012
                                 + 1.24999 w_k1_MSFT_s012
                                 + 1.07609 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.01792 w_k1_ADM_s012
                                 + 0.85694 w_k1_AXP_s012
                                 + 1.13191 w_k1_DIS_s012
                                 + 1.05306 w_k1_FDX_s012
                                 + 0.786292 w_k1_IBM_s012
                                 + 1.17042 w_k1_INTC_s012
                                 + 0.948927 w_k1_KO_s012
                                 + 0.368924 w_k1_MAT_s012
                                 + 0.983678 w_k1_MSFT_s012
                                 + 0.978513 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.932063 w_k1_ADM_s012
                                 + 1.0283 w_k1_AXP_s012
                                 + 0.919009 w_k1_DIS_s012
                                 + 0.884573 w_k1_FDX_s012
                                 + 1.28167 w_k1_IBM_s012
                                 + 1.27413 w_k1_INTC_s012
                                 + 1.17058 w_k1_KO_s012
                                 + 0.736724 w_k1_MAT_s012
                                 + 0.973168 w_k1_MSFT_s012
                                 + 0.976928 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.68146 w_k1_ADM_s012
                                 + 0.753678 w_k1_AXP_s012
                                 + 0.620451 w_k1_DIS_s012
                                 + 0.772422 w_k1_FDX_s012
                                 + 0.670128 w_k1_IBM_s012
                                 + 1.0093 w_k1_INTC_s012
                                 + 0.685433 w_k1_KO_s012
                                 + 0.579604 w_k1_MAT_s012
                                 + 0.783377 w_k1_MSFT_s012
                                 + 0.706591 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.923746 w_k1_ADM_s013
                                 + 0.983239 w_k1_AXP_s013
                                 + 0.969631 w_k1_DIS_s013
                                 + 0.930482 w_k1_FDX_s013
                                 + 0.866578 w_k1_IBM_s013
                                 + 1.05404 w_k1_INTC_s013
                                 + 1.09515 w_k1_KO_s013
                                 + 0.739722 w_k1_MAT_s013
                                 + 1.00327 w_k1_MSFT_s013
                                 + 0.918381 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.810114 w_k1_ADM_s013
                                 + 0.68736 w_k1_AXP_s013
                                 + 0.647354 w_k1_DIS_s013
                                 + 0.816572 w_k1_FDX_s013
                                 + 0.76946 w_k1_IBM_s013
                                 + 0.951532 w_k1_INTC_s013
                                 + 0.661056 w_k1_KO_s013
                                 + 0.648859 w_k1_MAT_s013
                                 + 0.852662 w_k1_MSFT_s013
                                 + 0.778053 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.15593 w_k1_ADM_s013
                                 + 1.60768 w_k1_AXP_s013
                                 + 1.52287 w_k1_DIS_s013 + 1.3941 w_k1_FDX_s013
                                 + 1.50537 w_k1_IBM_s013
                                 + 1.34423 w_k1_INTC_s013
                                 + 1.29277 w_k1_KO_s013 + 1.30692 w_k1_MAT_s013
                                 + 1.10972 w_k1_MSFT_s013
                                 + 1.29876 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.01698 w_k1_ADM_s013
                                 + 1.29503 w_k1_AXP_s013
                                 + 1.01448 w_k1_DIS_s013
                                 + 1.25724 w_k1_FDX_s013
                                 + 0.763567 w_k1_IBM_s013
                                 + 1.42859 w_k1_INTC_s013
                                 + 0.854419 w_k1_KO_s013
                                 + 0.679224 w_k1_MAT_s013
                                 + 1.07683 w_k1_MSFT_s013
                                 + 1.06784 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.18138 w_k1_ADM_s013
                                 + 1.36161 w_k1_AXP_s013
                                 + 1.34703 w_k1_DIS_s013
                                 + 1.22322 w_k1_FDX_s013
                                 + 0.992106 w_k1_IBM_s013
                                 + 1.19914 w_k1_INTC_s013 + 1.1279 w_k1_KO_s013
                                 + 1.0604 w_k1_MAT_s013
                                 + 1.15525 w_k1_MSFT_s013
                                 + 1.04658 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.913539 w_k1_ADM_s013
                                 + 0.973238 w_k1_AXP_s013
                                 + 0.658147 w_k1_DIS_s013
                                 + 0.923851 w_k1_FDX_s013
                                 + 0.634194 w_k1_IBM_s013
                                 + 0.870427 w_k1_INTC_s013
                                 + 0.651906 w_k1_KO_s013
                                 + 0.636716 w_k1_MAT_s013
                                 + 0.832509 w_k1_MSFT_s013
                                 + 0.841471 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.815538 w_k1_ADM_s013
                                 + 1.26267 w_k1_AXP_s013
                                 + 1.15528 w_k1_DIS_s013
                                 + 1.10374 w_k1_FDX_s013
                                 + 1.38669 w_k1_IBM_s013
                                 + 1.47371 w_k1_INTC_s013
                                 + 1.13111 w_k1_KO_s013 + 1.10563 w_k1_MAT_s013
                                 + 1.0459 w_k1_MSFT_s013 + 1.17636 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.921242 w_k1_ADM_s013
                                 + 0.815351 w_k1_AXP_s013
                                 + 0.732296 w_k1_DIS_s013
                                 + 0.839026 w_k1_FDX_s013
                                 + 1.19318 w_k1_IBM_s013
                                 + 0.89034 w_k1_INTC_s013
                                 + 0.963396 w_k1_KO_s013
                                 + 0.985573 w_k1_MAT_s013
                                 + 0.816289 w_k1_MSFT_s013
                                 + 0.921462 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.907885 w_k1_ADM_s014
                                 + 0.781634 w_k1_AXP_s014
                                 + 0.993754 w_k1_DIS_s014
                                 + 0.927759 w_k1_FDX_s014
                                 + 0.995172 w_k1_IBM_s014
                                 + 0.8703 w_k1_INTC_s014
                                 + 0.908047 w_k1_KO_s014
                                 + 1.07441 w_k1_MAT_s014
                                 + 0.915336 w_k1_MSFT_s014
                                 + 1.05502 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.005 w_k1_ADM_s014
                                 + 0.965209 w_k1_AXP_s014
                                 + 1.00788 w_k1_DIS_s014
                                 + 1.10788 w_k1_FDX_s014
                                 + 0.822079 w_k1_IBM_s014
                                 + 1.17631 w_k1_INTC_s014
                                 + 0.806428 w_k1_KO_s014
                                 + 0.759424 w_k1_MAT_s014
                                 + 1.1042 w_k1_MSFT_s014 + 1.16373 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.06554 w_k1_ADM_s014
                                 + 1.21896 w_k1_AXP_s014
                                 + 1.32156 w_k1_DIS_s014
                                 + 1.21052 w_k1_FDX_s014
                                 + 0.840531 w_k1_IBM_s014
                                 + 1.18236 w_k1_INTC_s014
                                 + 0.996197 w_k1_KO_s014
                                 + 0.801195 w_k1_MAT_s014
                                 + 1.1036 w_k1_MSFT_s014 + 1.16476 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.723351 w_k1_ADM_s014
                                 + 0.776068 w_k1_AXP_s014
                                 + 0.683175 w_k1_DIS_s014
                                 + 0.827862 w_k1_FDX_s014
                                 + 0.674152 w_k1_IBM_s014
                                 + 0.775271 w_k1_INTC_s014
                                 + 0.716816 w_k1_KO_s014
                                 + 0.88949 w_k1_MAT_s014
                                 + 0.801971 w_k1_MSFT_s014
                                 + 0.894799 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.805071 w_k1_ADM_s014
                                 + 0.882854 w_k1_AXP_s014
                                 + 0.816719 w_k1_DIS_s014
                                 + 0.813579 w_k1_FDX_s014
                                 + 1.20056 w_k1_IBM_s014
                                 + 0.819941 w_k1_INTC_s014
                                 + 0.90647 w_k1_KO_s014 + 1.34234 w_k1_MAT_s014
                                 + 0.841154 w_k1_MSFT_s014
                                 + 1.00522 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.27438 w_k1_ADM_s014
                                 + 1.15995 w_k1_AXP_s014
                                 + 1.23824 w_k1_DIS_s014
                                 + 1.04062 w_k1_FDX_s014
                                 + 1.36773 w_k1_IBM_s014
                                 + 0.896945 w_k1_INTC_s014
                                 + 1.15745 w_k1_KO_s014 + 1.5604 w_k1_MAT_s014
                                 + 1.00209 w_k1_MSFT_s014
                                 + 1.11834 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.94603 w_k1_ADM_s014
                                 + 1.41177 w_k1_AXP_s014
                                 + 1.51015 w_k1_DIS_s014 + 1.276 w_k1_FDX_s014
                                 + 1.38041 w_k1_IBM_s014
                                 + 1.24351 w_k1_INTC_s014
                                 + 1.29547 w_k1_KO_s014 + 1.82951 w_k1_MAT_s014
                                 + 1.20868 w_k1_MSFT_s014
                                 + 1.26385 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.789729 w_k1_ADM_s014
                                 + 0.589989 w_k1_AXP_s014
                                 + 0.699661 w_k1_DIS_s014
                                 + 0.721634 w_k1_FDX_s014
                                 + 0.831246 w_k1_IBM_s014
                                 + 0.811567 w_k1_INTC_s014
                                 + 0.804335 w_k1_KO_s014
                                 + 0.59141 w_k1_MAT_s014
                                 + 0.76088 w_k1_MSFT_s014
                                 + 0.973538 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.2831 w_k1_ADM_s015
                                 + 1.37526 w_k1_AXP_s015
                                 + 1.97628 w_k1_DIS_s015
                                 + 1.53207 w_k1_FDX_s015
                                 + 1.22401 w_k1_IBM_s015
                                 + 1.23914 w_k1_INTC_s015
                                 + 1.39124 w_k1_KO_s015 + 1.34346 w_k1_MAT_s015
                                 + 1.54778 w_k1_MSFT_s015
                                 + 1.24947 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.844994 w_k1_ADM_s015
                                 + 0.79596 w_k1_AXP_s015
                                 + 0.737255 w_k1_DIS_s015
                                 + 0.935135 w_k1_FDX_s015
                                 + 0.743068 w_k1_IBM_s015
                                 + 0.912988 w_k1_INTC_s015
                                 + 0.806135 w_k1_KO_s015
                                 + 0.571913 w_k1_MAT_s015
                                 + 0.923754 w_k1_MSFT_s015
                                 + 0.925921 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.68463 w_k1_ADM_s015
                                 + 0.617383 w_k1_AXP_s015 + 0.632 w_k1_DIS_s015
                                 + 0.733088 w_k1_FDX_s015
                                 + 0.691304 w_k1_IBM_s015
                                 + 0.856213 w_k1_INTC_s015
                                 + 0.745865 w_k1_KO_s015
                                 + 0.621298 w_k1_MAT_s015
                                 + 0.807907 w_k1_MSFT_s015
                                 + 0.740643 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.794134 w_k1_ADM_s015
                                 + 0.913253 w_k1_AXP_s015
                                 + 0.881251 w_k1_DIS_s015
                                 + 0.873708 w_k1_FDX_s015
                                 + 1.52989 w_k1_IBM_s015
                                 + 0.945601 w_k1_INTC_s015
                                 + 1.09267 w_k1_KO_s015 + 1.39854 w_k1_MAT_s015
                                 + 0.892966 w_k1_MSFT_s015
                                 + 0.981896 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.931512 w_k1_ADM_s015
                                 + 1.05079 w_k1_AXP_s015
                                 + 1.26123 w_k1_DIS_s015
                                 + 1.24756 w_k1_FDX_s015
                                 + 0.754494 w_k1_IBM_s015
                                 + 1.07127 w_k1_INTC_s015
                                 + 1.03884 w_k1_KO_s015
                                 + 0.946387 w_k1_MAT_s015
                                 + 1.18865 w_k1_MSFT_s015
                                 + 1.09388 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.01988 w_k1_ADM_s015
                                 + 1.21614 w_k1_AXP_s015
                                 + 1.41258 w_k1_DIS_s015
                                 + 1.24906 w_k1_FDX_s015 + 1.3956 w_k1_IBM_s015
                                 + 1.03022 w_k1_INTC_s015
                                 + 1.44417 w_k1_KO_s015 + 1.58931 w_k1_MAT_s015
                                 + 1.26067 w_k1_MSFT_s015
                                 + 1.19769 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.939428 w_k1_ADM_s015
                                 + 1.1287 w_k1_AXP_s015 + 0.92644 w_k1_DIS_s015
                                 + 1.00422 w_k1_FDX_s015
                                 + 1.02622 w_k1_IBM_s015
                                 + 1.0037 w_k1_INTC_s015 + 1.01342 w_k1_KO_s015
                                 + 1.47123 w_k1_MAT_s015
                                 + 1.01411 w_k1_MSFT_s015
                                 + 1.03553 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.688233 w_k1_ADM_s015
                                 + 0.686191 w_k1_AXP_s015
                                 + 0.580465 w_k1_DIS_s015
                                 + 0.648593 w_k1_FDX_s015
                                 + 1.02744 w_k1_IBM_s015
                                 + 0.757341 w_k1_INTC_s015
                                 + 0.861105 w_k1_KO_s015
                                 + 1.03888 w_k1_MAT_s015
                                 + 0.67699 w_k1_MSFT_s015
                                 + 0.817544 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.961301 w_k1_ADM_s016
                                 + 0.985104 w_k1_AXP_s016
                                 + 0.800166 w_k1_DIS_s016
                                 + 0.994766 w_k1_FDX_s016
                                 + 0.683798 w_k1_IBM_s016
                                 + 1.12086 w_k1_INTC_s016
                                 + 0.750605 w_k1_KO_s016 + 1.0032 w_k1_MAT_s016
                                 + 1.05861 w_k1_MSFT_s016
                                 + 0.902349 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.24044 w_k1_ADM_s016
                                 + 1.57247 w_k1_AXP_s016
                                 + 1.41446 w_k1_DIS_s016
                                 + 1.20846 w_k1_FDX_s016
                                 + 1.37146 w_k1_IBM_s016
                                 + 1.23048 w_k1_INTC_s016
                                 + 1.25402 w_k1_KO_s016 + 1.9887 w_k1_MAT_s016
                                 + 1.25784 w_k1_MSFT_s016
                                 + 1.15747 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.972103 w_k1_ADM_s016
                                 + 0.869405 w_k1_AXP_s016
                                 + 1.04443 w_k1_DIS_s016
                                 + 1.02705 w_k1_FDX_s016
                                 + 0.851269 w_k1_IBM_s016
                                 + 1.17374 w_k1_INTC_s016
                                 + 1.03238 w_k1_KO_s016
                                 + 0.746722 w_k1_MAT_s016
                                 + 1.09626 w_k1_MSFT_s016
                                 + 1.01602 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.739634 w_k1_ADM_s016
                                 + 0.814446 w_k1_AXP_s016
                                 + 0.671797 w_k1_DIS_s016
                                 + 0.767559 w_k1_FDX_s016
                                 + 0.944576 w_k1_IBM_s016
                                 + 1.00589 w_k1_INTC_s016
                                 + 0.867935 w_k1_KO_s016
                                 + 1.24949 w_k1_MAT_s016
                                 + 0.936712 w_k1_MSFT_s016
                                 + 0.854737 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.04803 w_k1_ADM_s016
                                 + 0.96048 w_k1_AXP_s016
                                 + 0.891716 w_k1_DIS_s016
                                 + 0.9403 w_k1_FDX_s016
                                 + 0.949793 w_k1_IBM_s016
                                 + 0.830421 w_k1_INTC_s016
                                 + 1.00231 w_k1_KO_s016 + 1.29789 w_k1_MAT_s016
                                 + 1.03392 w_k1_MSFT_s016
                                 + 0.999798 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.934394 w_k1_ADM_s016
                                 + 0.892964 w_k1_AXP_s016
                                 + 0.984388 w_k1_DIS_s016
                                 + 0.92497 w_k1_FDX_s016
                                 + 1.26302 w_k1_IBM_s016
                                 + 1.11861 w_k1_INTC_s016
                                 + 1.03211 w_k1_KO_s016 + 1.32975 w_k1_MAT_s016
                                 + 0.979398 w_k1_MSFT_s016
                                 + 1.02073 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.23522 w_k1_ADM_s016
                                 + 1.23108 w_k1_AXP_s016
                                 + 1.51213 w_k1_DIS_s016
                                 + 1.35644 w_k1_FDX_s016 + 1.0918 w_k1_IBM_s016
                                 + 1.46356 w_k1_INTC_s016
                                 + 1.25572 w_k1_KO_s016 + 1.62717 w_k1_MAT_s016
                                 + 1.4149 w_k1_MSFT_s016 + 1.11966 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.730912 w_k1_ADM_s016
                                 + 0.604961 w_k1_AXP_s016
                                 + 0.491427 w_k1_DIS_s016
                                 + 0.667619 w_k1_FDX_s016
                                 + 0.707226 w_k1_IBM_s016
                                 + 0.720435 w_k1_INTC_s016
                                 + 0.678699 w_k1_KO_s016 + 1.0065 w_k1_MAT_s016
                                 + 0.67648 w_k1_MSFT_s016
                                 + 0.782538 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.10228 w_k1_ADM_s017
                                 + 1.38165 w_k1_AXP_s017
                                 + 1.73047 w_k1_DIS_s017
                                 + 1.29118 w_k1_FDX_s017
                                 + 1.31887 w_k1_IBM_s017
                                 + 1.16478 w_k1_INTC_s017
                                 + 1.21678 w_k1_KO_s017 + 1.26794 w_k1_MAT_s017
                                 + 1.22353 w_k1_MSFT_s017
                                 + 1.14095 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.956494 w_k1_ADM_s017
                                 + 1.04921 w_k1_AXP_s017
                                 + 0.916307 w_k1_DIS_s017
                                 + 0.994498 w_k1_FDX_s017
                                 + 0.794061 w_k1_IBM_s017
                                 + 1.06819 w_k1_INTC_s017
                                 + 0.791884 w_k1_KO_s017
                                 + 0.661524 w_k1_MAT_s017
                                 + 0.966533 w_k1_MSFT_s017
                                 + 0.981667 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.574758 w_k1_ADM_s017
                                 + 0.894447 w_k1_AXP_s017
                                 + 0.68443 w_k1_DIS_s017
                                 + 0.790162 w_k1_FDX_s017
                                 + 1.16109 w_k1_IBM_s017
                                 + 0.80573 w_k1_INTC_s017
                                 + 0.751993 w_k1_KO_s017
                                 + 0.969165 w_k1_MAT_s017
                                 + 0.762758 w_k1_MSFT_s017
                                 + 0.911858 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.81364 w_k1_ADM_s017
                                 + 0.919434 w_k1_AXP_s017
                                 + 1.44182 w_k1_DIS_s017
                                 + 1.07344 w_k1_FDX_s017
                                 + 1.17564 w_k1_IBM_s017
                                 + 0.927632 w_k1_INTC_s017
                                 + 1.30986 w_k1_KO_s017 + 1.00928 w_k1_MAT_s017
                                 + 1.23219 w_k1_MSFT_s017
                                 + 1.13772 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.902216 w_k1_ADM_s017
                                 + 0.951543 w_k1_AXP_s017
                                 + 1.25427 w_k1_DIS_s017
                                 + 1.17879 w_k1_FDX_s017
                                 + 0.925019 w_k1_IBM_s017
                                 + 0.952183 w_k1_INTC_s017
                                 + 0.951286 w_k1_KO_s017
                                 + 0.763866 w_k1_MAT_s017
                                 + 1.09289 w_k1_MSFT_s017
                                 + 1.05357 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.997838 w_k1_ADM_s017
                                 + 1.46885 w_k1_AXP_s017
                                 + 0.912193 w_k1_DIS_s017
                                 + 1.02576 w_k1_FDX_s017
                                 + 1.03079 w_k1_IBM_s017
                                 + 1.09467 w_k1_INTC_s017
                                 + 0.955961 w_k1_KO_s017
                                 + 1.32907 w_k1_MAT_s017
                                 + 1.02682 w_k1_MSFT_s017
                                 + 1.08766 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.7437 w_k1_ADM_s017
                                 + 0.627457 w_k1_AXP_s017
                                 + 0.644008 w_k1_DIS_s017
                                 + 0.730507 w_k1_FDX_s017
                                 + 0.857734 w_k1_IBM_s017
                                 + 0.77132 w_k1_INTC_s017
                                 + 0.794974 w_k1_KO_s017
                                 + 0.796995 w_k1_MAT_s017
                                 + 0.748439 w_k1_MSFT_s017
                                 + 0.861162 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.849678 w_k1_ADM_s017
                                 + 1.10336 w_k1_AXP_s017
                                 + 1.06912 w_k1_DIS_s017
                                 + 0.918183 w_k1_FDX_s017
                                 + 1.51381 w_k1_IBM_s017
                                 + 0.877975 w_k1_INTC_s017
                                 + 1.24136 w_k1_KO_s017 + 1.56128 w_k1_MAT_s017
                                 + 0.880369 w_k1_MSFT_s017
                                 + 1.05773 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.924438 w_k1_ADM_s100
                                 + 0.832177 w_k1_AXP_s100
                                 + 1.37207 w_k1_DIS_s100
                                 + 1.22186 w_k1_FDX_s100
                                 + 0.845065 w_k1_IBM_s100
                                 + 1.32508 w_k1_INTC_s100
                                 + 1.14505 w_k1_KO_s100
                                 + 0.736864 w_k1_MAT_s100
                                 + 1.40887 w_k1_MSFT_s100
                                 + 1.08228 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.15525 w_k1_ADM_s100
                                 + 1.3839 w_k1_AXP_s100 + 1.66517 w_k1_DIS_s100
                                 + 1.33566 w_k1_FDX_s100
                                 + 1.50477 w_k1_IBM_s100
                                 + 0.998513 w_k1_INTC_s100
                                 + 1.45802 w_k1_KO_s100 + 2.35291 w_k1_MAT_s100
                                 + 1.46013 w_k1_MSFT_s100
                                 + 1.38417 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.801751 w_k1_ADM_s100
                                 + 0.821018 w_k1_AXP_s100
                                 + 0.848576 w_k1_DIS_s100
                                 + 0.692235 w_k1_FDX_s100
                                 + 1.18417 w_k1_IBM_s100
                                 + 0.814559 w_k1_INTC_s100
                                 + 1.03405 w_k1_KO_s100 + 1.06731 w_k1_MAT_s100
                                 + 0.878981 w_k1_MSFT_s100
                                 + 0.898705 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.9709 w_k1_ADM_s100
                                 + 1.11421 w_k1_AXP_s100
                                 + 1.21081 w_k1_DIS_s100
                                 + 1.12279 w_k1_FDX_s100
                                 + 1.29256 w_k1_IBM_s100
                                 + 1.29059 w_k1_INTC_s100
                                 + 1.29882 w_k1_KO_s100 + 1.04963 w_k1_MAT_s100
                                 + 1.33961 w_k1_MSFT_s100
                                 + 1.17249 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.776788 w_k1_ADM_s100
                                 + 0.899245 w_k1_AXP_s100
                                 + 0.712423 w_k1_DIS_s100
                                 + 0.872449 w_k1_FDX_s100
                                 + 0.76806 w_k1_IBM_s100
                                 + 0.834423 w_k1_INTC_s100
                                 + 0.670677 w_k1_KO_s100
                                 + 0.750987 w_k1_MAT_s100
                                 + 0.750046 w_k1_MSFT_s100
                                 + 0.820344 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.48308 w_k1_ADM_s100
                                 + 1.26114 w_k1_AXP_s100
                                 + 1.33162 w_k1_DIS_s100
                                 + 1.36657 w_k1_FDX_s100
                                 + 0.683414 w_k1_IBM_s100
                                 + 1.11108 w_k1_INTC_s100
                                 + 0.906397 w_k1_KO_s100
                                 + 0.669895 w_k1_MAT_s100
                                 + 1.32331 w_k1_MSFT_s100 + 1.1078 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.738761 w_k1_ADM_s100
                                 + 1.04387 w_k1_AXP_s100
                                 + 0.787691 w_k1_DIS_s100
                                 + 0.79466 w_k1_FDX_s100
                                 + 1.59534 w_k1_IBM_s100
                                 + 1.02872 w_k1_INTC_s100
                                 + 1.13066 w_k1_KO_s100 + 1.7006 w_k1_MAT_s100
                                 + 0.907765 w_k1_MSFT_s100
                                 + 1.03175 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.640042 w_k1_ADM_s100
                                 + 1.01028 w_k1_AXP_s100
                                 + 1.01309 w_k1_DIS_s100
                                 + 0.898784 w_k1_FDX_s100
                                 + 1.42679 w_k1_IBM_s100
                                 + 1.21737 w_k1_INTC_s100
                                 + 1.11317 w_k1_KO_s100 + 1.13053 w_k1_MAT_s100
                                 + 1.00914 w_k1_MSFT_s100
                                 + 1.07294 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.87227 w_k1_ADM_s101
                                 + 1.07702 w_k1_AXP_s101
                                 + 0.873625 w_k1_DIS_s101
                                 + 0.782264 w_k1_FDX_s101
                                 + 1.10865 w_k1_IBM_s101
                                 + 1.10488 w_k1_INTC_s101
                                 + 1.01513 w_k1_KO_s101 + 1.5009 w_k1_MAT_s101
                                 + 0.912148 w_k1_MSFT_s101
                                 + 0.950472 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.516522 w_k1_ADM_s101
                                 + 0.791514 w_k1_AXP_s101
                                 + 0.493861 w_k1_DIS_s101
                                 + 0.652255 w_k1_FDX_s101
                                 + 0.789309 w_k1_IBM_s101
                                 + 0.76738 w_k1_INTC_s101
                                 + 0.684834 w_k1_KO_s101
                                 + 1.25169 w_k1_MAT_s101
                                 + 0.712305 w_k1_MSFT_s101
                                 + 0.825432 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.658252 w_k1_ADM_s101
                                 + 1.10463 w_k1_AXP_s101
                                 + 1.05299 w_k1_DIS_s101
                                 + 0.997036 w_k1_FDX_s101
                                 + 0.857561 w_k1_IBM_s101
                                 + 0.957379 w_k1_INTC_s101
                                 + 0.943058 w_k1_KO_s101
                                 + 1.17735 w_k1_MAT_s101
                                 + 0.97257 w_k1_MSFT_s101
                                 + 1.10843 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09018 w_k1_ADM_s101
                                 + 1.19511 w_k1_AXP_s101
                                 + 1.48107 w_k1_DIS_s101
                                 + 1.26815 w_k1_FDX_s101
                                 + 0.898509 w_k1_IBM_s101
                                 + 1.07066 w_k1_INTC_s101
                                 + 0.976513 w_k1_KO_s101
                                 + 0.891339 w_k1_MAT_s101
                                 + 1.17508 w_k1_MSFT_s101
                                 + 0.969952 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.975228 w_k1_ADM_s101
                                 + 1.14173 w_k1_AXP_s101
                                 + 0.769306 w_k1_DIS_s101
                                 + 0.936523 w_k1_FDX_s101
                                 + 0.907054 w_k1_IBM_s101
                                 + 0.907584 w_k1_INTC_s101
                                 + 0.913863 w_k1_KO_s101
                                 + 1.32673 w_k1_MAT_s101
                                 + 0.829868 w_k1_MSFT_s101
                                 + 0.95511 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.723143 w_k1_ADM_s101
                                 + 1.45509 w_k1_AXP_s101
                                 + 1.16448 w_k1_DIS_s101
                                 + 1.24933 w_k1_FDX_s101
                                 + 0.887179 w_k1_IBM_s101
                                 + 1.14517 w_k1_INTC_s101
                                 + 0.887922 w_k1_KO_s101
                                 + 1.55448 w_k1_MAT_s101
                                 + 1.21725 w_k1_MSFT_s101
                                 + 1.14064 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.923219 w_k1_ADM_s101
                                 + 1.37776 w_k1_AXP_s101
                                 + 1.10682 w_k1_DIS_s101
                                 + 1.09215 w_k1_FDX_s101
                                 + 1.41732 w_k1_IBM_s101
                                 + 0.919322 w_k1_INTC_s101
                                 + 1.25222 w_k1_KO_s101 + 2.46315 w_k1_MAT_s101
                                 + 1.10046 w_k1_MSFT_s101
                                 + 1.32783 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.07682 w_k1_ADM_s101
                                 + 1.27722 w_k1_AXP_s101
                                 + 1.94327 w_k1_DIS_s101
                                 + 1.44834 w_k1_FDX_s101
                                 + 1.44605 w_k1_IBM_s101
                                 + 1.42166 w_k1_INTC_s101
                                 + 1.33954 w_k1_KO_s101 + 1.52048 w_k1_MAT_s101
                                 + 1.42775 w_k1_MSFT_s101
                                 + 1.10841 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.22735 w_k1_ADM_s102
                                 + 0.637871 w_k1_AXP_s102
                                 + 0.969195 w_k1_DIS_s102
                                 + 0.969745 w_k1_FDX_s102
                                 + 0.865497 w_k1_IBM_s102
                                 + 0.809882 w_k1_INTC_s102
                                 + 0.787653 w_k1_KO_s102
                                 + 0.523981 w_k1_MAT_s102
                                 + 0.941304 w_k1_MSFT_s102
                                 + 0.855649 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.2435 w_k1_ADM_s102
                                 + 1.35853 w_k1_AXP_s102
                                 + 1.48655 w_k1_DIS_s102
                                 + 1.41889 w_k1_FDX_s102
                                 + 1.24332 w_k1_IBM_s102
                                 + 1.29077 w_k1_INTC_s102 + 1.1979 w_k1_KO_s102
                                 + 1.37126 w_k1_MAT_s102
                                 + 1.36492 w_k1_MSFT_s102
                                 + 1.37782 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.00169 w_k1_ADM_s102
                                 + 0.910969 w_k1_AXP_s102
                                 + 0.961365 w_k1_DIS_s102
                                 + 0.879281 w_k1_FDX_s102
                                 + 1.38872 w_k1_IBM_s102
                                 + 0.734391 w_k1_INTC_s102
                                 + 0.955692 w_k1_KO_s102
                                 + 1.65595 w_k1_MAT_s102
                                 + 0.931163 w_k1_MSFT_s102
                                 + 1.09962 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.12534 w_k1_ADM_s102
                                 + 1.14397 w_k1_AXP_s102
                                 + 1.17128 w_k1_DIS_s102
                                 + 1.24963 w_k1_FDX_s102
                                 + 0.968328 w_k1_IBM_s102
                                 + 1.2406 w_k1_INTC_s102
                                 + 0.978695 w_k1_KO_s102
                                 + 1.27059 w_k1_MAT_s102
                                 + 1.37609 w_k1_MSFT_s102
                                 + 1.10251 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.984533 w_k1_ADM_s102
                                 + 0.91366 w_k1_AXP_s102
                                 + 1.15008 w_k1_DIS_s102
                                 + 1.04817 w_k1_FDX_s102
                                 + 1.03461 w_k1_IBM_s102
                                 + 0.974099 w_k1_INTC_s102 + 1.091 w_k1_KO_s102
                                 + 0.993287 w_k1_MAT_s102
                                 + 1.05112 w_k1_MSFT_s102
                                 + 1.05204 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.42182 w_k1_ADM_s102
                                 + 1.20094 w_k1_AXP_s102
                                 + 1.04519 w_k1_DIS_s102
                                 + 1.11705 w_k1_FDX_s102
                                 + 1.18036 w_k1_IBM_s102
                                 + 1.00329 w_k1_INTC_s102
                                 + 1.06913 w_k1_KO_s102 + 1.5912 w_k1_MAT_s102
                                 + 1.13489 w_k1_MSFT_s102 + 1.0617 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.946011 w_k1_ADM_s102
                                 + 0.741308 w_k1_AXP_s102
                                 + 0.813191 w_k1_DIS_s102
                                 + 1.09421 w_k1_FDX_s102
                                 + 0.657205 w_k1_IBM_s102
                                 + 1.0633 w_k1_INTC_s102
                                 + 0.792366 w_k1_KO_s102
                                 + 0.756156 w_k1_MAT_s102
                                 + 1.15452 w_k1_MSFT_s102 + 1.0345 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.983924 w_k1_ADM_s102
                                 + 0.836478 w_k1_AXP_s102
                                 + 0.769347 w_k1_DIS_s102
                                 + 0.855305 w_k1_FDX_s102
                                 + 0.797747 w_k1_IBM_s102
                                 + 0.729101 w_k1_INTC_s102
                                 + 0.780234 w_k1_KO_s102
                                 + 0.876214 w_k1_MAT_s102
                                 + 0.966922 w_k1_MSFT_s102
                                 + 1.0239 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.12596 w_k1_ADM_s103
                                 + 1.08014 w_k1_AXP_s103
                                 + 1.38229 w_k1_DIS_s103
                                 + 1.16018 w_k1_FDX_s103
                                 + 1.22136 w_k1_IBM_s103
                                 + 0.916549 w_k1_INTC_s103
                                 + 1.17428 w_k1_KO_s103 + 1.11075 w_k1_MAT_s103
                                 + 0.948861 w_k1_MSFT_s103
                                 + 1.08456 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.84485 w_k1_ADM_s103
                                 + 0.776964 w_k1_AXP_s103
                                 + 0.756466 w_k1_DIS_s103
                                 + 0.839962 w_k1_FDX_s103
                                 + 1.17649 w_k1_IBM_s103
                                 + 0.964872 w_k1_INTC_s103
                                 + 1.12436 w_k1_KO_s103 + 1.04775 w_k1_MAT_s103
                                 + 0.836721 w_k1_MSFT_s103
                                 + 0.91463 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.939038 w_k1_ADM_s103
                                 + 0.942815 w_k1_AXP_s103
                                 + 1.42433 w_k1_DIS_s103
                                 + 1.20341 w_k1_FDX_s103
                                 + 1.59025 w_k1_IBM_s103
                                 + 1.1308 w_k1_INTC_s103 + 1.3717 w_k1_KO_s103
                                 + 1.48777 w_k1_MAT_s103
                                 + 1.20046 w_k1_MSFT_s103
                                 + 1.19317 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.22139 w_k1_ADM_s103
                                 + 0.784166 w_k1_AXP_s103
                                 + 1.17662 w_k1_DIS_s103
                                 + 1.14289 w_k1_FDX_s103
                                 + 0.76812 w_k1_IBM_s103
                                 + 0.931546 w_k1_INTC_s103
                                 + 1.05466 w_k1_KO_s103
                                 + 0.669957 w_k1_MAT_s103
                                 + 1.0142 w_k1_MSFT_s103 + 1.18084 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.01596 w_k1_ADM_s103
                                 + 0.878593 w_k1_AXP_s103
                                 + 0.918502 w_k1_DIS_s103
                                 + 1.04181 w_k1_FDX_s103
                                 + 0.889832 w_k1_IBM_s103
                                 + 0.949958 w_k1_INTC_s103
                                 + 0.942969 w_k1_KO_s103
                                 + 0.691084 w_k1_MAT_s103
                                 + 0.895485 w_k1_MSFT_s103
                                 + 0.86216 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.12622 w_k1_ADM_s103
                                 + 1.01921 w_k1_AXP_s103 + 1.1845 w_k1_DIS_s103
                                 + 1.24489 w_k1_FDX_s103
                                 + 0.868433 w_k1_IBM_s103 + 1.14 w_k1_INTC_s103
                                 + 1.0414 w_k1_KO_s103 + 0.562219 w_k1_MAT_s103
                                 + 1.04305 w_k1_MSFT_s103
                                 + 1.05764 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.780868 w_k1_ADM_s103
                                 + 0.615977 w_k1_AXP_s103
                                 + 0.694571 w_k1_DIS_s103
                                 + 0.797389 w_k1_FDX_s103
                                 + 0.827928 w_k1_IBM_s103
                                 + 0.697644 w_k1_INTC_s103
                                 + 0.745056 w_k1_KO_s103
                                 + 0.785425 w_k1_MAT_s103
                                 + 0.793487 w_k1_MSFT_s103
                                 + 0.757623 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.25851 w_k1_ADM_s103
                                 + 1.15271 w_k1_AXP_s103
                                 + 1.60033 w_k1_DIS_s103
                                 + 1.41544 w_k1_FDX_s103
                                 + 1.04662 w_k1_IBM_s103
                                 + 1.35162 w_k1_INTC_s103
                                 + 1.39325 w_k1_KO_s103 + 1.05445 w_k1_MAT_s103
                                 + 1.25128 w_k1_MSFT_s103 + 1.2146 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.13884 w_k1_ADM_s104
                                 + 1.29603 w_k1_AXP_s104
                                 + 1.37616 w_k1_DIS_s104
                                 + 1.19131 w_k1_FDX_s104
                                 + 0.992864 w_k1_IBM_s104
                                 + 1.09044 w_k1_INTC_s104
                                 + 1.14818 w_k1_KO_s104
                                 + 0.863445 w_k1_MAT_s104
                                 + 1.17109 w_k1_MSFT_s104
                                 + 1.00228 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.10336 w_k1_ADM_s104
                                 + 1.54068 w_k1_AXP_s104
                                 + 1.49211 w_k1_DIS_s104
                                 + 1.16804 w_k1_FDX_s104
                                 + 1.48076 w_k1_IBM_s104
                                 + 1.27666 w_k1_INTC_s104
                                 + 1.56974 w_k1_KO_s104 + 1.46234 w_k1_MAT_s104
                                 + 1.43197 w_k1_MSFT_s104
                                 + 1.10329 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.20988 w_k1_ADM_s104
                                 + 1.15935 w_k1_AXP_s104
                                 + 0.639652 w_k1_DIS_s104
                                 + 0.959928 w_k1_FDX_s104
                                 + 0.723323 w_k1_IBM_s104
                                 + 0.874686 w_k1_INTC_s104
                                 + 0.746705 w_k1_KO_s104
                                 + 0.50554 w_k1_MAT_s104
                                 + 0.84119 w_k1_MSFT_s104
                                 + 0.943262 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.734795 w_k1_ADM_s104
                                 + 0.972903 w_k1_AXP_s104
                                 + 0.874478 w_k1_DIS_s104
                                 + 0.983448 w_k1_FDX_s104
                                 + 0.869394 w_k1_IBM_s104
                                 + 1.02374 w_k1_INTC_s104
                                 + 0.876168 w_k1_KO_s104
                                 + 0.717586 w_k1_MAT_s104
                                 + 1.00652 w_k1_MSFT_s104
                                 + 0.939199 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.853766 w_k1_ADM_s104
                                 + 1.0121 w_k1_AXP_s104
                                 + 0.818442 w_k1_DIS_s104
                                 + 0.889201 w_k1_FDX_s104
                                 + 1.30872 w_k1_IBM_s104
                                 + 0.934649 w_k1_INTC_s104
                                 + 1.13202 w_k1_KO_s104 + 1.5014 w_k1_MAT_s104
                                 + 0.913818 w_k1_MSFT_s104
                                 + 1.0518 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.06229 w_k1_ADM_s104
                                 + 1.02925 w_k1_AXP_s104
                                 + 1.00466 w_k1_DIS_s104
                                 + 1.02787 w_k1_FDX_s104
                                 + 0.841915 w_k1_IBM_s104
                                 + 1.08299 w_k1_INTC_s104
                                 + 0.864592 w_k1_KO_s104
                                 + 0.459782 w_k1_MAT_s104
                                 + 0.96721 w_k1_MSFT_s104
                                 + 0.903516 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.709474 w_k1_ADM_s104
                                 + 0.767932 w_k1_AXP_s104
                                 + 0.644066 w_k1_DIS_s104
                                 + 0.811528 w_k1_FDX_s104
                                 + 0.819819 w_k1_IBM_s104
                                 + 0.861445 w_k1_INTC_s104
                                 + 0.740945 w_k1_KO_s104
                                 + 0.551403 w_k1_MAT_s104
                                 + 0.886183 w_k1_MSFT_s104
                                 + 0.905285 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.981619 w_k1_ADM_s104
                                 + 1.41691 w_k1_AXP_s104
                                 + 1.34553 w_k1_DIS_s104
                                 + 1.40727 w_k1_FDX_s104
                                 + 0.905096 w_k1_IBM_s104
                                 + 1.07909 w_k1_INTC_s104
                                 + 0.995674 w_k1_KO_s104
                                 + 1.03658 w_k1_MAT_s104
                                 + 1.36716 w_k1_MSFT_s104 + 1.0838 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.08422 w_k1_ADM_s105
                                 + 1.50884 w_k1_AXP_s105
                                 + 1.35624 w_k1_DIS_s105 + 1.3133 w_k1_FDX_s105
                                 + 1.23196 w_k1_IBM_s105
                                 + 1.3712 w_k1_INTC_s105 + 1.28883 w_k1_KO_s105
                                 + 1.47542 w_k1_MAT_s105
                                 + 1.25348 w_k1_MSFT_s105
                                 + 1.16469 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.16595 w_k1_ADM_s105
                                 + 1.19844 w_k1_AXP_s105
                                 + 0.774611 w_k1_DIS_s105
                                 + 0.8627 w_k1_FDX_s105 + 1.25241 w_k1_IBM_s105
                                 + 1.14302 w_k1_INTC_s105
                                 + 0.977657 w_k1_KO_s105
                                 + 1.13922 w_k1_MAT_s105
                                 + 0.880561 w_k1_MSFT_s105
                                 + 0.943321 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.1682 w_k1_ADM_s105
                                 + 1.29911 w_k1_AXP_s105
                                 + 1.09768 w_k1_DIS_s105
                                 + 1.10084 w_k1_FDX_s105
                                 + 1.01287 w_k1_IBM_s105
                                 + 0.968009 w_k1_INTC_s105
                                 + 1.01638 w_k1_KO_s105
                                 + 0.879314 w_k1_MAT_s105
                                 + 0.965399 w_k1_MSFT_s105
                                 + 1.00262 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.853811 w_k1_ADM_s105
                                 + 0.702758 w_k1_AXP_s105
                                 + 0.606234 w_k1_DIS_s105
                                 + 0.794246 w_k1_FDX_s105
                                 + 0.743912 w_k1_IBM_s105
                                 + 0.901763 w_k1_INTC_s105
                                 + 0.733912 w_k1_KO_s105
                                 + 0.508664 w_k1_MAT_s105
                                 + 0.733893 w_k1_MSFT_s105
                                 + 0.899397 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.19388 w_k1_ADM_s105
                                 + 0.974574 w_k1_AXP_s105
                                 + 1.03115 w_k1_DIS_s105
                                 + 1.21079 w_k1_FDX_s105
                                 + 0.710861 w_k1_IBM_s105
                                 + 0.829353 w_k1_INTC_s105
                                 + 0.867442 w_k1_KO_s105
                                 + 0.760598 w_k1_MAT_s105
                                 + 0.936086 w_k1_MSFT_s105
                                 + 0.984053 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.01602 w_k1_ADM_s105
                                 + 1.59402 w_k1_AXP_s105
                                 + 1.24895 w_k1_DIS_s105
                                 + 1.10599 w_k1_FDX_s105
                                 + 1.60609 w_k1_IBM_s105
                                 + 1.25358 w_k1_INTC_s105
                                 + 1.25116 w_k1_KO_s105 + 1.54105 w_k1_MAT_s105
                                 + 1.01795 w_k1_MSFT_s105
                                 + 1.17476 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.957546 w_k1_ADM_s105
                                 + 1.03584 w_k1_AXP_s105
                                 + 0.807106 w_k1_DIS_s105
                                 + 0.933474 w_k1_FDX_s105
                                 + 0.872937 w_k1_IBM_s105
                                 + 0.916381 w_k1_INTC_s105
                                 + 0.838325 w_k1_KO_s105
                                 + 1.09411 w_k1_MAT_s105
                                 + 0.806446 w_k1_MSFT_s105
                                 + 0.857062 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.656026 w_k1_ADM_s105
                                 + 0.809968 w_k1_AXP_s105
                                 + 0.469295 w_k1_DIS_s105
                                 + 0.631497 w_k1_FDX_s105
                                 + 1.00756 w_k1_IBM_s105
                                 + 0.898557 w_k1_INTC_s105
                                 + 0.817099 w_k1_KO_s105
                                 + 0.760216 w_k1_MAT_s105
                                 + 0.56536 w_k1_MSFT_s105
                                 + 0.697707 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.27054 w_k1_ADM_s106
                                 + 1.19223 w_k1_AXP_s106
                                 + 1.28704 w_k1_DIS_s106 + 1.2004 w_k1_FDX_s106
                                 + 0.939895 w_k1_IBM_s106
                                 + 0.964442 w_k1_INTC_s106
                                 + 0.967674 w_k1_KO_s106
                                 + 0.920094 w_k1_MAT_s106
                                 + 1.00569 w_k1_MSFT_s106
                                 + 0.962539 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.805419 w_k1_ADM_s106
                                 + 0.884572 w_k1_AXP_s106
                                 + 0.539151 w_k1_DIS_s106
                                 + 0.699483 w_k1_FDX_s106
                                 + 0.923145 w_k1_IBM_s106
                                 + 0.725855 w_k1_INTC_s106
                                 + 0.774222 w_k1_KO_s106
                                 + 1.05412 w_k1_MAT_s106
                                 + 0.609805 w_k1_MSFT_s106
                                 + 0.83723 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.865536 w_k1_ADM_s106
                                 + 0.750065 w_k1_AXP_s106
                                 + 0.914388 w_k1_DIS_s106
                                 + 0.93841 w_k1_FDX_s106
                                 + 0.815162 w_k1_IBM_s106
                                 + 1.00499 w_k1_INTC_s106
                                 + 0.930417 w_k1_KO_s106
                                 + 0.681794 w_k1_MAT_s106
                                 + 0.854159 w_k1_MSFT_s106
                                 + 0.926954 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.26901 w_k1_ADM_s106
                                 + 1.177 w_k1_AXP_s106 + 1.01921 w_k1_DIS_s106
                                 + 0.944502 w_k1_FDX_s106
                                 + 1.49563 w_k1_IBM_s106
                                 + 0.837486 w_k1_INTC_s106
                                 + 1.12848 w_k1_KO_s106 + 1.88295 w_k1_MAT_s106
                                 + 0.894076 w_k1_MSFT_s106
                                 + 1.01844 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.42713 w_k1_ADM_s106
                                 + 2.25659 w_k1_AXP_s106
                                 + 1.34345 w_k1_DIS_s106
                                 + 1.34207 w_k1_FDX_s106
                                 + 1.64269 w_k1_IBM_s106
                                 + 1.01684 w_k1_INTC_s106
                                 + 1.20807 w_k1_KO_s106 + 2.18329 w_k1_MAT_s106
                                 + 0.997355 w_k1_MSFT_s106
                                 + 1.19738 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.16617 w_k1_ADM_s106
                                 + 1.1161 w_k1_AXP_s106 + 1.08163 w_k1_DIS_s106
                                 + 0.886637 w_k1_FDX_s106
                                 + 1.15064 w_k1_IBM_s106
                                 + 0.920282 w_k1_INTC_s106
                                 + 0.968347 w_k1_KO_s106
                                 + 1.09017 w_k1_MAT_s106
                                 + 0.798628 w_k1_MSFT_s106
                                 + 0.983466 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.46622 w_k1_ADM_s106
                                 + 1.06059 w_k1_AXP_s106
                                 + 1.18113 w_k1_DIS_s106
                                 + 1.10953 w_k1_FDX_s106
                                 + 1.23999 w_k1_IBM_s106
                                 + 1.06553 w_k1_INTC_s106
                                 + 1.03833 w_k1_KO_s106 + 1.19853 w_k1_MAT_s106
                                 + 0.954491 w_k1_MSFT_s106
                                 + 1.01705 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.918902 w_k1_ADM_s106
                                 + 1.04117 w_k1_AXP_s106
                                 + 0.931205 w_k1_DIS_s106
                                 + 0.943479 w_k1_FDX_s106
                                 + 1.13689 w_k1_IBM_s106
                                 + 1.05614 w_k1_INTC_s106
                                 + 0.941722 w_k1_KO_s106
                                 + 0.813634 w_k1_MAT_s106
                                 + 0.8007 w_k1_MSFT_s106
                                 + 0.836378 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.93543 w_k1_ADM_s107
                                 + 1.01823 w_k1_AXP_s107
                                 + 1.09442 w_k1_DIS_s107
                                 + 1.04431 w_k1_FDX_s107
                                 + 0.72954 w_k1_IBM_s107
                                 + 1.04852 w_k1_INTC_s107
                                 + 0.801981 w_k1_KO_s107
                                 + 0.644054 w_k1_MAT_s107
                                 + 0.890656 w_k1_MSFT_s107
                                 + 0.908226 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.793254 w_k1_ADM_s107
                                 + 0.668954 w_k1_AXP_s107
                                 + 0.678619 w_k1_DIS_s107
                                 + 0.788441 w_k1_FDX_s107
                                 + 0.665803 w_k1_IBM_s107
                                 + 0.910566 w_k1_INTC_s107
                                 + 0.726385 w_k1_KO_s107
                                 + 0.647281 w_k1_MAT_s107
                                 + 0.782141 w_k1_MSFT_s107
                                 + 0.924716 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.89317 w_k1_ADM_s107
                                 + 1.01261 w_k1_AXP_s107
                                 + 1.05863 w_k1_DIS_s107
                                 + 0.894704 w_k1_FDX_s107
                                 + 1.27471 w_k1_IBM_s107
                                 + 0.908857 w_k1_INTC_s107
                                 + 1.06562 w_k1_KO_s107 + 1.41002 w_k1_MAT_s107
                                 + 0.865964 w_k1_MSFT_s107
                                 + 1.04924 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.737357 w_k1_ADM_s107
                                 + 0.977182 w_k1_AXP_s107
                                 + 0.85523 w_k1_DIS_s107
                                 + 0.817962 w_k1_FDX_s107
                                 + 0.977143 w_k1_IBM_s107
                                 + 1.01501 w_k1_INTC_s107
                                 + 0.823402 w_k1_KO_s107
                                 + 0.714612 w_k1_MAT_s107
                                 + 0.743414 w_k1_MSFT_s107
                                 + 0.944681 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.740626 w_k1_ADM_s107
                                 + 1.06152 w_k1_AXP_s107
                                 + 1.09496 w_k1_DIS_s107
                                 + 0.83154 w_k1_FDX_s107
                                 + 1.66957 w_k1_IBM_s107
                                 + 0.885898 w_k1_INTC_s107
                                 + 1.21914 w_k1_KO_s107 + 1.36067 w_k1_MAT_s107
                                 + 0.784871 w_k1_MSFT_s107
                                 + 1.10375 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.147 w_k1_ADM_s107
                                 + 1.53391 w_k1_AXP_s107
                                 + 1.22273 w_k1_DIS_s107
                                 + 1.19585 w_k1_FDX_s107
                                 + 1.24588 w_k1_IBM_s107
                                 + 1.27193 w_k1_INTC_s107
                                 + 1.04944 w_k1_KO_s107 + 1.17947 w_k1_MAT_s107
                                 + 0.902464 w_k1_MSFT_s107
                                 + 1.27837 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.01456 w_k1_ADM_s107
                                 + 0.805512 w_k1_AXP_s107
                                 + 1.4155 w_k1_DIS_s107 + 1.09215 w_k1_FDX_s107
                                 + 0.957865 w_k1_IBM_s107
                                 + 1.08003 w_k1_INTC_s107
                                 + 1.03444 w_k1_KO_s107
                                 + 0.782782 w_k1_MAT_s107
                                 + 1.13442 w_k1_MSFT_s107
                                 + 0.947596 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.973882 w_k1_ADM_s107
                                 + 1.57245 w_k1_AXP_s107 + 1.6433 w_k1_DIS_s107
                                 + 1.36927 w_k1_FDX_s107
                                 + 1.08057 w_k1_IBM_s107
                                 + 1.32316 w_k1_INTC_s107
                                 + 1.22802 w_k1_KO_s107 + 1.38639 w_k1_MAT_s107
                                 + 1.19094 w_k1_MSFT_s107
                                 + 1.27284 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.30084 w_k1_ADM_s110
                                 + 1.40317 w_k1_AXP_s110
                                 + 1.55118 w_k1_DIS_s110
                                 + 1.30244 w_k1_FDX_s110
                                 + 1.24936 w_k1_IBM_s110
                                 + 1.15899 w_k1_INTC_s110
                                 + 1.41464 w_k1_KO_s110 + 1.64438 w_k1_MAT_s110
                                 + 1.21123 w_k1_MSFT_s110
                                 + 1.39622 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.983507 w_k1_ADM_s110
                                 + 0.922829 w_k1_AXP_s110
                                 + 0.975736 w_k1_DIS_s110
                                 + 1.09829 w_k1_FDX_s110
                                 + 0.756699 w_k1_IBM_s110
                                 + 0.903888 w_k1_INTC_s110
                                 + 0.834551 w_k1_KO_s110
                                 + 0.555118 w_k1_MAT_s110
                                 + 0.950052 w_k1_MSFT_s110
                                 + 0.962145 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.760594 w_k1_ADM_s110
                                 + 0.865217 w_k1_AXP_s110
                                 + 1.01116 w_k1_DIS_s110
                                 + 0.972843 w_k1_FDX_s110
                                 + 1.4126 w_k1_IBM_s110
                                 + 0.900529 w_k1_INTC_s110
                                 + 1.16037 w_k1_KO_s110 + 1.51787 w_k1_MAT_s110
                                 + 0.931468 w_k1_MSFT_s110
                                 + 0.943522 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.17124 w_k1_ADM_s110
                                 + 1.4251 w_k1_AXP_s110 + 1.20207 w_k1_DIS_s110
                                 + 1.39239 w_k1_FDX_s110
                                 + 0.874854 w_k1_IBM_s110
                                 + 0.980228 w_k1_INTC_s110
                                 + 0.933158 w_k1_KO_s110
                                 + 0.917399 w_k1_MAT_s110
                                 + 1.02497 w_k1_MSFT_s110 + 1.1495 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.632043 w_k1_ADM_s110
                                 + 0.673691 w_k1_AXP_s110
                                 + 0.565552 w_k1_DIS_s110
                                 + 0.653218 w_k1_FDX_s110
                                 + 0.944666 w_k1_IBM_s110
                                 + 0.586727 w_k1_INTC_s110
                                 + 0.742445 w_k1_KO_s110
                                 + 0.718672 w_k1_MAT_s110
                                 + 0.635303 w_k1_MSFT_s110
                                 + 0.795837 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.04383 w_k1_ADM_s110
                                 + 1.21396 w_k1_AXP_s110
                                 + 1.09108 w_k1_DIS_s110
                                 + 1.08319 w_k1_FDX_s110
                                 + 1.10998 w_k1_IBM_s110
                                 + 1.12685 w_k1_INTC_s110
                                 + 1.08051 w_k1_KO_s110
                                 + 0.938409 w_k1_MAT_s110
                                 + 0.978481 w_k1_MSFT_s110
                                 + 1.00309 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.663909 w_k1_ADM_s110
                                 + 0.690498 w_k1_AXP_s110
                                 + 0.647612 w_k1_DIS_s110
                                 + 0.809745 w_k1_FDX_s110
                                 + 0.811724 w_k1_IBM_s110
                                 + 0.864498 w_k1_INTC_s110
                                 + 0.765667 w_k1_KO_s110
                                 + 0.654103 w_k1_MAT_s110
                                 + 0.787063 w_k1_MSFT_s110
                                 + 0.903261 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.767783 w_k1_ADM_s110
                                 + 0.815896 w_k1_AXP_s110
                                 + 0.696677 w_k1_DIS_s110
                                 + 0.84901 w_k1_FDX_s110
                                 + 1.03232 w_k1_IBM_s110
                                 + 0.741374 w_k1_INTC_s110
                                 + 0.887667 w_k1_KO_s110
                                 + 0.924678 w_k1_MAT_s110
                                 + 0.832669 w_k1_MSFT_s110
                                 + 0.946682 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.28331 w_k1_ADM_s111
                                 + 0.984964 w_k1_AXP_s111
                                 + 0.907301 w_k1_DIS_s111
                                 + 1.05353 w_k1_FDX_s111
                                 + 0.722274 w_k1_IBM_s111
                                 + 0.822683 w_k1_INTC_s111
                                 + 0.786331 w_k1_KO_s111
                                 + 0.749828 w_k1_MAT_s111
                                 + 0.899199 w_k1_MSFT_s111
                                 + 0.934547 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.3281 w_k1_ADM_s111
                                 + 0.812928 w_k1_AXP_s111
                                 + 0.883669 w_k1_DIS_s111
                                 + 0.85902 w_k1_FDX_s111
                                 + 1.02773 w_k1_IBM_s111
                                 + 0.831099 w_k1_INTC_s111
                                 + 0.955764 w_k1_KO_s111
                                 + 0.888031 w_k1_MAT_s111
                                 + 0.848105 w_k1_MSFT_s111
                                 + 0.898449 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.55009 w_k1_ADM_s111
                                 + 1.72141 w_k1_AXP_s111
                                 + 1.85018 w_k1_DIS_s111
                                 + 1.53173 w_k1_FDX_s111
                                 + 0.861047 w_k1_IBM_s111
                                 + 1.27075 w_k1_INTC_s111
                                 + 1.10164 w_k1_KO_s111 + 1.10268 w_k1_MAT_s111
                                 + 1.39762 w_k1_MSFT_s111
                                 + 1.25629 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.903324 w_k1_ADM_s111
                                 + 0.933123 w_k1_AXP_s111
                                 + 0.555365 w_k1_DIS_s111
                                 + 0.823119 w_k1_FDX_s111
                                 + 0.70031 w_k1_IBM_s111
                                 + 0.718402 w_k1_INTC_s111
                                 + 0.681251 w_k1_KO_s111
                                 + 0.853145 w_k1_MAT_s111
                                 + 0.755888 w_k1_MSFT_s111
                                 + 0.94779 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.4322 w_k1_ADM_s111
                                 + 1.21379 w_k1_AXP_s111
                                 + 1.42803 w_k1_DIS_s111
                                 + 1.13376 w_k1_FDX_s111
                                 + 1.63001 w_k1_IBM_s111
                                 + 0.81699 w_k1_INTC_s111
                                 + 1.39899 w_k1_KO_s111 + 1.67341 w_k1_MAT_s111
                                 + 1.08996 w_k1_MSFT_s111
                                 + 1.18062 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.26607 w_k1_ADM_s111
                                 + 1.02373 w_k1_AXP_s111
                                 + 1.31938 w_k1_DIS_s111 + 1.1625 w_k1_FDX_s111
                                 + 1.09837 w_k1_IBM_s111
                                 + 0.946455 w_k1_INTC_s111
                                 + 1.08972 w_k1_KO_s111 + 1.05063 w_k1_MAT_s111
                                 + 1.04596 w_k1_MSFT_s111
                                 + 1.22015 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.787591 w_k1_ADM_s111
                                 + 0.917969 w_k1_AXP_s111
                                 + 0.528495 w_k1_DIS_s111
                                 + 0.731165 w_k1_FDX_s111
                                 + 1.13699 w_k1_IBM_s111
                                 + 0.758252 w_k1_INTC_s111
                                 + 0.769478 w_k1_KO_s111
                                 + 1.28223 w_k1_MAT_s111
                                 + 0.637243 w_k1_MSFT_s111
                                 + 0.913839 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.839834 w_k1_ADM_s111
                                 + 0.851198 w_k1_AXP_s111
                                 + 0.854474 w_k1_DIS_s111
                                 + 0.822882 w_k1_FDX_s111
                                 + 1.57683 w_k1_IBM_s111
                                 + 0.847705 w_k1_INTC_s111
                                 + 1.09646 w_k1_KO_s111 + 1.42679 w_k1_MAT_s111
                                 + 0.825011 w_k1_MSFT_s111
                                 + 0.994408 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.791341 w_k1_ADM_s112
                                 + 0.942268 w_k1_AXP_s112
                                 + 0.725058 w_k1_DIS_s112
                                 + 0.780425 w_k1_FDX_s112
                                 + 0.886616 w_k1_IBM_s112
                                 + 0.733236 w_k1_INTC_s112
                                 + 0.764653 w_k1_KO_s112
                                 + 1.11402 w_k1_MAT_s112
                                 + 0.767204 w_k1_MSFT_s112
                                 + 0.846027 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17163 w_k1_ADM_s112
                                 + 1.4508 w_k1_AXP_s112 + 1.12938 w_k1_DIS_s112
                                 + 1.07712 w_k1_FDX_s112
                                 + 1.42975 w_k1_IBM_s112
                                 + 0.988618 w_k1_INTC_s112
                                 + 1.07022 w_k1_KO_s112 + 1.60323 w_k1_MAT_s112
                                 + 1.08411 w_k1_MSFT_s112
                                 + 1.00539 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.965861 w_k1_ADM_s112
                                 + 1.14392 w_k1_AXP_s112
                                 + 1.03847 w_k1_DIS_s112
                                 + 1.10669 w_k1_FDX_s112
                                 + 0.885738 w_k1_IBM_s112
                                 + 1.30794 w_k1_INTC_s112
                                 + 0.949816 w_k1_KO_s112
                                 + 1.02624 w_k1_MAT_s112
                                 + 1.16064 w_k1_MSFT_s112
                                 + 0.996547 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.25764 w_k1_ADM_s112
                                 + 1.16446 w_k1_AXP_s112
                                 + 1.10039 w_k1_DIS_s112
                                 + 1.00225 w_k1_FDX_s112
                                 + 1.19888 w_k1_IBM_s112
                                 + 0.937729 w_k1_INTC_s112
                                 + 0.955406 w_k1_KO_s112
                                 + 1.37102 w_k1_MAT_s112
                                 + 0.998146 w_k1_MSFT_s112
                                 + 1.01031 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.885118 w_k1_ADM_s112
                                 + 0.770368 w_k1_AXP_s112
                                 + 0.692343 w_k1_DIS_s112
                                 + 0.892077 w_k1_FDX_s112
                                 + 0.576273 w_k1_IBM_s112
                                 + 0.846251 w_k1_INTC_s112
                                 + 0.696465 w_k1_KO_s112
                                 + 0.581272 w_k1_MAT_s112
                                 + 0.992984 w_k1_MSFT_s112
                                 + 0.840507 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.08288 w_k1_ADM_s112
                                 + 1.58461 w_k1_AXP_s112
                                 + 1.66298 w_k1_DIS_s112
                                 + 1.46357 w_k1_FDX_s112
                                 + 1.17124 w_k1_IBM_s112
                                 + 1.24672 w_k1_INTC_s112 + 1.3621 w_k1_KO_s112
                                 + 1.09114 w_k1_MAT_s112
                                 + 1.34343 w_k1_MSFT_s112
                                 + 1.26302 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.56508 w_k1_ADM_s112
                                 + 1.28127 w_k1_AXP_s112
                                 + 0.850777 w_k1_DIS_s112
                                 + 1.11024 w_k1_FDX_s112
                                 + 0.91901 w_k1_IBM_s112
                                 + 1.03439 w_k1_INTC_s112
                                 + 0.928512 w_k1_KO_s112
                                 + 0.970857 w_k1_MAT_s112
                                 + 1.14317 w_k1_MSFT_s112
                                 + 1.13205 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.766686 w_k1_ADM_s112
                                 + 0.800855 w_k1_AXP_s112
                                 + 0.834087 w_k1_DIS_s112
                                 + 0.94497 w_k1_FDX_s112
                                 + 0.747218 w_k1_IBM_s112
                                 + 0.926917 w_k1_INTC_s112
                                 + 0.755848 w_k1_KO_s112
                                 + 0.724137 w_k1_MAT_s112
                                 + 1.01721 w_k1_MSFT_s112
                                 + 0.870748 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.72588 w_k1_ADM_s113
                                 + 1.33585 w_k1_AXP_s113
                                 + 1.63895 w_k1_DIS_s113
                                 + 1.37061 w_k1_FDX_s113
                                 + 1.19102 w_k1_IBM_s113
                                 + 1.28799 w_k1_INTC_s113
                                 + 1.13371 w_k1_KO_s113 + 1.08018 w_k1_MAT_s113
                                 + 1.19173 w_k1_MSFT_s113
                                 + 1.12334 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.84835 w_k1_ADM_s113
                                 + 0.489478 w_k1_AXP_s113
                                 + 0.496912 w_k1_DIS_s113
                                 + 0.637808 w_k1_FDX_s113
                                 + 0.937221 w_k1_IBM_s113
                                 + 0.785107 w_k1_INTC_s113
                                 + 0.718388 w_k1_KO_s113
                                 + 0.707066 w_k1_MAT_s113
                                 + 0.724125 w_k1_MSFT_s113
                                 + 0.70115 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.942058 w_k1_ADM_s113
                                 + 0.929101 w_k1_AXP_s113
                                 + 1.09203 w_k1_DIS_s113
                                 + 0.959366 w_k1_FDX_s113
                                 + 1.13539 w_k1_IBM_s113
                                 + 1.21592 w_k1_INTC_s113
                                 + 1.08972 w_k1_KO_s113
                                 + 0.953686 w_k1_MAT_s113
                                 + 0.916009 w_k1_MSFT_s113
                                 + 0.976584 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.26223 w_k1_ADM_s113
                                 + 1.17115 w_k1_AXP_s113
                                 + 1.17481 w_k1_DIS_s113
                                 + 1.40478 w_k1_FDX_s113
                                 + 1.01007 w_k1_IBM_s113
                                 + 1.29518 w_k1_INTC_s113
                                 + 1.06215 w_k1_KO_s113 + 1.00445 w_k1_MAT_s113
                                 + 1.14924 w_k1_MSFT_s113
                                 + 1.03145 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.18736 w_k1_ADM_s113
                                 + 0.871273 w_k1_AXP_s113
                                 + 1.11662 w_k1_DIS_s113
                                 + 1.19248 w_k1_FDX_s113
                                 + 0.725635 w_k1_IBM_s113
                                 + 1.25842 w_k1_INTC_s113
                                 + 0.828492 w_k1_KO_s113
                                 + 0.726534 w_k1_MAT_s113
                                 + 1.18216 w_k1_MSFT_s113
                                 + 0.924097 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.912839 w_k1_ADM_s113
                                 + 0.784293 w_k1_AXP_s113
                                 + 0.726131 w_k1_DIS_s113
                                 + 0.872785 w_k1_FDX_s113
                                 + 0.833977 w_k1_IBM_s113
                                 + 1.08067 w_k1_INTC_s113
                                 + 0.770059 w_k1_KO_s113
                                 + 0.693691 w_k1_MAT_s113
                                 + 0.981886 w_k1_MSFT_s113
                                 + 0.855834 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.13714 w_k1_ADM_s113
                                 + 1.52652 w_k1_AXP_s113
                                 + 1.39088 w_k1_DIS_s113
                                 + 1.19371 w_k1_FDX_s113
                                 + 1.53942 w_k1_IBM_s113
                                 + 1.33446 w_k1_INTC_s113
                                 + 1.36638 w_k1_KO_s113 + 1.88889 w_k1_MAT_s113
                                 + 1.26144 w_k1_MSFT_s113
                                 + 1.02768 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.930771 w_k1_ADM_s113
                                 + 0.905952 w_k1_AXP_s113
                                 + 0.854728 w_k1_DIS_s113
                                 + 0.883091 w_k1_FDX_s113
                                 + 1.36486 w_k1_IBM_s113
                                 + 0.843181 w_k1_INTC_s113
                                 + 1.04371 w_k1_KO_s113 + 1.29222 w_k1_MAT_s113
                                 + 0.841683 w_k1_MSFT_s113
                                 + 0.946977 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.69639 w_k1_ADM_s114
                                 + 0.978715 w_k1_AXP_s114
                                 + 0.764614 w_k1_DIS_s114
                                 + 0.836949 w_k1_FDX_s114
                                 + 1.23813 w_k1_IBM_s114
                                 + 1.09327 w_k1_INTC_s114
                                 + 0.972089 w_k1_KO_s114
                                 + 1.20381 w_k1_MAT_s114
                                 + 0.953534 w_k1_MSFT_s114
                                 + 0.940016 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.28932 w_k1_ADM_s114
                                 + 1.29475 w_k1_AXP_s114
                                 + 1.34349 w_k1_DIS_s114
                                 + 1.09497 w_k1_FDX_s114 + 1.4948 w_k1_IBM_s114
                                 + 1.27372 w_k1_INTC_s114 + 1.3079 w_k1_KO_s114
                                 + 1.3636 w_k1_MAT_s114
                                 + 1.14259 w_k1_MSFT_s114
                                 + 1.12793 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.816627 w_k1_ADM_s114
                                 + 1.31787 w_k1_AXP_s114
                                 + 0.937391 w_k1_DIS_s114
                                 + 1.12072 w_k1_FDX_s114
                                 + 1.00346 w_k1_IBM_s114
                                 + 1.27827 w_k1_INTC_s114
                                 + 0.823176 w_k1_KO_s114
                                 + 0.914556 w_k1_MAT_s114
                                 + 1.00904 w_k1_MSFT_s114
                                 + 0.930031 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.19012 w_k1_ADM_s114
                                 + 1.3755 w_k1_AXP_s114 + 1.33986 w_k1_DIS_s114
                                 + 1.2862 w_k1_FDX_s114
                                 + 0.907467 w_k1_IBM_s114
                                 + 1.31199 w_k1_INTC_s114
                                 + 0.957276 w_k1_KO_s114
                                 + 0.858258 w_k1_MAT_s114
                                 + 1.2951 w_k1_MSFT_s114 + 1.10429 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.84302 w_k1_ADM_s114
                                 + 1.26265 w_k1_AXP_s114 + 1.315 w_k1_DIS_s114
                                 + 1.08158 w_k1_FDX_s114
                                 + 1.24312 w_k1_IBM_s114
                                 + 1.44785 w_k1_INTC_s114
                                 + 1.06915 w_k1_KO_s114
                                 + 0.947406 w_k1_MAT_s114
                                 + 1.23726 w_k1_MSFT_s114
                                 + 1.08645 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.817009 w_k1_ADM_s114
                                 + 0.924697 w_k1_AXP_s114
                                 + 1.45683 w_k1_DIS_s114
                                 + 1.23636 w_k1_FDX_s114
                                 + 0.93937 w_k1_IBM_s114
                                 + 1.43949 w_k1_INTC_s114
                                 + 1.03414 w_k1_KO_s114
                                 + 0.845732 w_k1_MAT_s114
                                 + 1.35458 w_k1_MSFT_s114
                                 + 1.04027 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.78535 w_k1_ADM_s114
                                 + 0.89943 w_k1_AXP_s114
                                 + 0.857358 w_k1_DIS_s114
                                 + 0.874711 w_k1_FDX_s114
                                 + 0.764738 w_k1_IBM_s114
                                 + 1.03841 w_k1_INTC_s114
                                 + 0.675206 w_k1_KO_s114
                                 + 0.699393 w_k1_MAT_s114
                                 + 1.01868 w_k1_MSFT_s114
                                 + 0.88694 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.625741 w_k1_ADM_s114
                                 + 0.778654 w_k1_AXP_s114
                                 + 1.02033 w_k1_DIS_s114
                                 + 0.853837 w_k1_FDX_s114
                                 + 1.05681 w_k1_IBM_s114
                                 + 0.889851 w_k1_INTC_s114
                                 + 0.875285 w_k1_KO_s114
                                 + 0.830143 w_k1_MAT_s114
                                 + 0.865083 w_k1_MSFT_s114
                                 + 0.955134 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.972167 w_k1_ADM_s115
                                 + 1.1724 w_k1_AXP_s115 + 1.01687 w_k1_DIS_s115
                                 + 1.00202 w_k1_FDX_s115
                                 + 1.37091 w_k1_IBM_s115
                                 + 1.21323 w_k1_INTC_s115
                                 + 1.07834 w_k1_KO_s115 + 1.61217 w_k1_MAT_s115
                                 + 1.09429 w_k1_MSFT_s115
                                 + 1.19175 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.17665 w_k1_ADM_s115
                                 + 1.09133 w_k1_AXP_s115
                                 + 0.790497 w_k1_DIS_s115
                                 + 1.03149 w_k1_FDX_s115
                                 + 0.748831 w_k1_IBM_s115
                                 + 0.894843 w_k1_INTC_s115
                                 + 0.78682 w_k1_KO_s115
                                 + 0.835397 w_k1_MAT_s115
                                 + 0.908907 w_k1_MSFT_s115
                                 + 1.00651 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.890666 w_k1_ADM_s115
                                 + 0.911908 w_k1_AXP_s115
                                 + 0.846412 w_k1_DIS_s115
                                 + 0.828985 w_k1_FDX_s115
                                 + 0.996119 w_k1_IBM_s115
                                 + 0.954123 w_k1_INTC_s115
                                 + 0.823503 w_k1_KO_s115
                                 + 1.26393 w_k1_MAT_s115
                                 + 1.00801 w_k1_MSFT_s115
                                 + 0.964195 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.15564 w_k1_ADM_s115
                                 + 1.21084 w_k1_AXP_s115
                                 + 1.25773 w_k1_DIS_s115
                                 + 1.24151 w_k1_FDX_s115
                                 + 0.871946 w_k1_IBM_s115
                                 + 1.25262 w_k1_INTC_s115
                                 + 1.07648 w_k1_KO_s115 + 1.17557 w_k1_MAT_s115
                                 + 1.20501 w_k1_MSFT_s115
                                 + 1.26251 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.44175 w_k1_ADM_s115
                                 + 1.39578 w_k1_AXP_s115
                                 + 1.70428 w_k1_DIS_s115
                                 + 1.38938 w_k1_FDX_s115
                                 + 1.09201 w_k1_IBM_s115
                                 + 1.4474 w_k1_INTC_s115 + 1.4031 w_k1_KO_s115
                                 + 1.29725 w_k1_MAT_s115
                                 + 1.3314 w_k1_MSFT_s115 + 1.28931 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.29189 w_k1_ADM_s115
                                 + 1.50249 w_k1_AXP_s115
                                 + 1.38722 w_k1_DIS_s115
                                 + 1.18197 w_k1_FDX_s115 + 1.5025 w_k1_IBM_s115
                                 + 1.2758 w_k1_INTC_s115 + 1.06834 w_k1_KO_s115
                                 + 1.39255 w_k1_MAT_s115
                                 + 1.33131 w_k1_MSFT_s115
                                 + 1.20594 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.703036 w_k1_ADM_s115
                                 + 0.672699 w_k1_AXP_s115
                                 + 0.54124 w_k1_DIS_s115
                                 + 0.685197 w_k1_FDX_s115
                                 + 0.708047 w_k1_IBM_s115
                                 + 0.918418 w_k1_INTC_s115
                                 + 0.604628 w_k1_KO_s115
                                 + 0.630948 w_k1_MAT_s115
                                 + 0.782668 w_k1_MSFT_s115
                                 + 0.790302 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.18632 w_k1_ADM_s115
                                 + 1.00543 w_k1_AXP_s115
                                 + 0.884691 w_k1_DIS_s115
                                 + 0.940399 w_k1_FDX_s115
                                 + 0.756782 w_k1_IBM_s115
                                 + 0.935565 w_k1_INTC_s115
                                 + 0.862913 w_k1_KO_s115
                                 + 0.785139 w_k1_MAT_s115
                                 + 1.06912 w_k1_MSFT_s115
                                 + 0.970282 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.11789 w_k1_ADM_s116
                                 + 1.09319 w_k1_AXP_s116
                                 + 1.10841 w_k1_DIS_s116
                                 + 1.14893 w_k1_FDX_s116
                                 + 0.902657 w_k1_IBM_s116
                                 + 1.13684 w_k1_INTC_s116
                                 + 0.955646 w_k1_KO_s116
                                 + 0.916508 w_k1_MAT_s116
                                 + 0.959081 w_k1_MSFT_s116
                                 + 1.03851 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.10959 w_k1_ADM_s116
                                 + 1.14889 w_k1_AXP_s116
                                 + 1.18715 w_k1_DIS_s116
                                 + 1.14164 w_k1_FDX_s116
                                 + 1.25458 w_k1_IBM_s116
                                 + 1.20303 w_k1_INTC_s116
                                 + 1.14895 w_k1_KO_s116 + 1.4923 w_k1_MAT_s116
                                 + 1.09503 w_k1_MSFT_s116
                                 + 1.07617 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.738825 w_k1_ADM_s116
                                 + 0.912592 w_k1_AXP_s116
                                 + 0.736505 w_k1_DIS_s116
                                 + 0.775128 w_k1_FDX_s116
                                 + 1.17054 w_k1_IBM_s116
                                 + 1.10888 w_k1_INTC_s116
                                 + 1.00342 w_k1_KO_s116 + 0.98276 w_k1_MAT_s116
                                 + 0.88861 w_k1_MSFT_s116
                                 + 0.917929 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.17317 w_k1_ADM_s116
                                 + 1.09092 w_k1_AXP_s116 + 1.4914 w_k1_DIS_s116
                                 + 1.21742 w_k1_FDX_s116
                                 + 1.66828 w_k1_IBM_s116
                                 + 0.969289 w_k1_INTC_s116
                                 + 1.32825 w_k1_KO_s116 + 1.68965 w_k1_MAT_s116
                                 + 1.03948 w_k1_MSFT_s116
                                 + 1.17742 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.03082 w_k1_ADM_s116
                                 + 1.36636 w_k1_AXP_s116
                                 + 1.13454 w_k1_DIS_s116
                                 + 1.13989 w_k1_FDX_s116
                                 + 0.817009 w_k1_IBM_s116
                                 + 1.09977 w_k1_INTC_s116
                                 + 0.957327 w_k1_KO_s116
                                 + 0.944654 w_k1_MAT_s116
                                 + 0.907508 w_k1_MSFT_s116
                                 + 1.09398 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.7989 w_k1_ADM_s116
                                 + 0.902985 w_k1_AXP_s116
                                 + 0.681287 w_k1_DIS_s116
                                 + 0.933878 w_k1_FDX_s116
                                 + 0.757324 w_k1_IBM_s116
                                 + 1.01853 w_k1_INTC_s116
                                 + 0.634019 w_k1_KO_s116
                                 + 0.71207 w_k1_MAT_s116
                                 + 0.759232 w_k1_MSFT_s116
                                 + 0.821031 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.962768 w_k1_ADM_s116
                                 + 0.820192 w_k1_AXP_s116
                                 + 1.03071 w_k1_DIS_s116
                                 + 0.992337 w_k1_FDX_s116
                                 + 1.22926 w_k1_IBM_s116
                                 + 1.08738 w_k1_INTC_s116
                                 + 1.17557 w_k1_KO_s116
                                 + 0.966144 w_k1_MAT_s116
                                 + 0.960945 w_k1_MSFT_s116
                                 + 1.13268 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.18099 w_k1_ADM_s116
                                 + 1.42599 w_k1_AXP_s116
                                 + 1.43792 w_k1_DIS_s116
                                 + 1.36743 w_k1_FDX_s116
                                 + 1.14204 w_k1_IBM_s116
                                 + 1.47607 w_k1_INTC_s116 + 1.326 w_k1_KO_s116
                                 + 1.17135 w_k1_MAT_s116
                                 + 1.4769 w_k1_MSFT_s116 + 1.15285 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.754347 w_k1_ADM_s117
                                 + 0.677485 w_k1_AXP_s117
                                 + 0.555253 w_k1_DIS_s117
                                 + 0.685874 w_k1_FDX_s117
                                 + 0.694632 w_k1_IBM_s117
                                 + 0.875976 w_k1_INTC_s117
                                 + 0.65753 w_k1_KO_s117
                                 + 0.444905 w_k1_MAT_s117
                                 + 0.756923 w_k1_MSFT_s117
                                 + 0.728197 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.49269 w_k1_ADM_s117
                                 + 1.65181 w_k1_AXP_s117
                                 + 1.60189 w_k1_DIS_s117
                                 + 1.40319 w_k1_FDX_s117 + 1.1407 w_k1_IBM_s117
                                 + 1.36508 w_k1_INTC_s117
                                 + 1.29181 w_k1_KO_s117 + 1.36632 w_k1_MAT_s117
                                 + 1.26304 w_k1_MSFT_s117
                                 + 1.27218 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.03372 w_k1_ADM_s117
                                 + 1.08579 w_k1_AXP_s117
                                 + 0.952739 w_k1_DIS_s117
                                 + 0.94677 w_k1_FDX_s117 + 1.2699 w_k1_IBM_s117
                                 + 1.08406 w_k1_INTC_s117
                                 + 1.02158 w_k1_KO_s117 + 1.01185 w_k1_MAT_s117
                                 + 0.899404 w_k1_MSFT_s117
                                 + 0.985983 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.11325 w_k1_ADM_s117
                                 + 0.893014 w_k1_AXP_s117
                                 + 0.827014 w_k1_DIS_s117
                                 + 0.915077 w_k1_FDX_s117
                                 + 0.763287 w_k1_IBM_s117
                                 + 1.09322 w_k1_INTC_s117
                                 + 0.856525 w_k1_KO_s117
                                 + 0.566909 w_k1_MAT_s117
                                 + 0.984408 w_k1_MSFT_s117
                                 + 0.897746 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.02358 w_k1_ADM_s117
                                 + 1.06117 w_k1_AXP_s117
                                 + 1.11889 w_k1_DIS_s117
                                 + 1.02195 w_k1_FDX_s117
                                 + 0.844013 w_k1_IBM_s117
                                 + 1.01305 w_k1_INTC_s117
                                 + 0.962307 w_k1_KO_s117
                                 + 1.02373 w_k1_MAT_s117
                                 + 1.06682 w_k1_MSFT_s117
                                 + 0.923832 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.18392 w_k1_ADM_s117
                                 + 1.32044 w_k1_AXP_s117 + 1.1874 w_k1_DIS_s117
                                 + 1.10415 w_k1_FDX_s117
                                 + 1.71121 w_k1_IBM_s117
                                 + 1.18415 w_k1_INTC_s117
                                 + 1.47275 w_k1_KO_s117 + 1.29663 w_k1_MAT_s117
                                 + 0.970413 w_k1_MSFT_s117
                                 + 1.06279 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.95983 w_k1_ADM_s117
                                 + 1.01008 w_k1_AXP_s117
                                 + 0.692962 w_k1_DIS_s117
                                 + 0.865422 w_k1_FDX_s117
                                 + 0.901662 w_k1_IBM_s117
                                 + 0.856048 w_k1_INTC_s117
                                 + 0.817356 w_k1_KO_s117
                                 + 1.28102 w_k1_MAT_s117
                                 + 0.84958 w_k1_MSFT_s117
                                 + 0.864538 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.77493 w_k1_ADM_s117
                                 + 0.731906 w_k1_AXP_s117
                                 + 0.632487 w_k1_DIS_s117
                                 + 0.727261 w_k1_FDX_s117
                                 + 0.992777 w_k1_IBM_s117
                                 + 0.885843 w_k1_INTC_s117
                                 + 0.959174 w_k1_KO_s117
                                 + 0.83548 w_k1_MAT_s117
                                 + 0.792574 w_k1_MSFT_s117
                                 + 0.854668 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0577531137141051 w_k1_ADM_s000
                                 + 0.0346976853597234 w_k1_ADM_s001
                                 + 0.0530740264648225 w_k1_ADM_s002
                                 + 0.102958224706903 w_k1_ADM_s003
                                 + 0.0313242748425162 w_k1_ADM_s004
                                 + 0.0554601816477215 w_k1_ADM_s005
                                 + 0.0252035256029405 w_k1_ADM_s006
                                 + 0.0222633271246618 w_k1_ADM_s007
                                 + 0.0355436443131107 w_k1_ADM_s010
                                 + 0.037347585926475 w_k1_ADM_s011
                                 + 0.0244153153759091 w_k1_ADM_s012
                                 + 0.0249970286846518 w_k1_ADM_s013
                                 + 0.105912676030292 w_k1_ADM_s014
                                 + 0.0379611333780322 w_k1_ADM_s015
                                 + 0.0594747980191284 w_k1_ADM_s016
                                 + 0.0403201446097385 w_k1_ADM_s017
                                 + 0.00414539269654951 w_k1_ADM_s100
                                 + 0.0021455713999668 w_k1_ADM_s101
                                 + 0.00403178416715119 w_k1_ADM_s102
                                 + 0.00990342908283229 w_k1_ADM_s103
                                 + 0.00617026188266673 w_k1_ADM_s104
                                 + 0.0039923612880432 w_k1_ADM_s105
                                 + 0.00296461687698888 w_k1_ADM_s106
                                 + 0.00549284829402044 w_k1_ADM_s107
                                 + 0.0238986513465434 w_k1_ADM_s110
                                 + 0.0252255269220711 w_k1_ADM_s111
                                 + 0.0432798846804039 w_k1_ADM_s112
                                 + 0.0117446246013834 w_k1_ADM_s113
                                 + 0.00849519620785993 w_k1_ADM_s114
                                 + 0.0140875294657311 w_k1_ADM_s115
                                 + 0.0138513517615082 w_k1_ADM_s116
                                 + 0.0522809574451351 w_k1_ADM_s117
                                 + 0.0670066904907229 w_k1_AXP_s000
                                 + 0.0353992351955281 w_k1_AXP_s001
                                 + 0.0497151901759876 w_k1_AXP_s002
                                 + 0.10978281511688 w_k1_AXP_s003
                                 + 0.0316732447617011 w_k1_AXP_s004
                                 + 0.0631451103529443 w_k1_AXP_s005
                                 + 0.0246331351376656 w_k1_AXP_s006
                                 + 0.0261310628522732 w_k1_AXP_s007
                                 + 0.0347677288445842 w_k1_AXP_s010
                                 + 0.0366003991765656 w_k1_AXP_s011
                                 + 0.0255176929204528 w_k1_AXP_s012
                                 + 0.0284209298022883 w_k1_AXP_s013
                                 + 0.114608955795084 w_k1_AXP_s014
                                 + 0.0421158544985739 w_k1_AXP_s015
                                 + 0.0585409222229541 w_k1_AXP_s016
                                 + 0.046534457324689 w_k1_AXP_s017
                                 + 0.00463243825203537 w_k1_AXP_s100
                                 + 0.00289750241529363 w_k1_AXP_s101
                                 + 0.00358610703144239 w_k1_AXP_s102
                                 + 0.00878000964942894 w_k1_AXP_s103
                                 + 0.00734903055598431 w_k1_AXP_s104
                                 + 0.00435353679138714 w_k1_AXP_s105
                                 + 0.00295282213928352 w_k1_AXP_s106
                                 + 0.00641471706100143 w_k1_AXP_s107
                                 + 0.025872895041522 w_k1_AXP_s110
                                 + 0.0218231980398549 w_k1_AXP_s111
                                 + 0.04821541079893 w_k1_AXP_s112
                                 + 0.0102060189213301 w_k1_AXP_s113
                                 + 0.0104033102082998 w_k1_AXP_s114
                                 + 0.0142183738517495 w_k1_AXP_s115
                                 + 0.0151230963017618 w_k1_AXP_s116
                                 + 0.0532962510522018 w_k1_AXP_s117
                                 + 0.0701994747403595 w_k1_DIS_s000
                                 + 0.0310123525320478 w_k1_DIS_s001
                                 + 0.0552880586355526 w_k1_DIS_s002
                                 + 0.114626343096372 w_k1_DIS_s003
                                 + 0.032341534771323 w_k1_DIS_s004
                                 + 0.0613716753773495 w_k1_DIS_s005
                                 + 0.0244416812955962 w_k1_DIS_s006
                                 + 0.025602586590519 w_k1_DIS_s007
                                 + 0.0337007996822393 w_k1_DIS_s010
                                 + 0.034301895202439 w_k1_DIS_s011
                                 + 0.0264434077176922 w_k1_DIS_s012
                                 + 0.0258708311141471 w_k1_DIS_s013
                                 + 0.120707939647423 w_k1_DIS_s014
                                 + 0.0431052839609307 w_k1_DIS_s015
                                 + 0.0614032995216509 w_k1_DIS_s016
                                 + 0.0486796021193951 w_k1_DIS_s017
                                 + 0.00481838963146846 w_k1_DIS_s100
                                 + 0.00252040973068645 w_k1_DIS_s101
                                 + 0.00383226212836024 w_k1_DIS_s102
                                 + 0.0108103002909403 w_k1_DIS_s103
                                 + 0.00673474248721725 w_k1_DIS_s104
                                 + 0.00358958094918097 w_k1_DIS_s105
                                 + 0.00279417060765796 w_k1_DIS_s106
                                 + 0.00661935311894979 w_k1_DIS_s107
                                 + 0.0254837721518317 w_k1_DIS_s110
                                 + 0.022018331058109 w_k1_DIS_s111
                                 + 0.042346825140688 w_k1_DIS_s112
                                 + 0.0109437918569774 w_k1_DIS_s113
                                 + 0.0105075621927971 w_k1_DIS_s114
                                 + 0.0132927032677066 w_k1_DIS_s115
                                 + 0.0144775190835569 w_k1_DIS_s116
                                 + 0.0493520806116077 w_k1_DIS_s117
                                 + 0.069297703940874 w_k1_FDX_s000
                                 + 0.0317114207968743 w_k1_FDX_s001
                                 + 0.0528295690612473 w_k1_FDX_s002
                                 + 0.107707124126747 w_k1_FDX_s003
                                 + 0.0333247711637034 w_k1_FDX_s004
                                 + 0.0583228640919556 w_k1_FDX_s005
                                 + 0.0232442104117978 w_k1_FDX_s006
                                 + 0.0250980378777693 w_k1_FDX_s007
                                 + 0.0349699286048435 w_k1_FDX_s010
                                 + 0.0369544145822321 w_k1_FDX_s011
                                 + 0.0249526336262719 w_k1_FDX_s012
                                 + 0.0270556545865285 w_k1_FDX_s013
                                 + 0.115845939233053 w_k1_FDX_s014
                                 + 0.0430026639075755 w_k1_FDX_s015
                                 + 0.0604279422621235 w_k1_FDX_s016
                                 + 0.0459582498467798 w_k1_FDX_s017
                                 + 0.00452683193508609 w_k1_FDX_s100
                                 + 0.00246551354537732 w_k1_FDX_s101
                                 + 0.00386497594619533 w_k1_FDX_s102
                                 + 0.0105947003847029 w_k1_FDX_s103
                                 + 0.00671327374988423 w_k1_FDX_s104
                                 + 0.00389459876551934 w_k1_FDX_s105
                                 + 0.0027484485817065 w_k1_FDX_s106
                                 + 0.00597891086086977 w_k1_FDX_s107
                                 + 0.0265215875944451 w_k1_FDX_s110
                                 + 0.0215060596460272 w_k1_FDX_s111
                                 + 0.0433474053788752 w_k1_FDX_s112
                                 + 0.0110696401368751 w_k1_FDX_s113
                                 + 0.00984442658881412 w_k1_FDX_s114
                                 + 0.0132979352695234 w_k1_FDX_s115
                                 + 0.0149364330552326 w_k1_FDX_s116
                                 + 0.0481529074219046 w_k1_FDX_s117
                                 + 0.0724292858953813 w_k1_IBM_s000
                                 + 0.0316303630949497 w_k1_IBM_s001
                                 + 0.0530569334135041 w_k1_IBM_s002
                                 + 0.115141029954345 w_k1_IBM_s003
                                 + 0.0312300760718509 w_k1_IBM_s004
                                 + 0.0604682801105218 w_k1_IBM_s005
                                 + 0.0250549562793611 w_k1_IBM_s006
                                 + 0.0255658621567591 w_k1_IBM_s007
                                 + 0.0352803731483991 w_k1_IBM_s010
                                 + 0.0362744149728365 w_k1_IBM_s011
                                 + 0.024594217188198 w_k1_IBM_s012
                                 + 0.0276092630349106 w_k1_IBM_s013
                                 + 0.118524917455979 w_k1_IBM_s014
                                 + 0.0448639858202615 w_k1_IBM_s015
                                 + 0.0601567249072527 w_k1_IBM_s016
                                 + 0.046978198786809 w_k1_IBM_s017
                                 + 0.0052866413910472 w_k1_IBM_s100
                                 + 0.00258902179130599 w_k1_IBM_s101
                                 + 0.00388521315514749 w_k1_IBM_s102
                                 + 0.0105380505613991 w_k1_IBM_s103
                                 + 0.00672598959176273 w_k1_IBM_s104
                                 + 0.00389989998258347 w_k1_IBM_s105
                                 + 0.00289771180285924 w_k1_IBM_s106
                                 + 0.00636038292416128 w_k1_IBM_s107
                                 + 0.0257422214411692 w_k1_IBM_s110
                                 + 0.0202150184314026 w_k1_IBM_s111
                                 + 0.0416898042944 w_k1_IBM_s112
                                 + 0.0108876951526477 w_k1_IBM_s113
                                 + 0.0103149383848909 w_k1_IBM_s114
                                 + 0.0128153549205188 w_k1_IBM_s115
                                 + 0.0152153504161692 w_k1_IBM_s116
                                 + 0.0486073846324491 w_k1_IBM_s117
                                 + 0.0741776353703849 w_k1_INTC_s000
                                 + 0.0317555351400933 w_k1_INTC_s001
                                 + 0.0496985889882995 w_k1_INTC_s002
                                 + 0.0974440819475231 w_k1_INTC_s003
                                 + 0.0311315723372402 w_k1_INTC_s004
                                 + 0.0641794487674125 w_k1_INTC_s005
                                 + 0.0239250715251929 w_k1_INTC_s006
                                 + 0.0265673457360018 w_k1_INTC_s007
                                 + 0.0330766415762415 w_k1_INTC_s010
                                 + 0.0389702779854055 w_k1_INTC_s011
                                 + 0.0304715049251079 w_k1_INTC_s012
                                 + 0.0300805697919833 w_k1_INTC_s013
                                 + 0.113195278731095 w_k1_INTC_s014
                                 + 0.0410495355664661 w_k1_INTC_s015
                                 + 0.0670476857489171 w_k1_INTC_s016
                                 + 0.0440123172992144 w_k1_INTC_s017
                                 + 0.0047734059983935 w_k1_INTC_s100
                                 + 0.00252834699713744 w_k1_INTC_s101
                                 + 0.00344704668355595 w_k1_INTC_s102
                                 + 0.00982284052877977 w_k1_INTC_s103
                                 + 0.00669618139845894 w_k1_INTC_s104
                                 + 0.00398099686008157 w_k1_INTC_s105
                                 + 0.00263086353674746 w_k1_INTC_s106
                                 + 0.00638999995786111 w_k1_INTC_s107
                                 + 0.0240540759248611 w_k1_INTC_s110
                                 + 0.0181288489866622 w_k1_INTC_s111
                                 + 0.043072894508261 w_k1_INTC_s112
                                 + 0.0120560462985223 w_k1_INTC_s113
                                 + 0.0115856588968634 w_k1_INTC_s114
                                 + 0.0142041298519761 w_k1_INTC_s115
                                 + 0.0161527065240473 w_k1_INTC_s116
                                 + 0.0515255635881354 w_k1_INTC_s117
                                 + 0.0676311062590399 w_k1_KO_s000
                                 + 0.0293805195588329 w_k1_KO_s001
                                 + 0.0541529740577849 w_k1_KO_s002
                                 + 0.110853484653023 w_k1_KO_s003
                                 + 0.0286769277340667 w_k1_KO_s004
                                 + 0.0563292309350943 w_k1_KO_s005
                                 + 0.0232715834000413 w_k1_KO_s006
                                 + 0.0254009726362231 w_k1_KO_s007
                                 + 0.0315219679743679 w_k1_KO_s010
                                 + 0.038952535842299 w_k1_KO_s011
                                 + 0.0249249799321303 w_k1_KO_s012
                                 + 0.0261522680199889 w_k1_KO_s013
                                 + 0.110191341204121 w_k1_KO_s014
                                 + 0.0442773862983565 w_k1_KO_s015
                                 + 0.0612446852583001 w_k1_KO_s016
                                 + 0.0436013062916417 w_k1_KO_s017
                                 + 0.00489280733970096 w_k1_KO_s100
                                 + 0.00246815882215066 w_k1_KO_s101
                                 + 0.00355159643810684 w_k1_KO_s102
                                 + 0.0107808586344907 w_k1_KO_s103
                                 + 0.00676891569610809 w_k1_KO_s104
                                 + 0.00371224046980304 w_k1_KO_s105
                                 + 0.00266751825967876 w_k1_KO_s106
                                 + 0.00582746245729202 w_k1_KO_s107
                                 + 0.0252105138709112 w_k1_KO_s110
                                 + 0.0194949651869916 w_k1_KO_s111
                                 + 0.0392340118087239 w_k1_KO_s112
                                 + 0.0101205740394297 w_k1_KO_s113
                                 + 0.00907272710944402 w_k1_KO_s114
                                 + 0.0123483484388288 w_k1_KO_s115
                                 + 0.0142813583221518 w_k1_KO_s116
                                 + 0.0482573172630654 w_k1_KO_s117
                                 + 0.0640489589483321 w_k1_MAT_s000
                                 + 0.0392565724448577 w_k1_MAT_s001
                                 + 0.0536195874996741 w_k1_MAT_s002
                                 + 0.118260375483684 w_k1_MAT_s003
                                 + 0.0291400866131647 w_k1_MAT_s004
                                 + 0.0522770970123963 w_k1_MAT_s005
                                 + 0.022783628718025 w_k1_MAT_s006
                                 + 0.0261373615186622 w_k1_MAT_s007
                                 + 0.0412961878821745 w_k1_MAT_s010
                                 + 0.0418578895616065 w_k1_MAT_s011
                                 + 0.0167634046709746 w_k1_MAT_s012
                                 + 0.023896115466355 w_k1_MAT_s013
                                 + 0.134570314665009 w_k1_MAT_s014
                                 + 0.0502487517068866 w_k1_MAT_s015
                                 + 0.0742272987277353 w_k1_MAT_s016
                                 + 0.0435343524691572 w_k1_MAT_s017
                                 + 0.00526631065564228 w_k1_MAT_s100
                                 + 0.00373678535239605 w_k1_MAT_s101
                                 + 0.00428894389862972 w_k1_MAT_s102
                                 + 0.00939814509766946 w_k1_MAT_s103
                                 + 0.00612359773503602 w_k1_MAT_s104
                                 + 0.003824830200446 w_k1_MAT_s105
                                 + 0.00292652671232446 w_k1_MAT_s106
                                 + 0.00579094223948985 w_k1_MAT_s107
                                 + 0.0246990778377273 w_k1_MAT_s110
                                 + 0.020887643419814 w_k1_MAT_s111
                                 + 0.0464466009380219 w_k1_MAT_s112
                                 + 0.0101028813373652 w_k1_MAT_s113
                                 + 0.00917999350890255 w_k1_MAT_s114
                                 + 0.0149440141573704 w_k1_MAT_s115
                                 + 0.0147972824541295 w_k1_MAT_s116
                                 + 0.0471457291550416 w_k1_MAT_s117
                                 + 0.0710764476301366 w_k1_MSFT_s000
                                 + 0.0324407136596367 w_k1_MSFT_s001
                                 + 0.0509943190940866 w_k1_MSFT_s002
                                 + 0.109057785087255 w_k1_MSFT_s003
                                 + 0.0316099839177062 w_k1_MSFT_s004
                                 + 0.0567037778100764 w_k1_MSFT_s005
                                 + 0.0237291152553333 w_k1_MSFT_s006
                                 + 0.0250596186322274 w_k1_MSFT_s007
                                 + 0.0315691656500084 w_k1_MSFT_s010
                                 + 0.0362516938300793 w_k1_MSFT_s011
                                 + 0.0250551398408483 w_k1_MSFT_s012
                                 + 0.0257104943834816 w_k1_MSFT_s013
                                 + 0.112859316667602 w_k1_MSFT_s014
                                 + 0.0434033358113433 w_k1_MSFT_s015
                                 + 0.0648647499238086 w_k1_MSFT_s016
                                 + 0.0448182263835732 w_k1_MSFT_s017
                                 + 0.00494754308240733 w_k1_MSFT_s100
                                 + 0.00246421341999407 w_k1_MSFT_s101
                                 + 0.00401585661933503 w_k1_MSFT_s102
                                 + 0.00967483962753622 w_k1_MSFT_s103
                                 + 0.00694924600078668 w_k1_MSFT_s104
                                 + 0.0035162577030831 w_k1_MSFT_s105
                                 + 0.00238337706526939 w_k1_MSFT_s106
                                 + 0.00541264379516927 w_k1_MSFT_s107
                                 + 0.0239325354834868 w_k1_MSFT_s110
                                 + 0.0196143836671081 w_k1_MSFT_s111
                                 + 0.0439517471399809 w_k1_MSFT_s112
                                 + 0.010908153756536 w_k1_MSFT_s113
                                 + 0.0106326468542817 w_k1_MSFT_s114
                                 + 0.0141249058811072 w_k1_MSFT_s115
                                 + 0.0139171011759321 w_k1_MSFT_s116
                                 + 0.0470234543441886 w_k1_MSFT_s117
                                 + 0.0657396480282709 w_k1_PG_s000
                                 + 0.0305170541165051 w_k1_PG_s001
                                 + 0.0507469932288243 w_k1_PG_s002
                                 + 0.112788982726703 w_k1_PG_s003
                                 + 0.0311350592152266 w_k1_PG_s004
                                 + 0.0568731718380428 w_k1_PG_s005
                                 + 0.0236950921662452 w_k1_PG_s006
                                 + 0.02610362691885 w_k1_PG_s007
                                 + 0.0337611024206136 w_k1_PG_s010
                                 + 0.0363903759360995 w_k1_PG_s011
                                 + 0.0237398539833176 w_k1_PG_s012
                                 + 0.0263170304978838 w_k1_PG_s013
                                 + 0.12498612714157 w_k1_PG_s014
                                 + 0.0426475228922241 w_k1_PG_s015
                                 + 0.0599191708653594 w_k1_PG_s016
                                 + 0.0461858717391979 w_k1_PG_s017
                                 + 0.0047667370912001 w_k1_PG_s100
                                 + 0.00255641300801016 w_k1_PG_s101
                                 + 0.00406811313077929 w_k1_PG_s102
                                 + 0.00986275541299019 w_k1_PG_s103
                                 + 0.0064173011053935 w_k1_PG_s104
                                 + 0.00375707599164139 w_k1_PG_s105
                                 + 0.00262899456227701 w_k1_PG_s106
                                 + 0.00628319373841569 w_k1_PG_s107
                                 + 0.0259871578731526 w_k1_PG_s110
                                 + 0.0210736251928036 w_k1_PG_s111
                                 + 0.0410807015660343 w_k1_PG_s112
                                 + 0.00994727860330319 w_k1_PG_s113
                                 + 0.00971015154241837 w_k1_PG_s114
                                 + 0.0140949743269544 w_k1_PG_s115
                                 + 0.0143492063624857 w_k1_PG_s116
                                 + 0.0465780899110579 w_k1_PG_s117 >= 1.17
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
