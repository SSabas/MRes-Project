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
 w_k1_ADM_s0_return_constraint:  1.14546 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.00183 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.09308 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.995244 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.777072 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.18145 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.990244 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.646752 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.853732 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.20692 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.1916 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.809577 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.21787 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.47202 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.881563 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.39077 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.20875 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.948145 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.967751 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.970174 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.15489 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.16043 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.927154 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.40878 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.986642 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.88486 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.15087 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.25077 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.35629 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.946579 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.787683 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.684378 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.82652 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.04521 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.758411 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.802067 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.07962 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.947163 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.25549 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.90241 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.22636 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.938968 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.865093 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.16659 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.791207 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.617575 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.910792 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.72548 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.14907 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.02604 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.16619 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.43993 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.668347 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.65296 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.01166 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.884182 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.937008 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.45764 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.01268 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.38454 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.887764 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.82105 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.962658 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.73952 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.919496 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.1743 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.3469 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.82839 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.969153 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.601269 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.871766 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.59189 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.773278 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.963433 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.33289 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.988327 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.34105 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.881945 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.20853 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.882584 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.847933 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.26032 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.905461 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.597165 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.70373 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 2.01234 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.02298 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.06688 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.49332 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.52226 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.75165 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.90908 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.37791 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.12589 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.75285 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.35831 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.0077 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.19128 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.853604 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.62683 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.23358 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.641411 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.03484 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.986042 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.38592 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.676656 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.873912 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.536683 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.946731 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.70583 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.675858 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.848316 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.08975 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.11264 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21402 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.918298 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.10308 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.94664 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.851872 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.18321 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.939441 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.667769 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.756278 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.4324 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.13338 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.884655 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.20717 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.40438 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.821465 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.59617 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.34269 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.01631 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.807473 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.18045 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.86589 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.26266 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.768515 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.37124 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.0511 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.825586 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.06183 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.03914 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.2594 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.861491 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.954368 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.665758 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.852789 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.29197 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.838467 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.857565 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.07936 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.16656 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18919 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.936019 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.3079 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.77847 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.07884 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.00722 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.781293 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.756668 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.10699 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.65952 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.776419 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.30549 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.17866 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.920966 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.71644 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.15748 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.962119 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.975622 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.20115 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.48841 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.66233 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.00022 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.24513 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.89899 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.01451 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.496292 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.609189 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.27228 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.963894 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.789789 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.991128 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.561328 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.05941 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.61648 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.607338 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.50127 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.11367 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.795016 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08175 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.890647 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.12062 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.06669 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.935672 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1715 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.851643 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.820892 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.981128 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.4903 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.08918 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.938544 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.2686 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.32434 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.805504 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.38978 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.05361 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.987075 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.753787 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.16247 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.955598 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.974589 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.884154 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.41137 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.06921 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.765122 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.985567 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.997994 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.15236 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.85321 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.04679 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.710449 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.939725 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.39121 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.831419 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.939547 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.0133 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.16439 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.11602 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.91775 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.36839 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.963608 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.979235 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.09415 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.832077 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.613843 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.867359 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.6557 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.85413 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.03679 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.14413 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.10174 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.776817 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.39796 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.02398 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.07356 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.901314 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.36391 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.16736 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.05957 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.05122 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.64726 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.12081 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.574776 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.788467 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.11298 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.03723 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.769128 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.947094 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.689212 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.05042 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.59907 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.672988 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.1059 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.08223 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.904513 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.30495 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.896421 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.28517 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.689892 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.09317 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.20548 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.760823 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.748617 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.20906 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 2.2306 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.80085 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.4179 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.14858 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.04634 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.720588 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.36133 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.83511 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.33646 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.37819 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.76084 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.5858 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.37533 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.03509 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 2.08106 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.04361 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.425055 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.730896 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.22484 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.03937 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.758791 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.856619 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.530753 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.884793 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.71644 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.526188 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.4762 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.27083 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.75106 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12504 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.938713 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.12335 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.04673 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.830785 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.17256 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.98071 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.730275 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.864211 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.56716 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.16556 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.984846 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.34296 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.28775 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.818828 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.33043 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.17557 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.05461 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.78533 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.12385 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.918148 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.0939 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.919127 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.38871 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.14644 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.872904 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.1391 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.09608 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.2579 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.885319 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.932434 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.685745 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.82866 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.35288 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.830802 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.863876 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.981598 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.04998 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10954 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.944309 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.08543 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.89972 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.945916 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.06907 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.02883 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.79128 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.907097 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.5625 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.984128 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.00292 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.09089 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.27889 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.918449 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.37029 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.14623 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.1191 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.924234 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.33241 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.13695 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.08147 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.96292 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.33562 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.01352 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.799498 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.990032 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.05492 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.09526 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.854586 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.992559 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.764576 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.991686 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.26947 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.87849 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.01749 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.14006 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.06289 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.573999383342026 z_k1_s0000
                                 - 0.155000137825458 z_k1_s0001
                                 - 0.324000808786711 z_k1_s0002
                                 - 0.298001059193075 z_k1_s0003
                                 - 0.166000228346066 z_k1_s0004
                                 - 0.431999375713677 z_k1_s0005
                                 - 0.263999618491691 z_k1_s0006
                                 - 0.341000252135949 z_k1_s0007
                                 - 0.67700003785553 z_k1_s0010
                                 - 0.39500043536598 z_k1_s0011
                                 - 0.272999791335524 z_k1_s0012
                                 - 0.555000250324566 z_k1_s0013
                                 - 0.245999786873784 z_k1_s0014
                                 - 0.300999616203361 z_k1_s0015
                                 - 0.214999929787994 z_k1_s0016
                                 - 0.192999883856231 z_k1_s0017
                                 - 0.323999079029071 z_k1_s0020
                                 - 0.147999443606338 z_k1_s0021
                                 - 0.391999021442113 z_k1_s0022
                                 - 0.143999468546673 z_k1_s0023
                                 - 0.126999482943413 z_k1_s0024
                                 - 0.134999616262113 z_k1_s0025
                                 - 0.209999606629254 z_k1_s0026
                                 - 0.351999270845466 z_k1_s0027
                                 - 0.200999882238843 z_k1_s0030
                                 - 0.110999889060279 z_k1_s0031
                                 - 0.343000034467164 z_k1_s0032
                                 - 0.100000005565065 z_k1_s0033
                                 - 0.0929999614045316 z_k1_s0034
                                 - 0.109999899651623 z_k1_s0035
                                 - 0.0849999278353427 z_k1_s0036
                                 - 0.139999700211242 z_k1_s0037
                                 - 0.316999350925923 z_k1_s0040
                                 - 0.095999565498387 z_k1_s0041
                                 - 0.178999295912195 z_k1_s0042
                                 - 0.210999208611429 z_k1_s0043
                                 - 0.422997767247283 z_k1_s0044
                                 - 0.138999405038154 z_k1_s0045
                                 - 0.190998400177841 z_k1_s0046
                                 - 0.169999277315708 z_k1_s0047
                                 - 0.552999981830473 z_k1_s0050
                                 - 0.43399981004418 z_k1_s0051
                                 - 0.314999985257864 z_k1_s0052
                                 - 0.571998230519163 z_k1_s0053
                                 - 0.589001229345571 z_k1_s0054
                                 - 0.209999874505251 z_k1_s0055
                                 - 0.292999839776373 z_k1_s0056
                                 - 0.504000115212573 z_k1_s0057
                                 - 0.122000006808175 z_k1_s0060
                                 - 0.198000610153973 z_k1_s0061
                                 - 0.460999388157416 z_k1_s0062
                                 - 0.123999906680487 z_k1_s0063
                                 - 0.126999898688888 z_k1_s0064
                                 - 0.133999974841781 z_k1_s0065
                                 - 0.117999922663684 z_k1_s0066
                                 - 0.137999916786338 z_k1_s0067
                                 - 0.160000491050716 z_k1_s0070
                                 - 0.104000025383059 z_k1_s0071
                                 - 0.150999930916782 z_k1_s0072
                                 - 0.21700061090005 z_k1_s0073
                                 - 0.186000313771539 z_k1_s0074
                                 - 0.265000170615452 z_k1_s0075
                                 - 0.22800047939734 z_k1_s0076
                                 - 0.157999713687827 z_k1_s0077
                                 - 0.387000164170664 z_k1_s0100
                                 - 0.441000087471174 z_k1_s0101
                                 - 0.39500004705217 z_k1_s0102
                                 - 0.441000087471174 z_k1_s0103
                                 - 0.765999493659361 z_k1_s0104
                                 - 1.34499875058708 z_k1_s0105
                                 - 0.351000215303658 z_k1_s0106
                                 - 0.632999656128464 z_k1_s0107
                                 - 0.295000142562058 z_k1_s0110
                                 - 0.205999669717589 z_k1_s0111
                                 - 0.364999558135641 z_k1_s0112
                                 - 0.146000170654517 z_k1_s0113
                                 - 0.106000157212209 z_k1_s0114
                                 - 0.142000030350182 z_k1_s0115
                                 - 0.314999975583082 z_k1_s0116
                                 - 0.162000037071336 z_k1_s0117
                                 - 0.0580000355956257 z_k1_s0120
                                 - 0.239000718982547 z_k1_s0121
                                 - 0.170999690986689 z_k1_s0122
                                 - 0.123000122990771 z_k1_s0123
                                 - 0.288999776977804 z_k1_s0124
                                 - 0.210000241984275 z_k1_s0125
                                 - 0.0820000819938473 z_k1_s0126
                                 - 0.139999723989079 z_k1_s0127
                                 - 0.711999759864578 z_k1_s0130
                                 - 0.649000916123675 z_k1_s0131
                                 - 0.254999893160489 z_k1_s0132
                                 - 0.218000126061894 z_k1_s0133
                                 - 0.800999241847683 z_k1_s0134
                                 - 0.294999884056252 z_k1_s0135
                                 - 0.673998226433708 z_k1_s0136
                                 - 0.300000126918205 z_k1_s0137
                                 - 0.313999509583656 z_k1_s0140
                                 - 0.606000316711399 z_k1_s0141
                                 - 0.208999809321497 z_k1_s0142
                                 - 0.792000648375106 z_k1_s0143
                                 - 0.271999807078651 z_k1_s0144
                                 - 0.278999905496034 z_k1_s0145
                                 - 0.230999695776466 z_k1_s0146
                                 - 0.256999723041301 z_k1_s0147
                                 - 0.0850001414003868 z_k1_s0150
                                 - 0.0860000971838664 z_k1_s0151
                                 - 0.208000667175444 z_k1_s0152
                                 - 0.327999949198434 z_k1_s0153
                                 - 0.102000077920356 z_k1_s0154
                                 - 0.121000614208098 z_k1_s0155
                                 - 0.135999950960293 z_k1_s0156
                                 - 0.0810000888661966 z_k1_s0157
                                 - 0.195999839774383 z_k1_s0160
                                 - 0.083000063016763 z_k1_s0161
                                 - 0.190000212026356 z_k1_s0162
                                 - 0.220000600767522 z_k1_s0163
                                 - 0.102000084219399 z_k1_s0164
                                 - 0.113999677215607 z_k1_s0165
                                 - 0.0519999988177124 z_k1_s0166
                                 - 0.167999701949398 z_k1_s0167
                                 - 0.228999791324015 z_k1_s0170
                                 - 0.137000058373076 z_k1_s0171
                                 - 0.115000159789195 z_k1_s0172
                                 - 0.0760000422901838 z_k1_s0173
                                 - 0.104000148797189 z_k1_s0174
                                 - 0.155999976395524 z_k1_s0175
                                 - 0.150999803126253 z_k1_s0176
                                 - 0.266000086315578 z_k1_s0177
                                 - 0.777000339587337 z_k1_s1000
                                 - 0.183999802446498 z_k1_s1001
                                 - 0.431997985442184 z_k1_s1002
                                 - 0.717999293725902 z_k1_s1003
                                 - 0.343999320513534 z_k1_s1004
                                 - 0.469000454894643 z_k1_s1005
                                 - 0.517000231054886 z_k1_s1006
                                 - 0.521997565742639 z_k1_s1007
                                 - 0.20599986949798 z_k1_s1010
                                 - 0.869998581510941 z_k1_s1011
                                 - 0.196000079324279 z_k1_s1012
                                 - 0.302999861422754 z_k1_s1013
                                 - 0.188000041025331 z_k1_s1014
                                 - 0.415999929945423 z_k1_s1015
                                 - 0.613999417544477 z_k1_s1016
                                 - 0.642999255728182 z_k1_s1017
                                 - 0.272999765949119 z_k1_s1020
                                 - 0.421999217598028 z_k1_s1021
                                 - 0.428000634623828 z_k1_s1022
                                 - 0.339999940888743 z_k1_s1023
                                 - 0.378999635567504 z_k1_s1024
                                 - 0.2609997866942 z_k1_s1025
                                 - 0.266999617721844 z_k1_s1026
                                 - 0.801998032004394 z_k1_s1027
                                 - 0.292999871254667 z_k1_s1030
                                 - 0.726000449392108 z_k1_s1031
                                 - 0.872997439269025 z_k1_s1032
                                 - 0.722998569394622 z_k1_s1033
                                 - 1.16800085402201 z_k1_s1034
                                 - 0.408999807657538 z_k1_s1035
                                 - 0.614999594485051 z_k1_s1036
                                 - 0.478000239099763 z_k1_s1037
                                 - 0.265999105318443 z_k1_s1040
                                 - 0.247000018138393 z_k1_s1041
                                 - 0.182999887132517 z_k1_s1042
                                 - 0.221000042292241 z_k1_s1043
                                 - 0.544999684159679 z_k1_s1044
                                 - 0.284999925698163 z_k1_s1045
                                 - 0.204999885740796 z_k1_s1046
                                 - 0.524000732160668 z_k1_s1047
                                 - 0.311999623477683 z_k1_s1050
                                 - 0.725001240646749 z_k1_s1051
                                 - 1.08100064766 z_k1_s1052
                                 - 0.390999431754876 z_k1_s1053
                                 - 0.505999294606263 z_k1_s1054
                                 - 0.448999123247255 z_k1_s1055
                                 - 0.332999230563296 z_k1_s1056
                                 - 1.29999800640768 z_k1_s1057
                                 - 0.517000143936699 z_k1_s1060
                                 - 1.47299750967747 z_k1_s1061
                                 - 0.455998447431215 z_k1_s1062
                                 - 0.439998617001704 z_k1_s1063
                                 - 0.634997956157149 z_k1_s1064
                                 - 0.477000567862923 z_k1_s1065
                                 - 0.349999102237113 z_k1_s1066
                                 - 0.337998760816385 z_k1_s1067
                                 - 0.963000789522316 z_k1_s1070
                                 - 0.820999028092861 z_k1_s1071
                                 - 0.960003282096131 z_k1_s1072
                                 - 0.678002023529761 z_k1_s1073
                                 - 0.510000860261676 z_k1_s1074
                                 - 0.245000592242757 z_k1_s1075
                                 - 1.08200006763806 z_k1_s1076
                                 - 0.63000068162243 z_k1_s1077
                                 - 0.260000262379845 z_k1_s1100
                                 - 0.167000075643866 z_k1_s1101
                                 - 0.559999797817205 z_k1_s1102
                                 - 0.359000681324861 z_k1_s1103
                                 - 0.125000025182496 z_k1_s1104
                                 - 0.454999671663782 z_k1_s1105
                                 - 0.516999621154224 z_k1_s1106
                                 - 0.182000131165828 z_k1_s1107
                                 - 0.213000437916401 z_k1_s1110
                                 - 0.329999970574727 z_k1_s1111
                                 - 0.231000697402288 z_k1_s1112
                                 - 0.152000595658645 z_k1_s1113
                                 - 0.21499934985931 z_k1_s1114
                                 - 0.0680000548578207 z_k1_s1115
                                 - 0.160000551430154 z_k1_s1116
                                 - 0.067000024486383 z_k1_s1117
                                 - 0.819005007575397 z_k1_s1120
                                 - 0.442001698579272 z_k1_s1121
                                 - 0.433001907855393 z_k1_s1122
                                 - 0.43500176237139 z_k1_s1123
                                 - 0.986003994708483 z_k1_s1124
                                 - 0.309001129021912 z_k1_s1125
                                 - 0.697004974061704 z_k1_s1126
                                 - 0.333001610223348 z_k1_s1127
                                 - 0.237000500267427 z_k1_s1130
                                 - 0.348001025471905 z_k1_s1131
                                 - 0.692002267288396 z_k1_s1132
                                 - 0.633003108015014 z_k1_s1133
                                 - 0.309000784475593 z_k1_s1134
                                 - 1.06100294040323 z_k1_s1135
                                 - 0.934000146769065 z_k1_s1136
                                 - 1.00900261907482 z_k1_s1137
                                 - 0.649001897933815 z_k1_s1140
                                 - 0.767003359017013 z_k1_s1141
                                 - 0.72900080781267 z_k1_s1142
                                 - 0.291001252004543 z_k1_s1143
                                 - 0.236000876158471 z_k1_s1144
                                 - 0.382001546497684 z_k1_s1145
                                 - 0.617002333982273 z_k1_s1146
                                 - 0.421002958771154 z_k1_s1147
                                 - 0.494999156485764 z_k1_s1150
                                 - 0.484001539565346 z_k1_s1151
                                 - 0.365999030310468 z_k1_s1152
                                 - 0.420001014292039 z_k1_s1153
                                 - 0.246000186299367 z_k1_s1154
                                 - 0.17500016312152 z_k1_s1155
                                 - 0.223000167826254 z_k1_s1156
                                 - 0.186000115543408 z_k1_s1157
                                 - 0.527999717666063 z_k1_s1160
                                 - 0.222000410656186 z_k1_s1161
                                 - 0.379000861951221 z_k1_s1162
                                 - 0.425999948662771 z_k1_s1163
                                 - 0.270000667317343 z_k1_s1164
                                 - 0.511000403166904 z_k1_s1165
                                 - 1.00100302324955 z_k1_s1166
                                 - 0.545002914173558 z_k1_s1167
                                 - 0.472000204159571 z_k1_s1170
                                 - 0.514999804538511 z_k1_s1171
                                 - 2.34500002066566 z_k1_s1172
                                 - 0.497000004379886 z_k1_s1173
                                 - 1.18699701046059 z_k1_s1174
                                 - 0.747999006591851 z_k1_s1175
                                 - 0.542000204776456 z_k1_s1176
                                 - 0.69400030611598 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.810416 w_k1_ADM_s000
                                 + 0.92697 w_k1_AXP_s000
                                 + 0.739039 w_k1_DIS_s000
                                 + 0.839995 w_k1_FDX_s000
                                 + 1.03535 w_k1_IBM_s000
                                 + 0.979495 w_k1_INTC_s000
                                 + 0.89622 w_k1_KO_s000
                                 + 0.778708 w_k1_MAT_s000
                                 + 0.755783 w_k1_MSFT_s000
                                 + 0.825597 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.85035 w_k1_ADM_s000
                                 + 1.13435 w_k1_AXP_s000
                                 + 0.844226 w_k1_DIS_s000
                                 + 0.926553 w_k1_FDX_s000
                                 + 1.43869 w_k1_IBM_s000
                                 + 1.2332 w_k1_INTC_s000 + 1.08189 w_k1_KO_s000
                                 + 1.64149 w_k1_MAT_s000
                                 + 1.00881 w_k1_MSFT_s000
                                 + 1.03072 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.19396 w_k1_ADM_s000
                                 + 1.50645 w_k1_AXP_s000
                                 + 1.36574 w_k1_DIS_s000
                                 + 1.21773 w_k1_FDX_s000
                                 + 1.35438 w_k1_IBM_s000
                                 + 1.19222 w_k1_INTC_s000
                                 + 1.21271 w_k1_KO_s000 + 1.12171 w_k1_MAT_s000
                                 + 1.04841 w_k1_MSFT_s000
                                 + 1.09261 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.94924 w_k1_ADM_s000
                                 + 0.933533 w_k1_AXP_s000
                                 + 1.28332 w_k1_DIS_s000
                                 + 1.01432 w_k1_FDX_s000
                                 + 1.41382 w_k1_IBM_s000
                                 + 1.06673 w_k1_INTC_s000
                                 + 1.28674 w_k1_KO_s000 + 1.34001 w_k1_MAT_s000
                                 + 1.11043 w_k1_MSFT_s000 + 1.0796 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.19609 w_k1_ADM_s000
                                 + 1.39304 w_k1_AXP_s000
                                 + 1.06492 w_k1_DIS_s000
                                 + 1.22735 w_k1_FDX_s000
                                 + 0.769099 w_k1_IBM_s000
                                 + 1.17215 w_k1_INTC_s000
                                 + 0.924871 w_k1_KO_s000
                                 + 0.805473 w_k1_MAT_s000
                                 + 1.13167 w_k1_MSFT_s000
                                 + 0.925866 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.872734 w_k1_ADM_s000
                                 + 1.01281 w_k1_AXP_s000
                                 + 0.848427 w_k1_DIS_s000
                                 + 0.976039 w_k1_FDX_s000
                                 + 0.889017 w_k1_IBM_s000
                                 + 1.22895 w_k1_INTC_s000
                                 + 0.98851 w_k1_KO_s000 + 1.08353 w_k1_MAT_s000
                                 + 0.976972 w_k1_MSFT_s000
                                 + 0.892754 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.887647 w_k1_ADM_s000
                                 + 0.787546 w_k1_AXP_s000
                                 + 0.685577 w_k1_DIS_s000
                                 + 0.833114 w_k1_FDX_s000
                                 + 0.622415 w_k1_IBM_s000
                                 + 1.00242 w_k1_INTC_s000
                                 + 0.591178 w_k1_KO_s000
                                 + 0.454776 w_k1_MAT_s000
                                 + 0.737551 w_k1_MSFT_s000
                                 + 0.725418 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.14774 w_k1_ADM_s000
                                 + 1.11663 w_k1_AXP_s000
                                 + 1.32991 w_k1_DIS_s000
                                 + 1.10863 w_k1_FDX_s000
                                 + 0.997711 w_k1_IBM_s000
                                 + 1.17834 w_k1_INTC_s000
                                 + 1.13987 w_k1_KO_s000 + 1.14529 w_k1_MAT_s000
                                 + 1.06574 w_k1_MSFT_s000
                                 + 1.05382 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.0044 w_k1_ADM_s001
                                 + 0.975778 w_k1_AXP_s001
                                 + 1.1955 w_k1_DIS_s001 + 1.17294 w_k1_FDX_s001
                                 + 0.914187 w_k1_IBM_s001
                                 + 0.996638 w_k1_INTC_s001
                                 + 1.25426 w_k1_KO_s001
                                 + 0.728476 w_k1_MAT_s001
                                 + 1.15522 w_k1_MSFT_s001
                                 + 1.13981 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.821208 w_k1_ADM_s001
                                 + 0.692288 w_k1_AXP_s001
                                 + 0.604671 w_k1_DIS_s001
                                 + 0.753473 w_k1_FDX_s001
                                 + 0.845561 w_k1_IBM_s001
                                 + 0.756607 w_k1_INTC_s001
                                 + 0.855112 w_k1_KO_s001
                                 + 0.778326 w_k1_MAT_s001
                                 + 0.791929 w_k1_MSFT_s001
                                 + 0.80514 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.31468 w_k1_ADM_s001
                                 + 1.25555 w_k1_AXP_s001
                                 + 1.08208 w_k1_DIS_s001
                                 + 1.00344 w_k1_FDX_s001 + 1.5708 w_k1_IBM_s001
                                 + 0.773785 w_k1_INTC_s001
                                 + 1.24655 w_k1_KO_s001 + 1.62661 w_k1_MAT_s001
                                 + 0.895918 w_k1_MSFT_s001
                                 + 1.11448 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.05499 w_k1_ADM_s001
                                 + 0.958722 w_k1_AXP_s001
                                 + 0.945323 w_k1_DIS_s001
                                 + 0.981736 w_k1_FDX_s001
                                 + 0.657466 w_k1_IBM_s001
                                 + 0.992057 w_k1_INTC_s001
                                 + 0.872291 w_k1_KO_s001
                                 + 0.682654 w_k1_MAT_s001
                                 + 0.940913 w_k1_MSFT_s001
                                 + 0.940787 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.46616 w_k1_ADM_s001
                                 + 1.52395 w_k1_AXP_s001
                                 + 1.44874 w_k1_DIS_s001
                                 + 1.42244 w_k1_FDX_s001
                                 + 1.13267 w_k1_IBM_s001
                                 + 1.30787 w_k1_INTC_s001
                                 + 1.32921 w_k1_KO_s001 + 1.12367 w_k1_MAT_s001
                                 + 1.23503 w_k1_MSFT_s001
                                 + 1.09903 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.29481 w_k1_ADM_s001
                                 + 1.05246 w_k1_AXP_s001
                                 + 0.84264 w_k1_DIS_s001
                                 + 0.935966 w_k1_FDX_s001
                                 + 1.13356 w_k1_IBM_s001
                                 + 0.919444 w_k1_INTC_s001
                                 + 1.13531 w_k1_KO_s001
                                 + 0.970442 w_k1_MAT_s001
                                 + 0.940705 w_k1_MSFT_s001
                                 + 1.09109 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.856027 w_k1_ADM_s001
                                 + 0.682042 w_k1_AXP_s001
                                 + 0.733103 w_k1_DIS_s001
                                 + 0.764924 w_k1_FDX_s001
                                 + 1.24886 w_k1_IBM_s001
                                 + 0.703332 w_k1_INTC_s001
                                 + 1.13483 w_k1_KO_s001
                                 + 0.976861 w_k1_MAT_s001
                                 + 0.827849 w_k1_MSFT_s001
                                 + 0.845252 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.960706 w_k1_ADM_s001
                                 + 1.31117 w_k1_AXP_s001
                                 + 1.73736 w_k1_DIS_s001 + 1.237 w_k1_FDX_s001
                                 + 1.50492 w_k1_IBM_s001
                                 + 1.15758 w_k1_INTC_s001 + 1.703 w_k1_KO_s001
                                 + 1.65359 w_k1_MAT_s001
                                 + 1.29303 w_k1_MSFT_s001
                                 + 1.19507 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.1222 w_k1_ADM_s002
                                 + 0.819399 w_k1_AXP_s002
                                 + 0.941543 w_k1_DIS_s002
                                 + 0.987898 w_k1_FDX_s002
                                 + 0.837842 w_k1_IBM_s002
                                 + 0.838398 w_k1_INTC_s002
                                 + 0.844491 w_k1_KO_s002
                                 + 0.759574 w_k1_MAT_s002
                                 + 0.846687 w_k1_MSFT_s002
                                 + 1.00774 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.05697 w_k1_ADM_s002
                                 + 1.25628 w_k1_AXP_s002
                                 + 1.22921 w_k1_DIS_s002
                                 + 1.34135 w_k1_FDX_s002
                                 + 0.911234 w_k1_IBM_s002
                                 + 0.994326 w_k1_INTC_s002
                                 + 0.978864 w_k1_KO_s002
                                 + 0.975488 w_k1_MAT_s002
                                 + 1.03553 w_k1_MSFT_s002 + 1.1051 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.868874 w_k1_ADM_s002
                                 + 0.730846 w_k1_AXP_s002
                                 + 0.901996 w_k1_DIS_s002
                                 + 0.905971 w_k1_FDX_s002
                                 + 1.23297 w_k1_IBM_s002
                                 + 0.890337 w_k1_INTC_s002
                                 + 0.998173 w_k1_KO_s002
                                 + 0.954203 w_k1_MAT_s002
                                 + 0.872129 w_k1_MSFT_s002
                                 + 0.953853 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.39295 w_k1_ADM_s002
                                 + 1.23236 w_k1_AXP_s002 + 1.5395 w_k1_DIS_s002
                                 + 1.42833 w_k1_FDX_s002 + 1.1617 w_k1_IBM_s002
                                 + 1.43072 w_k1_INTC_s002
                                 + 1.36413 w_k1_KO_s002 + 0.88108 w_k1_MAT_s002
                                 + 1.287 w_k1_MSFT_s002 + 1.34706 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.751332 w_k1_ADM_s002
                                 + 0.889561 w_k1_AXP_s002 + 0.644 w_k1_DIS_s002
                                 + 0.810216 w_k1_FDX_s002
                                 + 0.852393 w_k1_IBM_s002
                                 + 0.724005 w_k1_INTC_s002
                                 + 0.699979 w_k1_KO_s002
                                 + 1.02748 w_k1_MAT_s002
                                 + 0.722482 w_k1_MSFT_s002
                                 + 0.828781 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.94331 w_k1_ADM_s002
                                 + 0.901296 w_k1_AXP_s002
                                 + 0.940619 w_k1_DIS_s002
                                 + 1.04955 w_k1_FDX_s002
                                 + 0.622023 w_k1_IBM_s002
                                 + 0.941507 w_k1_INTC_s002
                                 + 0.79094 w_k1_KO_s002
                                 + 0.537846 w_k1_MAT_s002
                                 + 0.925678 w_k1_MSFT_s002
                                 + 0.904925 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.666522 w_k1_ADM_s002
                                 + 0.605658 w_k1_AXP_s002
                                 + 0.644267 w_k1_DIS_s002
                                 + 0.71665 w_k1_FDX_s002
                                 + 0.864963 w_k1_IBM_s002
                                 + 0.80705 w_k1_INTC_s002
                                 + 0.798132 w_k1_KO_s002
                                 + 0.698307 w_k1_MAT_s002
                                 + 0.794201 w_k1_MSFT_s002
                                 + 0.809509 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.32679 w_k1_ADM_s002
                                 + 1.21789 w_k1_AXP_s002
                                 + 1.19142 w_k1_DIS_s002
                                 + 1.08849 w_k1_FDX_s002
                                 + 1.33884 w_k1_IBM_s002
                                 + 0.884644 w_k1_INTC_s002
                                 + 1.31825 w_k1_KO_s002 + 1.17952 w_k1_MAT_s002
                                 + 0.999621 w_k1_MSFT_s002
                                 + 1.15093 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.943728 w_k1_ADM_s003
                                 + 0.736262 w_k1_AXP_s003
                                 + 0.739911 w_k1_DIS_s003
                                 + 0.828635 w_k1_FDX_s003
                                 + 0.776461 w_k1_IBM_s003
                                 + 0.85393 w_k1_INTC_s003
                                 + 0.767427 w_k1_KO_s003
                                 + 0.666585 w_k1_MAT_s003
                                 + 0.777892 w_k1_MSFT_s003
                                 + 0.80998 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.3844 w_k1_ADM_s003
                                 + 1.12743 w_k1_AXP_s003 + 1.18 w_k1_DIS_s003
                                 + 1.32937 w_k1_FDX_s003
                                 + 0.848211 w_k1_IBM_s003
                                 + 1.14252 w_k1_INTC_s003
                                 + 0.908385 w_k1_KO_s003
                                 + 0.604394 w_k1_MAT_s003
                                 + 1.06649 w_k1_MSFT_s003
                                 + 0.966038 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.03228 w_k1_ADM_s003
                                 + 0.873393 w_k1_AXP_s003
                                 + 0.805752 w_k1_DIS_s003
                                 + 0.826951 w_k1_FDX_s003
                                 + 1.41815 w_k1_IBM_s003
                                 + 0.829133 w_k1_INTC_s003
                                 + 0.99652 w_k1_KO_s003 + 1.11159 w_k1_MAT_s003
                                 + 0.84033 w_k1_MSFT_s003
                                 + 0.788454 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.30439 w_k1_ADM_s003
                                 + 1.11579 w_k1_AXP_s003
                                 + 1.09758 w_k1_DIS_s003
                                 + 1.09848 w_k1_FDX_s003
                                 + 1.04166 w_k1_IBM_s003
                                 + 0.830439 w_k1_INTC_s003
                                 + 0.96198 w_k1_KO_s003 + 1.09236 w_k1_MAT_s003
                                 + 0.935429 w_k1_MSFT_s003
                                 + 0.989181 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.88817 w_k1_ADM_s003
                                 + 1.61163 w_k1_AXP_s003
                                 + 1.53844 w_k1_DIS_s003
                                 + 1.57752 w_k1_FDX_s003
                                 + 0.896269 w_k1_IBM_s003
                                 + 1.07677 w_k1_INTC_s003
                                 + 0.873567 w_k1_KO_s003
                                 + 0.817238 w_k1_MAT_s003
                                 + 1.10594 w_k1_MSFT_s003
                                 + 0.917336 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.7579 w_k1_ADM_s003
                                 + 1.3946 w_k1_AXP_s003 + 1.66855 w_k1_DIS_s003
                                 + 1.34022 w_k1_FDX_s003
                                 + 1.56744 w_k1_IBM_s003
                                 + 1.1556 w_k1_INTC_s003 + 1.40384 w_k1_KO_s003
                                 + 1.31171 w_k1_MAT_s003
                                 + 1.27413 w_k1_MSFT_s003
                                 + 1.10802 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.12559 w_k1_ADM_s003
                                 + 0.838282 w_k1_AXP_s003
                                 + 1.0748 w_k1_DIS_s003 + 1.09624 w_k1_FDX_s003
                                 + 0.581919 w_k1_IBM_s003
                                 + 0.830261 w_k1_INTC_s003
                                 + 0.735969 w_k1_KO_s003
                                 + 0.540776 w_k1_MAT_s003
                                 + 0.899292 w_k1_MSFT_s003
                                 + 0.783514 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.930557 w_k1_ADM_s003
                                 + 1.10572 w_k1_AXP_s003
                                 + 0.861232 w_k1_DIS_s003
                                 + 0.986893 w_k1_FDX_s003
                                 + 0.838176 w_k1_IBM_s003
                                 + 0.902826 w_k1_INTC_s003
                                 + 0.796471 w_k1_KO_s003
                                 + 0.896674 w_k1_MAT_s003
                                 + 0.706543 w_k1_MSFT_s003
                                 + 0.791736 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.827764 w_k1_ADM_s004
                                 + 1.16657 w_k1_AXP_s004
                                 + 1.12447 w_k1_DIS_s004
                                 + 0.931969 w_k1_FDX_s004
                                 + 1.29463 w_k1_IBM_s004
                                 + 1.12164 w_k1_INTC_s004 + 1.2317 w_k1_KO_s004
                                 + 1.62088 w_k1_MAT_s004
                                 + 0.933572 w_k1_MSFT_s004
                                 + 1.10369 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.10684 w_k1_ADM_s004
                                 + 1.57887 w_k1_AXP_s004 + 1.3774 w_k1_DIS_s004
                                 + 1.15558 w_k1_FDX_s004
                                 + 1.33961 w_k1_IBM_s004
                                 + 1.08368 w_k1_INTC_s004
                                 + 1.21395 w_k1_KO_s004 + 1.44521 w_k1_MAT_s004
                                 + 0.977563 w_k1_MSFT_s004
                                 + 1.23097 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.795207 w_k1_ADM_s004
                                 + 0.825781 w_k1_AXP_s004
                                 + 0.689075 w_k1_DIS_s004
                                 + 0.798218 w_k1_FDX_s004
                                 + 0.859991 w_k1_IBM_s004
                                 + 0.82871 w_k1_INTC_s004
                                 + 0.886317 w_k1_KO_s004
                                 + 1.04858 w_k1_MAT_s004
                                 + 0.849489 w_k1_MSFT_s004
                                 + 0.971648 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.31915 w_k1_ADM_s004
                                 + 1.24494 w_k1_AXP_s004
                                 + 1.57368 w_k1_DIS_s004
                                 + 1.36493 w_k1_FDX_s004
                                 + 1.12114 w_k1_IBM_s004
                                 + 1.13711 w_k1_INTC_s004
                                 + 1.20082 w_k1_KO_s004 + 1.01377 w_k1_MAT_s004
                                 + 1.16083 w_k1_MSFT_s004
                                 + 1.25832 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.817666 w_k1_ADM_s004
                                 + 0.843983 w_k1_AXP_s004
                                 + 1.01421 w_k1_DIS_s004
                                 + 0.92889 w_k1_FDX_s004
                                 + 0.949691 w_k1_IBM_s004
                                 + 1.09094 w_k1_INTC_s004
                                 + 1.02411 w_k1_KO_s004
                                 + 0.782877 w_k1_MAT_s004
                                 + 0.943735 w_k1_MSFT_s004
                                 + 0.986521 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.874338 w_k1_ADM_s004
                                 + 0.620185 w_k1_AXP_s004
                                 + 0.74369 w_k1_DIS_s004
                                 + 0.832508 w_k1_FDX_s004
                                 + 0.599584 w_k1_IBM_s004
                                 + 0.852478 w_k1_INTC_s004
                                 + 0.69522 w_k1_KO_s004
                                 + 0.494099 w_k1_MAT_s004
                                 + 0.789333 w_k1_MSFT_s004
                                 + 0.826555 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.906883 w_k1_ADM_s004
                                 + 0.799751 w_k1_AXP_s004
                                 + 1.39035 w_k1_DIS_s004 + 1.106 w_k1_FDX_s004
                                 + 0.898528 w_k1_IBM_s004
                                 + 1.28973 w_k1_INTC_s004
                                 + 1.13564 w_k1_KO_s004
                                 + 0.775643 w_k1_MAT_s004
                                 + 1.07671 w_k1_MSFT_s004
                                 + 0.972028 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.04184 w_k1_ADM_s004
                                 + 1.21609 w_k1_AXP_s004
                                 + 0.89577 w_k1_DIS_s004
                                 + 1.12883 w_k1_FDX_s004
                                 + 0.944081 w_k1_IBM_s004
                                 + 1.03869 w_k1_INTC_s004
                                 + 0.936446 w_k1_KO_s004
                                 + 1.38047 w_k1_MAT_s004
                                 + 0.930814 w_k1_MSFT_s004
                                 + 1.17666 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.24437 w_k1_ADM_s005
                                 + 0.932098 w_k1_AXP_s005
                                 + 0.950795 w_k1_DIS_s005
                                 + 1.07354 w_k1_FDX_s005
                                 + 0.73292 w_k1_IBM_s005
                                 + 1.12731 w_k1_INTC_s005
                                 + 0.951294 w_k1_KO_s005
                                 + 0.651842 w_k1_MAT_s005
                                 + 1.08631 w_k1_MSFT_s005
                                 + 1.00156 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.16416 w_k1_ADM_s005
                                 + 0.849514 w_k1_AXP_s005
                                 + 0.887377 w_k1_DIS_s005
                                 + 0.972912 w_k1_FDX_s005
                                 + 0.940524 w_k1_IBM_s005
                                 + 1.15696 w_k1_INTC_s005
                                 + 0.947453 w_k1_KO_s005
                                 + 0.769711 w_k1_MAT_s005
                                 + 1.05456 w_k1_MSFT_s005
                                 + 0.930886 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.41892 w_k1_ADM_s005
                                 + 1.23906 w_k1_AXP_s005
                                 + 0.934734 w_k1_DIS_s005
                                 + 1.20915 w_k1_FDX_s005
                                 + 0.954005 w_k1_IBM_s005
                                 + 0.929343 w_k1_INTC_s005
                                 + 0.931295 w_k1_KO_s005
                                 + 1.02016 w_k1_MAT_s005
                                 + 0.811221 w_k1_MSFT_s005
                                 + 1.15211 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.31726 w_k1_ADM_s005
                                 + 1.08871 w_k1_AXP_s005
                                 + 1.00388 w_k1_DIS_s005
                                 + 1.05584 w_k1_FDX_s005
                                 + 1.10893 w_k1_IBM_s005
                                 + 1.18579 w_k1_INTC_s005
                                 + 1.16966 w_k1_KO_s005 + 1.03858 w_k1_MAT_s005
                                 + 1.00281 w_k1_MSFT_s005
                                 + 1.20806 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.0696 w_k1_ADM_s005
                                 + 0.842459 w_k1_AXP_s005
                                 + 0.838039 w_k1_DIS_s005
                                 + 0.86997 w_k1_FDX_s005 + 1.4975 w_k1_IBM_s005
                                 + 1.1363 w_k1_INTC_s005 + 1.26077 w_k1_KO_s005
                                 + 1.13593 w_k1_MAT_s005
                                 + 0.967297 w_k1_MSFT_s005
                                 + 1.0124 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.08227 w_k1_ADM_s005
                                 + 0.77873 w_k1_AXP_s005
                                 + 0.642168 w_k1_DIS_s005
                                 + 0.924787 w_k1_FDX_s005
                                 + 0.618264 w_k1_IBM_s005
                                 + 0.845169 w_k1_INTC_s005
                                 + 0.699224 w_k1_KO_s005
                                 + 0.620638 w_k1_MAT_s005
                                 + 0.772669 w_k1_MSFT_s005
                                 + 0.841502 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.954512 w_k1_ADM_s005
                                 + 0.655464 w_k1_AXP_s005
                                 + 0.661315 w_k1_DIS_s005
                                 + 0.739937 w_k1_FDX_s005
                                 + 0.823051 w_k1_IBM_s005
                                 + 0.849324 w_k1_INTC_s005
                                 + 0.886128 w_k1_KO_s005
                                 + 0.668941 w_k1_MAT_s005
                                 + 0.8339 w_k1_MSFT_s005
                                 + 0.834736 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.49534 w_k1_ADM_s005
                                 + 1.48538 w_k1_AXP_s005
                                 + 1.39815 w_k1_DIS_s005
                                 + 1.41359 w_k1_FDX_s005
                                 + 1.29936 w_k1_IBM_s005
                                 + 1.64181 w_k1_INTC_s005
                                 + 1.43208 w_k1_KO_s005 + 1.48206 w_k1_MAT_s005
                                 + 1.37659 w_k1_MSFT_s005
                                 + 1.19945 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.760482 w_k1_ADM_s006
                                 + 0.681453 w_k1_AXP_s006
                                 + 0.651186 w_k1_DIS_s006
                                 + 0.741842 w_k1_FDX_s006
                                 + 0.670235 w_k1_IBM_s006
                                 + 0.699808 w_k1_INTC_s006
                                 + 0.763009 w_k1_KO_s006
                                 + 0.639293 w_k1_MAT_s006
                                 + 0.73315 w_k1_MSFT_s006
                                 + 0.928049 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.00181 w_k1_ADM_s006
                                 + 1.09418 w_k1_AXP_s006
                                 + 1.08732 w_k1_DIS_s006 + 1.1954 w_k1_FDX_s006
                                 + 0.898139 w_k1_IBM_s006
                                 + 0.948408 w_k1_INTC_s006
                                 + 1.05136 w_k1_KO_s006 + 1.11814 w_k1_MAT_s006
                                 + 1.05914 w_k1_MSFT_s006
                                 + 1.22353 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.761686 w_k1_ADM_s006
                                 + 0.834264 w_k1_AXP_s006
                                 + 0.958666 w_k1_DIS_s006
                                 + 0.862906 w_k1_FDX_s006
                                 + 0.896113 w_k1_IBM_s006
                                 + 0.87918 w_k1_INTC_s006
                                 + 0.830985 w_k1_KO_s006
                                 + 0.852886 w_k1_MAT_s006
                                 + 0.934466 w_k1_MSFT_s006
                                 + 1.02151 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.830964 w_k1_ADM_s006
                                 + 0.964058 w_k1_AXP_s006
                                 + 0.717686 w_k1_DIS_s006
                                 + 0.81783 w_k1_FDX_s006
                                 + 1.17631 w_k1_IBM_s006
                                 + 0.84564 w_k1_INTC_s006
                                 + 0.976329 w_k1_KO_s006
                                 + 0.925742 w_k1_MAT_s006
                                 + 0.86019 w_k1_MSFT_s006
                                 + 0.98787 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.977983 w_k1_ADM_s006
                                 + 1.39947 w_k1_AXP_s006 + 1.0424 w_k1_DIS_s006
                                 + 1.01357 w_k1_FDX_s006 + 1.4887 w_k1_IBM_s006
                                 + 0.893559 w_k1_INTC_s006
                                 + 1.03016 w_k1_KO_s006 + 1.39468 w_k1_MAT_s006
                                 + 0.890307 w_k1_MSFT_s006
                                 + 1.13811 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.990993 w_k1_ADM_s006
                                 + 1.18471 w_k1_AXP_s006
                                 + 1.00283 w_k1_DIS_s006
                                 + 0.981585 w_k1_FDX_s006
                                 + 1.18543 w_k1_IBM_s006
                                 + 0.968981 w_k1_INTC_s006
                                 + 1.1602 w_k1_KO_s006 + 1.32667 w_k1_MAT_s006
                                 + 0.929201 w_k1_MSFT_s006
                                 + 1.24714 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.16014 w_k1_ADM_s006
                                 + 0.990008 w_k1_AXP_s006
                                 + 1.28647 w_k1_DIS_s006
                                 + 0.999715 w_k1_FDX_s006
                                 + 1.21497 w_k1_IBM_s006
                                 + 1.07016 w_k1_INTC_s006
                                 + 1.49018 w_k1_KO_s006 + 1.2771 w_k1_MAT_s006
                                 + 1.207 w_k1_MSFT_s006 + 1.26322 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.46926 w_k1_ADM_s006
                                 + 1.57616 w_k1_AXP_s006
                                 + 1.69794 w_k1_DIS_s006
                                 + 1.30269 w_k1_FDX_s006
                                 + 1.34255 w_k1_IBM_s006
                                 + 1.23422 w_k1_INTC_s006
                                 + 1.42663 w_k1_KO_s006 + 1.91299 w_k1_MAT_s006
                                 + 1.31403 w_k1_MSFT_s006
                                 + 1.58499 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.997313 w_k1_ADM_s007
                                 + 1.02816 w_k1_AXP_s007
                                 + 0.895745 w_k1_DIS_s007
                                 + 1.01245 w_k1_FDX_s007
                                 + 1.01674 w_k1_IBM_s007
                                 + 0.870845 w_k1_INTC_s007
                                 + 0.863839 w_k1_KO_s007
                                 + 1.06585 w_k1_MAT_s007
                                 + 0.873902 w_k1_MSFT_s007
                                 + 0.897503 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.02779 w_k1_ADM_s007
                                 + 1.08174 w_k1_AXP_s007
                                 + 1.14044 w_k1_DIS_s007
                                 + 0.944647 w_k1_FDX_s007
                                 + 1.49051 w_k1_IBM_s007
                                 + 1.31548 w_k1_INTC_s007
                                 + 1.24105 w_k1_KO_s007 + 1.35806 w_k1_MAT_s007
                                 + 1.20594 w_k1_MSFT_s007
                                 + 0.988914 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.0936 w_k1_ADM_s007
                                 + 0.936123 w_k1_AXP_s007
                                 + 1.39762 w_k1_DIS_s007 + 1.1727 w_k1_FDX_s007
                                 + 0.857158 w_k1_IBM_s007
                                 + 1.14384 w_k1_INTC_s007
                                 + 1.20698 w_k1_KO_s007
                                 + 0.659987 w_k1_MAT_s007
                                 + 1.30137 w_k1_MSFT_s007
                                 + 1.02871 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.802258 w_k1_ADM_s007
                                 + 0.985685 w_k1_AXP_s007
                                 + 0.729216 w_k1_DIS_s007
                                 + 0.867113 w_k1_FDX_s007
                                 + 1.02175 w_k1_IBM_s007
                                 + 0.70571 w_k1_INTC_s007
                                 + 0.887991 w_k1_KO_s007
                                 + 1.48171 w_k1_MAT_s007
                                 + 0.881382 w_k1_MSFT_s007
                                 + 0.844557 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.05578 w_k1_ADM_s007
                                 + 1.07218 w_k1_AXP_s007
                                 + 0.974807 w_k1_DIS_s007
                                 + 1.16375 w_k1_FDX_s007
                                 + 0.850903 w_k1_IBM_s007
                                 + 1.0491 w_k1_INTC_s007
                                 + 0.910924 w_k1_KO_s007
                                 + 0.691749 w_k1_MAT_s007
                                 + 1.15028 w_k1_MSFT_s007
                                 + 0.920793 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.746131 w_k1_ADM_s007
                                 + 0.569519 w_k1_AXP_s007
                                 + 0.649452 w_k1_DIS_s007
                                 + 0.754242 w_k1_FDX_s007
                                 + 0.729744 w_k1_IBM_s007
                                 + 0.866722 w_k1_INTC_s007
                                 + 0.733096 w_k1_KO_s007
                                 + 0.653238 w_k1_MAT_s007
                                 + 0.930701 w_k1_MSFT_s007
                                 + 0.794252 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.05316 w_k1_ADM_s007
                                 + 1.42853 w_k1_AXP_s007
                                 + 1.53299 w_k1_DIS_s007 + 1.3919 w_k1_FDX_s007
                                 + 1.23334 w_k1_IBM_s007
                                 + 1.29579 w_k1_INTC_s007
                                 + 1.33363 w_k1_KO_s007 + 1.24751 w_k1_MAT_s007
                                 + 1.39936 w_k1_MSFT_s007
                                 + 1.20839 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.906736 w_k1_ADM_s007
                                 + 0.797251 w_k1_AXP_s007
                                 + 1.00382 w_k1_DIS_s007
                                 + 0.936302 w_k1_FDX_s007
                                 + 0.912468 w_k1_IBM_s007
                                 + 1.0193 w_k1_INTC_s007
                                 + 0.850483 w_k1_KO_s007
                                 + 0.789076 w_k1_MAT_s007
                                 + 1.0772 w_k1_MSFT_s007
                                 + 0.894804 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.952909 w_k1_ADM_s010
                                 + 0.989688 w_k1_AXP_s010
                                 + 1.00045 w_k1_DIS_s010
                                 + 1.06964 w_k1_FDX_s010
                                 + 0.746843 w_k1_IBM_s010
                                 + 1.01669 w_k1_INTC_s010
                                 + 0.794893 w_k1_KO_s010
                                 + 0.597155 w_k1_MAT_s010
                                 + 1.0668 w_k1_MSFT_s010 + 0.88886 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.0393 w_k1_ADM_s010
                                 + 1.48593 w_k1_AXP_s010
                                 + 1.34231 w_k1_DIS_s010
                                 + 1.14258 w_k1_FDX_s010
                                 + 1.62733 w_k1_IBM_s010
                                 + 1.36077 w_k1_INTC_s010
                                 + 1.17121 w_k1_KO_s010 + 1.41863 w_k1_MAT_s010
                                 + 1.04402 w_k1_MSFT_s010
                                 + 1.10872 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.13507 w_k1_ADM_s010
                                 + 1.27485 w_k1_AXP_s010
                                 + 0.975267 w_k1_DIS_s010
                                 + 1.28427 w_k1_FDX_s010
                                 + 0.713599 w_k1_IBM_s010
                                 + 1.1131 w_k1_INTC_s010
                                 + 0.829834 w_k1_KO_s010
                                 + 0.905316 w_k1_MAT_s010
                                 + 1.10897 w_k1_MSFT_s010
                                 + 0.961583 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.97895 w_k1_ADM_s010
                                 + 1.11617 w_k1_AXP_s010
                                 + 0.869512 w_k1_DIS_s010
                                 + 0.985567 w_k1_FDX_s010
                                 + 0.864696 w_k1_IBM_s010
                                 + 0.872617 w_k1_INTC_s010
                                 + 0.782904 w_k1_KO_s010
                                 + 0.870005 w_k1_MAT_s010
                                 + 0.805638 w_k1_MSFT_s010
                                 + 0.957739 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.780896 w_k1_ADM_s010
                                 + 0.742465 w_k1_AXP_s010
                                 + 0.593375 w_k1_DIS_s010
                                 + 0.755962 w_k1_FDX_s010
                                 + 0.69399 w_k1_IBM_s010
                                 + 0.813754 w_k1_INTC_s010
                                 + 0.664291 w_k1_KO_s010
                                 + 0.803457 w_k1_MAT_s010
                                 + 0.710931 w_k1_MSFT_s010
                                 + 0.822908 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.844748 w_k1_ADM_s010
                                 + 0.928072 w_k1_AXP_s010
                                 + 0.79116 w_k1_DIS_s010
                                 + 0.868054 w_k1_FDX_s010
                                 + 1.24075 w_k1_IBM_s010
                                 + 0.926793 w_k1_INTC_s010
                                 + 1.0492 w_k1_KO_s010 + 1.16308 w_k1_MAT_s010
                                 + 0.931115 w_k1_MSFT_s010
                                 + 0.871857 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.41742 w_k1_ADM_s010
                                 + 1.63444 w_k1_AXP_s010
                                 + 1.67763 w_k1_DIS_s010 + 1.6004 w_k1_FDX_s010
                                 + 0.885784 w_k1_IBM_s010
                                 + 1.38837 w_k1_INTC_s010
                                 + 1.10327 w_k1_KO_s010
                                 + 0.967917 w_k1_MAT_s010
                                 + 1.30851 w_k1_MSFT_s010
                                 + 1.26529 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.20903 w_k1_ADM_s010
                                 + 1.08772 w_k1_AXP_s010
                                 + 1.33919 w_k1_DIS_s010
                                 + 1.22157 w_k1_FDX_s010
                                 + 1.13197 w_k1_IBM_s010
                                 + 1.0507 w_k1_INTC_s010 + 1.20159 w_k1_KO_s010
                                 + 1.40011 w_k1_MAT_s010
                                 + 1.27488 w_k1_MSFT_s010
                                 + 1.11114 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.28947 w_k1_ADM_s011
                                 + 1.36167 w_k1_AXP_s011
                                 + 1.35145 w_k1_DIS_s011
                                 + 1.34057 w_k1_FDX_s011
                                 + 1.18342 w_k1_IBM_s011
                                 + 1.10234 w_k1_INTC_s011 + 1.0433 w_k1_KO_s011
                                 + 1.50175 w_k1_MAT_s011
                                 + 1.24314 w_k1_MSFT_s011
                                 + 1.00267 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.14151 w_k1_ADM_s011
                                 + 1.1587 w_k1_AXP_s011 + 1.5178 w_k1_DIS_s011
                                 + 1.07905 w_k1_FDX_s011
                                 + 1.51583 w_k1_IBM_s011
                                 + 0.938964 w_k1_INTC_s011
                                 + 1.42013 w_k1_KO_s011 + 1.83608 w_k1_MAT_s011
                                 + 1.2375 w_k1_MSFT_s011 + 1.17808 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.02378 w_k1_ADM_s011
                                 + 0.896806 w_k1_AXP_s011
                                 + 0.819986 w_k1_DIS_s011
                                 + 0.913998 w_k1_FDX_s011
                                 + 1.24858 w_k1_IBM_s011
                                 + 0.833381 w_k1_INTC_s011
                                 + 0.959309 w_k1_KO_s011
                                 + 1.39133 w_k1_MAT_s011
                                 + 0.901198 w_k1_MSFT_s011
                                 + 0.950906 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.914535 w_k1_ADM_s011
                                 + 0.89743 w_k1_AXP_s011
                                 + 1.27631 w_k1_DIS_s011
                                 + 0.993357 w_k1_FDX_s011
                                 + 1.15619 w_k1_IBM_s011
                                 + 0.904369 w_k1_INTC_s011
                                 + 0.975202 w_k1_KO_s011
                                 + 1.00582 w_k1_MAT_s011
                                 + 0.972396 w_k1_MSFT_s011
                                 + 0.894072 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.983678 w_k1_ADM_s011
                                 + 0.766652 w_k1_AXP_s011
                                 + 1.06258 w_k1_DIS_s011
                                 + 0.934376 w_k1_FDX_s011
                                 + 0.886093 w_k1_IBM_s011
                                 + 0.965411 w_k1_INTC_s011
                                 + 0.946304 w_k1_KO_s011
                                 + 0.635327 w_k1_MAT_s011
                                 + 1.06886 w_k1_MSFT_s011
                                 + 0.982839 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.675415 w_k1_ADM_s011
                                 + 0.597525 w_k1_AXP_s011
                                 + 0.602372 w_k1_DIS_s011
                                 + 0.679802 w_k1_FDX_s011
                                 + 0.664527 w_k1_IBM_s011
                                 + 0.685751 w_k1_INTC_s011
                                 + 0.728785 w_k1_KO_s011
                                 + 0.708354 w_k1_MAT_s011
                                 + 0.86017 w_k1_MSFT_s011
                                 + 0.752078 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.21535 w_k1_ADM_s011
                                 + 0.94344 w_k1_AXP_s011
                                 + 0.975421 w_k1_DIS_s011
                                 + 1.08033 w_k1_FDX_s011
                                 + 0.61627 w_k1_IBM_s011
                                 + 0.887362 w_k1_INTC_s011
                                 + 0.764453 w_k1_KO_s011
                                 + 0.63465 w_k1_MAT_s011
                                 + 0.963385 w_k1_MSFT_s011
                                 + 1.0295 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.969237 w_k1_ADM_s011
                                 + 0.871034 w_k1_AXP_s011
                                 + 0.765961 w_k1_DIS_s011
                                 + 0.860541 w_k1_FDX_s011
                                 + 0.775122 w_k1_IBM_s011
                                 + 0.928391 w_k1_INTC_s011
                                 + 0.857201 w_k1_KO_s011
                                 + 0.840359 w_k1_MAT_s011
                                 + 0.850412 w_k1_MSFT_s011
                                 + 0.856242 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.832493 w_k1_ADM_s012
                                 + 0.705555 w_k1_AXP_s012
                                 + 0.600304 w_k1_DIS_s012
                                 + 0.703072 w_k1_FDX_s012
                                 + 0.912253 w_k1_IBM_s012
                                 + 0.796894 w_k1_INTC_s012
                                 + 0.915826 w_k1_KO_s012 + 0.7278 w_k1_MAT_s012
                                 + 0.819757 w_k1_MSFT_s012
                                 + 0.865114 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.06785 w_k1_ADM_s012
                                 + 0.962013 w_k1_AXP_s012
                                 + 0.846514 w_k1_DIS_s012
                                 + 0.890633 w_k1_FDX_s012
                                 + 0.822717 w_k1_IBM_s012
                                 + 1.03479 w_k1_INTC_s012
                                 + 0.956735 w_k1_KO_s012
                                 + 0.677664 w_k1_MAT_s012
                                 + 0.945499 w_k1_MSFT_s012
                                 + 0.965264 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.29032 w_k1_ADM_s012
                                 + 1.06596 w_k1_AXP_s012
                                 + 1.44618 w_k1_DIS_s012
                                 + 1.09934 w_k1_FDX_s012
                                 + 0.956373 w_k1_IBM_s012
                                 + 1.36554 w_k1_INTC_s012
                                 + 1.17689 w_k1_KO_s012
                                 + 0.884225 w_k1_MAT_s012
                                 + 1.29092 w_k1_MSFT_s012
                                 + 1.14151 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.896739 w_k1_ADM_s012
                                 + 0.771721 w_k1_AXP_s012
                                 + 0.715655 w_k1_DIS_s012
                                 + 0.852283 w_k1_FDX_s012
                                 + 0.618229 w_k1_IBM_s012
                                 + 0.986771 w_k1_INTC_s012
                                 + 0.762321 w_k1_KO_s012
                                 + 0.626026 w_k1_MAT_s012
                                 + 0.815678 w_k1_MSFT_s012
                                 + 0.949821 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.21206 w_k1_ADM_s012
                                 + 1.18754 w_k1_AXP_s012
                                 + 1.01111 w_k1_DIS_s012
                                 + 1.09641 w_k1_FDX_s012
                                 + 1.03485 w_k1_IBM_s012
                                 + 1.10886 w_k1_INTC_s012
                                 + 1.14955 w_k1_KO_s012 + 1.4816 w_k1_MAT_s012
                                 + 0.974564 w_k1_MSFT_s012
                                 + 1.13124 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.892157 w_k1_ADM_s012
                                 + 1.04336 w_k1_AXP_s012
                                 + 0.922195 w_k1_DIS_s012
                                 + 0.840041 w_k1_FDX_s012
                                 + 1.35431 w_k1_IBM_s012
                                 + 1.02821 w_k1_INTC_s012 + 1.0611 w_k1_KO_s012
                                 + 1.4276 w_k1_MAT_s012
                                 + 1.01701 w_k1_MSFT_s012
                                 + 1.04264 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.54391 w_k1_ADM_s012
                                 + 1.79412 w_k1_AXP_s012
                                 + 1.82135 w_k1_DIS_s012
                                 + 1.37545 w_k1_FDX_s012
                                 + 1.22364 w_k1_IBM_s012
                                 + 1.54718 w_k1_INTC_s012 + 1.483 w_k1_KO_s012
                                 + 1.66247 w_k1_MAT_s012
                                 + 1.38444 w_k1_MSFT_s012
                                 + 1.33873 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.20836 w_k1_ADM_s012
                                 + 1.17432 w_k1_AXP_s012 + 1.2982 w_k1_DIS_s012
                                 + 1.0406 w_k1_FDX_s012 + 1.63289 w_k1_IBM_s012
                                 + 1.43536 w_k1_INTC_s012 + 1.3418 w_k1_KO_s012
                                 + 1.7658 w_k1_MAT_s012
                                 + 1.17822 w_k1_MSFT_s012
                                 + 1.19936 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.935748 w_k1_ADM_s013
                                 + 0.785759 w_k1_AXP_s013
                                 + 0.761778 w_k1_DIS_s013
                                 + 0.906597 w_k1_FDX_s013
                                 + 0.721793 w_k1_IBM_s013
                                 + 0.940675 w_k1_INTC_s013
                                 + 0.780962 w_k1_KO_s013
                                 + 0.513855 w_k1_MAT_s013
                                 + 0.866453 w_k1_MSFT_s013
                                 + 0.884836 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.15553 w_k1_ADM_s013
                                 + 1.26077 w_k1_AXP_s013
                                 + 1.03397 w_k1_DIS_s013
                                 + 1.07187 w_k1_FDX_s013
                                 + 0.920202 w_k1_IBM_s013
                                 + 0.944134 w_k1_INTC_s013
                                 + 0.879903 w_k1_KO_s013
                                 + 0.935726 w_k1_MAT_s013
                                 + 0.940776 w_k1_MSFT_s013
                                 + 1.00632 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.17967 w_k1_ADM_s013
                                 + 1.26843 w_k1_AXP_s013
                                 + 1.11995 w_k1_DIS_s013
                                 + 1.03348 w_k1_FDX_s013
                                 + 1.50838 w_k1_IBM_s013
                                 + 1.16015 w_k1_INTC_s013
                                 + 1.12776 w_k1_KO_s013
                                 + 0.775403 w_k1_MAT_s013
                                 + 1.02227 w_k1_MSFT_s013
                                 + 1.01324 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.36488 w_k1_ADM_s013
                                 + 1.09478 w_k1_AXP_s013
                                 + 1.00908 w_k1_DIS_s013
                                 + 1.21009 w_k1_FDX_s013
                                 + 0.694456 w_k1_IBM_s013
                                 + 1.12214 w_k1_INTC_s013
                                 + 0.851613 w_k1_KO_s013
                                 + 0.616089 w_k1_MAT_s013
                                 + 1.15439 w_k1_MSFT_s013
                                 + 0.969673 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.953598 w_k1_ADM_s013
                                 + 0.81801 w_k1_AXP_s013
                                 + 0.932145 w_k1_DIS_s013
                                 + 0.886591 w_k1_FDX_s013
                                 + 1.17224 w_k1_IBM_s013
                                 + 0.907352 w_k1_INTC_s013
                                 + 0.967518 w_k1_KO_s013
                                 + 0.759238 w_k1_MAT_s013
                                 + 0.859101 w_k1_MSFT_s013
                                 + 0.904888 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.615744 w_k1_ADM_s013
                                 + 0.753821 w_k1_AXP_s013
                                 + 0.569117 w_k1_DIS_s013
                                 + 0.699187 w_k1_FDX_s013
                                 + 0.752433 w_k1_IBM_s013
                                 + 0.694416 w_k1_INTC_s013
                                 + 0.641264 w_k1_KO_s013
                                 + 0.58517 w_k1_MAT_s013
                                 + 0.681327 w_k1_MSFT_s013
                                 + 0.719796 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.18704 w_k1_ADM_s013
                                 + 1.54691 w_k1_AXP_s013
                                 + 1.47026 w_k1_DIS_s013
                                 + 1.32637 w_k1_FDX_s013
                                 + 1.25223 w_k1_IBM_s013
                                 + 1.20201 w_k1_INTC_s013
                                 + 1.17587 w_k1_KO_s013 + 1.23389 w_k1_MAT_s013
                                 + 1.0536 w_k1_MSFT_s013 + 1.18614 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.2487 w_k1_ADM_s013
                                 + 0.907538 w_k1_AXP_s013
                                 + 1.4158 w_k1_DIS_s013 + 1.09533 w_k1_FDX_s013
                                 + 0.995263 w_k1_IBM_s013
                                 + 1.04588 w_k1_INTC_s013
                                 + 1.09205 w_k1_KO_s013
                                 + 0.481697 w_k1_MAT_s013
                                 + 1.139 w_k1_MSFT_s013 + 1.00283 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.948998 w_k1_ADM_s014
                                 + 1.19608 w_k1_AXP_s014
                                 + 1.24395 w_k1_DIS_s014
                                 + 1.14271 w_k1_FDX_s014
                                 + 0.978244 w_k1_IBM_s014
                                 + 1.16288 w_k1_INTC_s014
                                 + 1.16656 w_k1_KO_s014
                                 + 0.880883 w_k1_MAT_s014
                                 + 1.072 w_k1_MSFT_s014 + 1.17407 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.907083 w_k1_ADM_s014
                                 + 0.895341 w_k1_AXP_s014
                                 + 0.915678 w_k1_DIS_s014
                                 + 0.92905 w_k1_FDX_s014
                                 + 1.03555 w_k1_IBM_s014
                                 + 1.03291 w_k1_INTC_s014
                                 + 1.04888 w_k1_KO_s014
                                 + 0.909553 w_k1_MAT_s014
                                 + 0.872586 w_k1_MSFT_s014
                                 + 0.987683 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.09023 w_k1_ADM_s014
                                 + 1.26977 w_k1_AXP_s014
                                 + 0.875697 w_k1_DIS_s014
                                 + 1.07612 w_k1_FDX_s014
                                 + 0.788843 w_k1_IBM_s014
                                 + 1.04021 w_k1_INTC_s014
                                 + 0.929723 w_k1_KO_s014
                                 + 0.903263 w_k1_MAT_s014
                                 + 0.903631 w_k1_MSFT_s014
                                 + 1.03071 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.697467 w_k1_ADM_s014
                                 + 0.757181 w_k1_AXP_s014
                                 + 0.717962 w_k1_DIS_s014
                                 + 0.796822 w_k1_FDX_s014
                                 + 0.794786 w_k1_IBM_s014
                                 + 1.14355 w_k1_INTC_s014
                                 + 0.796962 w_k1_KO_s014
                                 + 0.685875 w_k1_MAT_s014
                                 + 0.857386 w_k1_MSFT_s014
                                 + 0.895787 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.14512 w_k1_ADM_s014
                                 + 1.62628 w_k1_AXP_s014 + 1.8395 w_k1_DIS_s014
                                 + 1.45353 w_k1_FDX_s014
                                 + 1.18031 w_k1_IBM_s014
                                 + 1.3502 w_k1_INTC_s014 + 1.3894 w_k1_KO_s014
                                 + 1.50426 w_k1_MAT_s014
                                 + 1.30024 w_k1_MSFT_s014
                                 + 1.43492 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.01327 w_k1_ADM_s014
                                 + 1.07379 w_k1_AXP_s014
                                 + 1.02091 w_k1_DIS_s014
                                 + 1.21158 w_k1_FDX_s014
                                 + 0.766409 w_k1_IBM_s014
                                 + 1.24119 w_k1_INTC_s014
                                 + 0.89416 w_k1_KO_s014
                                 + 0.614649 w_k1_MAT_s014
                                 + 1.10124 w_k1_MSFT_s014
                                 + 1.00566 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.887017 w_k1_ADM_s014
                                 + 1.11794 w_k1_AXP_s014
                                 + 1.46745 w_k1_DIS_s014
                                 + 1.11038 w_k1_FDX_s014
                                 + 1.58356 w_k1_IBM_s014
                                 + 1.41915 w_k1_INTC_s014
                                 + 1.64638 w_k1_KO_s014 + 1.20515 w_k1_MAT_s014
                                 + 1.13974 w_k1_MSFT_s014
                                 + 1.36064 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.965159 w_k1_ADM_s014
                                 + 1.00651 w_k1_AXP_s014
                                 + 1.32176 w_k1_DIS_s014
                                 + 1.05396 w_k1_FDX_s014
                                 + 1.27203 w_k1_IBM_s014
                                 + 1.15013 w_k1_INTC_s014
                                 + 1.12749 w_k1_KO_s014 + 1.09402 w_k1_MAT_s014
                                 + 1.15124 w_k1_MSFT_s014
                                 + 1.20006 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.01544 w_k1_ADM_s015
                                 + 1.12838 w_k1_AXP_s015
                                 + 0.978692 w_k1_DIS_s015
                                 + 0.935068 w_k1_FDX_s015
                                 + 1.16643 w_k1_IBM_s015
                                 + 1.01063 w_k1_INTC_s015
                                 + 0.991274 w_k1_KO_s015
                                 + 1.47876 w_k1_MAT_s015
                                 + 0.93035 w_k1_MSFT_s015
                                 + 0.929314 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.85346 w_k1_ADM_s015
                                 + 0.853038 w_k1_AXP_s015
                                 + 0.726447 w_k1_DIS_s015
                                 + 0.793911 w_k1_FDX_s015 + 1.004 w_k1_IBM_s015
                                 + 0.774255 w_k1_INTC_s015
                                 + 0.708508 w_k1_KO_s015
                                 + 1.34394 w_k1_MAT_s015
                                 + 0.741569 w_k1_MSFT_s015
                                 + 0.8075 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.06072 w_k1_ADM_s015
                                 + 0.757565 w_k1_AXP_s015
                                 + 0.978227 w_k1_DIS_s015
                                 + 0.906276 w_k1_FDX_s015
                                 + 1.03065 w_k1_IBM_s015
                                 + 0.972601 w_k1_INTC_s015
                                 + 1.03322 w_k1_KO_s015
                                 + 0.914457 w_k1_MAT_s015
                                 + 0.888765 w_k1_MSFT_s015
                                 + 0.839705 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.0296 w_k1_ADM_s015
                                 + 1.03334 w_k1_AXP_s015 + 1.0919 w_k1_DIS_s015
                                 + 1.06762 w_k1_FDX_s015
                                 + 0.886162 w_k1_IBM_s015
                                 + 1.00262 w_k1_INTC_s015
                                 + 0.905077 w_k1_KO_s015
                                 + 1.13695 w_k1_MAT_s015
                                 + 1.1952 w_k1_MSFT_s015
                                 + 0.939821 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.60388 w_k1_ADM_s015
                                 + 1.53554 w_k1_AXP_s015
                                 + 1.40636 w_k1_DIS_s015
                                 + 1.27954 w_k1_FDX_s015
                                 + 1.34945 w_k1_IBM_s015
                                 + 1.14599 w_k1_INTC_s015
                                 + 1.15005 w_k1_KO_s015 + 1.55974 w_k1_MAT_s015
                                 + 1.15356 w_k1_MSFT_s015
                                 + 1.11588 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.31579 w_k1_ADM_s015
                                 + 0.992355 w_k1_AXP_s015
                                 + 1.42377 w_k1_DIS_s015
                                 + 1.20327 w_k1_FDX_s015
                                 + 1.45811 w_k1_IBM_s015
                                 + 1.05078 w_k1_INTC_s015
                                 + 1.20967 w_k1_KO_s015 + 1.33371 w_k1_MAT_s015
                                 + 1.15601 w_k1_MSFT_s015
                                 + 0.949413 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.884602 w_k1_ADM_s015
                                 + 0.672094 w_k1_AXP_s015
                                 + 0.618606 w_k1_DIS_s015
                                 + 0.766576 w_k1_FDX_s015
                                 + 0.68696 w_k1_IBM_s015
                                 + 1.08382 w_k1_INTC_s015
                                 + 0.575873 w_k1_KO_s015
                                 + 0.52681 w_k1_MAT_s015
                                 + 0.845499 w_k1_MSFT_s015
                                 + 0.765876 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.28608 w_k1_ADM_s015
                                 + 1.01066 w_k1_AXP_s015
                                 + 1.01062 w_k1_DIS_s015
                                 + 0.871818 w_k1_FDX_s015
                                 + 1.41562 w_k1_IBM_s015
                                 + 0.841928 w_k1_INTC_s015
                                 + 1.09054 w_k1_KO_s015 + 1.27031 w_k1_MAT_s015
                                 + 0.773108 w_k1_MSFT_s015
                                 + 0.985765 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.909625 w_k1_ADM_s016
                                 + 1.10614 w_k1_AXP_s016
                                 + 1.24381 w_k1_DIS_s016
                                 + 1.20991 w_k1_FDX_s016 + 1.2289 w_k1_IBM_s016
                                 + 0.927059 w_k1_INTC_s016
                                 + 1.25031 w_k1_KO_s016 + 1.41294 w_k1_MAT_s016
                                 + 1.08613 w_k1_MSFT_s016
                                 + 1.05538 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.01565 w_k1_ADM_s016
                                 + 1.48655 w_k1_AXP_s016
                                 + 1.76257 w_k1_DIS_s016
                                 + 1.33205 w_k1_FDX_s016
                                 + 1.51215 w_k1_IBM_s016
                                 + 1.43495 w_k1_INTC_s016
                                 + 1.63812 w_k1_KO_s016 + 2.24911 w_k1_MAT_s016
                                 + 1.35822 w_k1_MSFT_s016
                                 + 1.25365 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.570659 w_k1_ADM_s016
                                 + 0.725612 w_k1_AXP_s016
                                 + 0.696837 w_k1_DIS_s016
                                 + 0.795631 w_k1_FDX_s016
                                 + 1.06795 w_k1_IBM_s016
                                 + 0.859529 w_k1_INTC_s016
                                 + 0.974131 w_k1_KO_s016
                                 + 1.12999 w_k1_MAT_s016
                                 + 0.793126 w_k1_MSFT_s016
                                 + 0.841051 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.936931 w_k1_ADM_s016
                                 + 1.01802 w_k1_AXP_s016
                                 + 1.27045 w_k1_DIS_s016 + 1.153 w_k1_FDX_s016
                                 + 0.864922 w_k1_IBM_s016
                                 + 0.99913 w_k1_INTC_s016
                                 + 0.905757 w_k1_KO_s016
                                 + 0.877985 w_k1_MAT_s016
                                 + 1.12791 w_k1_MSFT_s016
                                 + 0.972322 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.800959 w_k1_ADM_s016
                                 + 0.859739 w_k1_AXP_s016
                                 + 0.863989 w_k1_DIS_s016
                                 + 0.869638 w_k1_FDX_s016
                                 + 1.34379 w_k1_IBM_s016
                                 + 0.78601 w_k1_INTC_s016
                                 + 0.984256 w_k1_KO_s016
                                 + 1.15007 w_k1_MAT_s016
                                 + 0.886746 w_k1_MSFT_s016
                                 + 1.00411 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.772634 w_k1_ADM_s016
                                 + 0.598737 w_k1_AXP_s016
                                 + 0.770425 w_k1_DIS_s016
                                 + 0.776978 w_k1_FDX_s016
                                 + 0.856496 w_k1_IBM_s016
                                 + 0.744001 w_k1_INTC_s016
                                 + 0.910455 w_k1_KO_s016
                                 + 0.536614 w_k1_MAT_s016
                                 + 0.897022 w_k1_MSFT_s016
                                 + 0.802459 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.932804 w_k1_ADM_s016
                                 + 0.813468 w_k1_AXP_s016
                                 + 1.00953 w_k1_DIS_s016
                                 + 1.12271 w_k1_FDX_s016
                                 + 0.605314 w_k1_IBM_s016
                                 + 1.15907 w_k1_INTC_s016
                                 + 0.791573 w_k1_KO_s016
                                 + 0.401039 w_k1_MAT_s016
                                 + 1.00092 w_k1_MSFT_s016
                                 + 0.935519 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.683203 w_k1_ADM_s016
                                 + 0.775674 w_k1_AXP_s016
                                 + 0.86122 w_k1_DIS_s016
                                 + 0.924199 w_k1_FDX_s016
                                 + 0.758763 w_k1_IBM_s016
                                 + 0.914548 w_k1_INTC_s016
                                 + 0.815234 w_k1_KO_s016
                                 + 0.692832 w_k1_MAT_s016
                                 + 0.899746 w_k1_MSFT_s016
                                 + 0.840702 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.869651 w_k1_ADM_s017
                                 + 0.983809 w_k1_AXP_s017
                                 + 0.783341 w_k1_DIS_s017
                                 + 0.900177 w_k1_FDX_s017
                                 + 0.943731 w_k1_IBM_s017
                                 + 1.08963 w_k1_INTC_s017
                                 + 0.813078 w_k1_KO_s017
                                 + 1.15349 w_k1_MAT_s017
                                 + 0.814777 w_k1_MSFT_s017
                                 + 1.00059 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.13962 w_k1_ADM_s017
                                 + 1.43868 w_k1_AXP_s017
                                 + 1.42548 w_k1_DIS_s017
                                 + 1.21183 w_k1_FDX_s017
                                 + 1.05785 w_k1_IBM_s017
                                 + 1.5376 w_k1_INTC_s017 + 1.05605 w_k1_KO_s017
                                 + 1.14379 w_k1_MAT_s017
                                 + 1.06801 w_k1_MSFT_s017 + 1.3033 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.29438 w_k1_ADM_s017
                                 + 1.36538 w_k1_AXP_s017
                                 + 1.25341 w_k1_DIS_s017
                                 + 1.17421 w_k1_FDX_s017
                                 + 1.19159 w_k1_IBM_s017
                                 + 1.30222 w_k1_INTC_s017
                                 + 1.11318 w_k1_KO_s017 + 1.12846 w_k1_MAT_s017
                                 + 0.923615 w_k1_MSFT_s017
                                 + 1.2396 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.507786 w_k1_ADM_s017
                                 + 0.736548 w_k1_AXP_s017
                                 + 0.810409 w_k1_DIS_s017
                                 + 0.752225 w_k1_FDX_s017
                                 + 1.29922 w_k1_IBM_s017
                                 + 1.00685 w_k1_INTC_s017
                                 + 0.952755 w_k1_KO_s017
                                 + 1.08804 w_k1_MAT_s017
                                 + 0.7883 w_k1_MSFT_s017
                                 + 0.949934 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.842198 w_k1_ADM_s017
                                 + 1.31781 w_k1_AXP_s017
                                 + 1.37128 w_k1_DIS_s017
                                 + 1.09767 w_k1_FDX_s017
                                 + 1.59551 w_k1_IBM_s017
                                 + 1.33898 w_k1_INTC_s017
                                 + 1.11271 w_k1_KO_s017 + 1.48691 w_k1_MAT_s017
                                 + 1.12997 w_k1_MSFT_s017
                                 + 1.14915 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.863803 w_k1_ADM_s017
                                 + 1.01661 w_k1_AXP_s017
                                 + 0.920761 w_k1_DIS_s017
                                 + 0.972892 w_k1_FDX_s017
                                 + 1.1706 w_k1_IBM_s017
                                 + 1.07302 w_k1_INTC_s017
                                 + 0.838779 w_k1_KO_s017
                                 + 1.14722 w_k1_MAT_s017
                                 + 1.01286 w_k1_MSFT_s017
                                 + 1.04746 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.27899 w_k1_ADM_s017
                                 + 1.0659 w_k1_AXP_s017 + 1.18675 w_k1_DIS_s017
                                 + 1.26108 w_k1_FDX_s017
                                 + 0.734949 w_k1_IBM_s017
                                 + 1.43345 w_k1_INTC_s017
                                 + 0.934454 w_k1_KO_s017
                                 + 0.524518 w_k1_MAT_s017
                                 + 1.14981 w_k1_MSFT_s017
                                 + 0.975556 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.04379 w_k1_ADM_s017
                                 + 0.740259 w_k1_AXP_s017
                                 + 0.823337 w_k1_DIS_s017
                                 + 0.844188 w_k1_FDX_s017
                                 + 0.79627 w_k1_IBM_s017
                                 + 0.942603 w_k1_INTC_s017
                                 + 0.813688 w_k1_KO_s017
                                 + 0.541087 w_k1_MAT_s017
                                 + 0.779386 w_k1_MSFT_s017
                                 + 0.946012 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.31797 w_k1_ADM_s100
                                 + 1.20738 w_k1_AXP_s100
                                 + 1.43015 w_k1_DIS_s100
                                 + 1.27233 w_k1_FDX_s100
                                 + 1.15101 w_k1_IBM_s100
                                 + 1.02654 w_k1_INTC_s100
                                 + 1.05449 w_k1_KO_s100 + 1.28648 w_k1_MAT_s100
                                 + 1.14778 w_k1_MSFT_s100
                                 + 1.12371 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.649731 w_k1_ADM_s100
                                 + 0.518098 w_k1_AXP_s100
                                 + 0.410254 w_k1_DIS_s100
                                 + 0.669966 w_k1_FDX_s100
                                 + 0.699744 w_k1_IBM_s100
                                 + 0.660731 w_k1_INTC_s100
                                 + 0.537178 w_k1_KO_s100
                                 + 0.714358 w_k1_MAT_s100
                                 + 0.645082 w_k1_MSFT_s100
                                 + 0.716335 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.1912 w_k1_ADM_s100
                                 + 1.08552 w_k1_AXP_s100
                                 + 0.983834 w_k1_DIS_s100
                                 + 1.13427 w_k1_FDX_s100
                                 + 0.878368 w_k1_IBM_s100
                                 + 1.013 w_k1_INTC_s100 + 0.88349 w_k1_KO_s100
                                 + 0.786885 w_k1_MAT_s100
                                 + 1.05314 w_k1_MSFT_s100
                                 + 0.942737 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.887864 w_k1_ADM_s100
                                 + 0.928197 w_k1_AXP_s100
                                 + 0.922278 w_k1_DIS_s100
                                 + 0.974001 w_k1_FDX_s100
                                 + 1.20478 w_k1_IBM_s100
                                 + 0.909716 w_k1_INTC_s100
                                 + 0.973691 w_k1_KO_s100
                                 + 1.27802 w_k1_MAT_s100
                                 + 0.901492 w_k1_MSFT_s100
                                 + 0.930453 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.19238 w_k1_ADM_s100
                                 + 0.918497 w_k1_AXP_s100
                                 + 1.07587 w_k1_DIS_s100
                                 + 1.23945 w_k1_FDX_s100
                                 + 0.73417 w_k1_IBM_s100
                                 + 1.20142 w_k1_INTC_s100
                                 + 0.861819 w_k1_KO_s100
                                 + 0.809929 w_k1_MAT_s100
                                 + 1.24744 w_k1_MSFT_s100
                                 + 0.968676 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.853246 w_k1_ADM_s100
                                 + 0.718294 w_k1_AXP_s100
                                 + 0.644971 w_k1_DIS_s100
                                 + 0.932518 w_k1_FDX_s100
                                 + 0.686465 w_k1_IBM_s100
                                 + 0.767022 w_k1_INTC_s100
                                 + 0.643683 w_k1_KO_s100
                                 + 0.752029 w_k1_MAT_s100
                                 + 0.764161 w_k1_MSFT_s100
                                 + 0.809555 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.836551 w_k1_ADM_s100
                                 + 0.80907 w_k1_AXP_s100
                                 + 0.79229 w_k1_DIS_s100
                                 + 0.953382 w_k1_FDX_s100
                                 + 0.875926 w_k1_IBM_s100
                                 + 0.873756 w_k1_INTC_s100
                                 + 0.908583 w_k1_KO_s100
                                 + 0.971019 w_k1_MAT_s100
                                 + 0.840925 w_k1_MSFT_s100
                                 + 0.852962 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.899483 w_k1_ADM_s100
                                 + 1.42402 w_k1_AXP_s100
                                 + 1.28894 w_k1_DIS_s100
                                 + 1.13612 w_k1_FDX_s100
                                 + 1.69896 w_k1_IBM_s100
                                 + 1.27338 w_k1_INTC_s100 + 1.2479 w_k1_KO_s100
                                 + 1.74289 w_k1_MAT_s100
                                 + 1.05242 w_k1_MSFT_s100
                                 + 1.07697 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.932412 w_k1_ADM_s101
                                 + 0.947043 w_k1_AXP_s101
                                 + 1.11863 w_k1_DIS_s101
                                 + 0.925855 w_k1_FDX_s101
                                 + 1.45052 w_k1_IBM_s101
                                 + 0.950165 w_k1_INTC_s101
                                 + 1.2852 w_k1_KO_s101 + 1.68304 w_k1_MAT_s101
                                 + 1.07101 w_k1_MSFT_s101
                                 + 1.09424 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.897738 w_k1_ADM_s101
                                 + 0.735383 w_k1_AXP_s101
                                 + 1.05181 w_k1_DIS_s101
                                 + 0.913227 w_k1_FDX_s101
                                 + 0.870691 w_k1_IBM_s101
                                 + 0.88577 w_k1_INTC_s101
                                 + 0.868746 w_k1_KO_s101
                                 + 0.678942 w_k1_MAT_s101
                                 + 0.95868 w_k1_MSFT_s101
                                 + 0.929716 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.46736 w_k1_ADM_s101
                                 + 1.33857 w_k1_AXP_s101
                                 + 2.21459 w_k1_DIS_s101
                                 + 1.54496 w_k1_FDX_s101
                                 + 1.41941 w_k1_IBM_s101
                                 + 1.12792 w_k1_INTC_s101
                                 + 1.43899 w_k1_KO_s101 + 1.23529 w_k1_MAT_s101
                                 + 1.54365 w_k1_MSFT_s101
                                 + 1.16323 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.852543 w_k1_ADM_s101
                                 + 0.678979 w_k1_AXP_s101
                                 + 1.13901 w_k1_DIS_s101
                                 + 1.05402 w_k1_FDX_s101
                                 + 0.565057 w_k1_IBM_s101
                                 + 0.960156 w_k1_INTC_s101
                                 + 0.788555 w_k1_KO_s101
                                 + 0.433195 w_k1_MAT_s101
                                 + 1.09624 w_k1_MSFT_s101
                                 + 0.911418 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.854516 w_k1_ADM_s101
                                 + 1.12453 w_k1_AXP_s101
                                 + 1.30607 w_k1_DIS_s101
                                 + 0.859413 w_k1_FDX_s101
                                 + 1.85892 w_k1_IBM_s101
                                 + 0.856098 w_k1_INTC_s101
                                 + 1.30918 w_k1_KO_s101 + 1.73198 w_k1_MAT_s101
                                 + 1.03298 w_k1_MSFT_s101
                                 + 1.00151 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.706838 w_k1_ADM_s101
                                 + 0.68075 w_k1_AXP_s101
                                 + 0.714391 w_k1_DIS_s101
                                 + 0.808997 w_k1_FDX_s101
                                 + 0.70027 w_k1_IBM_s101
                                 + 0.940526 w_k1_INTC_s101
                                 + 0.663037 w_k1_KO_s101
                                 + 0.507849 w_k1_MAT_s101
                                 + 0.885058 w_k1_MSFT_s101
                                 + 0.772458 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.16125 w_k1_ADM_s101
                                 + 1.2204 w_k1_AXP_s101 + 1.13856 w_k1_DIS_s101
                                 + 1.14143 w_k1_FDX_s101
                                 + 0.960864 w_k1_IBM_s101
                                 + 1.03724 w_k1_INTC_s101
                                 + 1.08848 w_k1_KO_s101 + 1.20095 w_k1_MAT_s101
                                 + 1.03135 w_k1_MSFT_s101
                                 + 1.18817 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.966359 w_k1_ADM_s101
                                 + 0.860938 w_k1_AXP_s101
                                 + 1.17407 w_k1_DIS_s101
                                 + 0.982242 w_k1_FDX_s101
                                 + 1.20039 w_k1_IBM_s101
                                 + 0.815734 w_k1_INTC_s101
                                 + 1.03886 w_k1_KO_s101 + 1.02699 w_k1_MAT_s101
                                 + 0.952066 w_k1_MSFT_s101
                                 + 0.853186 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.32567 w_k1_ADM_s102
                                 + 1.51274 w_k1_AXP_s102 + 1.7014 w_k1_DIS_s102
                                 + 1.23054 w_k1_FDX_s102 + 1.435 w_k1_IBM_s102
                                 + 1.36348 w_k1_INTC_s102
                                 + 1.42665 w_k1_KO_s102
                                 + 0.962758 w_k1_MAT_s102
                                 + 1.4186 w_k1_MSFT_s102 + 1.37358 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.978574 w_k1_ADM_s102
                                 + 0.830105 w_k1_AXP_s102
                                 + 0.703183 w_k1_DIS_s102
                                 + 0.814547 w_k1_FDX_s102
                                 + 0.775127 w_k1_IBM_s102
                                 + 0.744644 w_k1_INTC_s102
                                 + 0.758554 w_k1_KO_s102
                                 + 0.697229 w_k1_MAT_s102
                                 + 0.818078 w_k1_MSFT_s102
                                 + 0.900418 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.848504 w_k1_ADM_s102
                                 + 0.824098 w_k1_AXP_s102
                                 + 0.74119 w_k1_DIS_s102
                                 + 0.83733 w_k1_FDX_s102
                                 + 0.992037 w_k1_IBM_s102
                                 + 1.00043 w_k1_INTC_s102
                                 + 0.852907 w_k1_KO_s102
                                 + 1.02629 w_k1_MAT_s102
                                 + 0.982458 w_k1_MSFT_s102
                                 + 0.93351 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.28804 w_k1_ADM_s102
                                 + 1.39548 w_k1_AXP_s102
                                 + 1.11894 w_k1_DIS_s102
                                 + 1.15158 w_k1_FDX_s102
                                 + 1.30458 w_k1_IBM_s102
                                 + 0.982418 w_k1_INTC_s102
                                 + 1.10312 w_k1_KO_s102 + 1.17605 w_k1_MAT_s102
                                 + 1.12123 w_k1_MSFT_s102
                                 + 1.21341 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.07426 w_k1_ADM_s102
                                 + 0.846684 w_k1_AXP_s102
                                 + 0.831539 w_k1_DIS_s102
                                 + 0.992058 w_k1_FDX_s102
                                 + 0.691666 w_k1_IBM_s102
                                 + 0.896659 w_k1_INTC_s102
                                 + 0.823223 w_k1_KO_s102
                                 + 0.484741 w_k1_MAT_s102
                                 + 0.925652 w_k1_MSFT_s102
                                 + 0.985779 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.30572 w_k1_ADM_s102
                                 + 1.18004 w_k1_AXP_s102
                                 + 1.40284 w_k1_DIS_s102
                                 + 1.20962 w_k1_FDX_s102
                                 + 1.02312 w_k1_IBM_s102
                                 + 1.19728 w_k1_INTC_s102
                                 + 1.26481 w_k1_KO_s102
                                 + 0.611194 w_k1_MAT_s102
                                 + 1.22675 w_k1_MSFT_s102
                                 + 1.13126 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.18814 w_k1_ADM_s102
                                 + 1.06398 w_k1_AXP_s102
                                 + 0.931281 w_k1_DIS_s102
                                 + 0.887574 w_k1_FDX_s102
                                 + 1.37096 w_k1_IBM_s102
                                 + 0.839887 w_k1_INTC_s102
                                 + 1.09574 w_k1_KO_s102 + 1.25894 w_k1_MAT_s102
                                 + 0.952667 w_k1_MSFT_s102
                                 + 1.12902 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.22452 w_k1_ADM_s102
                                 + 1.32259 w_k1_AXP_s102
                                 + 1.01532 w_k1_DIS_s102
                                 + 1.13331 w_k1_FDX_s102
                                 + 0.981061 w_k1_IBM_s102
                                 + 1.08696 w_k1_INTC_s102
                                 + 0.881391 w_k1_KO_s102
                                 + 0.920423 w_k1_MAT_s102
                                 + 1.03946 w_k1_MSFT_s102
                                 + 1.05351 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.12426 w_k1_ADM_s103
                                 + 1.12237 w_k1_AXP_s103
                                 + 0.925254 w_k1_DIS_s103
                                 + 1.1707 w_k1_FDX_s103
                                 + 0.732949 w_k1_IBM_s103
                                 + 1.09586 w_k1_INTC_s103
                                 + 0.819927 w_k1_KO_s103
                                 + 0.936813 w_k1_MAT_s103
                                 + 0.983183 w_k1_MSFT_s103
                                 + 0.905333 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.00929 w_k1_ADM_s103
                                 + 0.886795 w_k1_AXP_s103
                                 + 0.894544 w_k1_DIS_s103
                                 + 0.964161 w_k1_FDX_s103
                                 + 1.07856 w_k1_IBM_s103
                                 + 1.01762 w_k1_INTC_s103
                                 + 1.07417 w_k1_KO_s103 + 1.15446 w_k1_MAT_s103
                                 + 0.955619 w_k1_MSFT_s103
                                 + 0.949739 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.91512 w_k1_ADM_s103
                                 + 0.68859 w_k1_AXP_s103
                                 + 0.830686 w_k1_DIS_s103
                                 + 0.920447 w_k1_FDX_s103
                                 + 0.897391 w_k1_IBM_s103
                                 + 0.976599 w_k1_INTC_s103
                                 + 0.948875 w_k1_KO_s103
                                 + 0.870771 w_k1_MAT_s103
                                 + 0.960304 w_k1_MSFT_s103
                                 + 0.899945 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.798087 w_k1_ADM_s103
                                 + 0.642386 w_k1_AXP_s103
                                 + 0.652132 w_k1_DIS_s103
                                 + 0.753371 w_k1_FDX_s103
                                 + 0.642688 w_k1_IBM_s103
                                 + 0.959305 w_k1_INTC_s103
                                 + 0.859122 w_k1_KO_s103
                                 + 0.877117 w_k1_MAT_s103
                                 + 0.845901 w_k1_MSFT_s103
                                 + 0.797051 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.44217 w_k1_ADM_s103
                                 + 1.53141 w_k1_AXP_s103
                                 + 1.44126 w_k1_DIS_s103
                                 + 1.39411 w_k1_FDX_s103
                                 + 1.08298 w_k1_IBM_s103
                                 + 1.50121 w_k1_INTC_s103
                                 + 1.20922 w_k1_KO_s103 + 1.16044 w_k1_MAT_s103
                                 + 1.12014 w_k1_MSFT_s103
                                 + 1.21776 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.13075 w_k1_ADM_s103
                                 + 1.32919 w_k1_AXP_s103
                                 + 1.31446 w_k1_DIS_s103
                                 + 1.15641 w_k1_FDX_s103
                                 + 1.52909 w_k1_IBM_s103
                                 + 1.38635 w_k1_INTC_s103
                                 + 1.42911 w_k1_KO_s103 + 1.94159 w_k1_MAT_s103
                                 + 1.10984 w_k1_MSFT_s103
                                 + 1.18145 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.11707 w_k1_ADM_s103
                                 + 0.887031 w_k1_AXP_s103
                                 + 0.81061 w_k1_DIS_s103
                                 + 0.803037 w_k1_FDX_s103
                                 + 1.39756 w_k1_IBM_s103
                                 + 0.978552 w_k1_INTC_s103
                                 + 1.06424 w_k1_KO_s103 + 1.48377 w_k1_MAT_s103
                                 + 0.844572 w_k1_MSFT_s103
                                 + 0.922375 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.19359 w_k1_ADM_s103
                                 + 0.950283 w_k1_AXP_s103
                                 + 1.10558 w_k1_DIS_s103
                                 + 1.02668 w_k1_FDX_s103
                                 + 1.40436 w_k1_IBM_s103
                                 + 1.23792 w_k1_INTC_s103
                                 + 1.30249 w_k1_KO_s103 + 0.98243 w_k1_MAT_s103
                                 + 1.01194 w_k1_MSFT_s103
                                 + 1.08231 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.05949 w_k1_ADM_s104
                                 + 1.04946 w_k1_AXP_s104
                                 + 1.05829 w_k1_DIS_s104
                                 + 1.24517 w_k1_FDX_s104
                                 + 0.672944 w_k1_IBM_s104
                                 + 1.21195 w_k1_INTC_s104
                                 + 1.01269 w_k1_KO_s104
                                 + 0.639046 w_k1_MAT_s104
                                 + 1.13002 w_k1_MSFT_s104
                                 + 0.966598 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.720799 w_k1_ADM_s104
                                 + 0.626015 w_k1_AXP_s104
                                 + 0.599082 w_k1_DIS_s104
                                 + 0.724585 w_k1_FDX_s104
                                 + 0.90223 w_k1_IBM_s104
                                 + 0.959751 w_k1_INTC_s104
                                 + 0.885939 w_k1_KO_s104
                                 + 0.570735 w_k1_MAT_s104
                                 + 0.807209 w_k1_MSFT_s104
                                 + 0.744325 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.643918 w_k1_ADM_s104
                                 + 0.913678 w_k1_AXP_s104
                                 + 0.798901 w_k1_DIS_s104
                                 + 0.851455 w_k1_FDX_s104
                                 + 0.997997 w_k1_IBM_s104
                                 + 0.924294 w_k1_INTC_s104
                                 + 0.863057 w_k1_KO_s104
                                 + 1.03495 w_k1_MAT_s104
                                 + 0.918373 w_k1_MSFT_s104
                                 + 0.849946 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.07507 w_k1_ADM_s104
                                 + 1.0629 w_k1_AXP_s104 + 1.14159 w_k1_DIS_s104
                                 + 1.0335 w_k1_FDX_s104 + 1.65279 w_k1_IBM_s104
                                 + 1.12368 w_k1_INTC_s104
                                 + 1.32818 w_k1_KO_s104 + 1.2019 w_k1_MAT_s104
                                 + 1.07861 w_k1_MSFT_s104 + 1.0386 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.16481 w_k1_ADM_s104
                                 + 1.43811 w_k1_AXP_s104
                                 + 1.26195 w_k1_DIS_s104
                                 + 1.32181 w_k1_FDX_s104
                                 + 1.10195 w_k1_IBM_s104
                                 + 1.28269 w_k1_INTC_s104
                                 + 1.18497 w_k1_KO_s104
                                 + 0.961688 w_k1_MAT_s104
                                 + 1.24771 w_k1_MSFT_s104
                                 + 1.11712 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.942102 w_k1_ADM_s104
                                 + 0.894504 w_k1_AXP_s104 + 1.31 w_k1_DIS_s104
                                 + 1.045 w_k1_FDX_s104 + 1.15961 w_k1_IBM_s104
                                 + 1.19511 w_k1_INTC_s104
                                 + 1.30418 w_k1_KO_s104
                                 + 0.877394 w_k1_MAT_s104
                                 + 1.17812 w_k1_MSFT_s104
                                 + 0.996823 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.790616 w_k1_ADM_s104
                                 + 0.716861 w_k1_AXP_s104
                                 + 0.992508 w_k1_DIS_s104
                                 + 0.837177 w_k1_FDX_s104
                                 + 1.19658 w_k1_IBM_s104
                                 + 0.916836 w_k1_INTC_s104
                                 + 1.07545 w_k1_KO_s104 + 0.76874 w_k1_MAT_s104
                                 + 0.902798 w_k1_MSFT_s104
                                 + 0.856265 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.854429 w_k1_ADM_s104
                                 + 0.696982 w_k1_AXP_s104
                                 + 0.834236 w_k1_DIS_s104
                                 + 0.953002 w_k1_FDX_s104
                                 + 0.77551 w_k1_IBM_s104
                                 + 0.891535 w_k1_INTC_s104
                                 + 0.97559 w_k1_KO_s104 + 0.62909 w_k1_MAT_s104
                                 + 0.990271 w_k1_MSFT_s104
                                 + 0.810659 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.08989 w_k1_ADM_s105
                                 + 1.18216 w_k1_AXP_s105
                                 + 1.41508 w_k1_DIS_s105 + 1.1697 w_k1_FDX_s105
                                 + 1.39589 w_k1_IBM_s105
                                 + 1.1381 w_k1_INTC_s105 + 1.1746 w_k1_KO_s105
                                 + 1.37503 w_k1_MAT_s105
                                 + 1.05204 w_k1_MSFT_s105
                                 + 1.28912 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.97024 w_k1_ADM_s105
                                 + 1.1352 w_k1_AXP_s105
                                 + 0.977145 w_k1_DIS_s105
                                 + 1.07853 w_k1_FDX_s105
                                 + 0.70472 w_k1_IBM_s105
                                 + 1.14026 w_k1_INTC_s105
                                 + 0.916615 w_k1_KO_s105
                                 + 0.965161 w_k1_MAT_s105
                                 + 1.03434 w_k1_MSFT_s105
                                 + 1.06274 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.990017 w_k1_ADM_s105
                                 + 1.119 w_k1_AXP_s105 + 1.16178 w_k1_DIS_s105
                                 + 0.999837 w_k1_FDX_s105
                                 + 1.12441 w_k1_IBM_s105
                                 + 1.16637 w_k1_INTC_s105
                                 + 1.04896 w_k1_KO_s105 + 1.20474 w_k1_MAT_s105
                                 + 0.988037 w_k1_MSFT_s105
                                 + 0.98566 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.792381 w_k1_ADM_s105
                                 + 1.18903 w_k1_AXP_s105
                                 + 1.19656 w_k1_DIS_s105
                                 + 0.96978 w_k1_FDX_s105
                                 + 1.66586 w_k1_IBM_s105
                                 + 1.15125 w_k1_INTC_s105
                                 + 1.09389 w_k1_KO_s105 + 1.75426 w_k1_MAT_s105
                                 + 0.962606 w_k1_MSFT_s105
                                 + 0.971241 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.17377 w_k1_ADM_s105
                                 + 1.75429 w_k1_AXP_s105
                                 + 1.80794 w_k1_DIS_s105
                                 + 1.36756 w_k1_FDX_s105
                                 + 1.35303 w_k1_IBM_s105
                                 + 1.43096 w_k1_INTC_s105
                                 + 1.34789 w_k1_KO_s105 + 1.70462 w_k1_MAT_s105
                                 + 1.17369 w_k1_MSFT_s105
                                 + 1.17503 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.723886 w_k1_ADM_s105
                                 + 0.681164 w_k1_AXP_s105
                                 + 0.678939 w_k1_DIS_s105
                                 + 0.721945 w_k1_FDX_s105
                                 + 0.664911 w_k1_IBM_s105
                                 + 0.703187 w_k1_INTC_s105
                                 + 0.641015 w_k1_KO_s105
                                 + 0.581812 w_k1_MAT_s105
                                 + 0.716417 w_k1_MSFT_s105
                                 + 0.787178 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.908486 w_k1_ADM_s105
                                 + 1.03273 w_k1_AXP_s105
                                 + 1.39837 w_k1_DIS_s105
                                 + 1.22537 w_k1_FDX_s105
                                 + 0.851885 w_k1_IBM_s105
                                 + 1.19491 w_k1_INTC_s105
                                 + 1.10011 w_k1_KO_s105
                                 + 0.753775 w_k1_MAT_s105
                                 + 1.12937 w_k1_MSFT_s105
                                 + 0.926183 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.885562 w_k1_ADM_s105
                                 + 0.965083 w_k1_AXP_s105
                                 + 0.906728 w_k1_DIS_s105
                                 + 0.880587 w_k1_FDX_s105
                                 + 0.940966 w_k1_IBM_s105
                                 + 0.984559 w_k1_INTC_s105
                                 + 0.918271 w_k1_KO_s105
                                 + 0.795156 w_k1_MAT_s105
                                 + 0.821018 w_k1_MSFT_s105
                                 + 0.896447 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.02598 w_k1_ADM_s106
                                 + 1.27366 w_k1_AXP_s106
                                 + 0.962458 w_k1_DIS_s106
                                 + 1.13929 w_k1_FDX_s106
                                 + 1.14354 w_k1_IBM_s106
                                 + 1.02817 w_k1_INTC_s106
                                 + 1.06875 w_k1_KO_s106 + 1.53098 w_k1_MAT_s106
                                 + 1.10277 w_k1_MSFT_s106
                                 + 1.06305 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.948403 w_k1_ADM_s106
                                 + 0.785573 w_k1_AXP_s106
                                 + 0.749221 w_k1_DIS_s106
                                 + 0.869368 w_k1_FDX_s106
                                 + 0.879088 w_k1_IBM_s106
                                 + 0.844541 w_k1_INTC_s106
                                 + 0.856989 w_k1_KO_s106
                                 + 0.693048 w_k1_MAT_s106
                                 + 0.791822 w_k1_MSFT_s106
                                 + 0.890304 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.04877 w_k1_ADM_s106
                                 + 1.18844 w_k1_AXP_s106
                                 + 0.97588 w_k1_DIS_s106
                                 + 1.04453 w_k1_FDX_s106
                                 + 1.58276 w_k1_IBM_s106
                                 + 0.984801 w_k1_INTC_s106
                                 + 1.39014 w_k1_KO_s106 + 1.61061 w_k1_MAT_s106
                                 + 0.998595 w_k1_MSFT_s106
                                 + 1.16219 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.972549 w_k1_ADM_s106
                                 + 0.905304 w_k1_AXP_s106
                                 + 1.26003 w_k1_DIS_s106
                                 + 1.08285 w_k1_FDX_s106
                                 + 1.13719 w_k1_IBM_s106
                                 + 1.10553 w_k1_INTC_s106
                                 + 1.15251 w_k1_KO_s106
                                 + 0.913488 w_k1_MAT_s106
                                 + 1.1283 w_k1_MSFT_s106 + 1.111 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.19958 w_k1_ADM_s106
                                 + 0.90218 w_k1_AXP_s106
                                 + 0.991959 w_k1_DIS_s106
                                 + 1.07353 w_k1_FDX_s106
                                 + 0.781037 w_k1_IBM_s106
                                 + 1.1574 w_k1_INTC_s106
                                 + 0.873421 w_k1_KO_s106
                                 + 0.65644 w_k1_MAT_s106
                                 + 1.0729 w_k1_MSFT_s106
                                 + 0.941396 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.29595 w_k1_ADM_s106
                                 + 1.26135 w_k1_AXP_s106
                                 + 1.44542 w_k1_DIS_s106
                                 + 1.31093 w_k1_FDX_s106
                                 + 0.902962 w_k1_IBM_s106
                                 + 1.39668 w_k1_INTC_s106
                                 + 1.17155 w_k1_KO_s106
                                 + 0.876074 w_k1_MAT_s106
                                 + 1.2261 w_k1_MSFT_s106 + 1.10344 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.987847 w_k1_ADM_s106
                                 + 1.11411 w_k1_AXP_s106
                                 + 0.863384 w_k1_DIS_s106
                                 + 1.06659 w_k1_FDX_s106
                                 + 0.787469 w_k1_IBM_s106
                                 + 1.10376 w_k1_INTC_s106
                                 + 0.86505 w_k1_KO_s106
                                 + 0.868892 w_k1_MAT_s106
                                 + 0.996189 w_k1_MSFT_s106
                                 + 0.965913 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.05947 w_k1_ADM_s106
                                 + 1.47751 w_k1_AXP_s106 + 1.6851 w_k1_DIS_s106
                                 + 1.36312 w_k1_FDX_s106
                                 + 1.50976 w_k1_IBM_s106
                                 + 1.15744 w_k1_INTC_s106 + 1.5198 w_k1_KO_s106
                                 + 1.81585 w_k1_MAT_s106
                                 + 1.22537 w_k1_MSFT_s106 + 1.2052 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.996748 w_k1_ADM_s107
                                 + 0.911318 w_k1_AXP_s107
                                 + 1.03512 w_k1_DIS_s107
                                 + 0.986515 w_k1_FDX_s107
                                 + 0.71467 w_k1_IBM_s107
                                 + 0.959833 w_k1_INTC_s107
                                 + 0.83815 w_k1_KO_s107
                                 + 0.629947 w_k1_MAT_s107
                                 + 0.889543 w_k1_MSFT_s107
                                 + 0.924599 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.66782 w_k1_ADM_s107
                                 + 0.658465 w_k1_AXP_s107
                                 + 0.643913 w_k1_DIS_s107
                                 + 0.655136 w_k1_FDX_s107
                                 + 0.799938 w_k1_IBM_s107
                                 + 0.780274 w_k1_INTC_s107
                                 + 0.807223 w_k1_KO_s107
                                 + 0.656098 w_k1_MAT_s107
                                 + 0.798272 w_k1_MSFT_s107
                                 + 0.823162 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.36445 w_k1_ADM_s107
                                 + 1.56622 w_k1_AXP_s107
                                 + 1.66429 w_k1_DIS_s107 + 1.3327 w_k1_FDX_s107
                                 + 1.31297 w_k1_IBM_s107
                                 + 1.21016 w_k1_INTC_s107
                                 + 1.34691 w_k1_KO_s107 + 1.42824 w_k1_MAT_s107
                                 + 1.30001 w_k1_MSFT_s107
                                 + 1.24643 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.767165 w_k1_ADM_s107
                                 + 0.932791 w_k1_AXP_s107
                                 + 0.940416 w_k1_DIS_s107
                                 + 0.829779 w_k1_FDX_s107
                                 + 1.27553 w_k1_IBM_s107
                                 + 0.874872 w_k1_INTC_s107
                                 + 1.03785 w_k1_KO_s107 + 1.07153 w_k1_MAT_s107
                                 + 0.867062 w_k1_MSFT_s107
                                 + 0.934214 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.958697 w_k1_ADM_s107
                                 + 0.83528 w_k1_AXP_s107
                                 + 1.01527 w_k1_DIS_s107
                                 + 0.87245 w_k1_FDX_s107
                                 + 1.02125 w_k1_IBM_s107
                                 + 1.12752 w_k1_INTC_s107
                                 + 1.04939 w_k1_KO_s107
                                 + 0.832527 w_k1_MAT_s107
                                 + 0.975476 w_k1_MSFT_s107
                                 + 0.927595 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.11413 w_k1_ADM_s107
                                 + 1.04638 w_k1_AXP_s107
                                 + 1.13917 w_k1_DIS_s107
                                 + 0.949169 w_k1_FDX_s107
                                 + 1.23767 w_k1_IBM_s107
                                 + 1.10714 w_k1_INTC_s107
                                 + 1.19359 w_k1_KO_s107 + 1.00738 w_k1_MAT_s107
                                 + 1.03053 w_k1_MSFT_s107 + 1.0376 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.969741 w_k1_ADM_s107
                                 + 1.14931 w_k1_AXP_s107
                                 + 1.17715 w_k1_DIS_s107
                                 + 1.06085 w_k1_FDX_s107
                                 + 0.967146 w_k1_IBM_s107
                                 + 1.20145 w_k1_INTC_s107
                                 + 1.00188 w_k1_KO_s107 + 1.12069 w_k1_MAT_s107
                                 + 1.06606 w_k1_MSFT_s107
                                 + 1.07595 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.1137 w_k1_ADM_s107
                                 + 1.46354 w_k1_AXP_s107
                                 + 1.29189 w_k1_DIS_s107
                                 + 1.13492 w_k1_FDX_s107 + 1.1599 w_k1_IBM_s107
                                 + 0.997179 w_k1_INTC_s107
                                 + 1.07912 w_k1_KO_s107 + 1.6704 w_k1_MAT_s107
                                 + 1.01106 w_k1_MSFT_s107 + 1.0982 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.784402 w_k1_ADM_s110
                                 + 0.633046 w_k1_AXP_s110
                                 + 0.556594 w_k1_DIS_s110
                                 + 0.705954 w_k1_FDX_s110
                                 + 0.789968 w_k1_IBM_s110
                                 + 0.699575 w_k1_INTC_s110
                                 + 0.661662 w_k1_KO_s110
                                 + 0.574236 w_k1_MAT_s110
                                 + 0.696554 w_k1_MSFT_s110
                                 + 0.729103 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.967692 w_k1_ADM_s110
                                 + 1.14442 w_k1_AXP_s110
                                 + 0.77188 w_k1_DIS_s110
                                 + 0.922721 w_k1_FDX_s110
                                 + 0.839999 w_k1_IBM_s110
                                 + 0.813363 w_k1_INTC_s110
                                 + 0.951877 w_k1_KO_s110
                                 + 1.20863 w_k1_MAT_s110
                                 + 0.75392 w_k1_MSFT_s110
                                 + 0.965588 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.949806 w_k1_ADM_s110
                                 + 0.939055 w_k1_AXP_s110
                                 + 1.07648 w_k1_DIS_s110
                                 + 0.939101 w_k1_FDX_s110
                                 + 1.28194 w_k1_IBM_s110
                                 + 0.96296 w_k1_INTC_s110
                                 + 1.16903 w_k1_KO_s110 + 1.60231 w_k1_MAT_s110
                                 + 1.06306 w_k1_MSFT_s110
                                 + 1.03744 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.24486 w_k1_ADM_s110
                                 + 1.36194 w_k1_AXP_s110
                                 + 1.22114 w_k1_DIS_s110
                                 + 1.21284 w_k1_FDX_s110
                                 + 1.09742 w_k1_IBM_s110
                                 + 0.985879 w_k1_INTC_s110
                                 + 1.03238 w_k1_KO_s110 + 1.25689 w_k1_MAT_s110
                                 + 1.16156 w_k1_MSFT_s110
                                 + 1.08155 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.20779 w_k1_ADM_s110
                                 + 0.990013 w_k1_AXP_s110
                                 + 0.854363 w_k1_DIS_s110
                                 + 1.01329 w_k1_FDX_s110
                                 + 0.861174 w_k1_IBM_s110
                                 + 0.852769 w_k1_INTC_s110
                                 + 0.952566 w_k1_KO_s110
                                 + 0.925067 w_k1_MAT_s110
                                 + 0.88599 w_k1_MSFT_s110
                                 + 1.03401 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.31495 w_k1_ADM_s110
                                 + 1.14316 w_k1_AXP_s110
                                 + 1.35267 w_k1_DIS_s110
                                 + 1.28792 w_k1_FDX_s110
                                 + 1.04756 w_k1_IBM_s110
                                 + 1.09828 w_k1_INTC_s110
                                 + 1.13047 w_k1_KO_s110 + 1.06486 w_k1_MAT_s110
                                 + 1.19375 w_k1_MSFT_s110
                                 + 1.11259 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.780283 w_k1_ADM_s110
                                 + 0.837955 w_k1_AXP_s110
                                 + 0.819271 w_k1_DIS_s110
                                 + 0.915893 w_k1_FDX_s110
                                 + 0.773744 w_k1_IBM_s110
                                 + 0.859939 w_k1_INTC_s110
                                 + 0.774713 w_k1_KO_s110
                                 + 0.926352 w_k1_MAT_s110
                                 + 0.93024 w_k1_MSFT_s110
                                 + 0.95828 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.01941 w_k1_ADM_s110
                                 + 0.797972 w_k1_AXP_s110
                                 + 1.19026 w_k1_DIS_s110
                                 + 1.03292 w_k1_FDX_s110
                                 + 0.74598 w_k1_IBM_s110
                                 + 0.931639 w_k1_INTC_s110
                                 + 0.95528 w_k1_KO_s110
                                 + 0.829826 w_k1_MAT_s110
                                 + 1.10796 w_k1_MSFT_s110
                                 + 0.904093 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.762426 w_k1_ADM_s111
                                 + 0.848559 w_k1_AXP_s111
                                 + 0.827251 w_k1_DIS_s111
                                 + 0.818615 w_k1_FDX_s111
                                 + 1.18573 w_k1_IBM_s111
                                 + 0.97379 w_k1_INTC_s111
                                 + 0.997268 w_k1_KO_s111
                                 + 0.984268 w_k1_MAT_s111
                                 + 0.920038 w_k1_MSFT_s111
                                 + 0.989033 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.08251 w_k1_ADM_s111
                                 + 0.99337 w_k1_AXP_s111
                                 + 0.982439 w_k1_DIS_s111
                                 + 1.08124 w_k1_FDX_s111
                                 + 1.10255 w_k1_IBM_s111
                                 + 1.13588 w_k1_INTC_s111
                                 + 0.968349 w_k1_KO_s111
                                 + 0.989337 w_k1_MAT_s111
                                 + 1.05148 w_k1_MSFT_s111
                                 + 1.00338 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.10171 w_k1_ADM_s111
                                 + 1.35729 w_k1_AXP_s111
                                 + 1.28615 w_k1_DIS_s111
                                 + 1.34986 w_k1_FDX_s111
                                 + 1.00269 w_k1_IBM_s111
                                 + 1.00704 w_k1_INTC_s111
                                 + 0.932396 w_k1_KO_s111
                                 + 1.15944 w_k1_MAT_s111
                                 + 1.17799 w_k1_MSFT_s111
                                 + 1.06664 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.937499 w_k1_ADM_s111
                                 + 0.858083 w_k1_AXP_s111
                                 + 1.18753 w_k1_DIS_s111
                                 + 1.04987 w_k1_FDX_s111
                                 + 1.00628 w_k1_IBM_s111
                                 + 1.03013 w_k1_INTC_s111
                                 + 0.964026 w_k1_KO_s111
                                 + 0.675522 w_k1_MAT_s111
                                 + 1.07471 w_k1_MSFT_s111
                                 + 1.02659 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.883132 w_k1_ADM_s111
                                 + 0.66563 w_k1_AXP_s111
                                 + 0.615459 w_k1_DIS_s111
                                 + 0.797624 w_k1_FDX_s111
                                 + 0.764163 w_k1_IBM_s111
                                 + 0.875368 w_k1_INTC_s111
                                 + 0.785487 w_k1_KO_s111
                                 + 0.605729 w_k1_MAT_s111
                                 + 0.84826 w_k1_MSFT_s111
                                 + 0.851911 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.00205 w_k1_ADM_s111
                                 + 0.944801 w_k1_AXP_s111
                                 + 0.757297 w_k1_DIS_s111
                                 + 1.08812 w_k1_FDX_s111
                                 + 0.688445 w_k1_IBM_s111
                                 + 0.997777 w_k1_INTC_s111
                                 + 0.711713 w_k1_KO_s111
                                 + 0.474397 w_k1_MAT_s111
                                 + 0.910222 w_k1_MSFT_s111
                                 + 0.969033 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.911475 w_k1_ADM_s111
                                 + 1.01853 w_k1_AXP_s111
                                 + 0.823469 w_k1_DIS_s111
                                 + 0.890874 w_k1_FDX_s111
                                 + 1.48065 w_k1_IBM_s111
                                 + 0.854516 w_k1_INTC_s111
                                 + 0.981434 w_k1_KO_s111
                                 + 1.24724 w_k1_MAT_s111
                                 + 0.768609 w_k1_MSFT_s111
                                 + 1.05748 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.81166 w_k1_ADM_s111
                                 + 0.774284 w_k1_AXP_s111
                                 + 1.04023 w_k1_DIS_s111
                                 + 1.03198 w_k1_FDX_s111
                                 + 0.626206 w_k1_IBM_s111
                                 + 1.12118 w_k1_INTC_s111
                                 + 0.769638 w_k1_KO_s111
                                 + 0.340962 w_k1_MAT_s111
                                 + 1.1512 w_k1_MSFT_s111
                                 + 0.939293 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.11577 w_k1_ADM_s112
                                 + 1.31177 w_k1_AXP_s112
                                 + 1.51349 w_k1_DIS_s112
                                 + 1.32925 w_k1_FDX_s112
                                 + 0.999319 w_k1_IBM_s112
                                 + 1.2459 w_k1_INTC_s112 + 1.15263 w_k1_KO_s112
                                 + 1.21626 w_k1_MAT_s112
                                 + 1.32606 w_k1_MSFT_s112
                                 + 1.09904 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.854432 w_k1_ADM_s112
                                 + 0.780826 w_k1_AXP_s112
                                 + 0.7125 w_k1_DIS_s112
                                 + 0.895452 w_k1_FDX_s112
                                 + 0.643179 w_k1_IBM_s112
                                 + 0.764955 w_k1_INTC_s112
                                 + 0.734042 w_k1_KO_s112
                                 + 0.602694 w_k1_MAT_s112
                                 + 0.943062 w_k1_MSFT_s112
                                 + 0.862928 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.03532 w_k1_ADM_s112
                                 + 0.961918 w_k1_AXP_s112
                                 + 1.05474 w_k1_DIS_s112
                                 + 1.09691 w_k1_FDX_s112
                                 + 0.801452 w_k1_IBM_s112
                                 + 0.978844 w_k1_INTC_s112
                                 + 0.995535 w_k1_KO_s112
                                 + 0.856283 w_k1_MAT_s112
                                 + 1.13634 w_k1_MSFT_s112
                                 + 0.936317 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.808759 w_k1_ADM_s112
                                 + 0.642506 w_k1_AXP_s112
                                 + 0.522781 w_k1_DIS_s112
                                 + 0.74274 w_k1_FDX_s112
                                 + 0.808665 w_k1_IBM_s112
                                 + 0.699935 w_k1_INTC_s112
                                 + 0.655452 w_k1_KO_s112
                                 + 0.61369 w_k1_MAT_s112
                                 + 0.718982 w_k1_MSFT_s112
                                 + 0.783512 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.06756 w_k1_ADM_s112
                                 + 1.07302 w_k1_AXP_s112
                                 + 1.24309 w_k1_DIS_s112
                                 + 1.11618 w_k1_FDX_s112
                                 + 1.48063 w_k1_IBM_s112
                                 + 1.06515 w_k1_INTC_s112
                                 + 1.20922 w_k1_KO_s112 + 1.46759 w_k1_MAT_s112
                                 + 1.24949 w_k1_MSFT_s112
                                 + 1.11587 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.88029 w_k1_ADM_s112
                                 + 0.896976 w_k1_AXP_s112
                                 + 0.768777 w_k1_DIS_s112
                                 + 0.901616 w_k1_FDX_s112
                                 + 1.02044 w_k1_IBM_s112
                                 + 0.865912 w_k1_INTC_s112
                                 + 0.931347 w_k1_KO_s112
                                 + 0.803003 w_k1_MAT_s112
                                 + 0.944867 w_k1_MSFT_s112
                                 + 0.992968 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.920772 w_k1_ADM_s112
                                 + 0.995013 w_k1_AXP_s112
                                 + 0.915013 w_k1_DIS_s112
                                 + 0.963348 w_k1_FDX_s112
                                 + 0.987177 w_k1_IBM_s112
                                 + 1.14288 w_k1_INTC_s112
                                 + 0.888918 w_k1_KO_s112
                                 + 0.667126 w_k1_MAT_s112
                                 + 1.08604 w_k1_MSFT_s112
                                 + 0.900728 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.754079 w_k1_ADM_s112
                                 + 0.733042 w_k1_AXP_s112
                                 + 0.802997 w_k1_DIS_s112
                                 + 0.812631 w_k1_FDX_s112
                                 + 0.863271 w_k1_IBM_s112
                                 + 1.0518 w_k1_INTC_s112
                                 + 0.904661 w_k1_KO_s112
                                 + 0.751449 w_k1_MAT_s112
                                 + 0.974618 w_k1_MSFT_s112
                                 + 0.8935 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.95946 w_k1_ADM_s113
                                 + 0.93984 w_k1_AXP_s113
                                 + 0.641274 w_k1_DIS_s113
                                 + 0.834218 w_k1_FDX_s113
                                 + 1.12681 w_k1_IBM_s113
                                 + 0.701037 w_k1_INTC_s113
                                 + 0.802502 w_k1_KO_s113
                                 + 1.06827 w_k1_MAT_s113
                                 + 0.703904 w_k1_MSFT_s113
                                 + 0.893293 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.00107 w_k1_ADM_s113
                                 + 0.803573 w_k1_AXP_s113
                                 + 0.763173 w_k1_DIS_s113
                                 + 0.803473 w_k1_FDX_s113
                                 + 1.52518 w_k1_IBM_s113
                                 + 0.791325 w_k1_INTC_s113
                                 + 1.05831 w_k1_KO_s113
                                 + 0.978633 w_k1_MAT_s113
                                 + 0.733124 w_k1_MSFT_s113
                                 + 0.977741 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.05897 w_k1_ADM_s113
                                 + 0.594949 w_k1_AXP_s113
                                 + 0.755563 w_k1_DIS_s113
                                 + 0.918363 w_k1_FDX_s113
                                 + 0.64704 w_k1_IBM_s113
                                 + 0.785572 w_k1_INTC_s113
                                 + 0.679244 w_k1_KO_s113
                                 + 0.525072 w_k1_MAT_s113
                                 + 0.915604 w_k1_MSFT_s113
                                 + 0.914685 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.983688 w_k1_ADM_s113
                                 + 0.989638 w_k1_AXP_s113
                                 + 1.0271 w_k1_DIS_s113
                                 + 0.923173 w_k1_FDX_s113
                                 + 1.52081 w_k1_IBM_s113
                                 + 0.956115 w_k1_INTC_s113
                                 + 1.1999 w_k1_KO_s113 + 1.54754 w_k1_MAT_s113
                                 + 0.875482 w_k1_MSFT_s113
                                 + 1.0421 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.02318 w_k1_ADM_s113
                                 + 0.973355 w_k1_AXP_s113
                                 + 0.686478 w_k1_DIS_s113
                                 + 0.878864 w_k1_FDX_s113
                                 + 0.782812 w_k1_IBM_s113
                                 + 0.829466 w_k1_INTC_s113
                                 + 0.699612 w_k1_KO_s113
                                 + 0.752377 w_k1_MAT_s113
                                 + 0.731342 w_k1_MSFT_s113
                                 + 0.883912 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.11461 w_k1_ADM_s113
                                 + 1.2042 w_k1_AXP_s113 + 1.0158 w_k1_DIS_s113
                                 + 1.22649 w_k1_FDX_s113
                                 + 0.718547 w_k1_IBM_s113
                                 + 0.960773 w_k1_INTC_s113
                                 + 0.872051 w_k1_KO_s113
                                 + 0.82456 w_k1_MAT_s113
                                 + 0.950006 w_k1_MSFT_s113
                                 + 1.04621 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.974963 w_k1_ADM_s113
                                 + 0.795651 w_k1_AXP_s113
                                 + 0.868504 w_k1_DIS_s113
                                 + 0.903571 w_k1_FDX_s113
                                 + 1.00862 w_k1_IBM_s113
                                 + 0.782167 w_k1_INTC_s113
                                 + 0.878547 w_k1_KO_s113
                                 + 0.781656 w_k1_MAT_s113
                                 + 0.82078 w_k1_MSFT_s113
                                 + 0.979236 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.56765 w_k1_ADM_s113
                                 + 1.5207 w_k1_AXP_s113 + 1.47473 w_k1_DIS_s113
                                 + 1.37363 w_k1_FDX_s113 + 1.3275 w_k1_IBM_s113
                                 + 1.16141 w_k1_INTC_s113
                                 + 1.22496 w_k1_KO_s113 + 1.36746 w_k1_MAT_s113
                                 + 1.18794 w_k1_MSFT_s113
                                 + 1.29183 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.984348 w_k1_ADM_s114
                                 + 0.714107 w_k1_AXP_s114
                                 + 0.900135 w_k1_DIS_s114
                                 + 0.934942 w_k1_FDX_s114
                                 + 0.921572 w_k1_IBM_s114
                                 + 1.1019 w_k1_INTC_s114
                                 + 0.998249 w_k1_KO_s114
                                 + 0.781485 w_k1_MAT_s114
                                 + 1.00491 w_k1_MSFT_s114
                                 + 0.950129 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.01664 w_k1_ADM_s114
                                 + 0.936215 w_k1_AXP_s114
                                 + 1.37711 w_k1_DIS_s114
                                 + 1.12254 w_k1_FDX_s114
                                 + 1.06778 w_k1_IBM_s114
                                 + 1.12356 w_k1_INTC_s114
                                 + 1.31937 w_k1_KO_s114 + 1.28406 w_k1_MAT_s114
                                 + 1.1278 w_k1_MSFT_s114 + 1.08182 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.872517 w_k1_ADM_s114
                                 + 0.844891 w_k1_AXP_s114
                                 + 0.800328 w_k1_DIS_s114
                                 + 0.837006 w_k1_FDX_s114
                                 + 1.29827 w_k1_IBM_s114
                                 + 0.96598 w_k1_INTC_s114
                                 + 1.13493 w_k1_KO_s114
                                 + 0.985492 w_k1_MAT_s114
                                 + 0.905712 w_k1_MSFT_s114
                                 + 0.908187 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.766212 w_k1_ADM_s114
                                 + 0.567926 w_k1_AXP_s114
                                 + 0.660949 w_k1_DIS_s114
                                 + 0.797293 w_k1_FDX_s114
                                 + 0.755612 w_k1_IBM_s114
                                 + 1.10768 w_k1_INTC_s114
                                 + 0.875947 w_k1_KO_s114
                                 + 0.724982 w_k1_MAT_s114
                                 + 0.929586 w_k1_MSFT_s114
                                 + 0.822762 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.14183 w_k1_ADM_s114
                                 + 0.815754 w_k1_AXP_s114
                                 + 0.610181 w_k1_DIS_s114
                                 + 0.847796 w_k1_FDX_s114
                                 + 0.775802 w_k1_IBM_s114
                                 + 0.831141 w_k1_INTC_s114
                                 + 0.747453 w_k1_KO_s114
                                 + 0.680336 w_k1_MAT_s114
                                 + 0.803535 w_k1_MSFT_s114
                                 + 0.831159 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.975852 w_k1_ADM_s114
                                 + 1.04421 w_k1_AXP_s114
                                 + 0.927575 w_k1_DIS_s114
                                 + 1.08275 w_k1_FDX_s114
                                 + 0.87809 w_k1_IBM_s114
                                 + 1.10327 w_k1_INTC_s114
                                 + 0.880404 w_k1_KO_s114
                                 + 0.829926 w_k1_MAT_s114
                                 + 0.943831 w_k1_MSFT_s114
                                 + 0.986759 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.46724 w_k1_ADM_s114
                                 + 1.37975 w_k1_AXP_s114
                                 + 1.36878 w_k1_DIS_s114
                                 + 1.33316 w_k1_FDX_s114 + 1.3995 w_k1_IBM_s114
                                 + 1.31441 w_k1_INTC_s114
                                 + 1.21568 w_k1_KO_s114 + 1.05789 w_k1_MAT_s114
                                 + 1.17661 w_k1_MSFT_s114
                                 + 1.16491 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.27292 w_k1_ADM_s114
                                 + 1.36348 w_k1_AXP_s114
                                 + 1.30415 w_k1_DIS_s114
                                 + 1.20135 w_k1_FDX_s114
                                 + 1.06664 w_k1_IBM_s114
                                 + 1.09641 w_k1_INTC_s114
                                 + 1.24805 w_k1_KO_s114 + 1.14443 w_k1_MAT_s114
                                 + 1.11028 w_k1_MSFT_s114 + 1.1802 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.875139 w_k1_ADM_s115
                                 + 0.890635 w_k1_AXP_s115
                                 + 0.851129 w_k1_DIS_s115
                                 + 0.884446 w_k1_FDX_s115
                                 + 1.10024 w_k1_IBM_s115
                                 + 1.10765 w_k1_INTC_s115 + 1.0328 w_k1_KO_s115
                                 + 1.007 w_k1_MAT_s115
                                 + 0.892686 w_k1_MSFT_s115
                                 + 0.909894 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.40979 w_k1_ADM_s115
                                 + 1.3264 w_k1_AXP_s115 + 1.36378 w_k1_DIS_s115
                                 + 1.23668 w_k1_FDX_s115
                                 + 1.18252 w_k1_IBM_s115
                                 + 1.15982 w_k1_INTC_s115
                                 + 1.32214 w_k1_KO_s115 + 1.04641 w_k1_MAT_s115
                                 + 1.17834 w_k1_MSFT_s115
                                 + 1.20086 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.11481 w_k1_ADM_s115
                                 + 1.04584 w_k1_AXP_s115 + 0.9589 w_k1_DIS_s115
                                 + 1.01538 w_k1_FDX_s115
                                 + 0.815767 w_k1_IBM_s115
                                 + 0.951225 w_k1_INTC_s115
                                 + 0.897408 w_k1_KO_s115
                                 + 0.861325 w_k1_MAT_s115
                                 + 1.03675 w_k1_MSFT_s115 + 1.078 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.845222 w_k1_ADM_s115
                                 + 0.610502 w_k1_AXP_s115
                                 + 0.657167 w_k1_DIS_s115
                                 + 0.776528 w_k1_FDX_s115
                                 + 0.666049 w_k1_IBM_s115
                                 + 0.865773 w_k1_INTC_s115
                                 + 0.691022 w_k1_KO_s115
                                 + 0.481029 w_k1_MAT_s115
                                 + 0.836251 w_k1_MSFT_s115
                                 + 0.874734 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.02971 w_k1_ADM_s115
                                 + 0.746757 w_k1_AXP_s115
                                 + 0.847088 w_k1_DIS_s115
                                 + 0.844687 w_k1_FDX_s115
                                 + 0.81451 w_k1_IBM_s115
                                 + 0.856275 w_k1_INTC_s115
                                 + 0.970912 w_k1_KO_s115
                                 + 0.610808 w_k1_MAT_s115
                                 + 0.896099 w_k1_MSFT_s115
                                 + 0.909356 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.27268 w_k1_ADM_s115
                                 + 1.48942 w_k1_AXP_s115
                                 + 1.90113 w_k1_DIS_s115
                                 + 1.44358 w_k1_FDX_s115
                                 + 1.05495 w_k1_IBM_s115
                                 + 1.44494 w_k1_INTC_s115 + 1.3824 w_k1_KO_s115
                                 + 1.20265 w_k1_MAT_s115
                                 + 1.5889 w_k1_MSFT_s115 + 1.17141 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.15539 w_k1_ADM_s115
                                 + 1.16054 w_k1_AXP_s115
                                 + 1.24322 w_k1_DIS_s115
                                 + 1.03682 w_k1_FDX_s115
                                 + 0.950908 w_k1_IBM_s115
                                 + 1.09321 w_k1_INTC_s115
                                 + 1.01129 w_k1_KO_s115
                                 + 0.883615 w_k1_MAT_s115
                                 + 1.1328 w_k1_MSFT_s115 + 1.00558 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.951517 w_k1_ADM_s115
                                 + 1.19108 w_k1_AXP_s115
                                 + 1.40491 w_k1_DIS_s115
                                 + 1.11986 w_k1_FDX_s115 + 1.5845 w_k1_IBM_s115
                                 + 1.24116 w_k1_INTC_s115
                                 + 1.46932 w_k1_KO_s115 + 1.34367 w_k1_MAT_s115
                                 + 1.31919 w_k1_MSFT_s115
                                 + 1.13817 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.795558 w_k1_ADM_s116
                                 + 0.89335 w_k1_AXP_s116
                                 + 0.856304 w_k1_DIS_s116
                                 + 0.831561 w_k1_FDX_s116
                                 + 1.4026 w_k1_IBM_s116
                                 + 1.11059 w_k1_INTC_s116
                                 + 1.07763 w_k1_KO_s116 + 1.10687 w_k1_MAT_s116
                                 + 0.907336 w_k1_MSFT_s116
                                 + 0.971867 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.11914 w_k1_ADM_s116
                                 + 1.42655 w_k1_AXP_s116
                                 + 1.67993 w_k1_DIS_s116 + 1.3906 w_k1_FDX_s116
                                 + 1.23952 w_k1_IBM_s116
                                 + 1.48061 w_k1_INTC_s116
                                 + 1.24876 w_k1_KO_s116
                                 + 0.913346 w_k1_MAT_s116
                                 + 1.45061 w_k1_MSFT_s116
                                 + 1.09944 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.00357 w_k1_ADM_s116
                                 + 0.971218 w_k1_AXP_s116
                                 + 1.26043 w_k1_DIS_s116
                                 + 1.03148 w_k1_FDX_s116
                                 + 1.19843 w_k1_IBM_s116
                                 + 1.06475 w_k1_INTC_s116 + 1.2433 w_k1_KO_s116
                                 + 0.954395 w_k1_MAT_s116
                                 + 0.978645 w_k1_MSFT_s116
                                 + 1.0522 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.934401 w_k1_ADM_s116
                                 + 1.47228 w_k1_AXP_s116
                                 + 1.37026 w_k1_DIS_s116 + 1.2685 w_k1_FDX_s116
                                 + 1.16899 w_k1_IBM_s116
                                 + 1.34819 w_k1_INTC_s116
                                 + 1.07707 w_k1_KO_s116 + 1.35533 w_k1_MAT_s116
                                 + 1.26066 w_k1_MSFT_s116
                                 + 1.11943 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.00159 w_k1_ADM_s116
                                 + 1.00371 w_k1_AXP_s116
                                 + 1.21209 w_k1_DIS_s116
                                 + 1.23096 w_k1_FDX_s116
                                 + 0.771616 w_k1_IBM_s116
                                 + 0.98726 w_k1_INTC_s116
                                 + 0.939268 w_k1_KO_s116
                                 + 0.896844 w_k1_MAT_s116
                                 + 1.10933 w_k1_MSFT_s116
                                 + 0.985099 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.818302 w_k1_ADM_s116
                                 + 0.687941 w_k1_AXP_s116
                                 + 0.621752 w_k1_DIS_s116
                                 + 0.805651 w_k1_FDX_s116
                                 + 0.699797 w_k1_IBM_s116
                                 + 0.805594 w_k1_INTC_s116
                                 + 0.667668 w_k1_KO_s116
                                 + 0.59419 w_k1_MAT_s116
                                 + 0.767705 w_k1_MSFT_s116
                                 + 0.754109 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.719649 w_k1_ADM_s116
                                 + 1.06925 w_k1_AXP_s116
                                 + 0.842058 w_k1_DIS_s116
                                 + 0.973438 w_k1_FDX_s116
                                 + 0.875075 w_k1_IBM_s116
                                 + 1.10305 w_k1_INTC_s116
                                 + 0.818226 w_k1_KO_s116
                                 + 0.955172 w_k1_MAT_s116
                                 + 0.893716 w_k1_MSFT_s116
                                 + 0.937055 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.00571 w_k1_ADM_s116
                                 + 1.15827 w_k1_AXP_s116 + 1.0682 w_k1_DIS_s116
                                 + 1.04517 w_k1_FDX_s116
                                 + 1.43647 w_k1_IBM_s116
                                 + 1.06065 w_k1_INTC_s116
                                 + 1.11759 w_k1_KO_s116
                                 + 0.912716 w_k1_MAT_s116
                                 + 1.01709 w_k1_MSFT_s116
                                 + 0.999067 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.41571 w_k1_ADM_s117
                                 + 1.75966 w_k1_AXP_s117
                                 + 1.46846 w_k1_DIS_s117
                                 + 1.32616 w_k1_FDX_s117
                                 + 1.40012 w_k1_IBM_s117
                                 + 1.43042 w_k1_INTC_s117
                                 + 1.32848 w_k1_KO_s117 + 1.66265 w_k1_MAT_s117
                                 + 1.42439 w_k1_MSFT_s117
                                 + 1.13067 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.25209 w_k1_ADM_s117
                                 + 1.51914 w_k1_AXP_s117
                                 + 1.78163 w_k1_DIS_s117 + 1.4123 w_k1_FDX_s117
                                 + 1.17559 w_k1_IBM_s117
                                 + 1.0259 w_k1_INTC_s117 + 1.41544 w_k1_KO_s117
                                 + 1.40133 w_k1_MAT_s117
                                 + 1.32295 w_k1_MSFT_s117 + 1.3413 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.961411 w_k1_ADM_s117
                                 + 0.703745 w_k1_AXP_s117
                                 + 0.734333 w_k1_DIS_s117
                                 + 0.801115 w_k1_FDX_s117
                                 + 0.790008 w_k1_IBM_s117
                                 + 0.800613 w_k1_INTC_s117
                                 + 0.954551 w_k1_KO_s117
                                 + 0.620565 w_k1_MAT_s117
                                 + 0.818241 w_k1_MSFT_s117
                                 + 0.819838 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.26599 w_k1_ADM_s117
                                 + 1.28155 w_k1_AXP_s117
                                 + 0.939848 w_k1_DIS_s117
                                 + 1.11257 w_k1_FDX_s117
                                 + 0.982577 w_k1_IBM_s117
                                 + 1.14998 w_k1_INTC_s117
                                 + 0.972792 w_k1_KO_s117
                                 + 0.982141 w_k1_MAT_s117
                                 + 1.08329 w_k1_MSFT_s117
                                 + 1.03122 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.09158 w_k1_ADM_s117
                                 + 1.01852 w_k1_AXP_s117
                                 + 0.835497 w_k1_DIS_s117
                                 + 0.923937 w_k1_FDX_s117
                                 + 1.08183 w_k1_IBM_s117
                                 + 0.787239 w_k1_INTC_s117
                                 + 1.03566 w_k1_KO_s117 + 1.14473 w_k1_MAT_s117
                                 + 0.938785 w_k1_MSFT_s117
                                 + 1.01148 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.06564 w_k1_ADM_s117
                                 + 1.10663 w_k1_AXP_s117
                                 + 1.17679 w_k1_DIS_s117 + 1.0151 w_k1_FDX_s117
                                 + 1.40459 w_k1_IBM_s117
                                 + 0.973861 w_k1_INTC_s117
                                 + 1.28923 w_k1_KO_s117 + 1.25192 w_k1_MAT_s117
                                 + 1.06084 w_k1_MSFT_s117
                                 + 0.99993 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.01011 w_k1_ADM_s117
                                 + 0.868989 w_k1_AXP_s117
                                 + 0.994404 w_k1_DIS_s117
                                 + 1.09146 w_k1_FDX_s117
                                 + 0.746203 w_k1_IBM_s117
                                 + 1.00717 w_k1_INTC_s117 + 0.8805 w_k1_KO_s117
                                 + 0.675596 w_k1_MAT_s117
                                 + 1.04465 w_k1_MSFT_s117
                                 + 0.936914 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.33308 w_k1_ADM_s117
                                 + 1.15462 w_k1_AXP_s117
                                 + 1.21559 w_k1_DIS_s117
                                 + 1.17816 w_k1_FDX_s117
                                 + 0.829737 w_k1_IBM_s117
                                 + 0.89534 w_k1_INTC_s117 + 1.1295 w_k1_KO_s117
                                 + 0.908435 w_k1_MAT_s117
                                 + 1.05123 w_k1_MSFT_s117
                                 + 1.08551 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0246799103756616 w_k1_ADM_s000
                                 + 0.0306865768720678 w_k1_ADM_s001
                                 + 0.0189096828977345 w_k1_ADM_s002
                                 + 0.0142278987664377 w_k1_ADM_s003
                                 + 0.0160706651232364 w_k1_ADM_s004
                                 + 0.042844076489026 w_k1_ADM_s005
                                 + 0.0134196500586314 w_k1_ADM_s006
                                 + 0.013831700232175 w_k1_ADM_s007
                                 + 0.0470435831397218 w_k1_ADM_s010
                                 + 0.0186277665390743 w_k1_ADM_s011
                                 + 0.0146785393860544 w_k1_ADM_s012
                                 + 0.0413470081275057 w_k1_ADM_s013
                                 + 0.0267505057637946 w_k1_ADM_s014
                                 + 0.0126533510795939 w_k1_ADM_s015
                                 + 0.00910197701460394 w_k1_ADM_s016
                                 + 0.0123584100157301 w_k1_ADM_s017
                                 + 0.0400807090703972 w_k1_ADM_s100
                                 + 0.0330809916707035 w_k1_ADM_s101
                                 + 0.0362319118236789 w_k1_ADM_s102
                                 + 0.0584253243125225 w_k1_ADM_s103
                                 + 0.0232840588830797 w_k1_ADM_s104
                                 + 0.0479620495717385 w_k1_ADM_s105
                                 + 0.0491732961596436 w_k1_ADM_s106
                                 + 0.0585095497381306 w_k1_ADM_s107
                                 + 0.0268256048690517 w_k1_ADM_s110
                                 + 0.0137485152808981 w_k1_ADM_s111
                                 + 0.043091099518675 w_k1_ADM_s112
                                 + 0.0592239953981285 w_k1_ADM_s113
                                 + 0.0436108770863762 w_k1_ADM_s114
                                 + 0.0278920956883609 w_k1_ADM_s115
                                 + 0.0340398093781024 w_k1_ADM_s116
                                 + 0.077621849776954 w_k1_ADM_s117
                                 + 0.0273165065680629 w_k1_AXP_s000
                                 + 0.0290028771344415 w_k1_AXP_s001
                                 + 0.0170589609916122 w_k1_AXP_s002
                                 + 0.0121362778140877 w_k1_AXP_s003
                                 + 0.0173456219243421 w_k1_AXP_s004
                                 + 0.0349760829403657 w_k1_AXP_s005
                                 + 0.0147473834854539 w_k1_AXP_s006
                                 + 0.0143427525098033 w_k1_AXP_s007
                                 + 0.0511330022845056 w_k1_AXP_s010
                                 + 0.0170314830622467 w_k1_AXP_s011
                                 + 0.0142187173291455 w_k1_AXP_s012
                                 + 0.0413229140588293 w_k1_AXP_s013
                                 + 0.0304206673972009 w_k1_AXP_s014
                                 + 0.0111575236840489 w_k1_AXP_s015
                                 + 0.0103058106287971 w_k1_AXP_s016
                                 + 0.0128889111714262 w_k1_AXP_s017
                                 + 0.0398331875040481 w_k1_AXP_s100
                                 + 0.0310047429121025 w_k1_AXP_s101
                                 + 0.0356413885711358 w_k1_AXP_s102
                                 + 0.0537033514843046 w_k1_AXP_s103
                                 + 0.0238676362396871 w_k1_AXP_s104
                                 + 0.056584204750636 w_k1_AXP_s105
                                 + 0.0481976995007451 w_k1_AXP_s106
                                 + 0.0640215153814316 w_k1_AXP_s107
                                 + 0.025928608515308 w_k1_AXP_s110
                                 + 0.0137471851875358 w_k1_AXP_s111
                                 + 0.0438827796834173 w_k1_AXP_s112
                                 + 0.0539649239795445 w_k1_AXP_s113
                                 + 0.0397947682478499 w_k1_AXP_s114
                                 + 0.0264672383737073 w_k1_AXP_s115
                                 + 0.0410779040331585 w_k1_AXP_s116
                                 + 0.0720916626262964 w_k1_AXP_s117
                                 + 0.0255778326977128 w_k1_DIS_s000
                                 + 0.0297121238893234 w_k1_DIS_s001
                                 + 0.0182569585101325 w_k1_DIS_s002
                                 + 0.012043782002751 w_k1_DIS_s003
                                 + 0.0189429114747253 w_k1_DIS_s004
                                 + 0.0330646399781268 w_k1_DIS_s005
                                 + 0.0147855498472044 w_k1_DIS_s006
                                 + 0.0149275129590877 w_k1_DIS_s007
                                 + 0.0470300798235879 w_k1_DIS_s010
                                 + 0.018264940377594 w_k1_DIS_s011
                                 + 0.0138942732280591 w_k1_DIS_s012
                                 + 0.0404922962725991 w_k1_DIS_s013
                                 + 0.0316099761961973 w_k1_DIS_s014
                                 + 0.0118899505371399 w_k1_DIS_s015
                                 + 0.0117511433697691 w_k1_DIS_s016
                                 + 0.0126486798503434 w_k1_DIS_s017
                                 + 0.0402895084604772 w_k1_DIS_s100
                                 + 0.0392141966241326 w_k1_DIS_s101
                                 + 0.0320312403342153 w_k1_DIS_s102
                                 + 0.0536521502644271 w_k1_DIS_s103
                                 + 0.0252968424462581 w_k1_DIS_s104
                                 + 0.0573773508380726 w_k1_DIS_s105
                                 + 0.0479170624912062 w_k1_DIS_s106
                                 + 0.0664525499601064 w_k1_DIS_s107
                                 + 0.0267728735390287 w_k1_DIS_s110
                                 + 0.0136328693265054 w_k1_DIS_s111
                                 + 0.0460700976086258 w_k1_DIS_s112
                                 + 0.0517965437002698 w_k1_DIS_s113
                                 + 0.0430814386265633 w_k1_DIS_s114
                                 + 0.0278798059031318 w_k1_DIS_s115
                                 + 0.0395656351438397 w_k1_DIS_s116
                                 + 0.0705432507781008 w_k1_DIS_s117
                                 + 0.0254595990057585 w_k1_FDX_s000
                                 + 0.0294535003005258 w_k1_FDX_s001
                                 + 0.0185764536695404 w_k1_FDX_s002
                                 + 0.0123308609721158 w_k1_FDX_s003
                                 + 0.0174902938329296 w_k1_FDX_s004
                                 + 0.036366015663157 w_k1_FDX_s005
                                 + 0.013843976970498 w_k1_FDX_s006
                                 + 0.0150709891788328 w_k1_FDX_s007
                                 + 0.0494134276835938 w_k1_FDX_s010
                                 + 0.0177167894743141 w_k1_FDX_s011
                                 + 0.0129819962680254 w_k1_FDX_s012
                                 + 0.0400746748976398 w_k1_FDX_s013
                                 + 0.0303856284466046 w_k1_FDX_s014
                                 + 0.0113742387642552 w_k1_FDX_s015
                                 + 0.0114345761712118 w_k1_FDX_s016
                                 + 0.0124527048028936 w_k1_FDX_s017
                                 + 0.042508807514319 w_k1_FDX_s100
                                 + 0.0343794300495771 w_k1_FDX_s101
                                 + 0.032671847227997 w_k1_FDX_s102
                                 + 0.0547714415077976 w_k1_FDX_s103
                                 + 0.0258361221668871 w_k1_FDX_s104
                                 + 0.0517583510012661 w_k1_FDX_s105
                                 + 0.0496338867729559 w_k1_FDX_s106
                                 + 0.058702063212908 w_k1_FDX_s107
                                 + 0.0267312182064227 w_k1_FDX_s110
                                 + 0.0145973786607707 w_k1_FDX_s111
                                 + 0.0460373512307293 w_k1_FDX_s112
                                 + 0.0550001104377894 w_k1_FDX_s113
                                 + 0.0425198990381306 w_k1_FDX_s114
                                 + 0.0263405070917137 w_k1_FDX_s115
                                 + 0.0396717357225033 w_k1_FDX_s116
                                 + 0.0705006319373072 w_k1_FDX_s117
                                 + 0.0269368990646607 w_k1_IBM_s000
                                 + 0.0292541536461503 w_k1_IBM_s001
                                 + 0.019020683075436 w_k1_IBM_s002
                                 + 0.0126339008798757 w_k1_IBM_s003
                                 + 0.0174666577394769 w_k1_IBM_s004
                                 + 0.0365620564691723 w_k1_IBM_s005
                                 + 0.0149512111318842 w_k1_IBM_s006
                                 + 0.0144586545358047 w_k1_IBM_s007
                                 + 0.0489773387002239 w_k1_IBM_s010
                                 + 0.017938883128254 w_k1_IBM_s011
                                 + 0.0140154196805854 w_k1_IBM_s012
                                 + 0.0395066744250833 w_k1_IBM_s013
                                 + 0.0295663619493602 w_k1_IBM_s014
                                 + 0.0121269587395725 w_k1_IBM_s015
                                 + 0.0115322208028435 w_k1_IBM_s016
                                 + 0.0126814542197777 w_k1_IBM_s017
                                 + 0.0418178562129529 w_k1_IBM_s100
                                 + 0.0350833265105689 w_k1_IBM_s101
                                 + 0.0326903729101626 w_k1_IBM_s102
                                 + 0.0566697388675325 w_k1_IBM_s103
                                 + 0.0253247106342945 w_k1_IBM_s104
                                 + 0.0530338593558779 w_k1_IBM_s105
                                 + 0.0482078364692058 w_k1_IBM_s106
                                 + 0.0607150161472221 w_k1_IBM_s107
                                 + 0.0257761249277783 w_k1_IBM_s110
                                 + 0.0149094920524717 w_k1_IBM_s111
                                 + 0.0455229654000111 w_k1_IBM_s112
                                 + 0.0549402154823305 w_k1_IBM_s113
                                 + 0.0441449179260397 w_k1_IBM_s114
                                 + 0.0258702641951979 w_k1_IBM_s115
                                 + 0.0419270847673119 w_k1_IBM_s116
                                 + 0.0692223590699093 w_k1_IBM_s117
                                 + 0.0284947814518716 w_k1_INTC_s000
                                 + 0.0270853625355505 w_k1_INTC_s001
                                 + 0.0167375959686043 w_k1_INTC_s002
                                 + 0.0109011892179506 w_k1_INTC_s003
                                 + 0.0185073181275667 w_k1_INTC_s004
                                 + 0.0401962734872933 w_k1_INTC_s005
                                 + 0.0132324905323336 w_k1_INTC_s006
                                 + 0.0148330909953226 w_k1_INTC_s007
                                 + 0.0484033909827288 w_k1_INTC_s010
                                 + 0.0158446722145897 w_k1_INTC_s011
                                 + 0.0151261883895005 w_k1_INTC_s012
                                 + 0.0387852663703594 w_k1_INTC_s013
                                 + 0.0345113591188556 w_k1_INTC_s014
                                 + 0.0114328246847166 w_k1_INTC_s015
                                 + 0.010628286384684 w_k1_INTC_s016
                                 + 0.0146028084409642 w_k1_INTC_s017
                                 + 0.0389944035153088 w_k1_INTC_s100
                                 + 0.0319193837392172 w_k1_INTC_s101
                                 + 0.0319698819072872 w_k1_INTC_s102
                                 + 0.0611999056961067 w_k1_INTC_s103
                                 + 0.0267170332936506 w_k1_INTC_s104
                                 + 0.0561038545223059 w_k1_INTC_s105
                                 + 0.0488976230827815 w_k1_INTC_s106
                                 + 0.060943252472679 w_k1_INTC_s107
                                 + 0.0243136974737493 w_k1_INTC_s110
                                 + 0.0143935993921592 w_k1_INTC_s111
                                 + 0.0455147810529904 w_k1_INTC_s112
                                 + 0.0476846785029906 w_k1_INTC_s113
                                 + 0.0449363269479335 w_k1_INTC_s114
                                 + 0.0275956390915537 w_k1_INTC_s115
                                 + 0.042533907163328 w_k1_INTC_s116
                                 + 0.0648262437849109 w_k1_INTC_s117
                                 + 0.0258382295412087 w_k1_KO_s000
                                 + 0.0325271371652333 w_k1_KO_s001
                                 + 0.0183350648026099 w_k1_KO_s002
                                 + 0.0110281474007884 w_k1_KO_s003
                                 + 0.018249409379214 w_k1_KO_s004
                                 + 0.0377049722750905 w_k1_KO_s005
                                 + 0.0146441823862827 w_k1_KO_s006
                                 + 0.0144437944651666 w_k1_KO_s007
                                 + 0.0456504595857616 w_k1_KO_s010
                                 + 0.0167631211423051 w_k1_KO_s011
                                 + 0.0144130039819871 w_k1_KO_s012
                                 + 0.0368463645367102 w_k1_KO_s013
                                 + 0.03124892214862 w_k1_KO_s014
                                 + 0.0108729438565467 w_k1_KO_s015
                                 + 0.011476834386841 w_k1_KO_s016
                                 + 0.0113541386017503 w_k1_KO_s017
                                 + 0.0371844623787203 w_k1_KO_s100
                                 + 0.0339979456191853 w_k1_KO_s101
                                 + 0.0309124214841386 w_k1_KO_s102
                                 + 0.0574356923796538 w_k1_KO_s103
                                 + 0.0268884523303937 w_k1_KO_s104
                                 + 0.0512259139277624 w_k1_KO_s105
                                 + 0.048857988596695 w_k1_KO_s106
                                 + 0.0605807145396345 w_k1_KO_s107
                                 + 0.0256409917725593 w_k1_KO_s110
                                 + 0.0131976177811552 w_k1_KO_s111
                                 + 0.0438556574442864 w_k1_KO_s112
                                 + 0.0498604536529757 w_k1_KO_s113
                                 + 0.0453031867039721 w_k1_KO_s114
                                 + 0.0274940961090113 w_k1_KO_s115
                                 + 0.0379917521545375 w_k1_KO_s116
                                 + 0.0753266650673074 w_k1_KO_s117
                                 + 0.0257656628935766 w_k1_MAT_s000
                                 + 0.0272124769432539 w_k1_MAT_s001
                                 + 0.0165632745930023 w_k1_MAT_s002
                                 + 0.0108337367083594 w_k1_MAT_s003
                                 + 0.0183681735392871 w_k1_MAT_s004
                                 + 0.0335229543429401 w_k1_MAT_s005
                                 + 0.0157694612574187 w_k1_MAT_s006
                                 + 0.0144384513707537 w_k1_MAT_s007
                                 + 0.0500378420677 w_k1_MAT_s010
                                 + 0.019799170395731 w_k1_MAT_s011
                                 + 0.0154389068985251 w_k1_MAT_s012
                                 + 0.0306210980695318 w_k1_MAT_s013
                                 + 0.0269997903861384 w_k1_MAT_s014
                                 + 0.0129941567581492 w_k1_MAT_s015
                                 + 0.0118719850906943 w_k1_MAT_s016
                                 + 0.0119004888999892 w_k1_MAT_s017
                                 + 0.0443170567336469 w_k1_MAT_s100
                                 + 0.0324537342586384 w_k1_MAT_s101
                                 + 0.0281372460226232 w_k1_MAT_s102
                                 + 0.0603858432737532 w_k1_MAT_s103
                                 + 0.0202738501745589 w_k1_MAT_s104
                                 + 0.0552546845337401 w_k1_MAT_s105
                                 + 0.0470133629811519 w_k1_MAT_s106
                                 + 0.061792478517727 w_k1_MAT_s107
                                 + 0.0292975687704795 w_k1_MAT_s110
                                 + 0.0129153580422221 w_k1_MAT_s111
                                 + 0.0431064550997985 w_k1_MAT_s112
                                 + 0.051538776642118 w_k1_MAT_s113
                                 + 0.0403358063792364 w_k1_MAT_s114
                                 + 0.0232989688080103 w_k1_MAT_s115
                                 + 0.0372562002020072 w_k1_MAT_s116
                                 + 0.0674165817721556 w_k1_MAT_s117
                                 + 0.024288192329116 w_k1_MSFT_s000
                                 + 0.0287619973853879 w_k1_MSFT_s001
                                 + 0.01690153478579 w_k1_MSFT_s002
                                 + 0.0107487470899605 w_k1_MSFT_s003
                                 + 0.016595816955284 w_k1_MSFT_s004
                                 + 0.0355768199869662 w_k1_MSFT_s005
                                 + 0.0139795015194607 w_k1_MSFT_s006
                                 + 0.0160285015942522 w_k1_MSFT_s007
                                 + 0.0472980150812015 w_k1_MSFT_s010
                                 + 0.0176923444338378 w_k1_MSFT_s011
                                 + 0.0136829208445145 w_k1_MSFT_s012
                                 + 0.0368077003171225 w_k1_MSFT_s013
                                 + 0.0295336268073908 w_k1_MSFT_s014
                                 + 0.011548949090212 w_k1_MSFT_s015
                                 + 0.0112036192112339 w_k1_MSFT_s016
                                 + 0.0115548891237377 w_k1_MSFT_s017
                                 + 0.0388437485445433 w_k1_MSFT_s100
                                 + 0.0349720462795353 w_k1_MSFT_s101
                                 + 0.0329322638327647 w_k1_MSFT_s102
                                 + 0.0519714828425961 w_k1_MSFT_s103
                                 + 0.026261418418667 w_k1_MSFT_s104
                                 + 0.0488153892426185 w_k1_MSFT_s105
                                 + 0.0471727079263031 w_k1_MSFT_s106
                                 + 0.0588831434550731 w_k1_MSFT_s107
                                 + 0.026558103595275 w_k1_MSFT_s110
                                 + 0.0142280872412023 w_k1_MSFT_s111
                                 + 0.0491317235780319 w_k1_MSFT_s112
                                 + 0.0480892399777735 w_k1_MSFT_s113
                                 + 0.0419157250195136 w_k1_MSFT_s114
                                 + 0.0273935531667894 w_k1_MSFT_s115
                                 + 0.0384980439804737 w_k1_MSFT_s116
                                 + 0.0701439836920742 w_k1_MSFT_s117
                                 + 0.0239960847717843 w_k1_PG_s000
                                 + 0.0292722820049284 w_k1_PG_s001
                                 + 0.0186048876230656 w_k1_PG_s002
                                 + 0.0102402958763587 w_k1_PG_s003
                                 + 0.0182534440696764 w_k1_PG_s004
                                 + 0.03633910435784 w_k1_PG_s005
                                 + 0.0162833787740453 w_k1_PG_s006
                                 + 0.0138368950323045 w_k1_PG_s007
                                 + 0.0458558488327589 w_k1_PG_s010
                                 + 0.0169006671619436 w_k1_PG_s011
                                 + 0.014164701886217 w_k1_PG_s012
                                 + 0.0379033027149854 w_k1_PG_s013
                                 + 0.0318567521512777 w_k1_PG_s014
                                 + 0.0104406289449728 w_k1_PG_s015
                                 + 0.0106840267623302 w_k1_PG_s016
                                 + 0.0130429984338585 w_k1_PG_s017
                                 + 0.0379631645024905 w_k1_PG_s100
                                 + 0.0332617764429267 w_k1_PG_s101
                                 + 0.0338229426626812 w_k1_PG_s102
                                 + 0.0530685203895939 w_k1_PG_s103
                                 + 0.0231927782249607 w_k1_PG_s104
                                 + 0.0503975067276375 w_k1_PG_s105
                                 + 0.0474935870888902 w_k1_PG_s106
                                 + 0.0597951222608588 w_k1_PG_s107
                                 + 0.0261551841975827 w_k1_PG_s110
                                 + 0.0142540135338671 w_k1_PG_s111
                                 + 0.0436022039914305 w_k1_PG_s112
                                 + 0.0544581020005609 w_k1_PG_s113
                                 + 0.0413660652394899 w_k1_PG_s114
                                 + 0.0265819391996971 w_k1_PG_s115
                                 + 0.0376670016738863 w_k1_PG_s116
                                 + 0.0686920441507568 w_k1_PG_s117 >= 1.07
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
