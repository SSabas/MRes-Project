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
 w_k1_ADM_s0_return_constraint:  1.11148 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.908318 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.889717 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.08879 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.891349 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.06322 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.04024 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.15048 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.749707 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.923456 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.958654 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.896572 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.16773 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.24652 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.79996 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.970844 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.774705 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.735411 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.988058 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.18101 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.831356 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.11177 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.35655 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.1081 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.920139 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.938375 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.20577 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.817729 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.03825 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.44581 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.17025 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.11637 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.31884 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.845583 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.825849 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.28697 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.27361 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.991492 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.13791 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.888922 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.759436 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.25136 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.862301 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.43216 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.01833 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.13637 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.7287 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.776182 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.13719 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.09826 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.4517 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.38439 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.80478 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.975808 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.836178 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.637359 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.07304 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.29928 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.767194 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.4694 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.70445 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.07666 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.760743 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.750574 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.18792 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.654869 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.94004 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.60518 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.12856 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.898471 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.41972 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.842225 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.596525 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.30786 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.2717 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.740081 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.22165 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.85888 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.692483 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.23668 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.816817 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.55048 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.973343 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.74942 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.703609 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.774423 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.05119 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.24564 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.60964 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.29337 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.770061 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.762626 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.770881 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.540881 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.19428 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.57921 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.755561 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.46095 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.7153 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.19746 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.654289 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.796896 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.27512 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.672676 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.93832 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.74506 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.06867 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.887096 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.55561 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.824969 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.580777 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.4505 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.21677 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.818221 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15214 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.856214 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.822933 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.10757 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.961398 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.2763 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.07635 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.26231 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.796399 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.779395 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.98405 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.984377 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.21011 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.28585 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.825422 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.959898 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.763443 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.671027 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.12577 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.47353 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.9256 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.21121 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.30741 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.931473 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.778869 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.768454 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.18011 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.669965 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.02204 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.48039 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.04644 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.993429 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.3383 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.780385 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.683457 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.3966 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.1597 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.822857 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.03278 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.05136 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.785695 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.20716 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.885852 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.46345 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.738555 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.11359 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.561715 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.988466 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.17697 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.84039 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.47736 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.97053 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.07848 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.85086 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.45129 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.794165 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.993314 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.831758 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.5849 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.65076 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.78656 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.32099 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.573504 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.971076 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.779349 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.734523 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.590373 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.04174 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.05768 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.72162 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.21664 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.42633 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.639034 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.806773 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.55311 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.969359 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.1691 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.90707 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.834461 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.03177 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.926469 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.18192 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.970672 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.14403 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.772493 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.824188 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.02093 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.09751 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.19318 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.04767 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.824222 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.893932 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.787274 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.809075 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.06418 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.24977 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.831013 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.37845 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.53037 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.980238 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.840138 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.87616 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.0951 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.725541 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.08911 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.41657 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.999709 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.08298 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.30666 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.886301 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.794357 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.16637 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.21099 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.900154 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.11251 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.925554 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.787341 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.0604 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.917376 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.39807 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.816796 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.33955 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.627351 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.851878 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.09242 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.32029 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.44999 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.07726 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.01345 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.899378 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.15953 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.817146 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.0938 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.23335 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.74973 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.51395 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.63663 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.24026 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.582417 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.93441 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.986162 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.769012 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.744225 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.3377 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.12437 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.89847 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.2966 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.956954 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.693889 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.0723 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.29283 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.899021 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.11343 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.01213 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.796558 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.08548 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.907657 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.6481 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.669861 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.19655 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.456061 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.781577 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.19426 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.70759 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.35729 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.05911 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.4962 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.01441 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.61112 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.743701 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.22805 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.0799 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.633758 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 2.08793 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.74473 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.26363 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.601017 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.721155 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.94118 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.631281 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.607126 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.24177 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.07324 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.693855 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.09921 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.05188 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.442432 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.8824 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.51391 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.802986 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12701 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.876286 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.81844 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.01702 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.961291 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.22218 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.00769 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.27849 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.853323 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.838034 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.04385 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.07504 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.42882 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.22908 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.914677 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.96163 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.837603 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.841388 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.22375 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.44824 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.930921 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.183 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.26216 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.00021 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.785323 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.868014 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.06765 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.755621 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.00257 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.37317 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.06247 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.04096 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.39224 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.873317 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.704912 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.33805 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.25553 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.925646 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08817 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.960719 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.815311 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.08817 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.945766 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.17869 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.05817 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.2801 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.875457 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.932579 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.10885 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.14706 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.30823 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.10358 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.917953 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.907399 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.913925 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.772821 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.04112 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11595 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.823311 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.13944 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.25666 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.05229 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.783454 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.848182 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.972878 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.773077 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.90426 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.14161 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.03155 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.850728 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.26797 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.928329 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.760944 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.995864 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.09998 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.892834 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.303000283493596 z_k1_s0000
                                 - 0.457000986259546 z_k1_s0001
                                 - 0.733001098748703 z_k1_s0002
                                 - 0.726001735895978 z_k1_s0003
                                 - 0.301000202678425 z_k1_s0004
                                 - 0.408001766288971 z_k1_s0005
                                 - 0.212000102404721 z_k1_s0006
                                 - 0.539999740087285 z_k1_s0007
                                 - 0.197999051133834 z_k1_s0010
                                 - 0.104999916540108 z_k1_s0011
                                 - 0.182000438535407 z_k1_s0012
                                 - 0.209999833080216 z_k1_s0013
                                 - 0.172000029913097 z_k1_s0014
                                 - 0.338999126516014 z_k1_s0015
                                 - 0.0919998227305196 z_k1_s0016
                                 - 0.159999247966715 z_k1_s0017
                                 - 0.238999059136439 z_k1_s0020
                                 - 0.322999025853436 z_k1_s0021
                                 - 0.116999614564385 z_k1_s0022
                                 - 0.377998438932415 z_k1_s0023
                                 - 0.295999687213098 z_k1_s0024
                                 - 0.457998285044354 z_k1_s0025
                                 - 0.571997488406292 z_k1_s0026
                                 - 0.182999166858081 z_k1_s0027
                                 - 1.16400037867596 z_k1_s0030
                                 - 0.567000374313592 z_k1_s0031
                                 - 0.332999576737282 z_k1_s0032
                                 - 1.08500043208189 z_k1_s0033
                                 - 0.508999247954703 z_k1_s0034
                                 - 0.310999848034844 z_k1_s0035
                                 - 0.325999663059234 z_k1_s0036
                                 - 0.308999872698259 z_k1_s0037
                                 - 0.900000804725107 z_k1_s0040
                                 - 0.389000424516855 z_k1_s0041
                                 - 0.431001600412398 z_k1_s0042
                                 - 0.389999501012943 z_k1_s0043
                                 - 0.324000507681365 z_k1_s0044
                                 - 0.484998960271879 z_k1_s0045
                                 - 0.408001042969731 z_k1_s0046
                                 - 0.306000418926754 z_k1_s0047
                                 - 0.446000424034346 z_k1_s0050
                                 - 0.264999615671182 z_k1_s0051
                                 - 0.415998822997522 z_k1_s0052
                                 - 0.30299975178613 z_k1_s0053
                                 - 0.102999888864615 z_k1_s0054
                                 - 0.279999402690471 z_k1_s0055
                                 - 0.104999928033795 z_k1_s0056
                                 - 0.109000006372155 z_k1_s0057
                                 - 0.201000217544142 z_k1_s0060
                                 - 0.0889999314917752 z_k1_s0061
                                 - 0.0809999738880002 z_k1_s0062
                                 - 0.286000316833949 z_k1_s0063
                                 - 0.215000509850547 z_k1_s0064
                                 - 0.216000321301119 z_k1_s0065
                                 - 0.12899986611057 z_k1_s0066
                                 - 0.248999963166792 z_k1_s0067
                                 - 0.165999990585515 z_k1_s0070
                                 - 0.26999873273025 z_k1_s0071
                                 - 0.479999648741454 z_k1_s0072
                                 - 0.19799988566833 z_k1_s0073
                                 - 0.341999046791234 z_k1_s0074
                                 - 0.461999570226208 z_k1_s0075
                                 - 0.131999842278939 z_k1_s0076
                                 - 0.394999277975014 z_k1_s0077
                                 - 0.603002932891039 z_k1_s0100
                                 - 0.407999963150562 z_k1_s0101
                                 - 0.379000689807183 z_k1_s0102
                                 - 0.522001348813434 z_k1_s0103
                                 - 0.319000739896908 z_k1_s0104
                                 - 0.551002900361962 z_k1_s0105
                                 - 0.512000028208717 z_k1_s0106
                                 - 0.503001117869622 z_k1_s0107
                                 - 0.59900423377114 z_k1_s0110
                                 - 0.857003551879708 z_k1_s0111
                                 - 0.4830038560142 z_k1_s0112
                                 - 0.933004103790629 z_k1_s0113
                                 - 0.340001168101617 z_k1_s0114
                                 - 0.366001496303271 z_k1_s0115
                                 - 0.404001772258732 z_k1_s0116
                                 - 0.432001650350945 z_k1_s0117
                                 - 0.147999962091488 z_k1_s0120
                                 - 0.189000650184674 z_k1_s0121
                                 - 0.210000722427416 z_k1_s0122
                                 - 0.198000065717036 z_k1_s0123
                                 - 0.0969999712734737 z_k1_s0124
                                 - 0.0730000938532781 z_k1_s0125
                                 - 0.181000691044609 z_k1_s0126
                                 - 0.340000417453616 z_k1_s0127
                                 - 0.10700005051967 z_k1_s0130
                                 - 0.284001252873366 z_k1_s0131
                                 - 0.206000108061239 z_k1_s0132
                                 - 0.193000032809279 z_k1_s0133
                                 - 0.30299999403319 z_k1_s0134
                                 - 0.44099912183002 z_k1_s0135
                                 - 0.349000473632545 z_k1_s0136
                                 - 0.427999739878431 z_k1_s0137
                                 - 1.03300358262196 z_k1_s0140
                                 - 0.550001126622322 z_k1_s0141
                                 - 0.68200134035558 z_k1_s0142
                                 - 1.11100486779365 z_k1_s0143
                                 - 1.4680020336026 z_k1_s0144
                                 - 0.623001094598784 z_k1_s0145
                                 - 1.02700511807316 z_k1_s0146
                                 - 0.588001080848374 z_k1_s0147
                                 - 0.216000289485825 z_k1_s0150
                                 - 0.789002499842247 z_k1_s0151
                                 - 0.809999570194363 z_k1_s0152
                                 - 0.721000440587934 z_k1_s0153
                                 - 0.410998659411599 z_k1_s0154
                                 - 0.325000381314693 z_k1_s0155
                                 - 0.420002022141491 z_k1_s0156
                                 - 0.349000727780299 z_k1_s0157
                                 - 1.87499748141443 z_k1_s0160
                                 - 0.753997936196246 z_k1_s0161
                                 - 0.55499984384253 z_k1_s0162
                                 - 0.584999901769137 z_k1_s0163
                                 - 0.40099999667917 z_k1_s0164
                                 - 0.413999939927387 z_k1_s0165
                                 - 0.420999626030341 z_k1_s0166
                                 - 1.13400194406581 z_k1_s0167
                                 - 0.504999387950727 z_k1_s0170
                                 - 0.584001448641705 z_k1_s0171
                                 - 0.288000221247585 z_k1_s0172
                                 - 0.559998430202102 z_k1_s0173
                                 - 0.375000138082678 z_k1_s0174
                                 - 1.2389994518244 z_k1_s0175
                                 - 0.693000532377003 z_k1_s0176
                                 - 0.256000246664559 z_k1_s0177
                                 - 0.33099946045124 z_k1_s1000
                                 - 0.310999357813577 z_k1_s1001
                                 - 0.192000278520908 z_k1_s1002
                                 - 0.252000350833678 z_k1_s1003
                                 - 0.164000134828179 z_k1_s1004
                                 - 0.363001038272818 z_k1_s1005
                                 - 0.501001510872475 z_k1_s1006
                                 - 0.241999121513748 z_k1_s1007
                                 - 0.951000748394926 z_k1_s1010
                                 - 0.7729992299944 z_k1_s1011
                                 - 0.689999288168386 z_k1_s1012
                                 - 0.753998535840017 z_k1_s1013
                                 - 0.887998795656231 z_k1_s1014
                                 - 2.03999377763392 z_k1_s1015
                                 - 0.906998542711912 z_k1_s1016
                                 - 2.46799715193893 z_k1_s1017
                                 - 0.278000106292954 z_k1_s1020
                                 - 0.157000055710768 z_k1_s1021
                                 - 0.182000192650094 z_k1_s1022
                                 - 0.390001055893222 z_k1_s1023
                                 - 0.522000074364435 z_k1_s1024
                                 - 0.755000774446814 z_k1_s1025
                                 - 0.463999732657191 z_k1_s1026
                                 - 0.252999969353627 z_k1_s1027
                                 - 0.516003009139114 z_k1_s1030
                                 - 0.877004394008663 z_k1_s1031
                                 - 0.541003553248272 z_k1_s1032
                                 - 0.571002456972047 z_k1_s1033
                                 - 0.400001708917671 z_k1_s1034
                                 - 0.333001387689817 z_k1_s1035
                                 - 0.434001924143961 z_k1_s1036
                                 - 0.701003537132455 z_k1_s1037
                                 - 0.87500109800787 z_k1_s1040
                                 - 0.761999855801001 z_k1_s1041
                                 - 0.45400009627831 z_k1_s1042
                                 - 0.691000412874729 z_k1_s1043
                                 - 0.732000170029347 z_k1_s1044
                                 - 0.765000384178237 z_k1_s1045
                                 - 0.878998070510382 z_k1_s1046
                                 - 0.440000056318181 z_k1_s1047
                                 - 0.224999403110248 z_k1_s1050
                                 - 0.325999889497853 z_k1_s1051
                                 - 0.218000709965174 z_k1_s1052
                                 - 0.114999970378695 z_k1_s1053
                                 - 0.234999715604189 z_k1_s1054
                                 - 0.120000126625665 z_k1_s1055
                                 - 0.356000826979677 z_k1_s1056
                                 - 0.31200059620685 z_k1_s1057
                                 - 0.504996264818981 z_k1_s1060
                                 - 0.249999103832346 z_k1_s1061
                                 - 0.419000171282367 z_k1_s1062
                                 - 0.942994745665049 z_k1_s1063
                                 - 0.302998882386928 z_k1_s1064
                                 - 0.480998400106939 z_k1_s1065
                                 - 0.269998822419766 z_k1_s1066
                                 - 0.573997615836362 z_k1_s1067
                                 - 0.391001275022874 z_k1_s1070
                                 - 0.319000061207733 z_k1_s1071
                                 - 0.401000476941458 z_k1_s1072
                                 - 0.403999077516811 z_k1_s1073
                                 - 0.236999935474062 z_k1_s1074
                                 - 0.218000001828474 z_k1_s1075
                                 - 0.43399958327312 z_k1_s1076
                                 - 0.495998695169121 z_k1_s1077
                                 - 0.31899821737579 z_k1_s1100
                                 - 0.22299839971908 z_k1_s1101
                                 - 0.134999120614226 z_k1_s1102
                                 - 0.223999131534619 z_k1_s1103
                                 - 0.147999331264437 z_k1_s1104
                                 - 0.0869996104838057 z_k1_s1105
                                 - 0.0869996104838057 z_k1_s1106
                                 - 0.10599942765204 z_k1_s1107
                                 - 0.293998342903059 z_k1_s1110
                                 - 0.601998114416093 z_k1_s1111
                                 - 0.321998322131517 z_k1_s1112
                                 - 0.431998034958208 z_k1_s1113
                                 - 0.645996848350173 z_k1_s1114
                                 - 0.180999475142256 z_k1_s1115
                                 - 0.137999692055149 z_k1_s1116
                                 - 0.262999208740489 z_k1_s1117
                                 - 0.126999771774805 z_k1_s1120
                                 - 0.223999486343106 z_k1_s1121
                                 - 0.0999998210888215 z_k1_s1122
                                 - 0.0479998899306656 z_k1_s1123
                                 - 0.111999743171553 z_k1_s1124
                                 - 0.0539999013719663 z_k1_s1125
                                 - 0.128999641188745 z_k1_s1126
                                 - 0.214000139273403 z_k1_s1127
                                 - 0.147999043235448 z_k1_s1130
                                 - 0.0839996599814928 z_k1_s1131
                                 - 0.135999489236548 z_k1_s1132
                                 - 0.0959996312787717 z_k1_s1133
                                 - 0.223998815085061 z_k1_s1134
                                 - 0.11399951867496 z_k1_s1135
                                 - 0.279998866604976 z_k1_s1136
                                 - 0.308998321983579 z_k1_s1137
                                 - 0.0529999445560693 z_k1_s1140
                                 - 0.14000020958183 z_k1_s1141
                                 - 0.0419999999745913 z_k1_s1142
                                 - 0.0519999610304727 z_k1_s1143
                                 - 0.0959998022563421 z_k1_s1144
                                 - 0.11899980074481 z_k1_s1145
                                 - 0.0309999924931558 z_k1_s1146
                                 - 0.0959998022563421 z_k1_s1147
                                 - 0.0779999200700019 z_k1_s1150
                                 - 0.0839999649368535 z_k1_s1151
                                 - 0.0649998919417265 z_k1_s1152
                                 - 0.226999776948582 z_k1_s1153
                                 - 0.0799998797590301 z_k1_s1154
                                 - 0.112000229582083 z_k1_s1155
                                 - 0.107999481205192 z_k1_s1156
                                 - 0.0749998561866344 z_k1_s1157
                                 - 0.251000841967778 z_k1_s1160
                                 - 0.248001265423581 z_k1_s1161
                                 - 0.174000348634364 z_k1_s1162
                                 - 0.1480005225747 z_k1_s1163
                                 - 0.116000363523283 z_k1_s1164
                                 - 0.157000563211025 z_k1_s1165
                                 - 0.0920002114596259 z_k1_s1166
                                 - 0.125000404159607 z_k1_s1167
                                 - 0.177000747021423 z_k1_s1170
                                 - 0.336000254090548 z_k1_s1171
                                 - 0.0690000073132003 z_k1_s1172
                                 - 0.300999425017949 z_k1_s1173
                                 - 0.262000367624318 z_k1_s1174
                                 - 0.13800018212654 z_k1_s1175
                                 - 0.11900013884461 z_k1_s1176
                                 - 0.273000101760983 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.656494 w_k1_ADM_s000
                                 + 0.696586 w_k1_AXP_s000
                                 + 0.59447 w_k1_DIS_s000
                                 + 0.722131 w_k1_FDX_s000
                                 + 0.786883 w_k1_IBM_s000
                                 + 0.770937 w_k1_INTC_s000
                                 + 0.778326 w_k1_KO_s000
                                 + 1.16285 w_k1_MAT_s000
                                 + 0.754631 w_k1_MSFT_s000
                                 + 0.771159 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.803711 w_k1_ADM_s000
                                 + 0.771689 w_k1_AXP_s000
                                 + 0.79403 w_k1_DIS_s000
                                 + 0.883846 w_k1_FDX_s000
                                 + 0.767893 w_k1_IBM_s000
                                 + 0.991373 w_k1_INTC_s000
                                 + 0.817952 w_k1_KO_s000
                                 + 0.83564 w_k1_MAT_s000
                                 + 0.965848 w_k1_MSFT_s000
                                 + 0.921235 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.10092 w_k1_ADM_s000
                                 + 1.06015 w_k1_AXP_s000
                                 + 1.33648 w_k1_DIS_s000
                                 + 1.16884 w_k1_FDX_s000
                                 + 0.962159 w_k1_IBM_s000
                                 + 1.30652 w_k1_INTC_s000
                                 + 1.19749 w_k1_KO_s000 + 1.04724 w_k1_MAT_s000
                                 + 1.18236 w_k1_MSFT_s000
                                 + 1.03904 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.11553 w_k1_ADM_s000
                                 + 0.99614 w_k1_AXP_s000
                                 + 0.877403 w_k1_DIS_s000
                                 + 0.981144 w_k1_FDX_s000
                                 + 0.965791 w_k1_IBM_s000
                                 + 0.78699 w_k1_INTC_s000
                                 + 0.947679 w_k1_KO_s000
                                 + 1.01874 w_k1_MAT_s000
                                 + 0.820004 w_k1_MSFT_s000
                                 + 0.912705 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.32168 w_k1_ADM_s000
                                 + 1.62497 w_k1_AXP_s000
                                 + 1.51015 w_k1_DIS_s000
                                 + 1.32201 w_k1_FDX_s000
                                 + 1.28683 w_k1_IBM_s000
                                 + 1.26514 w_k1_INTC_s000
                                 + 1.44147 w_k1_KO_s000 + 1.8849 w_k1_MAT_s000
                                 + 1.32442 w_k1_MSFT_s000
                                 + 1.16691 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.03567 w_k1_ADM_s000
                                 + 1.33559 w_k1_AXP_s000
                                 + 1.12087 w_k1_DIS_s000
                                 + 1.02514 w_k1_FDX_s000
                                 + 1.28097 w_k1_IBM_s000
                                 + 1.19162 w_k1_INTC_s000
                                 + 1.17096 w_k1_KO_s000 + 1.06787 w_k1_MAT_s000
                                 + 1.03675 w_k1_MSFT_s000
                                 + 0.899183 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.883501 w_k1_ADM_s000
                                 + 1.01177 w_k1_AXP_s000
                                 + 1.51878 w_k1_DIS_s000
                                 + 0.997766 w_k1_FDX_s000
                                 + 1.75967 w_k1_IBM_s000
                                 + 1.06742 w_k1_INTC_s000
                                 + 1.52656 w_k1_KO_s000 + 1.36356 w_k1_MAT_s000
                                 + 1.11963 w_k1_MSFT_s000
                                 + 1.08307 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.959158 w_k1_ADM_s000
                                 + 0.931753 w_k1_AXP_s000
                                 + 1.0394 w_k1_DIS_s000
                                 + 0.927458 w_k1_FDX_s000
                                 + 0.958503 w_k1_IBM_s000
                                 + 1.02344 w_k1_INTC_s000
                                 + 1.15284 w_k1_KO_s000
                                 + 0.908492 w_k1_MAT_s000
                                 + 1.05443 w_k1_MSFT_s000
                                 + 0.888992 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.38444 w_k1_ADM_s001
                                 + 1.49173 w_k1_AXP_s001
                                 + 1.41647 w_k1_DIS_s001
                                 + 1.34114 w_k1_FDX_s001
                                 + 1.14432 w_k1_IBM_s001
                                 + 1.37893 w_k1_INTC_s001
                                 + 1.14533 w_k1_KO_s001
                                 + 0.733105 w_k1_MAT_s001
                                 + 1.22223 w_k1_MSFT_s001
                                 + 1.06373 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.12477 w_k1_ADM_s001
                                 + 1.31539 w_k1_AXP_s001
                                 + 1.09724 w_k1_DIS_s001
                                 + 1.29257 w_k1_FDX_s001
                                 + 0.905523 w_k1_IBM_s001
                                 + 1.16834 w_k1_INTC_s001
                                 + 0.947023 w_k1_KO_s001
                                 + 0.751432 w_k1_MAT_s001
                                 + 1.1615 w_k1_MSFT_s001 + 1.177 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.874647 w_k1_ADM_s001
                                 + 1.03051 w_k1_AXP_s001
                                 + 0.76141 w_k1_DIS_s001
                                 + 0.907472 w_k1_FDX_s001
                                 + 1.0468 w_k1_IBM_s001
                                 + 0.828706 w_k1_INTC_s001
                                 + 0.852477 w_k1_KO_s001
                                 + 0.985748 w_k1_MAT_s001
                                 + 0.828716 w_k1_MSFT_s001
                                 + 0.863576 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.938457 w_k1_ADM_s001
                                 + 0.985998 w_k1_AXP_s001
                                 + 0.919493 w_k1_DIS_s001
                                 + 0.917124 w_k1_FDX_s001
                                 + 1.02749 w_k1_IBM_s001
                                 + 0.93589 w_k1_INTC_s001
                                 + 0.889317 w_k1_KO_s001
                                 + 0.648373 w_k1_MAT_s001
                                 + 0.952936 w_k1_MSFT_s001
                                 + 0.884128 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.00739 w_k1_ADM_s001
                                 + 0.90743 w_k1_AXP_s001
                                 + 1.16874 w_k1_DIS_s001
                                 + 1.03009 w_k1_FDX_s001
                                 + 0.850824 w_k1_IBM_s001
                                 + 1.20592 w_k1_INTC_s001
                                 + 1.09336 w_k1_KO_s001
                                 + 0.455125 w_k1_MAT_s001
                                 + 1.31605 w_k1_MSFT_s001
                                 + 1.03958 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.07379 w_k1_ADM_s001
                                 + 0.805107 w_k1_AXP_s001
                                 + 0.756189 w_k1_DIS_s001
                                 + 0.905199 w_k1_FDX_s001
                                 + 0.716819 w_k1_IBM_s001
                                 + 0.91838 w_k1_INTC_s001
                                 + 0.781868 w_k1_KO_s001
                                 + 0.769636 w_k1_MAT_s001
                                 + 0.924898 w_k1_MSFT_s001
                                 + 0.833937 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.56501 w_k1_ADM_s001
                                 + 1.81284 w_k1_AXP_s001
                                 + 2.02253 w_k1_DIS_s001 + 1.5983 w_k1_FDX_s001
                                 + 1.36658 w_k1_IBM_s001
                                 + 1.34716 w_k1_INTC_s001
                                 + 1.33415 w_k1_KO_s001 + 1.25811 w_k1_MAT_s001
                                 + 1.57358 w_k1_MSFT_s001
                                 + 1.25479 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.25073 w_k1_ADM_s001
                                 + 1.43863 w_k1_AXP_s001
                                 + 1.10462 w_k1_DIS_s001
                                 + 1.06111 w_k1_FDX_s001
                                 + 1.37678 w_k1_IBM_s001
                                 + 1.06192 w_k1_INTC_s001
                                 + 1.22408 w_k1_KO_s001 + 1.30663 w_k1_MAT_s001
                                 + 1.05911 w_k1_MSFT_s001
                                 + 1.03327 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.771248 w_k1_ADM_s002
                                 + 0.647278 w_k1_AXP_s002
                                 + 0.564257 w_k1_DIS_s002
                                 + 0.709262 w_k1_FDX_s002
                                 + 0.945264 w_k1_IBM_s002
                                 + 0.729942 w_k1_INTC_s002
                                 + 0.777482 w_k1_KO_s002
                                 + 0.784079 w_k1_MAT_s002
                                 + 0.699034 w_k1_MSFT_s002
                                 + 0.780224 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.07654 w_k1_ADM_s002
                                 + 0.875605 w_k1_AXP_s002
                                 + 0.840169 w_k1_DIS_s002
                                 + 0.898136 w_k1_FDX_s002
                                 + 1.16078 w_k1_IBM_s002
                                 + 0.974168 w_k1_INTC_s002
                                 + 1.12122 w_k1_KO_s002
                                 + 0.996782 w_k1_MAT_s002
                                 + 0.878041 w_k1_MSFT_s002
                                 + 0.888991 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.10663 w_k1_ADM_s002
                                 + 1.0945 w_k1_AXP_s002
                                 + 0.613957 w_k1_DIS_s002
                                 + 0.783724 w_k1_FDX_s002
                                 + 1.09267 w_k1_IBM_s002
                                 + 0.848011 w_k1_INTC_s002
                                 + 0.908609 w_k1_KO_s002 + 1.0103 w_k1_MAT_s002
                                 + 0.753725 w_k1_MSFT_s002
                                 + 0.950608 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.20351 w_k1_ADM_s002
                                 + 0.982327 w_k1_AXP_s002
                                 + 1.06277 w_k1_DIS_s002
                                 + 1.11951 w_k1_FDX_s002
                                 + 0.941235 w_k1_IBM_s002
                                 + 0.891769 w_k1_INTC_s002
                                 + 1.07529 w_k1_KO_s002
                                 + 0.970551 w_k1_MAT_s002
                                 + 0.871964 w_k1_MSFT_s002
                                 + 0.979289 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.15893 w_k1_ADM_s002
                                 + 1.30964 w_k1_AXP_s002
                                 + 1.20152 w_k1_DIS_s002 + 1.2941 w_k1_FDX_s002
                                 + 0.965177 w_k1_IBM_s002
                                 + 1.38679 w_k1_INTC_s002
                                 + 1.07095 w_k1_KO_s002 + 0.88357 w_k1_MAT_s002
                                 + 1.1146 w_k1_MSFT_s002 + 1.08619 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.21872 w_k1_ADM_s002
                                 + 1.35363 w_k1_AXP_s002
                                 + 1.64029 w_k1_DIS_s002
                                 + 1.26511 w_k1_FDX_s002
                                 + 1.47122 w_k1_IBM_s002
                                 + 1.51069 w_k1_INTC_s002
                                 + 1.62594 w_k1_KO_s002 + 1.86395 w_k1_MAT_s002
                                 + 1.33183 w_k1_MSFT_s002
                                 + 1.20488 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.925849 w_k1_ADM_s002
                                 + 0.906432 w_k1_AXP_s002
                                 + 0.984539 w_k1_DIS_s002
                                 + 0.943815 w_k1_FDX_s002
                                 + 1.23393 w_k1_IBM_s002
                                 + 1.07643 w_k1_INTC_s002 + 1.1912 w_k1_KO_s002
                                 + 1.51944 w_k1_MAT_s002
                                 + 1.05702 w_k1_MSFT_s002
                                 + 1.02238 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.76649 w_k1_ADM_s002
                                 + 0.480603 w_k1_AXP_s002
                                 + 0.598188 w_k1_DIS_s002
                                 + 0.767024 w_k1_FDX_s002
                                 + 0.664945 w_k1_IBM_s002
                                 + 0.922815 w_k1_INTC_s002
                                 + 0.758241 w_k1_KO_s002
                                 + 0.406191 w_k1_MAT_s002
                                 + 0.921639 w_k1_MSFT_s002
                                 + 0.815419 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.836189 w_k1_ADM_s003
                                 + 0.927182 w_k1_AXP_s003
                                 + 0.936922 w_k1_DIS_s003
                                 + 0.914873 w_k1_FDX_s003
                                 + 0.916319 w_k1_IBM_s003
                                 + 0.954802 w_k1_INTC_s003
                                 + 0.97716 w_k1_KO_s003
                                 + 0.887806 w_k1_MAT_s003
                                 + 0.930713 w_k1_MSFT_s003
                                 + 0.946011 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.08613 w_k1_ADM_s003
                                 + 1.30219 w_k1_AXP_s003
                                 + 1.63738 w_k1_DIS_s003
                                 + 1.24325 w_k1_FDX_s003
                                 + 1.31637 w_k1_IBM_s003
                                 + 1.25468 w_k1_INTC_s003
                                 + 1.47346 w_k1_KO_s003 + 1.43728 w_k1_MAT_s003
                                 + 1.30517 w_k1_MSFT_s003 + 1.0923 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.876005 w_k1_ADM_s003
                                 + 1.39354 w_k1_AXP_s003
                                 + 1.42187 w_k1_DIS_s003
                                 + 1.09186 w_k1_FDX_s003
                                 + 1.68901 w_k1_IBM_s003
                                 + 0.982232 w_k1_INTC_s003
                                 + 1.26642 w_k1_KO_s003 + 1.74083 w_k1_MAT_s003
                                 + 1.0002 w_k1_MSFT_s003 + 1.06656 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.0549 w_k1_ADM_s003
                                 + 1.13062 w_k1_AXP_s003
                                 + 1.11254 w_k1_DIS_s003
                                 + 1.19961 w_k1_FDX_s003
                                 + 1.01649 w_k1_IBM_s003
                                 + 0.938062 w_k1_INTC_s003
                                 + 1.0518 w_k1_KO_s003 + 1.04128 w_k1_MAT_s003
                                 + 1.14799 w_k1_MSFT_s003
                                 + 1.06915 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.898775 w_k1_ADM_s003
                                 + 1.07501 w_k1_AXP_s003
                                 + 0.869041 w_k1_DIS_s003
                                 + 1.01099 w_k1_FDX_s003
                                 + 0.628214 w_k1_IBM_s003
                                 + 0.932711 w_k1_INTC_s003
                                 + 0.918338 w_k1_KO_s003
                                 + 0.803405 w_k1_MAT_s003
                                 + 0.991259 w_k1_MSFT_s003
                                 + 0.95703 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.633568 w_k1_ADM_s003
                                 + 0.625925 w_k1_AXP_s003
                                 + 0.552211 w_k1_DIS_s003
                                 + 0.675961 w_k1_FDX_s003
                                 + 0.733199 w_k1_IBM_s003
                                 + 0.600897 w_k1_INTC_s003
                                 + 0.642223 w_k1_KO_s003
                                 + 0.66051 w_k1_MAT_s003
                                 + 0.796779 w_k1_MSFT_s003
                                 + 0.723849 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.659238 w_k1_ADM_s003
                                 + 0.852038 w_k1_AXP_s003
                                 + 0.802605 w_k1_DIS_s003
                                 + 0.805401 w_k1_FDX_s003
                                 + 1.45106 w_k1_IBM_s003
                                 + 1.01957 w_k1_INTC_s003
                                 + 1.01923 w_k1_KO_s003 + 1.19771 w_k1_MAT_s003
                                 + 0.880735 w_k1_MSFT_s003
                                 + 0.866424 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.861361 w_k1_ADM_s003
                                 + 1.198 w_k1_AXP_s003 + 0.759784 w_k1_DIS_s003
                                 + 0.979244 w_k1_FDX_s003
                                 + 1.19081 w_k1_IBM_s003
                                 + 1.02513 w_k1_INTC_s003 + 1.0059 w_k1_KO_s003
                                 + 1.43224 w_k1_MAT_s003
                                 + 0.996404 w_k1_MSFT_s003
                                 + 0.97286 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.988379 w_k1_ADM_s004
                                 + 1.14281 w_k1_AXP_s004
                                 + 1.08782 w_k1_DIS_s004
                                 + 0.985129 w_k1_FDX_s004
                                 + 1.31165 w_k1_IBM_s004
                                 + 0.946723 w_k1_INTC_s004
                                 + 1.02511 w_k1_KO_s004 + 1.24318 w_k1_MAT_s004
                                 + 0.971733 w_k1_MSFT_s004
                                 + 1.04974 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.12851 w_k1_ADM_s004
                                 + 0.916212 w_k1_AXP_s004
                                 + 1.52547 w_k1_DIS_s004
                                 + 1.25477 w_k1_FDX_s004
                                 + 1.10089 w_k1_IBM_s004
                                 + 1.10717 w_k1_INTC_s004 + 1.264 w_k1_KO_s004
                                 + 0.974973 w_k1_MAT_s004
                                 + 1.28554 w_k1_MSFT_s004
                                 + 1.11248 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.808914 w_k1_ADM_s004
                                 + 0.968557 w_k1_AXP_s004
                                 + 0.857547 w_k1_DIS_s004
                                 + 0.939331 w_k1_FDX_s004
                                 + 0.889899 w_k1_IBM_s004
                                 + 0.9775 w_k1_INTC_s004
                                 + 0.853606 w_k1_KO_s004
                                 + 0.893959 w_k1_MAT_s004
                                 + 0.87564 w_k1_MSFT_s004
                                 + 0.930135 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.842067 w_k1_ADM_s004
                                 + 0.783482 w_k1_AXP_s004
                                 + 1.0389 w_k1_DIS_s004
                                 + 0.912982 w_k1_FDX_s004
                                 + 1.05123 w_k1_IBM_s004
                                 + 1.01192 w_k1_INTC_s004
                                 + 1.00439 w_k1_KO_s004
                                 + 0.830102 w_k1_MAT_s004
                                 + 1.03506 w_k1_MSFT_s004
                                 + 0.965942 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.661942 w_k1_ADM_s004
                                 + 0.723715 w_k1_AXP_s004
                                 + 0.75235 w_k1_DIS_s004
                                 + 0.753886 w_k1_FDX_s004
                                 + 1.17946 w_k1_IBM_s004
                                 + 0.776405 w_k1_INTC_s004
                                 + 0.915995 w_k1_KO_s004
                                 + 1.06879 w_k1_MAT_s004
                                 + 0.782362 w_k1_MSFT_s004
                                 + 0.811821 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.29329 w_k1_ADM_s004
                                 + 1.01106 w_k1_AXP_s004
                                 + 1.06925 w_k1_DIS_s004
                                 + 1.07672 w_k1_FDX_s004
                                 + 0.80327 w_k1_IBM_s004
                                 + 1.00823 w_k1_INTC_s004
                                 + 0.826761 w_k1_KO_s004
                                 + 0.605892 w_k1_MAT_s004
                                 + 0.950243 w_k1_MSFT_s004
                                 + 0.907637 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.782005 w_k1_ADM_s004
                                 + 0.68371 w_k1_AXP_s004
                                 + 0.637879 w_k1_DIS_s004
                                 + 0.828276 w_k1_FDX_s004
                                 + 0.73149 w_k1_IBM_s004
                                 + 0.838709 w_k1_INTC_s004
                                 + 0.696896 w_k1_KO_s004
                                 + 0.705425 w_k1_MAT_s004
                                 + 0.767985 w_k1_MSFT_s004
                                 + 0.818403 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.36023 w_k1_ADM_s004
                                 + 1.6569 w_k1_AXP_s004 + 1.49228 w_k1_DIS_s004
                                 + 1.34214 w_k1_FDX_s004
                                 + 1.31627 w_k1_IBM_s004
                                 + 1.07957 w_k1_INTC_s004
                                 + 1.20976 w_k1_KO_s004 + 1.313 w_k1_MAT_s004
                                 + 1.22075 w_k1_MSFT_s004
                                 + 1.13052 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.28006 w_k1_ADM_s005
                                 + 1.13037 w_k1_AXP_s005
                                 + 1.30159 w_k1_DIS_s005
                                 + 1.19727 w_k1_FDX_s005
                                 + 0.942527 w_k1_IBM_s005
                                 + 1.13831 w_k1_INTC_s005
                                 + 1.18291 w_k1_KO_s005 + 1.11194 w_k1_MAT_s005
                                 + 1.23836 w_k1_MSFT_s005
                                 + 1.20344 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.936404 w_k1_ADM_s005
                                 + 0.89095 w_k1_AXP_s005
                                 + 0.724389 w_k1_DIS_s005
                                 + 0.834445 w_k1_FDX_s005
                                 + 0.781742 w_k1_IBM_s005
                                 + 0.803147 w_k1_INTC_s005
                                 + 0.88418 w_k1_KO_s005 + 0.80285 w_k1_MAT_s005
                                 + 0.891657 w_k1_MSFT_s005
                                 + 0.949007 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.16753 w_k1_ADM_s005
                                 + 1.03504 w_k1_AXP_s005
                                 + 0.873277 w_k1_DIS_s005
                                 + 0.851664 w_k1_FDX_s005
                                 + 1.27236 w_k1_IBM_s005
                                 + 0.854082 w_k1_INTC_s005
                                 + 1.14715 w_k1_KO_s005 + 1.32871 w_k1_MAT_s005
                                 + 0.882715 w_k1_MSFT_s005
                                 + 1.00779 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.895277 w_k1_ADM_s005
                                 + 1.02278 w_k1_AXP_s005
                                 + 1.02593 w_k1_DIS_s005
                                 + 1.02625 w_k1_FDX_s005
                                 + 1.04332 w_k1_IBM_s005
                                 + 0.897269 w_k1_INTC_s005
                                 + 1.12815 w_k1_KO_s005 + 1.3081 w_k1_MAT_s005
                                 + 0.942676 w_k1_MSFT_s005
                                 + 1.02722 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.986908 w_k1_ADM_s005
                                 + 0.633441 w_k1_AXP_s005
                                 + 0.456159 w_k1_DIS_s005
                                 + 0.625506 w_k1_FDX_s005
                                 + 0.799263 w_k1_IBM_s005
                                 + 0.741517 w_k1_INTC_s005
                                 + 0.636373 w_k1_KO_s005
                                 + 0.469807 w_k1_MAT_s005
                                 + 0.699059 w_k1_MSFT_s005
                                 + 0.802398 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.22217 w_k1_ADM_s005
                                 + 1.66436 w_k1_AXP_s005
                                 + 1.58723 w_k1_DIS_s005
                                 + 1.33099 w_k1_FDX_s005
                                 + 1.53188 w_k1_IBM_s005
                                 + 1.18577 w_k1_INTC_s005
                                 + 1.61497 w_k1_KO_s005 + 1.91837 w_k1_MAT_s005
                                 + 1.20539 w_k1_MSFT_s005
                                 + 1.45026 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.22437 w_k1_ADM_s005
                                 + 1.34682 w_k1_AXP_s005
                                 + 0.929214 w_k1_DIS_s005
                                 + 1.10329 w_k1_FDX_s005
                                 + 0.835822 w_k1_IBM_s005
                                 + 0.974289 w_k1_INTC_s005
                                 + 0.979624 w_k1_KO_s005
                                 + 1.04439 w_k1_MAT_s005
                                 + 1.08344 w_k1_MSFT_s005
                                 + 1.02049 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.04168 w_k1_ADM_s005
                                 + 0.862455 w_k1_AXP_s005
                                 + 0.826814 w_k1_DIS_s005
                                 + 0.962735 w_k1_FDX_s005
                                 + 0.578753 w_k1_IBM_s005
                                 + 0.79144 w_k1_INTC_s005
                                 + 0.810883 w_k1_KO_s005
                                 + 0.515227 w_k1_MAT_s005
                                 + 0.954666 w_k1_MSFT_s005
                                 + 0.957411 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.702275 w_k1_ADM_s006
                                 + 0.588978 w_k1_AXP_s006
                                 + 0.653764 w_k1_DIS_s006
                                 + 0.78556 w_k1_FDX_s006
                                 + 0.689562 w_k1_IBM_s006
                                 + 0.837853 w_k1_INTC_s006
                                 + 0.667225 w_k1_KO_s006
                                 + 0.551576 w_k1_MAT_s006
                                 + 0.71255 w_k1_MSFT_s006
                                 + 0.791954 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.14871 w_k1_ADM_s006
                                 + 1.36768 w_k1_AXP_s006
                                 + 1.89984 w_k1_DIS_s006
                                 + 1.33431 w_k1_FDX_s006
                                 + 1.50809 w_k1_IBM_s006
                                 + 1.49394 w_k1_INTC_s006
                                 + 1.62546 w_k1_KO_s006 + 1.62881 w_k1_MAT_s006
                                 + 1.35276 w_k1_MSFT_s006
                                 + 1.28526 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.71298 w_k1_ADM_s006
                                 + 0.924479 w_k1_AXP_s006
                                 + 1.10868 w_k1_DIS_s006
                                 + 0.886197 w_k1_FDX_s006
                                 + 1.17267 w_k1_IBM_s006
                                 + 0.882178 w_k1_INTC_s006
                                 + 0.923113 w_k1_KO_s006
                                 + 0.959266 w_k1_MAT_s006
                                 + 0.820697 w_k1_MSFT_s006
                                 + 0.895646 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.05674 w_k1_ADM_s006
                                 + 0.911502 w_k1_AXP_s006
                                 + 0.977574 w_k1_DIS_s006
                                 + 1.07144 w_k1_FDX_s006
                                 + 1.07124 w_k1_IBM_s006
                                 + 0.991333 w_k1_INTC_s006
                                 + 1.08259 w_k1_KO_s006 + 1.22564 w_k1_MAT_s006
                                 + 0.810405 w_k1_MSFT_s006
                                 + 0.998298 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.902291 w_k1_ADM_s006
                                 + 0.668499 w_k1_AXP_s006
                                 + 0.982083 w_k1_DIS_s006
                                 + 0.974881 w_k1_FDX_s006
                                 + 0.776237 w_k1_IBM_s006
                                 + 0.834589 w_k1_INTC_s006
                                 + 0.872438 w_k1_KO_s006
                                 + 0.78339 w_k1_MAT_s006
                                 + 0.920694 w_k1_MSFT_s006
                                 + 0.964111 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.901966 w_k1_ADM_s006
                                 + 0.82431 w_k1_AXP_s006
                                 + 0.837535 w_k1_DIS_s006
                                 + 0.887945 w_k1_FDX_s006
                                 + 0.928536 w_k1_IBM_s006
                                 + 0.845015 w_k1_INTC_s006
                                 + 0.854581 w_k1_KO_s006
                                 + 0.876247 w_k1_MAT_s006
                                 + 0.737854 w_k1_MSFT_s006
                                 + 0.904575 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.16671 w_k1_ADM_s006
                                 + 1.45395 w_k1_AXP_s006
                                 + 1.43434 w_k1_DIS_s006
                                 + 1.43392 w_k1_FDX_s006
                                 + 1.05874 w_k1_IBM_s006
                                 + 1.24934 w_k1_INTC_s006 + 1.1024 w_k1_KO_s006
                                 + 1.10558 w_k1_MAT_s006
                                 + 1.18267 w_k1_MSFT_s006
                                 + 1.00259 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.901892 w_k1_ADM_s006
                                 + 1.10685 w_k1_AXP_s006
                                 + 1.07425 w_k1_DIS_s006
                                 + 1.11435 w_k1_FDX_s006
                                 + 1.37845 w_k1_IBM_s006
                                 + 1.09053 w_k1_INTC_s006
                                 + 1.10577 w_k1_KO_s006 + 1.50663 w_k1_MAT_s006
                                 + 1.0084 w_k1_MSFT_s006 + 1.0814 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.796117 w_k1_ADM_s007
                                 + 0.682705 w_k1_AXP_s007
                                 + 0.688734 w_k1_DIS_s007
                                 + 0.711754 w_k1_FDX_s007
                                 + 0.950838 w_k1_IBM_s007
                                 + 0.933317 w_k1_INTC_s007
                                 + 0.802522 w_k1_KO_s007
                                 + 0.620705 w_k1_MAT_s007
                                 + 0.745334 w_k1_MSFT_s007
                                 + 0.90454 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.19809 w_k1_ADM_s007
                                 + 1.44441 w_k1_AXP_s007
                                 + 1.30482 w_k1_DIS_s007
                                 + 1.21316 w_k1_FDX_s007
                                 + 1.49886 w_k1_IBM_s007
                                 + 1.11117 w_k1_INTC_s007
                                 + 1.19592 w_k1_KO_s007 + 1.9257 w_k1_MAT_s007
                                 + 1.02722 w_k1_MSFT_s007
                                 + 1.21738 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.04844 w_k1_ADM_s007
                                 + 1.05191 w_k1_AXP_s007
                                 + 1.04616 w_k1_DIS_s007
                                 + 0.978629 w_k1_FDX_s007
                                 + 1.04973 w_k1_IBM_s007
                                 + 1.09167 w_k1_INTC_s007
                                 + 0.911078 w_k1_KO_s007 + 1.0297 w_k1_MAT_s007
                                 + 0.967851 w_k1_MSFT_s007
                                 + 1.06838 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.873235 w_k1_ADM_s007
                                 + 0.875996 w_k1_AXP_s007
                                 + 1.08506 w_k1_DIS_s007
                                 + 0.974941 w_k1_FDX_s007
                                 + 1.40805 w_k1_IBM_s007
                                 + 1.20811 w_k1_INTC_s007
                                 + 1.13669 w_k1_KO_s007 + 1.16703 w_k1_MAT_s007
                                 + 1.08469 w_k1_MSFT_s007
                                 + 1.14962 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.02919 w_k1_ADM_s007
                                 + 1.12513 w_k1_AXP_s007
                                 + 1.01627 w_k1_DIS_s007
                                 + 1.25705 w_k1_FDX_s007
                                 + 0.844212 w_k1_IBM_s007
                                 + 1.29295 w_k1_INTC_s007
                                 + 0.953988 w_k1_KO_s007
                                 + 0.792323 w_k1_MAT_s007
                                 + 1.00252 w_k1_MSFT_s007
                                 + 1.09015 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.773954 w_k1_ADM_s007
                                 + 0.727491 w_k1_AXP_s007
                                 + 0.837752 w_k1_DIS_s007
                                 + 0.810415 w_k1_FDX_s007
                                 + 0.810882 w_k1_IBM_s007
                                 + 0.886474 w_k1_INTC_s007
                                 + 0.816042 w_k1_KO_s007
                                 + 0.485632 w_k1_MAT_s007
                                 + 0.934253 w_k1_MSFT_s007
                                 + 0.914859 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.79548 w_k1_ADM_s007
                                 + 0.805537 w_k1_AXP_s007
                                 + 0.752185 w_k1_DIS_s007
                                 + 0.851962 w_k1_FDX_s007
                                 + 0.738431 w_k1_IBM_s007
                                 + 1.05632 w_k1_INTC_s007
                                 + 0.683155 w_k1_KO_s007
                                 + 0.544622 w_k1_MAT_s007
                                 + 0.801404 w_k1_MSFT_s007
                                 + 0.853603 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.07202 w_k1_ADM_s007
                                 + 0.822401 w_k1_AXP_s007
                                 + 1.50579 w_k1_DIS_s007
                                 + 1.17496 w_k1_FDX_s007
                                 + 0.962479 w_k1_IBM_s007
                                 + 1.29768 w_k1_INTC_s007
                                 + 1.11569 w_k1_KO_s007
                                 + 0.736598 w_k1_MAT_s007
                                 + 1.20434 w_k1_MSFT_s007
                                 + 1.05303 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.940915 w_k1_ADM_s010
                                 + 0.770606 w_k1_AXP_s010
                                 + 0.756748 w_k1_DIS_s010
                                 + 0.840535 w_k1_FDX_s010
                                 + 0.714811 w_k1_IBM_s010
                                 + 0.904075 w_k1_INTC_s010
                                 + 0.811114 w_k1_KO_s010
                                 + 0.56107 w_k1_MAT_s010
                                 + 0.895703 w_k1_MSFT_s010
                                 + 0.863705 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.50585 w_k1_ADM_s010
                                 + 1.19573 w_k1_AXP_s010
                                 + 1.37369 w_k1_DIS_s010
                                 + 1.28722 w_k1_FDX_s010
                                 + 0.881501 w_k1_IBM_s010
                                 + 1.1878 w_k1_INTC_s010 + 1.11067 w_k1_KO_s010
                                 + 0.725083 w_k1_MAT_s010
                                 + 1.2891 w_k1_MSFT_s010 + 1.11461 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.4561 w_k1_ADM_s010
                                 + 1.49589 w_k1_AXP_s010
                                 + 1.16775 w_k1_DIS_s010
                                 + 1.11915 w_k1_FDX_s010
                                 + 1.11206 w_k1_IBM_s010
                                 + 1.09469 w_k1_INTC_s010
                                 + 1.14594 w_k1_KO_s010
                                 + 0.975064 w_k1_MAT_s010
                                 + 1.13928 w_k1_MSFT_s010
                                 + 1.04326 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.03893 w_k1_ADM_s010
                                 + 1.02081 w_k1_AXP_s010 + 1.0312 w_k1_DIS_s010
                                 + 0.963828 w_k1_FDX_s010
                                 + 1.21765 w_k1_IBM_s010
                                 + 1.11285 w_k1_INTC_s010
                                 + 1.03395 w_k1_KO_s010
                                 + 0.965393 w_k1_MAT_s010
                                 + 0.961627 w_k1_MSFT_s010
                                 + 1.07448 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.3815 w_k1_ADM_s010
                                 + 1.61756 w_k1_AXP_s010
                                 + 1.21739 w_k1_DIS_s010
                                 + 1.02758 w_k1_FDX_s010
                                 + 1.58396 w_k1_IBM_s010
                                 + 1.25768 w_k1_INTC_s010
                                 + 1.27551 w_k1_KO_s010 + 1.6158 w_k1_MAT_s010
                                 + 1.1673 w_k1_MSFT_s010 + 1.25874 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.746422 w_k1_ADM_s010
                                 + 0.826697 w_k1_AXP_s010
                                 + 0.747244 w_k1_DIS_s010
                                 + 0.73277 w_k1_FDX_s010
                                 + 1.05408 w_k1_IBM_s010
                                 + 0.817591 w_k1_INTC_s010
                                 + 0.875051 w_k1_KO_s010
                                 + 0.848568 w_k1_MAT_s010
                                 + 0.781236 w_k1_MSFT_s010
                                 + 0.872935 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.28034 w_k1_ADM_s010
                                 + 1.12838 w_k1_AXP_s010
                                 + 1.35793 w_k1_DIS_s010
                                 + 1.10509 w_k1_FDX_s010
                                 + 1.22608 w_k1_IBM_s010
                                 + 1.29266 w_k1_INTC_s010
                                 + 1.24141 w_k1_KO_s010
                                 + 0.796039 w_k1_MAT_s010
                                 + 1.07792 w_k1_MSFT_s010
                                 + 1.21119 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.13462 w_k1_ADM_s010
                                 + 0.796588 w_k1_AXP_s010
                                 + 1.12291 w_k1_DIS_s010 + 1.0652 w_k1_FDX_s010
                                 + 0.775382 w_k1_IBM_s010
                                 + 1.0355 w_k1_INTC_s010
                                 + 0.986002 w_k1_KO_s010
                                 + 0.599645 w_k1_MAT_s010
                                 + 1.07161 w_k1_MSFT_s010
                                 + 0.982216 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.12028 w_k1_ADM_s011
                                 + 0.996862 w_k1_AXP_s011
                                 + 0.847046 w_k1_DIS_s011
                                 + 0.791054 w_k1_FDX_s011
                                 + 1.30006 w_k1_IBM_s011
                                 + 1.01419 w_k1_INTC_s011
                                 + 1.08716 w_k1_KO_s011 + 1.4218 w_k1_MAT_s011
                                 + 0.877839 w_k1_MSFT_s011
                                 + 1.00544 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.40592 w_k1_ADM_s011
                                 + 1.51538 w_k1_AXP_s011
                                 + 1.66646 w_k1_DIS_s011
                                 + 1.28274 w_k1_FDX_s011
                                 + 1.23784 w_k1_IBM_s011
                                 + 1.18998 w_k1_INTC_s011
                                 + 1.38737 w_k1_KO_s011 + 1.35468 w_k1_MAT_s011
                                 + 1.25381 w_k1_MSFT_s011
                                 + 1.25103 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.10377 w_k1_ADM_s011
                                 + 0.970649 w_k1_AXP_s011
                                 + 0.95573 w_k1_DIS_s011
                                 + 1.02026 w_k1_FDX_s011
                                 + 0.71213 w_k1_IBM_s011
                                 + 0.955931 w_k1_INTC_s011
                                 + 0.832065 w_k1_KO_s011
                                 + 0.777469 w_k1_MAT_s011
                                 + 0.923954 w_k1_MSFT_s011
                                 + 0.936981 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.924435 w_k1_ADM_s011
                                 + 0.995297 w_k1_AXP_s011
                                 + 1.13953 w_k1_DIS_s011
                                 + 1.05982 w_k1_FDX_s011
                                 + 0.977843 w_k1_IBM_s011
                                 + 1.18072 w_k1_INTC_s011
                                 + 1.06908 w_k1_KO_s011
                                 + 0.705685 w_k1_MAT_s011
                                 + 1.13783 w_k1_MSFT_s011
                                 + 1.05703 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.858804 w_k1_ADM_s011
                                 + 0.768066 w_k1_AXP_s011
                                 + 0.787497 w_k1_DIS_s011
                                 + 0.904035 w_k1_FDX_s011
                                 + 0.67429 w_k1_IBM_s011
                                 + 0.865677 w_k1_INTC_s011
                                 + 0.721705 w_k1_KO_s011
                                 + 0.735534 w_k1_MAT_s011
                                 + 0.831422 w_k1_MSFT_s011
                                 + 0.92425 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.9054 w_k1_ADM_s011
                                 + 1.43725 w_k1_AXP_s011
                                 + 1.04039 w_k1_DIS_s011 + 1.1119 w_k1_FDX_s011
                                 + 1.02435 w_k1_IBM_s011
                                 + 1.19236 w_k1_INTC_s011
                                 + 0.897722 w_k1_KO_s011
                                 + 1.32407 w_k1_MAT_s011
                                 + 0.93834 w_k1_MSFT_s011
                                 + 1.04874 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.95737 w_k1_ADM_s011
                                 + 0.799438 w_k1_AXP_s011
                                 + 0.693331 w_k1_DIS_s011
                                 + 0.723554 w_k1_FDX_s011
                                 + 0.777794 w_k1_IBM_s011
                                 + 0.817333 w_k1_INTC_s011
                                 + 0.848754 w_k1_KO_s011
                                 + 0.788959 w_k1_MAT_s011
                                 + 0.828857 w_k1_MSFT_s011
                                 + 0.81132 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.978152 w_k1_ADM_s011
                                 + 0.825743 w_k1_AXP_s011
                                 + 0.86829 w_k1_DIS_s011
                                 + 0.841576 w_k1_FDX_s011
                                 + 0.906271 w_k1_IBM_s011
                                 + 0.917843 w_k1_INTC_s011
                                 + 0.970086 w_k1_KO_s011
                                 + 0.735949 w_k1_MAT_s011
                                 + 0.912703 w_k1_MSFT_s011
                                 + 0.979092 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.967513 w_k1_ADM_s012
                                 + 0.871605 w_k1_AXP_s012
                                 + 0.659579 w_k1_DIS_s012
                                 + 0.869552 w_k1_FDX_s012
                                 + 0.701407 w_k1_IBM_s012
                                 + 0.865746 w_k1_INTC_s012
                                 + 0.728263 w_k1_KO_s012
                                 + 0.892986 w_k1_MAT_s012
                                 + 0.801646 w_k1_MSFT_s012
                                 + 0.818614 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.21008 w_k1_ADM_s012
                                 + 0.979727 w_k1_AXP_s012
                                 + 1.39008 w_k1_DIS_s012 + 1.1289 w_k1_FDX_s012
                                 + 0.924128 w_k1_IBM_s012
                                 + 1.06841 w_k1_INTC_s012
                                 + 1.15886 w_k1_KO_s012
                                 + 0.696407 w_k1_MAT_s012
                                 + 1.12194 w_k1_MSFT_s012
                                 + 1.12144 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.14222 w_k1_ADM_s012
                                 + 0.896178 w_k1_AXP_s012
                                 + 0.925754 w_k1_DIS_s012
                                 + 0.843059 w_k1_FDX_s012
                                 + 0.932281 w_k1_IBM_s012
                                 + 0.951857 w_k1_INTC_s012
                                 + 0.879177 w_k1_KO_s012
                                 + 0.815096 w_k1_MAT_s012
                                 + 1.0438 w_k1_MSFT_s012
                                 + 0.927842 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.41835 w_k1_ADM_s012
                                 + 1.29555 w_k1_AXP_s012 + 1.4505 w_k1_DIS_s012
                                 + 1.29803 w_k1_FDX_s012 + 1.1574 w_k1_IBM_s012
                                 + 1.46758 w_k1_INTC_s012
                                 + 1.10637 w_k1_KO_s012 + 1.38418 w_k1_MAT_s012
                                 + 1.40029 w_k1_MSFT_s012
                                 + 1.28166 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.17116 w_k1_ADM_s012
                                 + 0.847416 w_k1_AXP_s012
                                 + 1.03671 w_k1_DIS_s012
                                 + 0.966868 w_k1_FDX_s012 + 0.715 w_k1_IBM_s012
                                 + 0.794668 w_k1_INTC_s012
                                 + 0.753744 w_k1_KO_s012
                                 + 0.583742 w_k1_MAT_s012
                                 + 1.02447 w_k1_MSFT_s012
                                 + 0.981986 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.16883 w_k1_ADM_s012
                                 + 1.39486 w_k1_AXP_s012
                                 + 1.11095 w_k1_DIS_s012
                                 + 1.14752 w_k1_FDX_s012
                                 + 1.16148 w_k1_IBM_s012
                                 + 1.15233 w_k1_INTC_s012
                                 + 1.11142 w_k1_KO_s012 + 1.77385 w_k1_MAT_s012
                                 + 1.23711 w_k1_MSFT_s012
                                 + 1.07285 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.06063 w_k1_ADM_s012
                                 + 1.35199 w_k1_AXP_s012
                                 + 1.40833 w_k1_DIS_s012 + 1.1439 w_k1_FDX_s012
                                 + 1.53176 w_k1_IBM_s012
                                 + 1.25028 w_k1_INTC_s012
                                 + 1.55164 w_k1_KO_s012 + 1.34546 w_k1_MAT_s012
                                 + 1.20183 w_k1_MSFT_s012
                                 + 1.40072 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.990354 w_k1_ADM_s012
                                 + 0.917294 w_k1_AXP_s012
                                 + 0.919249 w_k1_DIS_s012
                                 + 0.975193 w_k1_FDX_s012
                                 + 1.21695 w_k1_IBM_s012
                                 + 1.17006 w_k1_INTC_s012
                                 + 0.92885 w_k1_KO_s012 + 1.06235 w_k1_MAT_s012
                                 + 1.03786 w_k1_MSFT_s012
                                 + 0.970185 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.14467 w_k1_ADM_s013
                                 + 1.03802 w_k1_AXP_s013
                                 + 0.900431 w_k1_DIS_s013
                                 + 0.8875 w_k1_FDX_s013 + 1.53862 w_k1_IBM_s013
                                 + 0.827956 w_k1_INTC_s013
                                 + 1.05536 w_k1_KO_s013 + 1.74231 w_k1_MAT_s013
                                 + 0.741709 w_k1_MSFT_s013
                                 + 1.02824 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.985533 w_k1_ADM_s013
                                 + 0.706714 w_k1_AXP_s013
                                 + 1.23545 w_k1_DIS_s013
                                 + 1.01568 w_k1_FDX_s013
                                 + 0.699866 w_k1_IBM_s013
                                 + 1.16234 w_k1_INTC_s013 + 1.0287 w_k1_KO_s013
                                 + 0.425304 w_k1_MAT_s013
                                 + 1.16409 w_k1_MSFT_s013
                                 + 1.06496 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.29696 w_k1_ADM_s013
                                 + 1.51431 w_k1_AXP_s013
                                 + 1.53754 w_k1_DIS_s013 + 1.2996 w_k1_FDX_s013
                                 + 1.46117 w_k1_IBM_s013
                                 + 1.46006 w_k1_INTC_s013
                                 + 1.39333 w_k1_KO_s013 + 1.35115 w_k1_MAT_s013
                                 + 1.26849 w_k1_MSFT_s013
                                 + 1.28123 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.07959 w_k1_ADM_s013
                                 + 1.09151 w_k1_AXP_s013
                                 + 1.47855 w_k1_DIS_s013 + 1.3108 w_k1_FDX_s013
                                 + 1.1593 w_k1_IBM_s013 + 1.136 w_k1_INTC_s013
                                 + 1.13222 w_k1_KO_s013 + 1.33339 w_k1_MAT_s013
                                 + 1.17771 w_k1_MSFT_s013
                                 + 1.20315 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.1491 w_k1_ADM_s013
                                 + 1.05875 w_k1_AXP_s013
                                 + 1.17039 w_k1_DIS_s013 + 1.1834 w_k1_FDX_s013
                                 + 0.748226 w_k1_IBM_s013
                                 + 0.971155 w_k1_INTC_s013
                                 + 0.928995 w_k1_KO_s013
                                 + 0.880508 w_k1_MAT_s013
                                 + 0.999701 w_k1_MSFT_s013
                                 + 1.15582 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.946109 w_k1_ADM_s013
                                 + 1.11024 w_k1_AXP_s013
                                 + 0.913071 w_k1_DIS_s013
                                 + 1.03227 w_k1_FDX_s013
                                 + 0.898762 w_k1_IBM_s013
                                 + 1.0068 w_k1_INTC_s013 + 0.87273 w_k1_KO_s013
                                 + 1.13561 w_k1_MAT_s013
                                 + 0.950032 w_k1_MSFT_s013
                                 + 0.960798 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.78506 w_k1_ADM_s013
                                 + 0.922943 w_k1_AXP_s013
                                 + 0.74717 w_k1_DIS_s013
                                 + 0.781318 w_k1_FDX_s013
                                 + 1.02758 w_k1_IBM_s013
                                 + 0.89217 w_k1_INTC_s013
                                 + 0.869069 w_k1_KO_s013
                                 + 0.953093 w_k1_MAT_s013
                                 + 0.797015 w_k1_MSFT_s013
                                 + 0.864762 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.943067 w_k1_ADM_s013
                                 + 0.843794 w_k1_AXP_s013
                                 + 0.985586 w_k1_DIS_s013
                                 + 1.03033 w_k1_FDX_s013 + 1.0918 w_k1_IBM_s013
                                 + 0.970355 w_k1_INTC_s013
                                 + 0.96952 w_k1_KO_s013
                                 + 0.827195 w_k1_MAT_s013
                                 + 0.898025 w_k1_MSFT_s013
                                 + 0.968299 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.10037 w_k1_ADM_s014
                                 + 1.14179 w_k1_AXP_s014
                                 + 0.953723 w_k1_DIS_s014
                                 + 0.939714 w_k1_FDX_s014
                                 + 1.23218 w_k1_IBM_s014
                                 + 0.956132 w_k1_INTC_s014
                                 + 1.12529 w_k1_KO_s014 + 1.13028 w_k1_MAT_s014
                                 + 0.94644 w_k1_MSFT_s014
                                 + 1.09157 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.02274 w_k1_ADM_s014
                                 + 0.756377 w_k1_AXP_s014
                                 + 0.942935 w_k1_DIS_s014
                                 + 0.995248 w_k1_FDX_s014
                                 + 0.804988 w_k1_IBM_s014
                                 + 1.22083 w_k1_INTC_s014
                                 + 0.946969 w_k1_KO_s014
                                 + 0.557858 w_k1_MAT_s014
                                 + 1.00987 w_k1_MSFT_s014
                                 + 0.88626 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.5147 w_k1_ADM_s014
                                 + 1.32653 w_k1_AXP_s014
                                 + 1.24215 w_k1_DIS_s014
                                 + 1.11321 w_k1_FDX_s014
                                 + 1.65554 w_k1_IBM_s014
                                 + 1.14078 w_k1_INTC_s014
                                 + 1.52174 w_k1_KO_s014 + 1.41194 w_k1_MAT_s014
                                 + 1.17942 w_k1_MSFT_s014 + 1.0989 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.887957 w_k1_ADM_s014
                                 + 0.835788 w_k1_AXP_s014
                                 + 0.927444 w_k1_DIS_s014
                                 + 0.868269 w_k1_FDX_s014
                                 + 1.03758 w_k1_IBM_s014
                                 + 0.947433 w_k1_INTC_s014
                                 + 1.06808 w_k1_KO_s014
                                 + 0.784004 w_k1_MAT_s014
                                 + 0.845495 w_k1_MSFT_s014
                                 + 0.932312 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.06663 w_k1_ADM_s014
                                 + 1.12914 w_k1_AXP_s014
                                 + 1.38225 w_k1_DIS_s014
                                 + 1.10716 w_k1_FDX_s014
                                 + 1.22781 w_k1_IBM_s014
                                 + 1.19791 w_k1_INTC_s014
                                 + 1.19285 w_k1_KO_s014 + 1.04389 w_k1_MAT_s014
                                 + 1.10219 w_k1_MSFT_s014
                                 + 1.19456 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.13337 w_k1_ADM_s014
                                 + 1.17287 w_k1_AXP_s014 + 1.0425 w_k1_DIS_s014
                                 + 1.19598 w_k1_FDX_s014
                                 + 0.775336 w_k1_IBM_s014
                                 + 1.0691 w_k1_INTC_s014
                                 + 0.866939 w_k1_KO_s014
                                 + 0.866397 w_k1_MAT_s014
                                 + 0.978489 w_k1_MSFT_s014
                                 + 0.921518 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.838094 w_k1_ADM_s014
                                 + 0.650517 w_k1_AXP_s014
                                 + 0.606301 w_k1_DIS_s014
                                 + 0.762358 w_k1_FDX_s014
                                 + 0.763916 w_k1_IBM_s014
                                 + 0.902864 w_k1_INTC_s014
                                 + 0.76147 w_k1_KO_s014
                                 + 0.544042 w_k1_MAT_s014
                                 + 0.828599 w_k1_MSFT_s014
                                 + 0.871019 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.15104 w_k1_ADM_s014
                                 + 1.06493 w_k1_AXP_s014
                                 + 0.834393 w_k1_DIS_s014
                                 + 0.957413 w_k1_FDX_s014
                                 + 0.787611 w_k1_IBM_s014
                                 + 0.814265 w_k1_INTC_s014
                                 + 0.839832 w_k1_KO_s014
                                 + 0.954658 w_k1_MAT_s014
                                 + 0.809764 w_k1_MSFT_s014
                                 + 0.90124 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 2.08413 w_k1_ADM_s015
                                 + 1.62722 w_k1_AXP_s015
                                 + 2.07166 w_k1_DIS_s015
                                 + 1.39789 w_k1_FDX_s015
                                 + 1.84722 w_k1_IBM_s015
                                 + 1.25336 w_k1_INTC_s015
                                 + 1.89038 w_k1_KO_s015 + 1.90376 w_k1_MAT_s015
                                 + 1.45002 w_k1_MSFT_s015
                                 + 1.56592 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.17791 w_k1_ADM_s015
                                 + 0.908859 w_k1_AXP_s015
                                 + 1.02143 w_k1_DIS_s015
                                 + 1.00837 w_k1_FDX_s015
                                 + 0.900524 w_k1_IBM_s015
                                 + 1.11419 w_k1_INTC_s015
                                 + 1.01014 w_k1_KO_s015
                                 + 0.802416 w_k1_MAT_s015
                                 + 1.04755 w_k1_MSFT_s015
                                 + 1.12179 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.96969 w_k1_ADM_s015
                                 + 0.766324 w_k1_AXP_s015
                                 + 0.725317 w_k1_DIS_s015
                                 + 0.786891 w_k1_FDX_s015
                                 + 1.04154 w_k1_IBM_s015
                                 + 0.948184 w_k1_INTC_s015
                                 + 0.927623 w_k1_KO_s015
                                 + 0.811624 w_k1_MAT_s015
                                 + 0.910457 w_k1_MSFT_s015
                                 + 0.904374 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.887903 w_k1_ADM_s015
                                 + 0.715038 w_k1_AXP_s015
                                 + 0.622544 w_k1_DIS_s015
                                 + 0.843603 w_k1_FDX_s015
                                 + 0.580479 w_k1_IBM_s015
                                 + 0.787898 w_k1_INTC_s015
                                 + 0.748404 w_k1_KO_s015
                                 + 0.717623 w_k1_MAT_s015
                                 + 0.789266 w_k1_MSFT_s015
                                 + 0.87156 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.02978 w_k1_ADM_s015
                                 + 1.10762 w_k1_AXP_s015
                                 + 1.04599 w_k1_DIS_s015
                                 + 0.935055 w_k1_FDX_s015
                                 + 1.57247 w_k1_IBM_s015
                                 + 1.22722 w_k1_INTC_s015
                                 + 1.21785 w_k1_KO_s015 + 1.34046 w_k1_MAT_s015
                                 + 1.05897 w_k1_MSFT_s015
                                 + 1.19598 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.70103 w_k1_ADM_s015
                                 + 1.7248 w_k1_AXP_s015 + 1.35283 w_k1_DIS_s015
                                 + 1.34964 w_k1_FDX_s015
                                 + 1.16644 w_k1_IBM_s015
                                 + 1.29544 w_k1_INTC_s015
                                 + 1.25047 w_k1_KO_s015 + 1.28107 w_k1_MAT_s015
                                 + 1.07558 w_k1_MSFT_s015
                                 + 1.27739 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.27734 w_k1_ADM_s015
                                 + 1.22283 w_k1_AXP_s015
                                 + 0.900275 w_k1_DIS_s015
                                 + 1.06018 w_k1_FDX_s015
                                 + 1.03969 w_k1_IBM_s015
                                 + 1.05454 w_k1_INTC_s015
                                 + 0.906836 w_k1_KO_s015
                                 + 1.15836 w_k1_MAT_s015
                                 + 0.951177 w_k1_MSFT_s015
                                 + 1.07228 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.13065 w_k1_ADM_s015
                                 + 1.06194 w_k1_AXP_s015
                                 + 1.44479 w_k1_DIS_s015
                                 + 1.06435 w_k1_FDX_s015
                                 + 1.29644 w_k1_IBM_s015
                                 + 1.14667 w_k1_INTC_s015
                                 + 1.44788 w_k1_KO_s015 + 1.17454 w_k1_MAT_s015
                                 + 1.20735 w_k1_MSFT_s015
                                 + 1.14469 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.09082 w_k1_ADM_s016
                                 + 0.82464 w_k1_AXP_s016
                                 + 0.804363 w_k1_DIS_s016
                                 + 0.905281 w_k1_FDX_s016
                                 + 0.847038 w_k1_IBM_s016
                                 + 0.920165 w_k1_INTC_s016
                                 + 0.874725 w_k1_KO_s016
                                 + 0.759032 w_k1_MAT_s016
                                 + 0.849946 w_k1_MSFT_s016
                                 + 1.00033 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.14932 w_k1_ADM_s016
                                 + 1.11242 w_k1_AXP_s016
                                 + 0.975117 w_k1_DIS_s016
                                 + 1.03372 w_k1_FDX_s016
                                 + 1.15531 w_k1_IBM_s016
                                 + 1.13225 w_k1_INTC_s016
                                 + 1.02128 w_k1_KO_s016 + 1.5132 w_k1_MAT_s016
                                 + 1.09894 w_k1_MSFT_s016
                                 + 1.00073 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.820657 w_k1_ADM_s016
                                 + 0.734343 w_k1_AXP_s016
                                 + 0.504006 w_k1_DIS_s016
                                 + 0.665622 w_k1_FDX_s016
                                 + 0.813943 w_k1_IBM_s016
                                 + 0.762606 w_k1_INTC_s016
                                 + 0.709648 w_k1_KO_s016
                                 + 0.798416 w_k1_MAT_s016
                                 + 0.706995 w_k1_MSFT_s016
                                 + 0.823598 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.22991 w_k1_ADM_s016
                                 + 1.10144 w_k1_AXP_s016 + 1.1423 w_k1_DIS_s016
                                 + 1.05178 w_k1_FDX_s016
                                 + 1.47623 w_k1_IBM_s016
                                 + 1.26263 w_k1_INTC_s016
                                 + 1.28985 w_k1_KO_s016 + 1.37722 w_k1_MAT_s016
                                 + 1.04962 w_k1_MSFT_s016
                                 + 1.00666 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.887689 w_k1_ADM_s016
                                 + 0.601838 w_k1_AXP_s016
                                 + 0.750715 w_k1_DIS_s016
                                 + 0.733575 w_k1_FDX_s016
                                 + 0.925576 w_k1_IBM_s016
                                 + 0.950516 w_k1_INTC_s016
                                 + 0.934079 w_k1_KO_s016
                                 + 0.911171 w_k1_MAT_s016
                                 + 0.887048 w_k1_MSFT_s016
                                 + 0.963091 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.48523 w_k1_ADM_s016
                                 + 1.31379 w_k1_AXP_s016
                                 + 1.64237 w_k1_DIS_s016
                                 + 1.37439 w_k1_FDX_s016
                                 + 1.54386 w_k1_IBM_s016
                                 + 1.24622 w_k1_INTC_s016
                                 + 1.51295 w_k1_KO_s016 + 1.38359 w_k1_MAT_s016
                                 + 1.50539 w_k1_MSFT_s016
                                 + 1.35228 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.963177 w_k1_ADM_s016
                                 + 0.896964 w_k1_AXP_s016
                                 + 1.18308 w_k1_DIS_s016
                                 + 1.06408 w_k1_FDX_s016
                                 + 0.924228 w_k1_IBM_s016
                                 + 1.27669 w_k1_INTC_s016
                                 + 1.03796 w_k1_KO_s016
                                 + 0.539762 w_k1_MAT_s016
                                 + 1.21328 w_k1_MSFT_s016
                                 + 1.03834 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.56157 w_k1_ADM_s016
                                 + 1.14903 w_k1_AXP_s016
                                 + 1.21813 w_k1_DIS_s016 + 1.2762 w_k1_FDX_s016
                                 + 0.917859 w_k1_IBM_s016
                                 + 1.14522 w_k1_INTC_s016
                                 + 1.12473 w_k1_KO_s016
                                 + 0.784269 w_k1_MAT_s016
                                 + 1.16575 w_k1_MSFT_s016
                                 + 1.10901 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.905271 w_k1_ADM_s017
                                 + 0.745611 w_k1_AXP_s017
                                 + 0.67764 w_k1_DIS_s017
                                 + 0.737811 w_k1_FDX_s017
                                 + 1.36079 w_k1_IBM_s017
                                 + 0.678473 w_k1_INTC_s017
                                 + 0.981601 w_k1_KO_s017
                                 + 1.34288 w_k1_MAT_s017
                                 + 0.817847 w_k1_MSFT_s017
                                 + 1.10554 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.18649 w_k1_ADM_s017
                                 + 0.870191 w_k1_AXP_s017
                                 + 0.908664 w_k1_DIS_s017
                                 + 0.925592 w_k1_FDX_s017
                                 + 1.03968 w_k1_IBM_s017
                                 + 0.843602 w_k1_INTC_s017
                                 + 0.975435 w_k1_KO_s017
                                 + 0.821178 w_k1_MAT_s017
                                 + 1.01842 w_k1_MSFT_s017
                                 + 1.05162 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.943088 w_k1_ADM_s017
                                 + 0.855695 w_k1_AXP_s017
                                 + 1.1784 w_k1_DIS_s017 + 1.00854 w_k1_FDX_s017
                                 + 0.945248 w_k1_IBM_s017
                                 + 0.894877 w_k1_INTC_s017
                                 + 1.08142 w_k1_KO_s017
                                 + 0.690276 w_k1_MAT_s017
                                 + 1.04651 w_k1_MSFT_s017
                                 + 0.930988 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.28681 w_k1_ADM_s017
                                 + 1.25689 w_k1_AXP_s017
                                 + 1.23354 w_k1_DIS_s017
                                 + 1.22716 w_k1_FDX_s017
                                 + 0.819366 w_k1_IBM_s017
                                 + 0.825853 w_k1_INTC_s017
                                 + 0.911466 w_k1_KO_s017
                                 + 0.71601 w_k1_MAT_s017
                                 + 1.16385 w_k1_MSFT_s017
                                 + 1.05531 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.16463 w_k1_ADM_s017
                                 + 0.871676 w_k1_AXP_s017
                                 + 0.867035 w_k1_DIS_s017
                                 + 0.923657 w_k1_FDX_s017
                                 + 0.780756 w_k1_IBM_s017
                                 + 0.823731 w_k1_INTC_s017
                                 + 0.885289 w_k1_KO_s017
                                 + 0.757713 w_k1_MAT_s017
                                 + 0.966683 w_k1_MSFT_s017
                                 + 1.08903 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.0053 w_k1_ADM_s017
                                 + 1.3064 w_k1_AXP_s017 + 1.48808 w_k1_DIS_s017
                                 + 1.22774 w_k1_FDX_s017
                                 + 1.43723 w_k1_IBM_s017
                                 + 0.98391 w_k1_INTC_s017
                                 + 1.26874 w_k1_KO_s017 + 1.74735 w_k1_MAT_s017
                                 + 1.41613 w_k1_MSFT_s017
                                 + 1.32111 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.907653 w_k1_ADM_s017
                                 + 0.626132 w_k1_AXP_s017
                                 + 0.590453 w_k1_DIS_s017
                                 + 0.776051 w_k1_FDX_s017
                                 + 0.721631 w_k1_IBM_s017
                                 + 0.65816 w_k1_INTC_s017
                                 + 0.774944 w_k1_KO_s017
                                 + 0.828229 w_k1_MAT_s017
                                 + 0.954012 w_k1_MSFT_s017
                                 + 0.851603 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.926941 w_k1_ADM_s017
                                 + 0.60508 w_k1_AXP_s017
                                 + 0.815191 w_k1_DIS_s017
                                 + 0.75629 w_k1_FDX_s017
                                 + 1.03634 w_k1_IBM_s017
                                 + 0.590329 w_k1_INTC_s017
                                 + 0.974213 w_k1_KO_s017
                                 + 0.858156 w_k1_MAT_s017
                                 + 0.864373 w_k1_MSFT_s017
                                 + 0.932165 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.834011 w_k1_ADM_s100
                                 + 0.922709 w_k1_AXP_s100
                                 + 1.06772 w_k1_DIS_s100
                                 + 1.03061 w_k1_FDX_s100
                                 + 0.914416 w_k1_IBM_s100
                                 + 0.895702 w_k1_INTC_s100
                                 + 0.844298 w_k1_KO_s100 + 1.0354 w_k1_MAT_s100
                                 + 1.00346 w_k1_MSFT_s100
                                 + 1.09547 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.909934 w_k1_ADM_s100
                                 + 0.831279 w_k1_AXP_s100
                                 + 0.66842 w_k1_DIS_s100
                                 + 0.813683 w_k1_FDX_s100
                                 + 0.850216 w_k1_IBM_s100
                                 + 0.59464 w_k1_INTC_s100
                                 + 0.791129 w_k1_KO_s100
                                 + 0.905087 w_k1_MAT_s100
                                 + 0.88493 w_k1_MSFT_s100
                                 + 0.940148 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.22624 w_k1_ADM_s100
                                 + 1.12106 w_k1_AXP_s100
                                 + 1.41271 w_k1_DIS_s100
                                 + 1.31235 w_k1_FDX_s100
                                 + 0.749777 w_k1_IBM_s100
                                 + 0.883556 w_k1_INTC_s100
                                 + 0.980127 w_k1_KO_s100
                                 + 0.934813 w_k1_MAT_s100
                                 + 1.17868 w_k1_MSFT_s100
                                 + 1.07085 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.91257 w_k1_ADM_s100
                                 + 0.899803 w_k1_AXP_s100
                                 + 1.25998 w_k1_DIS_s100
                                 + 0.981582 w_k1_FDX_s100
                                 + 1.34233 w_k1_IBM_s100
                                 + 0.855079 w_k1_INTC_s100
                                 + 1.17118 w_k1_KO_s100 + 1.42967 w_k1_MAT_s100
                                 + 1.11338 w_k1_MSFT_s100
                                 + 1.11466 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.740975 w_k1_ADM_s100
                                 + 0.674862 w_k1_AXP_s100
                                 + 0.579877 w_k1_DIS_s100
                                 + 0.728066 w_k1_FDX_s100
                                 + 0.66599 w_k1_IBM_s100
                                 + 0.567916 w_k1_INTC_s100
                                 + 0.549071 w_k1_KO_s100
                                 + 0.775986 w_k1_MAT_s100
                                 + 0.74378 w_k1_MSFT_s100
                                 + 0.780783 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.765014 w_k1_ADM_s100
                                 + 0.800337 w_k1_AXP_s100
                                 + 0.713994 w_k1_DIS_s100
                                 + 0.770908 w_k1_FDX_s100
                                 + 1.17375 w_k1_IBM_s100
                                 + 0.719007 w_k1_INTC_s100
                                 + 0.93225 w_k1_KO_s100 + 1.14547 w_k1_MAT_s100
                                 + 0.90758 w_k1_MSFT_s100
                                 + 1.06504 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.30453 w_k1_ADM_s100
                                 + 1.37899 w_k1_AXP_s100
                                 + 1.48524 w_k1_DIS_s100 + 1.2708 w_k1_FDX_s100
                                 + 1.24497 w_k1_IBM_s100
                                 + 0.842436 w_k1_INTC_s100
                                 + 1.00616 w_k1_KO_s100 + 1.1718 w_k1_MAT_s100
                                 + 1.0807 w_k1_MSFT_s100 + 1.14978 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.953976 w_k1_ADM_s100
                                 + 0.843375 w_k1_AXP_s100
                                 + 0.958125 w_k1_DIS_s100
                                 + 1.05662 w_k1_FDX_s100
                                 + 0.622257 w_k1_IBM_s100
                                 + 0.852313 w_k1_INTC_s100
                                 + 0.784359 w_k1_KO_s100
                                 + 0.698319 w_k1_MAT_s100
                                 + 1.15661 w_k1_MSFT_s100
                                 + 0.900524 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.10269 w_k1_ADM_s101
                                 + 1.43675 w_k1_AXP_s101
                                 + 1.49857 w_k1_DIS_s101
                                 + 1.33786 w_k1_FDX_s101
                                 + 1.25422 w_k1_IBM_s101
                                 + 1.3142 w_k1_INTC_s101 + 1.3493 w_k1_KO_s101
                                 + 1.33108 w_k1_MAT_s101
                                 + 1.46264 w_k1_MSFT_s101
                                 + 1.18739 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.01488 w_k1_ADM_s101
                                 + 1.08355 w_k1_AXP_s101
                                 + 1.16759 w_k1_DIS_s101
                                 + 1.28326 w_k1_FDX_s101
                                 + 0.810182 w_k1_IBM_s101
                                 + 1.31589 w_k1_INTC_s101
                                 + 0.920743 w_k1_KO_s101
                                 + 0.864319 w_k1_MAT_s101
                                 + 1.28978 w_k1_MSFT_s101
                                 + 1.06561 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.14139 w_k1_ADM_s101
                                 + 1.8312 w_k1_AXP_s101 + 1.20634 w_k1_DIS_s101
                                 + 1.36448 w_k1_FDX_s101
                                 + 1.03191 w_k1_IBM_s101
                                 + 1.26626 w_k1_INTC_s101
                                 + 1.04799 w_k1_KO_s101 + 1.53112 w_k1_MAT_s101
                                 + 1.21958 w_k1_MSFT_s101
                                 + 1.02814 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.2773 w_k1_ADM_s101
                                 + 1.25592 w_k1_AXP_s101
                                 + 1.15698 w_k1_DIS_s101
                                 + 1.11824 w_k1_FDX_s101
                                 + 1.74855 w_k1_IBM_s101
                                 + 0.971797 w_k1_INTC_s101
                                 + 1.28623 w_k1_KO_s101 + 1.69096 w_k1_MAT_s101
                                 + 0.981816 w_k1_MSFT_s101
                                 + 1.07358 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.900673 w_k1_ADM_s101
                                 + 0.913773 w_k1_AXP_s101
                                 + 0.816573 w_k1_DIS_s101
                                 + 0.870943 w_k1_FDX_s101
                                 + 1.3673 w_k1_IBM_s101
                                 + 0.92875 w_k1_INTC_s101
                                 + 0.940836 w_k1_KO_s101
                                 + 1.17239 w_k1_MAT_s101
                                 + 0.789341 w_k1_MSFT_s101
                                 + 0.954533 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.837455 w_k1_ADM_s101
                                 + 0.810345 w_k1_AXP_s101
                                 + 0.775893 w_k1_DIS_s101
                                 + 0.927933 w_k1_FDX_s101
                                 + 0.644982 w_k1_IBM_s101
                                 + 0.868414 w_k1_INTC_s101
                                 + 0.677982 w_k1_KO_s101
                                 + 0.554721 w_k1_MAT_s101
                                 + 0.920028 w_k1_MSFT_s101
                                 + 0.821251 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.977634 w_k1_ADM_s101
                                 + 1.20113 w_k1_AXP_s101
                                 + 0.855996 w_k1_DIS_s101
                                 + 1.12126 w_k1_FDX_s101
                                 + 0.870685 w_k1_IBM_s101
                                 + 1.02068 w_k1_INTC_s101
                                 + 0.770631 w_k1_KO_s101
                                 + 0.97623 w_k1_MAT_s101
                                 + 0.905936 w_k1_MSFT_s101
                                 + 0.945971 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.720115 w_k1_ADM_s101
                                 + 0.889222 w_k1_AXP_s101
                                 + 0.886909 w_k1_DIS_s101
                                 + 0.898303 w_k1_FDX_s101
                                 + 1.02494 w_k1_IBM_s101
                                 + 1.12472 w_k1_INTC_s101
                                 + 0.854534 w_k1_KO_s101
                                 + 1.02248 w_k1_MAT_s101
                                 + 0.965074 w_k1_MSFT_s101
                                 + 0.903169 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.01474 w_k1_ADM_s102
                                 + 0.856112 w_k1_AXP_s102
                                 + 1.15677 w_k1_DIS_s102
                                 + 1.02508 w_k1_FDX_s102
                                 + 0.875114 w_k1_IBM_s102
                                 + 1.02485 w_k1_INTC_s102
                                 + 1.13694 w_k1_KO_s102
                                 + 0.694658 w_k1_MAT_s102
                                 + 1.16408 w_k1_MSFT_s102
                                 + 1.00464 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.24843 w_k1_ADM_s102
                                 + 1.1795 w_k1_AXP_s102 + 1.26193 w_k1_DIS_s102
                                 + 1.30956 w_k1_FDX_s102
                                 + 0.751213 w_k1_IBM_s102
                                 + 1.04528 w_k1_INTC_s102
                                 + 1.01263 w_k1_KO_s102
                                 + 0.643635 w_k1_MAT_s102
                                 + 1.2388 w_k1_MSFT_s102 + 1.06275 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.21005 w_k1_ADM_s102
                                 + 1.63288 w_k1_AXP_s102
                                 + 1.73432 w_k1_DIS_s102
                                 + 1.34308 w_k1_FDX_s102
                                 + 1.47891 w_k1_IBM_s102
                                 + 1.25579 w_k1_INTC_s102
                                 + 1.54712 w_k1_KO_s102 + 1.57347 w_k1_MAT_s102
                                 + 1.26922 w_k1_MSFT_s102
                                 + 1.18247 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.01152 w_k1_ADM_s102
                                 + 1.11414 w_k1_AXP_s102
                                 + 1.35795 w_k1_DIS_s102 + 1.1226 w_k1_FDX_s102
                                 + 1.38538 w_k1_IBM_s102
                                 + 1.11118 w_k1_INTC_s102
                                 + 1.38529 w_k1_KO_s102 + 1.21227 w_k1_MAT_s102
                                 + 1.0877 w_k1_MSFT_s102 + 1.11694 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.914684 w_k1_ADM_s102
                                 + 0.68629 w_k1_AXP_s102
                                 + 0.779068 w_k1_DIS_s102
                                 + 0.888102 w_k1_FDX_s102
                                 + 0.692051 w_k1_IBM_s102
                                 + 0.763305 w_k1_INTC_s102
                                 + 0.802724 w_k1_KO_s102
                                 + 0.510652 w_k1_MAT_s102
                                 + 0.828754 w_k1_MSFT_s102
                                 + 0.830779 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.813518 w_k1_ADM_s102
                                 + 0.676895 w_k1_AXP_s102
                                 + 0.711267 w_k1_DIS_s102
                                 + 0.776731 w_k1_FDX_s102
                                 + 0.886873 w_k1_IBM_s102
                                 + 0.869159 w_k1_INTC_s102
                                 + 1.01715 w_k1_KO_s102
                                 + 0.775801 w_k1_MAT_s102
                                 + 0.901084 w_k1_MSFT_s102
                                 + 0.788071 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.876164 w_k1_ADM_s102
                                 + 0.850186 w_k1_AXP_s102
                                 + 0.962522 w_k1_DIS_s102
                                 + 0.929192 w_k1_FDX_s102
                                 + 1.24155 w_k1_IBM_s102
                                 + 0.898193 w_k1_INTC_s102
                                 + 1.15477 w_k1_KO_s102
                                 + 0.878134 w_k1_MAT_s102
                                 + 0.966648 w_k1_MSFT_s102
                                 + 0.979202 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.895982 w_k1_ADM_s102
                                 + 1.06061 w_k1_AXP_s102
                                 + 1.38686 w_k1_DIS_s102 + 1.2631 w_k1_FDX_s102
                                 + 1.07623 w_k1_IBM_s102
                                 + 1.15255 w_k1_INTC_s102
                                 + 1.45838 w_k1_KO_s102 + 1.13102 w_k1_MAT_s102
                                 + 1.25418 w_k1_MSFT_s102
                                 + 1.11132 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.06953 w_k1_ADM_s103
                                 + 1.10952 w_k1_AXP_s103 + 1.1026 w_k1_DIS_s103
                                 + 1.19224 w_k1_FDX_s103
                                 + 0.810472 w_k1_IBM_s103
                                 + 1.03678 w_k1_INTC_s103
                                 + 0.888956 w_k1_KO_s103
                                 + 1.05494 w_k1_MAT_s103
                                 + 1.10304 w_k1_MSFT_s103
                                 + 1.09408 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.771887 w_k1_ADM_s103
                                 + 0.968776 w_k1_AXP_s103
                                 + 0.805988 w_k1_DIS_s103
                                 + 0.970035 w_k1_FDX_s103
                                 + 0.815478 w_k1_IBM_s103
                                 + 0.997734 w_k1_INTC_s103
                                 + 0.740509 w_k1_KO_s103
                                 + 0.96589 w_k1_MAT_s103
                                 + 0.900238 w_k1_MSFT_s103
                                 + 0.99604 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.802594 w_k1_ADM_s103
                                 + 1.06112 w_k1_AXP_s103
                                 + 0.785269 w_k1_DIS_s103
                                 + 0.828539 w_k1_FDX_s103
                                 + 1.01697 w_k1_IBM_s103
                                 + 1.00607 w_k1_INTC_s103
                                 + 0.899619 w_k1_KO_s103
                                 + 1.07267 w_k1_MAT_s103
                                 + 0.855546 w_k1_MSFT_s103
                                 + 0.955468 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.22577 w_k1_ADM_s103
                                 + 1.38474 w_k1_AXP_s103
                                 + 1.34235 w_k1_DIS_s103 + 1.2628 w_k1_FDX_s103
                                 + 0.999912 w_k1_IBM_s103
                                 + 1.16516 w_k1_INTC_s103
                                 + 1.07668 w_k1_KO_s103
                                 + 0.937554 w_k1_MAT_s103
                                 + 1.19367 w_k1_MSFT_s103
                                 + 1.26583 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.864942 w_k1_ADM_s103
                                 + 0.836724 w_k1_AXP_s103
                                 + 0.500873 w_k1_DIS_s103
                                 + 0.719396 w_k1_FDX_s103
                                 + 0.774046 w_k1_IBM_s103
                                 + 0.692783 w_k1_INTC_s103
                                 + 0.682275 w_k1_KO_s103
                                 + 0.919934 w_k1_MAT_s103
                                 + 0.637828 w_k1_MSFT_s103
                                 + 0.816662 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.01216 w_k1_ADM_s103
                                 + 1.26533 w_k1_AXP_s103
                                 + 1.29573 w_k1_DIS_s103
                                 + 1.15019 w_k1_FDX_s103
                                 + 1.18551 w_k1_IBM_s103
                                 + 0.987636 w_k1_INTC_s103
                                 + 1.12462 w_k1_KO_s103 + 1.31147 w_k1_MAT_s103
                                 + 0.972048 w_k1_MSFT_s103
                                 + 1.1308 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.11057 w_k1_ADM_s103
                                 + 1.7139 w_k1_AXP_s103 + 1.30153 w_k1_DIS_s103
                                 + 1.19951 w_k1_FDX_s103
                                 + 1.58365 w_k1_IBM_s103
                                 + 1.21676 w_k1_INTC_s103
                                 + 1.21187 w_k1_KO_s103 + 2.38616 w_k1_MAT_s103
                                 + 1.09651 w_k1_MSFT_s103
                                 + 1.14603 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.749326 w_k1_ADM_s103
                                 + 0.99506 w_k1_AXP_s103
                                 + 0.919357 w_k1_DIS_s103
                                 + 0.846731 w_k1_FDX_s103
                                 + 1.37832 w_k1_IBM_s103
                                 + 0.781792 w_k1_INTC_s103
                                 + 0.96454 w_k1_KO_s103 + 1.28834 w_k1_MAT_s103
                                 + 0.807994 w_k1_MSFT_s103
                                 + 1.02528 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.26636 w_k1_ADM_s104
                                 + 1.23597 w_k1_AXP_s104
                                 + 1.03658 w_k1_DIS_s104
                                 + 1.20094 w_k1_FDX_s104
                                 + 0.843476 w_k1_IBM_s104
                                 + 1.08384 w_k1_INTC_s104
                                 + 0.894475 w_k1_KO_s104
                                 + 0.903138 w_k1_MAT_s104
                                 + 1.16323 w_k1_MSFT_s104
                                 + 1.08108 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.20223 w_k1_ADM_s104
                                 + 1.32477 w_k1_AXP_s104
                                 + 1.17915 w_k1_DIS_s104
                                 + 1.11411 w_k1_FDX_s104
                                 + 1.40469 w_k1_IBM_s104
                                 + 1.17213 w_k1_INTC_s104
                                 + 1.29228 w_k1_KO_s104 + 1.36362 w_k1_MAT_s104
                                 + 1.10738 w_k1_MSFT_s104
                                 + 1.24399 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.01602 w_k1_ADM_s104
                                 + 1.40179 w_k1_AXP_s104
                                 + 1.82898 w_k1_DIS_s104
                                 + 1.40753 w_k1_FDX_s104
                                 + 1.28868 w_k1_IBM_s104
                                 + 1.2555 w_k1_INTC_s104 + 1.29471 w_k1_KO_s104
                                 + 1.36367 w_k1_MAT_s104
                                 + 1.49462 w_k1_MSFT_s104
                                 + 1.37475 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.924973 w_k1_ADM_s104
                                 + 1.00365 w_k1_AXP_s104
                                 + 0.93525 w_k1_DIS_s104
                                 + 0.885297 w_k1_FDX_s104
                                 + 1.36128 w_k1_IBM_s104
                                 + 0.909731 w_k1_INTC_s104
                                 + 1.17255 w_k1_KO_s104
                                 + 0.942276 w_k1_MAT_s104
                                 + 1.07123 w_k1_MSFT_s104
                                 + 1.09576 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.975324 w_k1_ADM_s104
                                 + 0.881357 w_k1_AXP_s104
                                 + 0.642975 w_k1_DIS_s104
                                 + 0.803458 w_k1_FDX_s104
                                 + 1.08949 w_k1_IBM_s104
                                 + 1.07405 w_k1_INTC_s104
                                 + 0.870078 w_k1_KO_s104
                                 + 0.804873 w_k1_MAT_s104
                                 + 0.876579 w_k1_MSFT_s104
                                 + 0.927481 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.02525 w_k1_ADM_s104
                                 + 0.764937 w_k1_AXP_s104
                                 + 0.989053 w_k1_DIS_s104
                                 + 0.930595 w_k1_FDX_s104
                                 + 0.919936 w_k1_IBM_s104
                                 + 0.926208 w_k1_INTC_s104
                                 + 0.928583 w_k1_KO_s104
                                 + 0.716014 w_k1_MAT_s104
                                 + 1.03675 w_k1_MSFT_s104
                                 + 0.954594 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.889067 w_k1_ADM_s104
                                 + 0.768931 w_k1_AXP_s104
                                 + 0.755616 w_k1_DIS_s104
                                 + 0.890286 w_k1_FDX_s104
                                 + 0.765653 w_k1_IBM_s104
                                 + 0.933272 w_k1_INTC_s104
                                 + 0.889375 w_k1_KO_s104
                                 + 0.715558 w_k1_MAT_s104
                                 + 1.0225 w_k1_MSFT_s104
                                 + 0.942352 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.05634 w_k1_ADM_s104
                                 + 0.967738 w_k1_AXP_s104
                                 + 1.37919 w_k1_DIS_s104
                                 + 1.06886 w_k1_FDX_s104
                                 + 1.11629 w_k1_IBM_s104
                                 + 1.20207 w_k1_INTC_s104
                                 + 1.15356 w_k1_KO_s104
                                 + 0.628246 w_k1_MAT_s104
                                 + 1.28223 w_k1_MSFT_s104
                                 + 1.15925 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.980554 w_k1_ADM_s105
                                 + 0.684715 w_k1_AXP_s105
                                 + 0.602583 w_k1_DIS_s105
                                 + 0.79208 w_k1_FDX_s105
                                 + 0.657649 w_k1_IBM_s105
                                 + 0.861972 w_k1_INTC_s105
                                 + 0.678309 w_k1_KO_s105
                                 + 0.485567 w_k1_MAT_s105
                                 + 0.728724 w_k1_MSFT_s105
                                 + 0.79777 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.1707 w_k1_ADM_s105
                                 + 1.25303 w_k1_AXP_s105
                                 + 1.30748 w_k1_DIS_s105
                                 + 1.15577 w_k1_FDX_s105
                                 + 1.50533 w_k1_IBM_s105
                                 + 1.27653 w_k1_INTC_s105
                                 + 1.33823 w_k1_KO_s105 + 1.19345 w_k1_MAT_s105
                                 + 1.10861 w_k1_MSFT_s105
                                 + 1.17858 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.921405 w_k1_ADM_s105
                                 + 0.655894 w_k1_AXP_s105
                                 + 0.738759 w_k1_DIS_s105
                                 + 0.894498 w_k1_FDX_s105
                                 + 0.892535 w_k1_IBM_s105
                                 + 0.86873 w_k1_INTC_s105
                                 + 0.752676 w_k1_KO_s105
                                 + 0.839911 w_k1_MAT_s105
                                 + 0.865588 w_k1_MSFT_s105
                                 + 0.904914 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.751362 w_k1_ADM_s105
                                 + 0.892837 w_k1_AXP_s105
                                 + 0.500548 w_k1_DIS_s105
                                 + 0.721835 w_k1_FDX_s105
                                 + 0.861373 w_k1_IBM_s105
                                 + 0.970836 w_k1_INTC_s105
                                 + 0.653819 w_k1_KO_s105
                                 + 0.761376 w_k1_MAT_s105
                                 + 0.698503 w_k1_MSFT_s105
                                 + 0.857109 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.880564 w_k1_ADM_s105
                                 + 0.950999 w_k1_AXP_s105
                                 + 0.583422 w_k1_DIS_s105
                                 + 0.842808 w_k1_FDX_s105
                                 + 1.19854 w_k1_IBM_s105
                                 + 0.738887 w_k1_INTC_s105
                                 + 0.820809 w_k1_KO_s105
                                 + 1.16199 w_k1_MAT_s105
                                 + 0.745157 w_k1_MSFT_s105
                                 + 0.871575 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.99306 w_k1_ADM_s105
                                 + 0.773124 w_k1_AXP_s105 + 1.285 w_k1_DIS_s105
                                 + 1.05856 w_k1_FDX_s105
                                 + 1.16184 w_k1_IBM_s105
                                 + 1.05039 w_k1_INTC_s105
                                 + 1.08223 w_k1_KO_s105 + 1.07977 w_k1_MAT_s105
                                 + 1.07377 w_k1_MSFT_s105
                                 + 1.09299 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.951044 w_k1_ADM_s105
                                 + 0.912655 w_k1_AXP_s105
                                 + 1.02816 w_k1_DIS_s105
                                 + 1.13522 w_k1_FDX_s105
                                 + 0.872131 w_k1_IBM_s105
                                 + 1.05463 w_k1_INTC_s105
                                 + 1.00254 w_k1_KO_s105
                                 + 0.900183 w_k1_MAT_s105
                                 + 1.04344 w_k1_MSFT_s105
                                 + 1.02562 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.18229 w_k1_ADM_s105
                                 + 1.42012 w_k1_AXP_s105
                                 + 1.26236 w_k1_DIS_s105
                                 + 1.27393 w_k1_FDX_s105
                                 + 1.07148 w_k1_IBM_s105
                                 + 1.20218 w_k1_INTC_s105
                                 + 1.06798 w_k1_KO_s105 + 1.05714 w_k1_MAT_s105
                                 + 1.24352 w_k1_MSFT_s105
                                 + 1.14798 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.971751 w_k1_ADM_s106
                                 + 0.712612 w_k1_AXP_s106
                                 + 0.779096 w_k1_DIS_s106
                                 + 0.863511 w_k1_FDX_s106
                                 + 0.794806 w_k1_IBM_s106
                                 + 0.822728 w_k1_INTC_s106
                                 + 0.991386 w_k1_KO_s106
                                 + 0.812102 w_k1_MAT_s106
                                 + 0.905878 w_k1_MSFT_s106
                                 + 0.859147 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.852918 w_k1_ADM_s106
                                 + 0.777927 w_k1_AXP_s106
                                 + 1.2059 w_k1_DIS_s106
                                 + 0.892788 w_k1_FDX_s106
                                 + 1.36425 w_k1_IBM_s106
                                 + 1.0024 w_k1_INTC_s106 + 1.44613 w_k1_KO_s106
                                 + 1.22016 w_k1_MAT_s106
                                 + 1.13025 w_k1_MSFT_s106
                                 + 1.00583 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.947639 w_k1_ADM_s106
                                 + 0.86143 w_k1_AXP_s106
                                 + 0.744783 w_k1_DIS_s106
                                 + 0.772111 w_k1_FDX_s106
                                 + 1.24596 w_k1_IBM_s106
                                 + 0.916794 w_k1_INTC_s106
                                 + 0.982626 w_k1_KO_s106
                                 + 0.913261 w_k1_MAT_s106
                                 + 0.869572 w_k1_MSFT_s106
                                 + 0.899553 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.12995 w_k1_ADM_s106
                                 + 1.2935 w_k1_AXP_s106 + 1.11274 w_k1_DIS_s106
                                 + 1.08376 w_k1_FDX_s106
                                 + 1.11178 w_k1_IBM_s106
                                 + 0.919266 w_k1_INTC_s106
                                 + 1.00803 w_k1_KO_s106 + 1.16245 w_k1_MAT_s106
                                 + 0.959903 w_k1_MSFT_s106
                                 + 1.05321 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.2722 w_k1_ADM_s106
                                 + 0.973593 w_k1_AXP_s106
                                 + 0.830734 w_k1_DIS_s106
                                 + 0.983234 w_k1_FDX_s106
                                 + 0.74383 w_k1_IBM_s106
                                 + 0.847665 w_k1_INTC_s106
                                 + 0.841479 w_k1_KO_s106
                                 + 0.706228 w_k1_MAT_s106
                                 + 0.875333 w_k1_MSFT_s106
                                 + 0.925489 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.11163 w_k1_ADM_s106
                                 + 0.793551 w_k1_AXP_s106
                                 + 1.17832 w_k1_DIS_s106
                                 + 1.05602 w_k1_FDX_s106
                                 + 0.905658 w_k1_IBM_s106
                                 + 0.86336 w_k1_INTC_s106
                                 + 1.07097 w_k1_KO_s106
                                 + 0.587077 w_k1_MAT_s106
                                 + 1.06427 w_k1_MSFT_s106
                                 + 0.954349 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.763786 w_k1_ADM_s106
                                 + 0.671332 w_k1_AXP_s106
                                 + 0.723226 w_k1_DIS_s106
                                 + 0.805397 w_k1_FDX_s106
                                 + 0.643291 w_k1_IBM_s106
                                 + 0.765746 w_k1_INTC_s106
                                 + 0.718639 w_k1_KO_s106
                                 + 0.521656 w_k1_MAT_s106
                                 + 0.684732 w_k1_MSFT_s106
                                 + 0.743677 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.20014 w_k1_ADM_s106
                                 + 1.31674 w_k1_AXP_s106 + 1.5332 w_k1_DIS_s106
                                 + 1.22894 w_k1_FDX_s106
                                 + 1.63714 w_k1_IBM_s106
                                 + 1.26208 w_k1_INTC_s106
                                 + 1.58471 w_k1_KO_s106 + 1.94391 w_k1_MAT_s106
                                 + 1.36302 w_k1_MSFT_s106
                                 + 1.25459 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.20983 w_k1_ADM_s107
                                 + 1.5537 w_k1_AXP_s107 + 1.50808 w_k1_DIS_s107
                                 + 1.28975 w_k1_FDX_s107
                                 + 1.28021 w_k1_IBM_s107
                                 + 1.12315 w_k1_INTC_s107
                                 + 1.15044 w_k1_KO_s107 + 1.50051 w_k1_MAT_s107
                                 + 1.26194 w_k1_MSFT_s107
                                 + 1.25709 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.721529 w_k1_ADM_s107
                                 + 0.696958 w_k1_AXP_s107
                                 + 0.69454 w_k1_DIS_s107
                                 + 0.720142 w_k1_FDX_s107
                                 + 0.966783 w_k1_IBM_s107
                                 + 0.789897 w_k1_INTC_s107
                                 + 0.73815 w_k1_KO_s107
                                 + 0.701019 w_k1_MAT_s107
                                 + 0.718405 w_k1_MSFT_s107
                                 + 0.885348 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.986978 w_k1_ADM_s107
                                 + 1.04718 w_k1_AXP_s107
                                 + 1.25456 w_k1_DIS_s107
                                 + 1.18233 w_k1_FDX_s107
                                 + 0.797633 w_k1_IBM_s107
                                 + 1.02784 w_k1_INTC_s107
                                 + 0.84532 w_k1_KO_s107
                                 + 0.927857 w_k1_MAT_s107
                                 + 1.09007 w_k1_MSFT_s107
                                 + 1.01499 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.03817 w_k1_ADM_s107
                                 + 1.1099 w_k1_AXP_s107 + 1.33382 w_k1_DIS_s107
                                 + 1.02921 w_k1_FDX_s107
                                 + 1.37572 w_k1_IBM_s107
                                 + 0.97031 w_k1_INTC_s107
                                 + 1.15527 w_k1_KO_s107 + 1.52702 w_k1_MAT_s107
                                 + 1.1479 w_k1_MSFT_s107 + 1.11805 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.883247 w_k1_ADM_s107
                                 + 0.913431 w_k1_AXP_s107
                                 + 0.746852 w_k1_DIS_s107
                                 + 0.791445 w_k1_FDX_s107
                                 + 1.27061 w_k1_IBM_s107
                                 + 0.684224 w_k1_INTC_s107
                                 + 0.902251 w_k1_KO_s107
                                 + 1.21274 w_k1_MAT_s107
                                 + 0.820457 w_k1_MSFT_s107
                                 + 0.975365 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.801519 w_k1_ADM_s107
                                 + 0.897895 w_k1_AXP_s107
                                 + 0.624357 w_k1_DIS_s107
                                 + 0.77649 w_k1_FDX_s107
                                 + 0.732373 w_k1_IBM_s107
                                 + 0.680956 w_k1_INTC_s107
                                 + 0.662891 w_k1_KO_s107
                                 + 0.812623 w_k1_MAT_s107
                                 + 0.675417 w_k1_MSFT_s107
                                 + 0.837207 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.927845 w_k1_ADM_s107
                                 + 0.761177 w_k1_AXP_s107
                                 + 0.992473 w_k1_DIS_s107
                                 + 0.863913 w_k1_FDX_s107
                                 + 0.907718 w_k1_IBM_s107
                                 + 1.0131 w_k1_INTC_s107
                                 + 0.918961 w_k1_KO_s107
                                 + 0.657019 w_k1_MAT_s107
                                 + 0.995031 w_k1_MSFT_s107
                                 + 0.907223 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.957154 w_k1_ADM_s107
                                 + 1.09558 w_k1_AXP_s107
                                 + 0.972317 w_k1_DIS_s107
                                 + 0.995603 w_k1_FDX_s107
                                 + 1.08042 w_k1_IBM_s107
                                 + 0.983932 w_k1_INTC_s107
                                 + 1.1088 w_k1_KO_s107 + 0.998113 w_k1_MAT_s107
                                 + 0.886232 w_k1_MSFT_s107
                                 + 1.10649 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.2399 w_k1_ADM_s110
                                 + 1.00895 w_k1_AXP_s110
                                 + 0.905094 w_k1_DIS_s110
                                 + 0.993633 w_k1_FDX_s110 + 1.118 w_k1_IBM_s110
                                 + 0.885631 w_k1_INTC_s110
                                 + 0.951595 w_k1_KO_s110
                                 + 1.18631 w_k1_MAT_s110
                                 + 0.997361 w_k1_MSFT_s110
                                 + 0.988353 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.06033 w_k1_ADM_s110
                                 + 0.933503 w_k1_AXP_s110
                                 + 0.864835 w_k1_DIS_s110
                                 + 1.12007 w_k1_FDX_s110
                                 + 0.83759 w_k1_IBM_s110
                                 + 1.11284 w_k1_INTC_s110
                                 + 0.850021 w_k1_KO_s110
                                 + 0.704071 w_k1_MAT_s110
                                 + 1.03997 w_k1_MSFT_s110
                                 + 0.985208 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.26759 w_k1_ADM_s110
                                 + 1.54663 w_k1_AXP_s110
                                 + 1.34215 w_k1_DIS_s110
                                 + 1.40259 w_k1_FDX_s110 + 1.2295 w_k1_IBM_s110
                                 + 1.4136 w_k1_INTC_s110 + 1.03471 w_k1_KO_s110
                                 + 1.15216 w_k1_MAT_s110
                                 + 1.08561 w_k1_MSFT_s110
                                 + 1.14923 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.02992 w_k1_ADM_s110
                                 + 0.88806 w_k1_AXP_s110
                                 + 0.666207 w_k1_DIS_s110
                                 + 0.87506 w_k1_FDX_s110
                                 + 0.749007 w_k1_IBM_s110
                                 + 0.904827 w_k1_INTC_s110
                                 + 0.757381 w_k1_KO_s110
                                 + 0.82354 w_k1_MAT_s110
                                 + 0.847712 w_k1_MSFT_s110
                                 + 0.905142 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.12148 w_k1_ADM_s110
                                 + 1.07052 w_k1_AXP_s110
                                 + 1.01778 w_k1_DIS_s110
                                 + 1.11312 w_k1_FDX_s110
                                 + 1.48673 w_k1_IBM_s110
                                 + 1.10241 w_k1_INTC_s110
                                 + 1.20395 w_k1_KO_s110 + 1.59016 w_k1_MAT_s110
                                 + 1.12991 w_k1_MSFT_s110 + 1.0447 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.1715 w_k1_ADM_s110
                                 + 0.886876 w_k1_AXP_s110
                                 + 0.817155 w_k1_DIS_s110
                                 + 0.999104 w_k1_FDX_s110
                                 + 0.668342 w_k1_IBM_s110
                                 + 1.13069 w_k1_INTC_s110
                                 + 0.751759 w_k1_KO_s110
                                 + 0.620887 w_k1_MAT_s110
                                 + 0.973045 w_k1_MSFT_s110
                                 + 1.01338 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.21666 w_k1_ADM_s110
                                 + 1.11318 w_k1_AXP_s110
                                 + 1.38369 w_k1_DIS_s110
                                 + 1.27221 w_k1_FDX_s110
                                 + 0.949459 w_k1_IBM_s110
                                 + 1.33994 w_k1_INTC_s110
                                 + 1.17367 w_k1_KO_s110 + 1.2272 w_k1_MAT_s110
                                 + 1.35805 w_k1_MSFT_s110
                                 + 1.26159 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.05278 w_k1_ADM_s110
                                 + 1.42359 w_k1_AXP_s110
                                 + 0.862998 w_k1_DIS_s110
                                 + 1.12601 w_k1_FDX_s110
                                 + 1.09684 w_k1_IBM_s110
                                 + 1.3693 w_k1_INTC_s110
                                 + 0.908209 w_k1_KO_s110
                                 + 1.27281 w_k1_MAT_s110
                                 + 1.01521 w_k1_MSFT_s110
                                 + 1.10992 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.625539 w_k1_ADM_s111
                                 + 0.72672 w_k1_AXP_s111
                                 + 0.623548 w_k1_DIS_s111
                                 + 0.737947 w_k1_FDX_s111
                                 + 1.03223 w_k1_IBM_s111
                                 + 0.838392 w_k1_INTC_s111
                                 + 0.71589 w_k1_KO_s111
                                 + 0.744241 w_k1_MAT_s111
                                 + 0.76826 w_k1_MSFT_s111
                                 + 0.854833 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.904215 w_k1_ADM_s111
                                 + 0.927672 w_k1_AXP_s111
                                 + 0.914675 w_k1_DIS_s111
                                 + 0.904498 w_k1_FDX_s111
                                 + 1.16671 w_k1_IBM_s111
                                 + 0.7914 w_k1_INTC_s111 + 0.95271 w_k1_KO_s111
                                 + 1.17388 w_k1_MAT_s111
                                 + 0.89955 w_k1_MSFT_s111
                                 + 1.13117 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.933994 w_k1_ADM_s111
                                 + 1.0514 w_k1_AXP_s111 + 1.25648 w_k1_DIS_s111
                                 + 1.0882 w_k1_FDX_s111 + 1.14121 w_k1_IBM_s111
                                 + 1.09157 w_k1_INTC_s111
                                 + 1.00968 w_k1_KO_s111
                                 + 0.940181 w_k1_MAT_s111
                                 + 1.01551 w_k1_MSFT_s111
                                 + 1.16703 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.29873 w_k1_ADM_s111
                                 + 1.42837 w_k1_AXP_s111
                                 + 1.38522 w_k1_DIS_s111
                                 + 1.20801 w_k1_FDX_s111
                                 + 1.42401 w_k1_IBM_s111
                                 + 1.13972 w_k1_INTC_s111
                                 + 1.29927 w_k1_KO_s111 + 1.31857 w_k1_MAT_s111
                                 + 1.14877 w_k1_MSFT_s111
                                 + 1.48039 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.850684 w_k1_ADM_s111
                                 + 0.892654 w_k1_AXP_s111
                                 + 0.775898 w_k1_DIS_s111
                                 + 0.909372 w_k1_FDX_s111
                                 + 0.729397 w_k1_IBM_s111
                                 + 0.979859 w_k1_INTC_s111
                                 + 0.695715 w_k1_KO_s111
                                 + 0.719397 w_k1_MAT_s111
                                 + 0.920847 w_k1_MSFT_s111
                                 + 1.06598 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.08419 w_k1_ADM_s111
                                 + 1.29696 w_k1_AXP_s111
                                 + 1.44677 w_k1_DIS_s111
                                 + 1.45151 w_k1_FDX_s111
                                 + 0.949907 w_k1_IBM_s111
                                 + 0.934927 w_k1_INTC_s111
                                 + 1.05182 w_k1_KO_s111
                                 + 0.939065 w_k1_MAT_s111
                                 + 1.14123 w_k1_MSFT_s111
                                 + 1.21446 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.959553 w_k1_ADM_s111
                                 + 0.565991 w_k1_AXP_s111
                                 + 0.539999 w_k1_DIS_s111
                                 + 0.706184 w_k1_FDX_s111
                                 + 0.638056 w_k1_IBM_s111
                                 + 0.734637 w_k1_INTC_s111
                                 + 0.70142 w_k1_KO_s111
                                 + 0.648157 w_k1_MAT_s111
                                 + 0.692182 w_k1_MSFT_s111
                                 + 0.773257 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.920673 w_k1_ADM_s111
                                 + 1.04977 w_k1_AXP_s111 + 1.0419 w_k1_DIS_s111
                                 + 1.16125 w_k1_FDX_s111
                                 + 0.749422 w_k1_IBM_s111
                                 + 1.09004 w_k1_INTC_s111
                                 + 0.817284 w_k1_KO_s111
                                 + 0.708156 w_k1_MAT_s111
                                 + 1.07089 w_k1_MSFT_s111
                                 + 1.12048 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.01292 w_k1_ADM_s112
                                 + 0.783484 w_k1_AXP_s112
                                 + 0.798813 w_k1_DIS_s112
                                 + 0.93871 w_k1_FDX_s112
                                 + 0.595507 w_k1_IBM_s112
                                 + 0.854816 w_k1_INTC_s112
                                 + 0.753281 w_k1_KO_s112
                                 + 0.807667 w_k1_MAT_s112
                                 + 0.847842 w_k1_MSFT_s112
                                 + 0.906167 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.961166 w_k1_ADM_s112
                                 + 1.07564 w_k1_AXP_s112
                                 + 0.840655 w_k1_DIS_s112
                                 + 0.927888 w_k1_FDX_s112
                                 + 1.10551 w_k1_IBM_s112
                                 + 1.12152 w_k1_INTC_s112
                                 + 0.975808 w_k1_KO_s112
                                 + 1.25782 w_k1_MAT_s112
                                 + 0.83579 w_k1_MSFT_s112
                                 + 0.95042 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.55545 w_k1_ADM_s112
                                 + 1.62497 w_k1_AXP_s112
                                 + 1.50693 w_k1_DIS_s112
                                 + 1.43184 w_k1_FDX_s112
                                 + 1.12205 w_k1_IBM_s112
                                 + 1.22234 w_k1_INTC_s112
                                 + 1.27479 w_k1_KO_s112 + 1.24427 w_k1_MAT_s112
                                 + 1.25427 w_k1_MSFT_s112
                                 + 1.24302 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.06112 w_k1_ADM_s112
                                 + 0.628352 w_k1_AXP_s112
                                 + 0.611281 w_k1_DIS_s112
                                 + 0.806056 w_k1_FDX_s112
                                 + 0.808433 w_k1_IBM_s112
                                 + 0.790651 w_k1_INTC_s112
                                 + 0.84213 w_k1_KO_s112
                                 + 0.705755 w_k1_MAT_s112
                                 + 0.751257 w_k1_MSFT_s112
                                 + 0.868763 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.00856 w_k1_ADM_s112
                                 + 0.978987 w_k1_AXP_s112
                                 + 0.805518 w_k1_DIS_s112
                                 + 0.894891 w_k1_FDX_s112
                                 + 0.786615 w_k1_IBM_s112
                                 + 0.75923 w_k1_INTC_s112
                                 + 0.828044 w_k1_KO_s112
                                 + 0.90085 w_k1_MAT_s112
                                 + 0.785606 w_k1_MSFT_s112
                                 + 0.756159 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.31077 w_k1_ADM_s112
                                 + 1.25287 w_k1_AXP_s112
                                 + 1.62877 w_k1_DIS_s112
                                 + 1.18909 w_k1_FDX_s112
                                 + 1.76327 w_k1_IBM_s112
                                 + 1.03127 w_k1_INTC_s112
                                 + 1.61479 w_k1_KO_s112 + 2.60976 w_k1_MAT_s112
                                 + 1.11144 w_k1_MSFT_s112
                                 + 1.24656 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.02768 w_k1_ADM_s112
                                 + 0.729842 w_k1_AXP_s112
                                 + 0.916515 w_k1_DIS_s112
                                 + 0.816386 w_k1_FDX_s112
                                 + 0.929212 w_k1_IBM_s112
                                 + 0.785431 w_k1_INTC_s112
                                 + 0.975839 w_k1_KO_s112
                                 + 0.803404 w_k1_MAT_s112
                                 + 0.757545 w_k1_MSFT_s112
                                 + 0.812803 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.32424 w_k1_ADM_s112
                                 + 1.10805 w_k1_AXP_s112
                                 + 1.14504 w_k1_DIS_s112
                                 + 1.09892 w_k1_FDX_s112
                                 + 1.19471 w_k1_IBM_s112
                                 + 1.06181 w_k1_INTC_s112
                                 + 1.21517 w_k1_KO_s112 + 1.5689 w_k1_MAT_s112
                                 + 1.1009 w_k1_MSFT_s112 + 1.13945 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.913371 w_k1_ADM_s113
                                 + 0.58962 w_k1_AXP_s113
                                 + 0.576222 w_k1_DIS_s113
                                 + 0.754772 w_k1_FDX_s113
                                 + 0.670442 w_k1_IBM_s113
                                 + 0.891661 w_k1_INTC_s113
                                 + 0.88108 w_k1_KO_s113 + 0.50098 w_k1_MAT_s113
                                 + 0.862344 w_k1_MSFT_s113
                                 + 0.828795 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.23492 w_k1_ADM_s113
                                 + 1.14538 w_k1_AXP_s113
                                 + 1.04459 w_k1_DIS_s113
                                 + 0.913159 w_k1_FDX_s113
                                 + 1.44256 w_k1_IBM_s113
                                 + 0.921753 w_k1_INTC_s113
                                 + 1.32323 w_k1_KO_s113 + 1.63255 w_k1_MAT_s113
                                 + 0.868749 w_k1_MSFT_s113
                                 + 1.10076 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.901116 w_k1_ADM_s113
                                 + 0.695211 w_k1_AXP_s113
                                 + 0.67397 w_k1_DIS_s113
                                 + 0.830156 w_k1_FDX_s113
                                 + 0.879571 w_k1_IBM_s113
                                 + 0.908619 w_k1_INTC_s113
                                 + 0.923921 w_k1_KO_s113
                                 + 1.03843 w_k1_MAT_s113
                                 + 0.898275 w_k1_MSFT_s113
                                 + 0.870414 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.59402 w_k1_ADM_s113
                                 + 1.36632 w_k1_AXP_s113
                                 + 1.38165 w_k1_DIS_s113
                                 + 1.06615 w_k1_FDX_s113
                                 + 1.82468 w_k1_IBM_s113
                                 + 1.1657 w_k1_INTC_s113 + 1.72063 w_k1_KO_s113
                                 + 2.06426 w_k1_MAT_s113
                                 + 0.98893 w_k1_MSFT_s113
                                 + 1.27349 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.09748 w_k1_ADM_s113
                                 + 1.24395 w_k1_AXP_s113
                                 + 1.48922 w_k1_DIS_s113
                                 + 1.34805 w_k1_FDX_s113
                                 + 1.11948 w_k1_IBM_s113
                                 + 1.41678 w_k1_INTC_s113
                                 + 1.35101 w_k1_KO_s113 + 1.36807 w_k1_MAT_s113
                                 + 1.39525 w_k1_MSFT_s113 + 1.2749 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.996193 w_k1_ADM_s113
                                 + 1.26858 w_k1_AXP_s113
                                 + 1.07047 w_k1_DIS_s113 + 1.0947 w_k1_FDX_s113
                                 + 1.29338 w_k1_IBM_s113
                                 + 1.10422 w_k1_INTC_s113
                                 + 1.26161 w_k1_KO_s113 + 1.95633 w_k1_MAT_s113
                                 + 0.884114 w_k1_MSFT_s113
                                 + 1.11032 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.08551 w_k1_ADM_s113
                                 + 1.06845 w_k1_AXP_s113
                                 + 0.930667 w_k1_DIS_s113
                                 + 0.954246 w_k1_FDX_s113
                                 + 0.980843 w_k1_IBM_s113
                                 + 1.22895 w_k1_INTC_s113
                                 + 1.17426 w_k1_KO_s113
                                 + 0.952089 w_k1_MAT_s113
                                 + 1.09552 w_k1_MSFT_s113
                                 + 1.03668 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.2879 w_k1_ADM_s113
                                 + 1.13213 w_k1_AXP_s113
                                 + 0.961581 w_k1_DIS_s113
                                 + 1.05976 w_k1_FDX_s113
                                 + 0.743635 w_k1_IBM_s113
                                 + 0.978955 w_k1_INTC_s113
                                 + 0.98386 w_k1_KO_s113
                                 + 0.991288 w_k1_MAT_s113
                                 + 0.942292 w_k1_MSFT_s113
                                 + 0.963763 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.862447 w_k1_ADM_s114
                                 + 0.675078 w_k1_AXP_s114
                                 + 0.791137 w_k1_DIS_s114
                                 + 0.86718 w_k1_FDX_s114
                                 + 0.584574 w_k1_IBM_s114
                                 + 0.915771 w_k1_INTC_s114
                                 + 0.729739 w_k1_KO_s114
                                 + 0.471647 w_k1_MAT_s114
                                 + 0.954244 w_k1_MSFT_s114
                                 + 0.802538 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.26457 w_k1_ADM_s114
                                 + 1.17524 w_k1_AXP_s114
                                 + 1.16962 w_k1_DIS_s114
                                 + 1.28855 w_k1_FDX_s114
                                 + 1.37627 w_k1_IBM_s114
                                 + 1.05877 w_k1_INTC_s114
                                 + 1.20567 w_k1_KO_s114 + 1.49287 w_k1_MAT_s114
                                 + 1.16281 w_k1_MSFT_s114
                                 + 1.09092 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.02246 w_k1_ADM_s114
                                 + 1.10383 w_k1_AXP_s114
                                 + 0.908777 w_k1_DIS_s114
                                 + 1.09796 w_k1_FDX_s114
                                 + 0.760559 w_k1_IBM_s114
                                 + 0.967915 w_k1_INTC_s114
                                 + 0.89166 w_k1_KO_s114 + 1.15853 w_k1_MAT_s114
                                 + 1.05231 w_k1_MSFT_s114
                                 + 1.04042 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.1051 w_k1_ADM_s114
                                 + 1.07164 w_k1_AXP_s114
                                 + 1.11332 w_k1_DIS_s114
                                 + 1.19051 w_k1_FDX_s114
                                 + 0.842548 w_k1_IBM_s114
                                 + 1.13438 w_k1_INTC_s114
                                 + 0.886248 w_k1_KO_s114
                                 + 0.921842 w_k1_MAT_s114
                                 + 1.25761 w_k1_MSFT_s114
                                 + 1.04331 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.754414 w_k1_ADM_s114
                                 + 0.766375 w_k1_AXP_s114
                                 + 0.73498 w_k1_DIS_s114
                                 + 0.835492 w_k1_FDX_s114
                                 + 0.907399 w_k1_IBM_s114
                                 + 0.874427 w_k1_INTC_s114
                                 + 0.831809 w_k1_KO_s114
                                 + 0.871843 w_k1_MAT_s114
                                 + 0.85714 w_k1_MSFT_s114
                                 + 0.798233 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.0752 w_k1_ADM_s114
                                 + 0.914011 w_k1_AXP_s114
                                 + 0.836431 w_k1_DIS_s114
                                 + 0.809811 w_k1_FDX_s114
                                 + 1.47631 w_k1_IBM_s114
                                 + 0.780497 w_k1_INTC_s114
                                 + 1.06862 w_k1_KO_s114 + 1.32182 w_k1_MAT_s114
                                 + 0.913293 w_k1_MSFT_s114
                                 + 0.986305 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.54358 w_k1_ADM_s114
                                 + 1.07802 w_k1_AXP_s114
                                 + 1.29723 w_k1_DIS_s114
                                 + 1.28072 w_k1_FDX_s114
                                 + 0.617404 w_k1_IBM_s114
                                 + 0.968303 w_k1_INTC_s114
                                 + 0.866376 w_k1_KO_s114
                                 + 0.78668 w_k1_MAT_s114
                                 + 1.28581 w_k1_MSFT_s114
                                 + 1.13021 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.988711 w_k1_ADM_s114
                                 + 1.12238 w_k1_AXP_s114
                                 + 0.889849 w_k1_DIS_s114
                                 + 0.957521 w_k1_FDX_s114
                                 + 1.08429 w_k1_IBM_s114
                                 + 0.978867 w_k1_INTC_s114
                                 + 0.95572 w_k1_KO_s114 + 1.12177 w_k1_MAT_s114
                                 + 1.01798 w_k1_MSFT_s114
                                 + 0.906043 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.38244 w_k1_ADM_s115
                                 + 1.76395 w_k1_AXP_s115
                                 + 1.79321 w_k1_DIS_s115
                                 + 1.46946 w_k1_FDX_s115
                                 + 1.42233 w_k1_IBM_s115
                                 + 1.63918 w_k1_INTC_s115
                                 + 1.32292 w_k1_KO_s115 + 1.46019 w_k1_MAT_s115
                                 + 1.23588 w_k1_MSFT_s115
                                 + 1.32235 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.17822 w_k1_ADM_s115
                                 + 0.765321 w_k1_AXP_s115
                                 + 0.964843 w_k1_DIS_s115
                                 + 0.895064 w_k1_FDX_s115
                                 + 0.999819 w_k1_IBM_s115
                                 + 1.14518 w_k1_INTC_s115
                                 + 1.02045 w_k1_KO_s115
                                 + 0.743805 w_k1_MAT_s115
                                 + 0.933062 w_k1_MSFT_s115
                                 + 1.03272 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.0089 w_k1_ADM_s115
                                 + 0.676805 w_k1_AXP_s115
                                 + 0.667787 w_k1_DIS_s115
                                 + 0.732108 w_k1_FDX_s115
                                 + 1.0243 w_k1_IBM_s115 + 0.7715 w_k1_INTC_s115
                                 + 0.975563 w_k1_KO_s115
                                 + 0.660432 w_k1_MAT_s115
                                 + 0.789651 w_k1_MSFT_s115
                                 + 1.07839 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.0969 w_k1_ADM_s115
                                 + 0.95924 w_k1_AXP_s115
                                 + 0.921733 w_k1_DIS_s115
                                 + 0.982583 w_k1_FDX_s115
                                 + 0.757899 w_k1_IBM_s115
                                 + 1.3403 w_k1_INTC_s115
                                 + 0.849885 w_k1_KO_s115
                                 + 0.554638 w_k1_MAT_s115
                                 + 1.05616 w_k1_MSFT_s115 + 1.0633 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.976145 w_k1_ADM_s115
                                 + 1.06105 w_k1_AXP_s115
                                 + 1.14091 w_k1_DIS_s115
                                 + 1.04841 w_k1_FDX_s115 + 1.1656 w_k1_IBM_s115
                                 + 1.27657 w_k1_INTC_s115
                                 + 1.13407 w_k1_KO_s115 + 1.16557 w_k1_MAT_s115
                                 + 1.18406 w_k1_MSFT_s115
                                 + 1.28321 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.37232 w_k1_ADM_s115
                                 + 1.12706 w_k1_AXP_s115
                                 + 1.27381 w_k1_DIS_s115
                                 + 1.01074 w_k1_FDX_s115
                                 + 1.46003 w_k1_IBM_s115
                                 + 0.910436 w_k1_INTC_s115 + 1.429 w_k1_KO_s115
                                 + 1.86586 w_k1_MAT_s115
                                 + 1.25356 w_k1_MSFT_s115
                                 + 1.28067 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.41591 w_k1_ADM_s115
                                 + 1.38196 w_k1_AXP_s115
                                 + 1.11094 w_k1_DIS_s115
                                 + 1.16197 w_k1_FDX_s115
                                 + 0.979224 w_k1_IBM_s115
                                 + 1.09714 w_k1_INTC_s115
                                 + 1.10117 w_k1_KO_s115 + 1.39909 w_k1_MAT_s115
                                 + 1.21192 w_k1_MSFT_s115 + 1.3231 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.871888 w_k1_ADM_s115
                                 + 0.629057 w_k1_AXP_s115
                                 + 0.598079 w_k1_DIS_s115
                                 + 0.707745 w_k1_FDX_s115
                                 + 0.726805 w_k1_IBM_s115
                                 + 1.04057 w_k1_INTC_s115
                                 + 0.662587 w_k1_KO_s115
                                 + 0.474474 w_k1_MAT_s115
                                 + 0.823008 w_k1_MSFT_s115
                                 + 0.87618 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.838644 w_k1_ADM_s116
                                 + 1.18741 w_k1_AXP_s116
                                 + 1.66893 w_k1_DIS_s116
                                 + 1.22677 w_k1_FDX_s116
                                 + 1.38435 w_k1_IBM_s116
                                 + 1.13879 w_k1_INTC_s116
                                 + 1.41528 w_k1_KO_s116 + 1.482 w_k1_MAT_s116
                                 + 1.26333 w_k1_MSFT_s116
                                 + 1.17895 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.1349 w_k1_ADM_s116
                                 + 0.967843 w_k1_AXP_s116
                                 + 0.967799 w_k1_DIS_s116
                                 + 1.03796 w_k1_FDX_s116
                                 + 0.791478 w_k1_IBM_s116
                                 + 1.0446 w_k1_INTC_s116
                                 + 0.974641 w_k1_KO_s116
                                 + 0.813473 w_k1_MAT_s116
                                 + 0.955042 w_k1_MSFT_s116
                                 + 1.02872 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.899189 w_k1_ADM_s116
                                 + 0.891008 w_k1_AXP_s116
                                 + 1.23498 w_k1_DIS_s116
                                 + 0.965786 w_k1_FDX_s116
                                 + 1.00716 w_k1_IBM_s116
                                 + 1.04244 w_k1_INTC_s116
                                 + 1.20267 w_k1_KO_s116
                                 + 0.918079 w_k1_MAT_s116
                                 + 1.08254 w_k1_MSFT_s116
                                 + 1.13876 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.98741 w_k1_ADM_s116
                                 + 1.08618 w_k1_AXP_s116
                                 + 0.929522 w_k1_DIS_s116
                                 + 1.01672 w_k1_FDX_s116
                                 + 1.17202 w_k1_IBM_s116
                                 + 0.813192 w_k1_INTC_s116
                                 + 1.05664 w_k1_KO_s116 + 1.25739 w_k1_MAT_s116
                                 + 0.844685 w_k1_MSFT_s116
                                 + 1.07265 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.741482 w_k1_ADM_s116
                                 + 0.637042 w_k1_AXP_s116
                                 + 0.747847 w_k1_DIS_s116
                                 + 0.825455 w_k1_FDX_s116
                                 + 0.786887 w_k1_IBM_s116
                                 + 1.02098 w_k1_INTC_s116
                                 + 0.834945 w_k1_KO_s116
                                 + 0.554019 w_k1_MAT_s116
                                 + 0.874379 w_k1_MSFT_s116
                                 + 0.842772 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.732308 w_k1_ADM_s116
                                 + 0.772095 w_k1_AXP_s116
                                 + 0.672344 w_k1_DIS_s116
                                 + 0.714317 w_k1_FDX_s116
                                 + 0.63728 w_k1_IBM_s116
                                 + 0.764843 w_k1_INTC_s116
                                 + 0.776825 w_k1_KO_s116
                                 + 0.73885 w_k1_MAT_s116
                                 + 0.738619 w_k1_MSFT_s116
                                 + 0.923127 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.676552 w_k1_ADM_s116
                                 + 0.737894 w_k1_AXP_s116
                                 + 0.710062 w_k1_DIS_s116
                                 + 0.740004 w_k1_FDX_s116
                                 + 0.931652 w_k1_IBM_s116
                                 + 0.802459 w_k1_INTC_s116
                                 + 0.859878 w_k1_KO_s116
                                 + 0.956387 w_k1_MAT_s116
                                 + 0.756824 w_k1_MSFT_s116
                                 + 0.970939 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.865955 w_k1_ADM_s116
                                 + 0.868448 w_k1_AXP_s116
                                 + 1.09106 w_k1_DIS_s116
                                 + 0.825189 w_k1_FDX_s116
                                 + 1.42098 w_k1_IBM_s116
                                 + 0.809971 w_k1_INTC_s116
                                 + 1.13711 w_k1_KO_s116 + 1.551 w_k1_MAT_s116
                                 + 0.806961 w_k1_MSFT_s116
                                 + 1.14346 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.883752 w_k1_ADM_s117
                                 + 1.25335 w_k1_AXP_s117
                                 + 0.854732 w_k1_DIS_s117
                                 + 0.93818 w_k1_FDX_s117
                                 + 1.45291 w_k1_IBM_s117
                                 + 0.874244 w_k1_INTC_s117
                                 + 1.06289 w_k1_KO_s117 + 1.7548 w_k1_MAT_s117
                                 + 0.838682 w_k1_MSFT_s117
                                 + 1.14432 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.23878 w_k1_ADM_s117
                                 + 1.22267 w_k1_AXP_s117
                                 + 1.11717 w_k1_DIS_s117
                                 + 1.12587 w_k1_FDX_s117
                                 + 0.929825 w_k1_IBM_s117
                                 + 0.963515 w_k1_INTC_s117
                                 + 1.0933 w_k1_KO_s117 + 0.987821 w_k1_MAT_s117
                                 + 1.05383 w_k1_MSFT_s117 + 1.1486 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.708036 w_k1_ADM_s117
                                 + 0.750797 w_k1_AXP_s117
                                 + 0.518947 w_k1_DIS_s117
                                 + 0.700594 w_k1_FDX_s117
                                 + 0.897605 w_k1_IBM_s117
                                 + 0.870044 w_k1_INTC_s117
                                 + 0.794637 w_k1_KO_s117
                                 + 0.917814 w_k1_MAT_s117
                                 + 0.795709 w_k1_MSFT_s117
                                 + 0.787085 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.964616 w_k1_ADM_s117
                                 + 1.06266 w_k1_AXP_s117
                                 + 0.949901 w_k1_DIS_s117
                                 + 0.966203 w_k1_FDX_s117
                                 + 1.08888 w_k1_IBM_s117
                                 + 0.914109 w_k1_INTC_s117
                                 + 0.971285 w_k1_KO_s117
                                 + 1.20623 w_k1_MAT_s117
                                 + 0.886332 w_k1_MSFT_s117
                                 + 0.94895 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.814582 w_k1_ADM_s117
                                 + 0.800316 w_k1_AXP_s117
                                 + 0.659418 w_k1_DIS_s117
                                 + 0.810394 w_k1_FDX_s117
                                 + 0.978074 w_k1_IBM_s117
                                 + 0.834992 w_k1_INTC_s117
                                 + 0.952202 w_k1_KO_s117
                                 + 0.917718 w_k1_MAT_s117
                                 + 0.801702 w_k1_MSFT_s117
                                 + 0.839668 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.12535 w_k1_ADM_s117
                                 + 1.18189 w_k1_AXP_s117
                                 + 1.32052 w_k1_DIS_s117
                                 + 1.09869 w_k1_FDX_s117
                                 + 1.16811 w_k1_IBM_s117
                                 + 0.942985 w_k1_INTC_s117
                                 + 1.18995 w_k1_KO_s117 + 1.03686 w_k1_MAT_s117
                                 + 1.15953 w_k1_MSFT_s117 + 1.0603 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.09413 w_k1_ADM_s117
                                 + 1.48956 w_k1_AXP_s117
                                 + 1.59361 w_k1_DIS_s117 + 1.2363 w_k1_FDX_s117
                                 + 1.44537 w_k1_IBM_s117
                                 + 1.40015 w_k1_INTC_s117
                                 + 1.61803 w_k1_KO_s117 + 1.29341 w_k1_MAT_s117
                                 + 1.33214 w_k1_MSFT_s117 + 1.1229 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.878766 w_k1_ADM_s117
                                 + 0.852056 w_k1_AXP_s117
                                 + 0.695017 w_k1_DIS_s117
                                 + 0.838275 w_k1_FDX_s117
                                 + 0.693004 w_k1_IBM_s117
                                 + 0.920535 w_k1_INTC_s117
                                 + 0.7821 w_k1_KO_s117 + 0.859062 w_k1_MAT_s117
                                 + 0.846236 w_k1_MSFT_s117
                                 + 0.930694 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0370869538420379 w_k1_ADM_s000
                                 + 0.0162986199757187 w_k1_ADM_s001
                                 + 0.0268751402385771 w_k1_ADM_s002
                                 + 0.0416102383524414 w_k1_ADM_s003
                                 + 0.035825853576903 w_k1_ADM_s004
                                 + 0.0226197399032679 w_k1_ADM_s005
                                 + 0.0136726635468004 w_k1_ADM_s006
                                 + 0.0236978823294258 w_k1_ADM_s007
                                 + 0.0435417646839796 w_k1_ADM_s010
                                 + 0.0470427228443308 w_k1_ADM_s011
                                 + 0.0162022093490397 w_k1_ADM_s012
                                 + 0.0232093643396499 w_k1_ADM_s013
                                 + 0.0752819163529257 w_k1_ADM_s014
                                 + 0.0471233020928001 w_k1_ADM_s015
                                 + 0.0723400038616508 w_k1_ADM_s016
                                 + 0.0469089800894413 w_k1_ADM_s017
                                 + 0.0230810619347813 w_k1_ADM_s100
                                 + 0.0875595500183791 w_k1_ADM_s101
                                 + 0.0281771972493571 w_k1_ADM_s102
                                 + 0.0405325153258176 w_k1_ADM_s103
                                 + 0.058691271945896 w_k1_ADM_s104
                                 + 0.0192309293796327 w_k1_ADM_s105
                                 + 0.0398182342608874 w_k1_ADM_s106
                                 + 0.027799033426516 w_k1_ADM_s107
                                 + 0.0151914386099086 w_k1_ADM_s110
                                 + 0.0271036775618589 w_k1_ADM_s111
                                 + 0.0115011747141512 w_k1_ADM_s112
                                 + 0.0157578779287516 w_k1_ADM_s113
                                 + 0.0066629739460373 w_k1_ADM_s114
                                 + 0.00971475752235939 w_k1_ADM_s115
                                 + 0.0116602598376149 w_k1_ADM_s116
                                 + 0.0165068262723949 w_k1_ADM_s117
                                 + 0.0381569997341699 w_k1_AXP_s000
                                 + 0.0165405888573524 w_k1_AXP_s001
                                 + 0.0255093257064093 w_k1_AXP_s002
                                 + 0.0489814164504703 w_k1_AXP_s003
                                 + 0.0361875754613627 w_k1_AXP_s004
                                 + 0.0224741217681949 w_k1_AXP_s005
                                 + 0.013606245833351 w_k1_AXP_s006
                                 + 0.0233375390648456 w_k1_AXP_s007
                                 + 0.0400227354065922 w_k1_AXP_s010
                                 + 0.047601296915776 w_k1_AXP_s011
                                 + 0.0149949926530746 w_k1_AXP_s012
                                 + 0.0232805260866347 w_k1_AXP_s013
                                 + 0.0711128452668412 w_k1_AXP_s014
                                 + 0.0410483673615134 w_k1_AXP_s015
                                 + 0.059027326288197 w_k1_AXP_s016
                                 + 0.043693406223183 w_k1_AXP_s017
                                 + 0.0230211062378052 w_k1_AXP_s100
                                 + 0.101629707828319 w_k1_AXP_s101
                                 + 0.0268700109815667 w_k1_AXP_s102
                                 + 0.0498432025342779 w_k1_AXP_s103
                                 + 0.0575290922800354 w_k1_AXP_s104
                                 + 0.0189245338792265 w_k1_AXP_s105
                                 + 0.0374880929460976 w_k1_AXP_s106
                                 + 0.0298412753219398 w_k1_AXP_s107
                                 + 0.0142108134748265 w_k1_AXP_s110
                                 + 0.0289330935264662 w_k1_AXP_s111
                                 + 0.0104167599223828 w_k1_AXP_s112
                                 + 0.0148144117825963 w_k1_AXP_s113
                                 + 0.00625904886387335 w_k1_AXP_s114
                                 + 0.0087115941282932 w_k1_AXP_s115
                                 + 0.0122541731719692 w_k1_AXP_s116
                                 + 0.0178697970742109 w_k1_AXP_s117
                                 + 0.0395476412241622 w_k1_DIS_s000
                                 + 0.0154752165100548 w_k1_DIS_s001
                                 + 0.0265931008598495 w_k1_DIS_s002
                                 + 0.0481006199594006 w_k1_DIS_s003
                                 + 0.0382646248668428 w_k1_DIS_s004
                                 + 0.0212570915064962 w_k1_DIS_s005
                                 + 0.0151445591405209 w_k1_DIS_s006
                                 + 0.026123064718129 w_k1_DIS_s007
                                 + 0.0405781968485209 w_k1_DIS_s010
                                 + 0.0476409758332598 w_k1_DIS_s011
                                 + 0.0159106583636558 w_k1_DIS_s012
                                 + 0.0248919402270733 w_k1_DIS_s013
                                 + 0.071732800197009 w_k1_DIS_s014
                                 + 0.0404167177525514 w_k1_DIS_s015
                                 + 0.0605179926331383 w_k1_DIS_s016
                                 + 0.0468977069130485 w_k1_DIS_s017
                                 + 0.0248030241198836 w_k1_DIS_s100
                                 + 0.0930563107302793 w_k1_DIS_s101
                                 + 0.0310611994917456 w_k1_DIS_s102
                                 + 0.0430828738000781 w_k1_DIS_s103
                                 + 0.0578044841839045 w_k1_DIS_s104
                                 + 0.0183161919219762 w_k1_DIS_s105
                                 + 0.0395009148842486 w_k1_DIS_s106
                                 + 0.0307927524792374 w_k1_DIS_s107
                                 + 0.0124557976390765 w_k1_DIS_s110
                                 + 0.0284858231839024 w_k1_DIS_s111
                                 + 0.01011229562088 w_k1_DIS_s112
                                 + 0.0141065217662326 w_k1_DIS_s113
                                 + 0.00597470998157759 w_k1_DIS_s114
                                 + 0.00872332740982704 w_k1_DIS_s115
                                 + 0.0140539260980528 w_k1_DIS_s116
                                 + 0.0158276388989419 w_k1_DIS_s117
                                 + 0.0372033542460063 w_k1_FDX_s000
                                 + 0.0155987767824884 w_k1_FDX_s001
                                 + 0.0261717441015449 w_k1_FDX_s002
                                 + 0.0472496566392723 w_k1_FDX_s003
                                 + 0.036507390455428 w_k1_FDX_s004
                                 + 0.0207824219404846 w_k1_FDX_s005
                                 + 0.0151871025712047 w_k1_FDX_s006
                                 + 0.0248937108743077 w_k1_FDX_s007
                                 + 0.037924689983155 w_k1_FDX_s010
                                 + 0.044249726936612 w_k1_FDX_s011
                                 + 0.014922774748892 w_k1_FDX_s012
                                 + 0.0243158080127435 w_k1_FDX_s013
                                 + 0.069582926748242 w_k1_FDX_s014
                                 + 0.0398284713646699 w_k1_FDX_s015
                                 + 0.062198851574019 w_k1_FDX_s016
                                 + 0.0448976346124391 w_k1_FDX_s017
                                 + 0.0238513397665565 w_k1_FDX_s100
                                 + 0.0994927071729375 w_k1_FDX_s101
                                 + 0.0297356026346342 w_k1_FDX_s102
                                 + 0.044201544001524 w_k1_FDX_s103
                                 + 0.0570342988754384 w_k1_FDX_s104
                                 + 0.0195970377486069 w_k1_FDX_s105
                                 + 0.0373348826154544 w_k1_FDX_s106
                                 + 0.0284953692725404 w_k1_FDX_s107
                                 + 0.0143380293435606 w_k1_FDX_s110
                                 + 0.0288675280582259 w_k1_FDX_s111
                                 + 0.010138575732632 w_k1_FDX_s112
                                 + 0.0142507011418101 w_k1_FDX_s113
                                 + 0.0064257710702513 w_k1_FDX_s114
                                 + 0.00836084886264213 w_k1_FDX_s115
                                 + 0.0126298862275056 w_k1_FDX_s116
                                 + 0.0162343068346612 w_k1_FDX_s117
                                 + 0.0379639888154902 w_k1_IBM_s000
                                 + 0.0146329655185262 w_k1_IBM_s001
                                 + 0.0287147291428417 w_k1_IBM_s002
                                 + 0.0486710009065558 w_k1_IBM_s003
                                 + 0.0387521840946938 w_k1_IBM_s004
                                 + 0.0213504955038452 w_k1_IBM_s005
                                 + 0.0152144998005573 w_k1_IBM_s006
                                 + 0.0248619129027095 w_k1_IBM_s007
                                 + 0.0395162851101239 w_k1_IBM_s010
                                 + 0.0440578153115653 w_k1_IBM_s011
                                 + 0.0154857021683473 w_k1_IBM_s012
                                 + 0.0233712310346996 w_k1_IBM_s013
                                 + 0.0753055210571569 w_k1_IBM_s014
                                 + 0.0428619522167367 w_k1_IBM_s015
                                 + 0.0621489556814227 w_k1_IBM_s016
                                 + 0.0486435280291065 w_k1_IBM_s017
                                 + 0.0235892527320758 w_k1_IBM_s100
                                 + 0.0969863457316017 w_k1_IBM_s101
                                 + 0.0304988933599255 w_k1_IBM_s102
                                 + 0.0461243091271516 w_k1_IBM_s103
                                 + 0.0599955532091101 w_k1_IBM_s104
                                 + 0.0199819844158497 w_k1_IBM_s105
                                 + 0.0408729746739896 w_k1_IBM_s106
                                 + 0.0307523635013326 w_k1_IBM_s107
                                 + 0.0135422708779233 w_k1_IBM_s110
                                 + 0.0291673811268245 w_k1_IBM_s111
                                 + 0.0103312562455314 w_k1_IBM_s112
                                 + 0.0141781292445772 w_k1_IBM_s113
                                 + 0.00685438284536029 w_k1_IBM_s114
                                 + 0.00850586190025084 w_k1_IBM_s115
                                 + 0.0134713407701597 w_k1_IBM_s116
                                 + 0.0173791913060277 w_k1_IBM_s117
                                 + 0.0386163021444235 w_k1_INTC_s000
                                 + 0.0155565720974185 w_k1_INTC_s001
                                 + 0.0281238748434967 w_k1_INTC_s002
                                 + 0.044784470239899 w_k1_INTC_s003
                                 + 0.0351178255622236 w_k1_INTC_s004
                                 + 0.0194464808028556 w_k1_INTC_s005
                                 + 0.0145101433455414 w_k1_INTC_s006
                                 + 0.0272190697448166 w_k1_INTC_s007
                                 + 0.0405997517581742 w_k1_INTC_s010
                                 + 0.0464810656164483 w_k1_INTC_s011
                                 + 0.016058575577042 w_k1_INTC_s012
                                 + 0.0240365522035438 w_k1_INTC_s013
                                 + 0.0732038125616247 w_k1_INTC_s014
                                 + 0.0425443012145905 w_k1_INTC_s015
                                 + 0.0647416549454758 w_k1_INTC_s016
                                 + 0.0369068718677523 w_k1_INTC_s017
                                 + 0.0184899203709931 w_k1_INTC_s100
                                 + 0.101712882212476 w_k1_INTC_s101
                                 + 0.0287395023381643 w_k1_INTC_s102
                                 + 0.0430170905041393 w_k1_INTC_s103
                                 + 0.0588415511193598 w_k1_INTC_s104
                                 + 0.0196133688922455 w_k1_INTC_s105
                                 + 0.0352037127516496 w_k1_INTC_s106
                                 + 0.0273362236755865 w_k1_INTC_s107
                                 + 0.0144743627519824 w_k1_INTC_s110
                                 + 0.0275701413473542 w_k1_INTC_s111
                                 + 0.00989236131472669 w_k1_INTC_s112
                                 + 0.0153470805272867 w_k1_INTC_s113
                                 + 0.00597216210630052 w_k1_INTC_s114
                                 + 0.00979073845189327 w_k1_INTC_s115
                                 + 0.0126022506803318 w_k1_INTC_s116
                                 + 0.0158048695744938 w_k1_INTC_s117
                                 + 0.0403321664127394 w_k1_KO_s000
                                 + 0.0143982316329705 w_k1_KO_s001
                                 + 0.0294253475452537 w_k1_KO_s002
                                 + 0.0484604397529822 w_k1_KO_s003
                                 + 0.0352619594453427 w_k1_KO_s004
                                 + 0.0228991162686447 w_k1_KO_s005
                                 + 0.0145288193775251 w_k1_KO_s006
                                 + 0.0235264427554065 w_k1_KO_s007
                                 + 0.0393689912101552 w_k1_KO_s010
                                 + 0.0457546610277611 w_k1_KO_s011
                                 + 0.0148140153837678 w_k1_KO_s012
                                 + 0.0229523873333744 w_k1_KO_s013
                                 + 0.0747479929995428 w_k1_KO_s014
                                 + 0.0428942152914528 w_k1_KO_s015
                                 + 0.0627192050008364 w_k1_KO_s016
                                 + 0.0457761912032875 w_k1_KO_s017
                                 + 0.0213118252637394 w_k1_KO_s100
                                 + 0.0871433579905454 w_k1_KO_s101
                                 + 0.0338864720004343 w_k1_KO_s102
                                 + 0.0405912866883116 w_k1_KO_s103
                                 + 0.0580200374917061 w_k1_KO_s104
                                 + 0.0184102810467445 w_k1_KO_s105
                                 + 0.0409821838544971 w_k1_KO_s106
                                 + 0.0279813085171006 w_k1_KO_s107
                                 + 0.0124441305980137 w_k1_KO_s110
                                 + 0.0262194680466778 w_k1_KO_s111
                                 + 0.0104801657125495 w_k1_KO_s112
                                 + 0.0161163262684322 w_k1_KO_s113
                                 + 0.00616630407763251 w_k1_KO_s114
                                 + 0.00864634123634086 w_k1_KO_s115
                                 + 0.0140266099786105 w_k1_KO_s116
                                 + 0.0172241735215479 w_k1_KO_s117
                                 + 0.0402417569460064 w_k1_MAT_s000
                                 + 0.0120361234142188 w_k1_MAT_s001
                                 + 0.0305309605718418 w_k1_MAT_s002
                                 + 0.050051958058186 w_k1_MAT_s003
                                 + 0.0353690321523311 w_k1_MAT_s004
                                 + 0.0240912956840829 w_k1_MAT_s005
                                 + 0.0155953419311272 w_k1_MAT_s006
                                 + 0.0220648221095396 w_k1_MAT_s007
                                 + 0.0319984599003942 w_k1_MAT_s010
                                 + 0.0441792335435354 w_k1_MAT_s011
                                 + 0.0149986455672078 w_k1_MAT_s012
                                 + 0.0229716157135966 w_k1_MAT_s013
                                 + 0.065006558272344 w_k1_MAT_s014
                                 + 0.0408284689002099 w_k1_MAT_s015
                                 + 0.0586771500107522 w_k1_MAT_s016
                                 + 0.0500024629379403 w_k1_MAT_s017
                                 + 0.0246309477796985 w_k1_MAT_s100
                                 + 0.0984705428353591 w_k1_MAT_s101
                                 + 0.0259921685621796 w_k1_MAT_s102
                                 + 0.0525053133794666 w_k1_MAT_s103
                                 + 0.0514186493600275 w_k1_MAT_s104
                                 + 0.0182191032401452 w_k1_MAT_s105
                                 + 0.0394700374426655 w_k1_MAT_s106
                                 + 0.0304409203304208 w_k1_MAT_s107
                                 + 0.0140649291038803 w_k1_MAT_s110
                                 + 0.0270822392163131 w_k1_MAT_s111
                                 + 0.0122383289576257 w_k1_MAT_s112
                                 + 0.0165302926088664 w_k1_MAT_s113
                                 + 0.00703663230265972 w_k1_MAT_s114
                                 + 0.00834113692271366 w_k1_MAT_s115
                                 + 0.0138169572895431 w_k1_MAT_s116
                                 + 0.0184088203223791 w_k1_MAT_s117
                                 + 0.0376044252229511 w_k1_MSFT_s000
                                 + 0.0156902699773594 w_k1_MSFT_s001
                                 + 0.0258163048293404 w_k1_MSFT_s002
                                 + 0.0474937424150603 w_k1_MSFT_s003
                                 + 0.035569524572011 w_k1_MSFT_s004
                                 + 0.0206876804390578 w_k1_MSFT_s005
                                 + 0.013228529896697 w_k1_MSFT_s006
                                 + 0.0243639417214771 w_k1_MSFT_s007
                                 + 0.0389357174931148 w_k1_MSFT_s010
                                 + 0.0446348949960347 w_k1_MSFT_s011
                                 + 0.0158723483362555 w_k1_MSFT_s012
                                 + 0.0228295881828463 w_k1_MSFT_s013
                                 + 0.0683155503054791 w_k1_MSFT_s014
                                 + 0.0405191667348309 w_k1_MSFT_s015
                                 + 0.0624034562002051 w_k1_MSFT_s016
                                 + 0.0496042135159005 w_k1_MSFT_s017
                                 + 0.0238699985852464 w_k1_MSFT_s100
                                 + 0.0975103560763487 w_k1_MSFT_s101
                                 + 0.0305206914836741 w_k1_MSFT_s102
                                 + 0.041242462943571 w_k1_MSFT_s103
                                 + 0.0617815584838656 w_k1_MSFT_s104
                                 + 0.0185780452872122 w_k1_MSFT_s105
                                 + 0.0375394494778729 w_k1_MSFT_s106
                                 + 0.0283656263721061 w_k1_MSFT_s107
                                 + 0.0136415313429385 w_k1_MSFT_s110
                                 + 0.0276924388579195 w_k1_MSFT_s111
                                 + 0.00937700407223825 w_k1_MSFT_s112
                                 + 0.0142893648646643 w_k1_MSFT_s113
                                 + 0.00651514148337771 w_k1_MSFT_s114
                                 + 0.00893586908918755 w_k1_MSFT_s115
                                 + 0.0125521497707173 w_k1_MSFT_s116
                                 + 0.0158383276645646 w_k1_MSFT_s117
                                 + 0.035066843737759 w_k1_PG_s000
                                 + 0.0141931496173754 w_k1_PG_s001
                                 + 0.0256237117554896 w_k1_PG_s002
                                 + 0.0453099593910956 w_k1_PG_s003
                                 + 0.0353821089725099 w_k1_PG_s004
                                 + 0.0221893611965863 w_k1_PG_s005
                                 + 0.0143290720826751 w_k1_PG_s006
                                 + 0.025434076187517 w_k1_PG_s007
                                 + 0.0392856752432205 w_k1_PG_s010
                                 + 0.0456200930071815 w_k1_PG_s011
                                 + 0.0153868934208474 w_k1_PG_s012
                                 + 0.0239877180919561 w_k1_PG_s013
                                 + 0.0715248845937769 w_k1_PG_s014
                                 + 0.0434082780805883 w_k1_PG_s015
                                 + 0.0631696092202443 w_k1_PG_s016
                                 + 0.049055780886129 w_k1_PG_s017
                                 + 0.0245011028965126 w_k1_PG_s100
                                 + 0.0908180202681258 w_k1_PG_s101
                                 + 0.0286113369042405 w_k1_PG_s102
                                 + 0.045971119484569 w_k1_PG_s103
                                 + 0.0602275031388414 w_k1_PG_s104
                                 + 0.0191882495578112 w_k1_PG_s105
                                 + 0.0371579303961544 w_k1_PG_s106
                                 + 0.0298887707146507 w_k1_PG_s107
                                 + 0.013630660038013 w_k1_PG_s110
                                 + 0.0325741988966644 w_k1_PG_s111
                                 + 0.00994676384889049 w_k1_PG_s112
                                 + 0.0145597717237013 w_k1_PG_s113
                                 + 0.00609229976436945 w_k1_PG_s114
                                 + 0.00956055201695287 w_k1_PG_s115
                                 + 0.0138289157469237 w_k1_PG_s116
                                 + 0.0168243739221495 w_k1_PG_s117 >= 1.15
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
