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
 w_k1_ADM_s0_return_constraint:  0.97984 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.05011 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.03042 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.09462 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.17475 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.839671 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.08098 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.13617 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.75553 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.678694 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.808228 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.937034 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.824777 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.16597 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.19857 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.00283 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.668282 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.02002 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.59073 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.24629 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.83162 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.838521 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.23872 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.13668 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.8383 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.928819 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.745042 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.790049 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.923405 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.867106 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.37568 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.31048 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.67287 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.858673 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.990086 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.783764 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.06756 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.931525 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.990131 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.0924 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.01665 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.988273 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.38912 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.854735 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.32427 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.44708 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 2.02 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.612381 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.581735 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.825096 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.935423 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.08229 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.50434 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.1334 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.580708 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.07618 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.84865 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.37252 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.949663 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.884034 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.84702 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.24565 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.35017 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.831876 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.683167 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.91001 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.916295 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.16792 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.34804 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.29655 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.64505 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.688487 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.24224 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.639613 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.82572 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.889573 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.02515 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.06043 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.05655 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.996041 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.464 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.886151 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.08032 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.46145 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 2.21709 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.410609 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.656879 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.654438 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.787446 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.977396 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.48256 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.04993 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.566784 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.04381 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 2.26744 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.42386 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.957969 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.852123 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.88135 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.33326 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.21262 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.873386 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.65799 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.980428 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.792854 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.12825 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.41599 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.18873 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 2.07914 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.659974 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.1688 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.475244 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.842485 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.85002 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.04536 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.98101 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.944108 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.02529 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.23434 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.951477 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.02416 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.30788 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.71471 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.569483 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.784427 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.838449 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.786386 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.1075 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.18308 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.05724 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.663604 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.16043 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.64745 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.31575 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.884027 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.86716 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.54028 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.33915 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.993286 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.06835 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.7601 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.968263 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.895688 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.05271 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.38305 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.16998 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.55287 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.697274 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.944887 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.669781 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.958982 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.876597 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.854773 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.2015 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.38361 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.940458 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.42819 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.891103 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.18942 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.25048 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.57294 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.695776 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.636995 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.73596 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.12718 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.882049 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.62345 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.972455 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.759185 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.760045 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.56325 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.14025 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.38446 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.848318 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.38518 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.865641 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.90824 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.628597 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.794142 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.23475 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.919759 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.47063 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.73322 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.07411 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.34538 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.971837 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.68097 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.66375 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.643265 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.04489 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.98696 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.0119 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.04279 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.98693 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.15139 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.978035 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.07126 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.23524 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.29649 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.610168 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.749509 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.838284 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.71781 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.992369 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.2561 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.05816 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.772708 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.22093 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.42208 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.21406 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.848663 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.880161 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.38483 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.3429 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.11243 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.05368 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.734469 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.11593 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.967845 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.04334 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.10313 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.00415 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.21591 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.696011 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.931 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.7521 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.805192 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.954004 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.983683 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.08744 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.08401 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.907461 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.34913 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.899285 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.17438 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.27154 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.80413 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.636026 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.705332 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.769542 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.977527 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.995095 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.41581 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.984804 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.70761 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.970639 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.73265 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.27046 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.09428 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.82331 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.25737 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.04353 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.37503 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.85424 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.826334 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.14206 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.849923 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.10741 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.02704 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.12454 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.36088 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.783467 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.26664 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.63766 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.780532 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.974061 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.87542 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.18946 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.26527 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.856149 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.40081 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.846198 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.40019 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.41139 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.33302 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.569359 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.524514 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.731609 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.15116 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.859965 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.79364 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.970688 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.774162 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.732418 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.99888 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.12802 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.38825 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.859414 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.74141 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.931188 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 2.43117 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.767818 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.743809 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.25244 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.22226 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.80835 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.01503 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.25194 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.77751 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.765724 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 2.05776 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.709373 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.692163 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.16537 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.05523 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.963288 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.940296 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.93357 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.26648 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.07207 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.05974 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.36522 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.41543 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.604067 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.81647 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.822486 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.802502 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.11049 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.19914 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.993728 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.790597 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.13595 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.62464 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.29299 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.903438 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.950634 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.3328 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.22861 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.03618 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.00506 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.811109 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.991565 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.901155 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.997876 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.08092 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.06118 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.31205 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.686245 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.997608 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.607226 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.846175 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.852933 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.01031 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.01154 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.973173 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.967689 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.27175 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.967876 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.08857 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.29397 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.52661 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.771421 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.862685 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.843324 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.985476 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.04313 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.26012 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.02636 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.72812 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.00407 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.44347 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.17545 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.954511 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.910065 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.24162 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.06738 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.08029 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.8679 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.744866 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.949557 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.872698 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.00667 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.13043 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.0841 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.34622 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.839841 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.09949 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.734025 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.882102 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.959387 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.173000352483025 z_k1_s0000
                                 - 0.180000447797382 z_k1_s0001
                                 - 0.150999986152175 z_k1_s0002
                                 - 0.0709999429311458 z_k1_s0003
                                 - 0.313999378957392 z_k1_s0004
                                 - 0.238000124687562 z_k1_s0005
                                 - 0.269999892695927 z_k1_s0006
                                 - 0.161000634229925 z_k1_s0007
                                 - 0.265998823948036 z_k1_s0010
                                 - 0.16599977913278 z_k1_s0011
                                 - 0.168999698538335 z_k1_s0012
                                 - 0.284000131378024 z_k1_s0013
                                 - 0.131999677638388 z_k1_s0014
                                 - 0.399999403055013 z_k1_s0015
                                 - 0.238000173161747 z_k1_s0016
                                 - 0.135999689578905 z_k1_s0017
                                 - 0.187999277864549 z_k1_s0020
                                 - 0.108999774506259 z_k1_s0021
                                 - 0.139000045654228 z_k1_s0022
                                 - 0.0619998642402696 z_k1_s0023
                                 - 0.0479998890376376 z_k1_s0024
                                 - 0.143999485513209 z_k1_s0025
                                 - 0.130999670681929 z_k1_s0026
                                 - 0.0869998783305888 z_k1_s0027
                                 - 0.174999866329841 z_k1_s0030
                                 - 0.214999393377425 z_k1_s0031
                                 - 0.0869998008270643 z_k1_s0032
                                 - 0.167000182119943 z_k1_s0033
                                 - 0.218999788481421 z_k1_s0034
                                 - 0.0709998794082215 z_k1_s0035
                                 - 0.107999829026569 z_k1_s0036
                                 - 0.0639999068749897 z_k1_s0037
                                 - 0.0859998316418963 z_k1_s0040
                                 - 0.091999763724024 z_k1_s0041
                                 - 0.0419998961740013 z_k1_s0042
                                 - 0.164999293505722 z_k1_s0043
                                 - 0.0869998048389548 z_k1_s0044
                                 - 0.188999950832001 z_k1_s0045
                                 - 0.104999972684445 z_k1_s0046
                                 - 0.162999161312052 z_k1_s0047
                                 - 0.238999622950286 z_k1_s0050
                                 - 0.198999218427605 z_k1_s0051
                                 - 0.284000269413147 z_k1_s0052
                                 - 0.154000102963502 z_k1_s0053
                                 - 0.185999507982392 z_k1_s0054
                                 - 0.154999845151786 z_k1_s0055
                                 - 0.165000329032146 z_k1_s0056
                                 - 0.148999861023322 z_k1_s0057
                                 - 0.13800020421952 z_k1_s0060
                                 - 0.181000999412219 z_k1_s0061
                                 - 0.308000141953236 z_k1_s0062
                                 - 0.119000200413971 z_k1_s0063
                                 - 0.0930001309957535 z_k1_s0064
                                 - 0.306999740436546 z_k1_s0065
                                 - 0.175000115314391 z_k1_s0066
                                 - 0.203999559279904 z_k1_s0067
                                 - 0.451999907113517 z_k1_s0070
                                 - 0.152999908479625 z_k1_s0071
                                 - 0.353998704961594 z_k1_s0072
                                 - 0.424998835388598 z_k1_s0073
                                 - 0.393999338962261 z_k1_s0074
                                 - 0.0959998829029511 z_k1_s0075
                                 - 0.296000477808407 z_k1_s0076
                                 - 0.170999842630214 z_k1_s0077
                                 - 0.191000005799548 z_k1_s0100
                                 - 0.452000621427235 z_k1_s0101
                                 - 0.421999438872271 z_k1_s0102
                                 - 0.285000870771622 z_k1_s0103
                                 - 0.354001057347908 z_k1_s0104
                                 - 0.143999918763529 z_k1_s0105
                                 - 0.197999974662358 z_k1_s0106
                                 - 0.257000995320386 z_k1_s0107
                                 - 0.433001521140325 z_k1_s0110
                                 - 0.466999004793231 z_k1_s0111
                                 - 0.712999733828948 z_k1_s0112
                                 - 0.199000140676442 z_k1_s0113
                                 - 0.417001414243637 z_k1_s0114
                                 - 0.225000131483389 z_k1_s0115
                                 - 0.669999903794526 z_k1_s0116
                                 - 0.533998995172684 z_k1_s0117
                                 - 1.39700312788192 z_k1_s0120
                                 - 1.26499775831714 z_k1_s0121
                                 - 0.887999508882155 z_k1_s0122
                                 - 0.647999924630285 z_k1_s0123
                                 - 1.04599836926861 z_k1_s0124
                                 - 1.37800137862617 z_k1_s0125
                                 - 0.385000088227128 z_k1_s0126
                                 - 1.04800406471782 z_k1_s0127
                                 - 0.409000835159754 z_k1_s0130
                                 - 0.525000503654292 z_k1_s0131
                                 - 0.213000326488452 z_k1_s0132
                                 - 0.257000200745001 z_k1_s0133
                                 - 0.346999676342182 z_k1_s0134
                                 - 0.274999882104193 z_k1_s0135
                                 - 0.176000298627111 z_k1_s0136
                                 - 0.469999992832843 z_k1_s0137
                                 - 0.521000933474721 z_k1_s0140
                                 - 0.753000962845187 z_k1_s0141
                                 - 0.30100019921609 z_k1_s0142
                                 - 0.816999359842955 z_k1_s0143
                                 - 0.249000019129501 z_k1_s0144
                                 - 0.39499881069465 z_k1_s0145
                                 - 0.334000327324896 z_k1_s0146
                                 - 0.22400013989569 z_k1_s0147
                                 - 1.27099708285734 z_k1_s0150
                                 - 1.26299639111808 z_k1_s0151
                                 - 0.462999827371228 z_k1_s0152
                                 - 0.979999913083729 z_k1_s0153
                                 - 0.636000026686968 z_k1_s0154
                                 - 1.15499723163515 z_k1_s0155
                                 - 0.720998724269086 z_k1_s0156
                                 - 0.297000063677391 z_k1_s0157
                                 - 0.496000421721199 z_k1_s0160
                                 - 0.188000080576547 z_k1_s0161
                                 - 0.362999906463114 z_k1_s0162
                                 - 0.205000035578658 z_k1_s0163
                                 - 0.500001567533984 z_k1_s0164
                                 - 0.501999249439067 z_k1_s0165
                                 - 0.137999923823149 z_k1_s0166
                                 - 0.499001281080788 z_k1_s0167
                                 - 0.862999464144744 z_k1_s0170
                                 - 1.28599589766981 z_k1_s0171
                                 - 0.573999600914047 z_k1_s0172
                                 - 1.76799620419101 z_k1_s0173
                                 - 0.58499927244498 z_k1_s0174
                                 - 1.08299828455752 z_k1_s0175
                                 - 1.2000017322426 z_k1_s0176
                                 - 1.62399887087877 z_k1_s0177
                                 - 0.0899998808577204 z_k1_s1000
                                 - 0.0849997489766007 z_k1_s1001
                                 - 0.154000239334754 z_k1_s1002
                                 - 0.048000019557532 z_k1_s1003
                                 - 0.122000060096237 z_k1_s1004
                                 - 0.0869999679292108 z_k1_s1005
                                 - 0.181999980668051 z_k1_s1006
                                 - 0.0630000828005665 z_k1_s1007
                                 - 0.597001613425074 z_k1_s1010
                                 - 0.594001230060082 z_k1_s1011
                                 - 0.184000576788854 z_k1_s1012
                                 - 0.302001253971693 z_k1_s1013
                                 - 0.17100051574502 z_k1_s1014
                                 - 0.221000504343696 z_k1_s1015
                                 - 0.222000632132027 z_k1_s1016
                                 - 0.376000042278403 z_k1_s1017
                                 - 0.12399975720826 z_k1_s1020
                                 - 0.0300000364213895 z_k1_s1021
                                 - 0.0300000364213895 z_k1_s1022
                                 - 0.0530000151777693 z_k1_s1023
                                 - 0.0779997996955003 z_k1_s1024
                                 - 0.138999922919011 z_k1_s1025
                                 - 0.103999929594075 z_k1_s1026
                                 - 0.032000019182808 z_k1_s1027
                                 - 0.0610000266068962 z_k1_s1030
                                 - 0.187000239027714 z_k1_s1031
                                 - 0.115999895803262 z_k1_s1032
                                 - 0.0800000095401892 z_k1_s1033
                                 - 0.135000285736597 z_k1_s1034
                                 - 0.0839999245471899 z_k1_s1035
                                 - 0.0440000418771078 z_k1_s1036
                                 - 0.107000310887533 z_k1_s1037
                                 - 0.32300048816466 z_k1_s1040
                                 - 0.385999785748334 z_k1_s1041
                                 - 0.251999355333409 z_k1_s1042
                                 - 0.113999955793538 z_k1_s1043
                                 - 0.195999931592394 z_k1_s1044
                                 - 0.158999887067333 z_k1_s1045
                                 - 0.232999326617841 z_k1_s1046
                                 - 0.502000416853637 z_k1_s1047
                                 - 0.651000295404743 z_k1_s1050
                                 - 0.36000184669777 z_k1_s1051
                                 - 0.402001426077702 z_k1_s1052
                                 - 0.292001619032917 z_k1_s1053
                                 - 0.152000476827245 z_k1_s1054
                                 - 0.230000513477586 z_k1_s1055
                                 - 0.327000814181806 z_k1_s1056
                                 - 0.311999601400099 z_k1_s1057
                                 - 0.15400004226862 z_k1_s1060
                                 - 0.306000274531079 z_k1_s1061
                                 - 0.138999911123026 z_k1_s1062
                                 - 0.428998815705959 z_k1_s1063
                                 - 0.299000050309866 z_k1_s1064
                                 - 0.145000008041246 z_k1_s1065
                                 - 0.512999283361393 z_k1_s1066
                                 - 0.238000954523878 z_k1_s1067
                                 - 0.0990004137950352 z_k1_s1070
                                 - 0.0560000352090983 z_k1_s1071
                                 - 0.0310000150407461 z_k1_s1072
                                 - 0.0460000645217978 z_k1_s1073
                                 - 0.0749999359049224 z_k1_s1074
                                 - 0.13499988462886 z_k1_s1075
                                 - 0.103999994188249 z_k1_s1076
                                 - 0.0770003910628799 z_k1_s1077
                                 - 0.293999321413485 z_k1_s1100
                                 - 0.103999955663312 z_k1_s1101
                                 - 0.109000008856752 z_k1_s1102
                                 - 0.443000641017928 z_k1_s1103
                                 - 0.19299997304635 z_k1_s1104
                                 - 0.295999873810966 z_k1_s1105
                                 - 0.3160009717849 z_k1_s1106
                                 - 0.458000357998158 z_k1_s1107
                                 - 0.539999711582419 z_k1_s1110
                                 - 0.652999913221063 z_k1_s1111
                                 - 1.11599828491736 z_k1_s1112
                                 - 0.505999898483101 z_k1_s1113
                                 - 0.971996443509337 z_k1_s1114
                                 - 1.58099890246583 z_k1_s1115
                                 - 1.62400122855238 z_k1_s1116
                                 - 1.00100303112422 z_k1_s1117
                                 - 0.159000028189569 z_k1_s1120
                                 - 0.260000193444602 z_k1_s1121
                                 - 0.386999238283911 z_k1_s1122
                                 - 0.790999899304042 z_k1_s1123
                                 - 0.214999928602229 z_k1_s1124
                                 - 0.558001257805088 z_k1_s1125
                                 - 0.399998836376075 z_k1_s1126
                                 - 0.396000177732528 z_k1_s1127
                                 - 0.726001559790754 z_k1_s1130
                                 - 0.441003127347858 z_k1_s1131
                                 - 0.458002101883128 z_k1_s1132
                                 - 0.53700232750657 z_k1_s1133
                                 - 0.29000110374656 z_k1_s1134
                                 - 0.20500069754985 z_k1_s1135
                                 - 0.299001190155783 z_k1_s1136
                                 - 0.733003307548554 z_k1_s1137
                                 - 0.722996974965074 z_k1_s1140
                                 - 0.42699795264717 z_k1_s1141
                                 - 0.341998375784317 z_k1_s1142
                                 - 0.388996654321987 z_k1_s1143
                                 - 0.296998686785959 z_k1_s1144
                                 - 0.424997884314265 z_k1_s1145
                                 - 0.750997931625735 z_k1_s1146
                                 - 0.342998409950769 z_k1_s1147
                                 - 0.758999872748796 z_k1_s1150
                                 - 0.228000194650251 z_k1_s1151
                                 - 0.242000005444548 z_k1_s1152
                                 - 0.624999365317474 z_k1_s1153
                                 - 0.432998532938259 z_k1_s1154
                                 - 0.312999953101414 z_k1_s1155
                                 - 0.39600039378202 z_k1_s1156
                                 - 0.532001277383989 z_k1_s1157
                                 - 0.868002718777123 z_k1_s1160
                                 - 0.43900047333364 z_k1_s1161
                                 - 0.401000895338483 z_k1_s1162
                                 - 0.196000193116578 z_k1_s1163
                                 - 0.730001084241041 z_k1_s1164
                                 - 0.175000548140549 z_k1_s1165
                                 - 0.731999888217398 z_k1_s1166
                                 - 0.667002990914629 z_k1_s1167
                                 - 2.1880012775159 z_k1_s1170
                                 - 0.502000034129787 z_k1_s1171
                                 - 0.956998846070079 z_k1_s1172
                                 - 0.480000022790231 z_k1_s1173
                                 - 1.01800292064034 z_k1_s1174
                                 - 0.517000266479568 z_k1_s1175
                                 - 1.17699895946564 z_k1_s1176
                                 - 1.3969990728612 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.02944 w_k1_ADM_s000
                                 + 1.18838 w_k1_AXP_s000
                                 + 0.944801 w_k1_DIS_s000
                                 + 1.09041 w_k1_FDX_s000
                                 + 0.745878 w_k1_IBM_s000
                                 + 1.075 w_k1_INTC_s000 + 0.801709 w_k1_KO_s000
                                 + 0.782531 w_k1_MAT_s000
                                 + 1.01012 w_k1_MSFT_s000
                                 + 0.935432 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.89078 w_k1_ADM_s000
                                 + 0.931897 w_k1_AXP_s000
                                 + 1.0646 w_k1_DIS_s000
                                 + 0.954957 w_k1_FDX_s000
                                 + 1.0824 w_k1_IBM_s000
                                 + 1.00864 w_k1_INTC_s000
                                 + 1.05213 w_k1_KO_s000
                                 + 0.981936 w_k1_MAT_s000
                                 + 1.01632 w_k1_MSFT_s000
                                 + 0.958744 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.14114 w_k1_ADM_s000
                                 + 0.898061 w_k1_AXP_s000
                                 + 0.77363 w_k1_DIS_s000
                                 + 1.04389 w_k1_FDX_s000
                                 + 0.914853 w_k1_IBM_s000
                                 + 0.974498 w_k1_INTC_s000
                                 + 0.742278 w_k1_KO_s000
                                 + 0.703032 w_k1_MAT_s000
                                 + 0.990573 w_k1_MSFT_s000
                                 + 0.899514 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.97842 w_k1_ADM_s000
                                 + 1.02159 w_k1_AXP_s000
                                 + 1.51533 w_k1_DIS_s000
                                 + 1.14498 w_k1_FDX_s000
                                 + 1.14298 w_k1_IBM_s000
                                 + 0.971451 w_k1_INTC_s000
                                 + 1.33973 w_k1_KO_s000
                                 + 0.838407 w_k1_MAT_s000
                                 + 1.29369 w_k1_MSFT_s000
                                 + 1.20987 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.728183 w_k1_ADM_s000
                                 + 0.671688 w_k1_AXP_s000
                                 + 0.6986 w_k1_DIS_s000
                                 + 0.804226 w_k1_FDX_s000
                                 + 0.769362 w_k1_IBM_s000
                                 + 0.824647 w_k1_INTC_s000
                                 + 0.829238 w_k1_KO_s000
                                 + 0.957541 w_k1_MAT_s000
                                 + 0.891274 w_k1_MSFT_s000
                                 + 0.887548 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.04781 w_k1_ADM_s000
                                 + 0.848037 w_k1_AXP_s000
                                 + 0.899639 w_k1_DIS_s000
                                 + 0.916014 w_k1_FDX_s000
                                 + 1.16142 w_k1_IBM_s000
                                 + 0.880538 w_k1_INTC_s000
                                 + 1.07731 w_k1_KO_s000 + 1.20364 w_k1_MAT_s000
                                 + 0.941169 w_k1_MSFT_s000
                                 + 0.998865 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.40013 w_k1_ADM_s000
                                 + 1.34748 w_k1_AXP_s000
                                 + 1.25357 w_k1_DIS_s000
                                 + 1.32245 w_k1_FDX_s000
                                 + 1.05798 w_k1_IBM_s000
                                 + 1.21947 w_k1_INTC_s000
                                 + 1.04212 w_k1_KO_s000 + 1.02176 w_k1_MAT_s000
                                 + 1.10363 w_k1_MSFT_s000
                                 + 1.13636 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.08948 w_k1_ADM_s000
                                 + 1.24362 w_k1_AXP_s000 + 1.1889 w_k1_DIS_s000
                                 + 1.07917 w_k1_FDX_s000 + 1.4919 w_k1_IBM_s000
                                 + 1.11942 w_k1_INTC_s000
                                 + 1.34459 w_k1_KO_s000 + 1.35292 w_k1_MAT_s000
                                 + 1.00333 w_k1_MSFT_s000
                                 + 1.16448 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.17825 w_k1_ADM_s001
                                 + 1.05433 w_k1_AXP_s001
                                 + 0.960772 w_k1_DIS_s001
                                 + 1.08029 w_k1_FDX_s001
                                 + 0.671269 w_k1_IBM_s001
                                 + 0.90974 w_k1_INTC_s001
                                 + 0.986546 w_k1_KO_s001
                                 + 0.794074 w_k1_MAT_s001
                                 + 0.895505 w_k1_MSFT_s001
                                 + 1.01703 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.45637 w_k1_ADM_s001
                                 + 1.38348 w_k1_AXP_s001
                                 + 1.80262 w_k1_DIS_s001
                                 + 1.42046 w_k1_FDX_s001
                                 + 1.36289 w_k1_IBM_s001
                                 + 1.4296 w_k1_INTC_s001 + 1.48503 w_k1_KO_s001
                                 + 1.41265 w_k1_MAT_s001
                                 + 1.42432 w_k1_MSFT_s001
                                 + 1.22814 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.888603 w_k1_ADM_s001
                                 + 0.857872 w_k1_AXP_s001
                                 + 0.695566 w_k1_DIS_s001
                                 + 0.775603 w_k1_FDX_s001
                                 + 1.16226 w_k1_IBM_s001
                                 + 0.903234 w_k1_INTC_s001
                                 + 1.03895 w_k1_KO_s001 + 1.23408 w_k1_MAT_s001
                                 + 0.764826 w_k1_MSFT_s001
                                 + 0.902426 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.927315 w_k1_ADM_s001
                                 + 0.776864 w_k1_AXP_s001
                                 + 0.658805 w_k1_DIS_s001
                                 + 0.835977 w_k1_FDX_s001
                                 + 0.744023 w_k1_IBM_s001
                                 + 0.857676 w_k1_INTC_s001
                                 + 0.791551 w_k1_KO_s001
                                 + 0.791135 w_k1_MAT_s001
                                 + 0.814972 w_k1_MSFT_s001
                                 + 0.718577 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.0141 w_k1_ADM_s001
                                 + 1.39478 w_k1_AXP_s001
                                 + 1.02739 w_k1_DIS_s001
                                 + 1.00531 w_k1_FDX_s001
                                 + 1.37416 w_k1_IBM_s001
                                 + 1.06271 w_k1_INTC_s001 + 1.1403 w_k1_KO_s001
                                 + 1.49085 w_k1_MAT_s001
                                 + 0.881725 w_k1_MSFT_s001
                                 + 1.01556 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.37832 w_k1_ADM_s001
                                 + 1.25436 w_k1_AXP_s001
                                 + 1.31737 w_k1_DIS_s001
                                 + 1.18886 w_k1_FDX_s001
                                 + 1.07391 w_k1_IBM_s001
                                 + 0.95313 w_k1_INTC_s001
                                 + 1.15196 w_k1_KO_s001 + 1.16628 w_k1_MAT_s001
                                 + 1.01079 w_k1_MSFT_s001
                                 + 1.07326 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.04037 w_k1_ADM_s001
                                 + 1.02925 w_k1_AXP_s001 + 1.0494 w_k1_DIS_s001
                                 + 0.934993 w_k1_FDX_s001
                                 + 1.08534 w_k1_IBM_s001
                                 + 0.888184 w_k1_INTC_s001
                                 + 1.03164 w_k1_KO_s001 + 1.01411 w_k1_MAT_s001
                                 + 0.82842 w_k1_MSFT_s001
                                 + 0.860768 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.30512 w_k1_ADM_s001
                                 + 1.02524 w_k1_AXP_s001
                                 + 1.10463 w_k1_DIS_s001
                                 + 0.913828 w_k1_FDX_s001
                                 + 1.81616 w_k1_IBM_s001
                                 + 0.827978 w_k1_INTC_s001
                                 + 1.44352 w_k1_KO_s001 + 1.65681 w_k1_MAT_s001
                                 + 0.878018 w_k1_MSFT_s001
                                 + 1.17633 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.5878 w_k1_ADM_s002
                                 + 0.750658 w_k1_AXP_s002
                                 + 0.653008 w_k1_DIS_s002
                                 + 0.713898 w_k1_FDX_s002
                                 + 1.01298 w_k1_IBM_s002
                                 + 0.830068 w_k1_INTC_s002
                                 + 0.811772 w_k1_KO_s002
                                 + 0.783573 w_k1_MAT_s002
                                 + 0.84482 w_k1_MSFT_s002
                                 + 0.836595 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.42312 w_k1_ADM_s002
                                 + 1.2988 w_k1_AXP_s002 + 1.66871 w_k1_DIS_s002
                                 + 1.38763 w_k1_FDX_s002 + 1.1931 w_k1_IBM_s002
                                 + 1.41923 w_k1_INTC_s002
                                 + 1.32875 w_k1_KO_s002 + 1.02314 w_k1_MAT_s002
                                 + 1.5925 w_k1_MSFT_s002 + 1.17873 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.770309 w_k1_ADM_s002
                                 + 0.937054 w_k1_AXP_s002
                                 + 0.760249 w_k1_DIS_s002
                                 + 0.874585 w_k1_FDX_s002
                                 + 0.690275 w_k1_IBM_s002
                                 + 1.12008 w_k1_INTC_s002
                                 + 0.745225 w_k1_KO_s002
                                 + 0.488987 w_k1_MAT_s002
                                 + 1.13447 w_k1_MSFT_s002
                                 + 0.893696 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.783342 w_k1_ADM_s002
                                 + 0.944096 w_k1_AXP_s002
                                 + 0.841792 w_k1_DIS_s002
                                 + 0.765262 w_k1_FDX_s002
                                 + 1.64917 w_k1_IBM_s002
                                 + 0.978918 w_k1_INTC_s002
                                 + 1.06368 w_k1_KO_s002 + 1.31404 w_k1_MAT_s002
                                 + 0.978328 w_k1_MSFT_s002
                                 + 1.00446 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.09744 w_k1_ADM_s002
                                 + 1.47512 w_k1_AXP_s002
                                 + 1.01062 w_k1_DIS_s002
                                 + 1.14179 w_k1_FDX_s002
                                 + 0.676467 w_k1_IBM_s002
                                 + 1.13427 w_k1_INTC_s002
                                 + 0.812686 w_k1_KO_s002
                                 + 0.854413 w_k1_MAT_s002
                                 + 1.09831 w_k1_MSFT_s002
                                 + 1.05982 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.891878 w_k1_ADM_s002
                                 + 0.989768 w_k1_AXP_s002
                                 + 1.07732 w_k1_DIS_s002
                                 + 1.04076 w_k1_FDX_s002
                                 + 1.00291 w_k1_IBM_s002
                                 + 1.15198 w_k1_INTC_s002
                                 + 0.989878 w_k1_KO_s002
                                 + 0.777498 w_k1_MAT_s002
                                 + 1.10787 w_k1_MSFT_s002
                                 + 0.956695 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.79741 w_k1_ADM_s002
                                 + 1.19913 w_k1_AXP_s002
                                 + 1.11723 w_k1_DIS_s002
                                 + 1.06195 w_k1_FDX_s002
                                 + 1.41399 w_k1_IBM_s002
                                 + 1.41936 w_k1_INTC_s002
                                 + 1.12874 w_k1_KO_s002 + 1.43335 w_k1_MAT_s002
                                 + 1.18441 w_k1_MSFT_s002
                                 + 1.04106 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.937225 w_k1_ADM_s002
                                 + 1.28431 w_k1_AXP_s002
                                 + 1.08981 w_k1_DIS_s002
                                 + 1.08653 w_k1_FDX_s002
                                 + 1.01273 w_k1_IBM_s002
                                 + 1.08557 w_k1_INTC_s002
                                 + 0.940174 w_k1_KO_s002
                                 + 0.899626 w_k1_MAT_s002
                                 + 1.01489 w_k1_MSFT_s002
                                 + 1.01471 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.2452 w_k1_ADM_s003
                                 + 1.22747 w_k1_AXP_s003
                                 + 1.45423 w_k1_DIS_s003 + 1.2348 w_k1_FDX_s003
                                 + 1.22756 w_k1_IBM_s003
                                 + 1.41232 w_k1_INTC_s003
                                 + 1.26286 w_k1_KO_s003 + 1.40434 w_k1_MAT_s003
                                 + 1.29601 w_k1_MSFT_s003
                                 + 1.02164 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.0263 w_k1_ADM_s003
                                 + 0.964864 w_k1_AXP_s003
                                 + 0.951166 w_k1_DIS_s003
                                 + 1.05185 w_k1_FDX_s003
                                 + 0.776804 w_k1_IBM_s003
                                 + 1.11684 w_k1_INTC_s003
                                 + 0.829407 w_k1_KO_s003
                                 + 0.722987 w_k1_MAT_s003
                                 + 0.929986 w_k1_MSFT_s003
                                 + 0.928943 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.02151 w_k1_ADM_s003
                                 + 1.13243 w_k1_AXP_s003
                                 + 1.37013 w_k1_DIS_s003
                                 + 1.27419 w_k1_FDX_s003
                                 + 1.01551 w_k1_IBM_s003
                                 + 1.32788 w_k1_INTC_s003
                                 + 1.18172 w_k1_KO_s003
                                 + 0.877857 w_k1_MAT_s003
                                 + 1.30406 w_k1_MSFT_s003
                                 + 0.954765 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.730005 w_k1_ADM_s003
                                 + 0.519912 w_k1_AXP_s003
                                 + 0.760683 w_k1_DIS_s003
                                 + 0.841134 w_k1_FDX_s003
                                 + 0.622509 w_k1_IBM_s003
                                 + 0.773511 w_k1_INTC_s003
                                 + 0.738464 w_k1_KO_s003
                                 + 0.603156 w_k1_MAT_s003
                                 + 0.93105 w_k1_MSFT_s003
                                 + 0.778667 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.820662 w_k1_ADM_s003
                                 + 0.799339 w_k1_AXP_s003
                                 + 1.10457 w_k1_DIS_s003
                                 + 0.888548 w_k1_FDX_s003
                                 + 0.976938 w_k1_IBM_s003
                                 + 1.18944 w_k1_INTC_s003
                                 + 0.981144 w_k1_KO_s003
                                 + 0.92152 w_k1_MAT_s003
                                 + 1.1562 w_k1_MSFT_s003
                                 + 0.884714 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.878789 w_k1_ADM_s003
                                 + 0.844427 w_k1_AXP_s003
                                 + 0.848619 w_k1_DIS_s003
                                 + 0.942805 w_k1_FDX_s003
                                 + 1.25055 w_k1_IBM_s003
                                 + 1.05416 w_k1_INTC_s003
                                 + 1.03805 w_k1_KO_s003 + 1.70465 w_k1_MAT_s003
                                 + 0.997184 w_k1_MSFT_s003
                                 + 0.979869 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.07981 w_k1_ADM_s003
                                 + 1.17034 w_k1_AXP_s003
                                 + 1.64454 w_k1_DIS_s003
                                 + 1.19466 w_k1_FDX_s003
                                 + 1.66428 w_k1_IBM_s003
                                 + 1.25162 w_k1_INTC_s003
                                 + 1.51285 w_k1_KO_s003 + 2.11784 w_k1_MAT_s003
                                 + 1.34327 w_k1_MSFT_s003
                                 + 1.09347 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.33072 w_k1_ADM_s003
                                 + 1.00774 w_k1_AXP_s003
                                 + 1.63294 w_k1_DIS_s003
                                 + 1.29589 w_k1_FDX_s003
                                 + 0.756306 w_k1_IBM_s003
                                 + 1.1348 w_k1_INTC_s003 + 1.25274 w_k1_KO_s003
                                 + 1.01475 w_k1_MAT_s003
                                 + 1.30686 w_k1_MSFT_s003 + 1.0396 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.998569 w_k1_ADM_s004
                                 + 1.17193 w_k1_AXP_s004
                                 + 1.04613 w_k1_DIS_s004
                                 + 1.08802 w_k1_FDX_s004
                                 + 1.10268 w_k1_IBM_s004
                                 + 1.15387 w_k1_INTC_s004
                                 + 1.04638 w_k1_KO_s004 + 1.60457 w_k1_MAT_s004
                                 + 1.00152 w_k1_MSFT_s004
                                 + 1.04162 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.1538 w_k1_ADM_s004
                                 + 1.36644 w_k1_AXP_s004 + 1.0672 w_k1_DIS_s004
                                 + 1.11138 w_k1_FDX_s004 + 1.0022 w_k1_IBM_s004
                                 + 1.19183 w_k1_INTC_s004
                                 + 1.18956 w_k1_KO_s004 + 1.14064 w_k1_MAT_s004
                                 + 0.994113 w_k1_MSFT_s004
                                 + 1.05834 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.918306 w_k1_ADM_s004
                                 + 0.961719 w_k1_AXP_s004
                                 + 1.37596 w_k1_DIS_s004
                                 + 1.09954 w_k1_FDX_s004
                                 + 0.964615 w_k1_IBM_s004
                                 + 1.00758 w_k1_INTC_s004
                                 + 1.24799 w_k1_KO_s004 + 1.17553 w_k1_MAT_s004
                                 + 1.21499 w_k1_MSFT_s004
                                 + 1.04378 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.1588 w_k1_ADM_s004
                                 + 1.03943 w_k1_AXP_s004
                                 + 0.90849 w_k1_DIS_s004
                                 + 0.950343 w_k1_FDX_s004
                                 + 0.921301 w_k1_IBM_s004
                                 + 0.980535 w_k1_INTC_s004
                                 + 1.01516 w_k1_KO_s004 + 1.2293 w_k1_MAT_s004
                                 + 0.965647 w_k1_MSFT_s004
                                 + 0.956355 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.952372 w_k1_ADM_s004
                                 + 1.3275 w_k1_AXP_s004 + 1.4673 w_k1_DIS_s004
                                 + 1.18354 w_k1_FDX_s004
                                 + 1.20658 w_k1_IBM_s004
                                 + 1.60079 w_k1_INTC_s004
                                 + 1.44449 w_k1_KO_s004 + 1.23061 w_k1_MAT_s004
                                 + 1.30086 w_k1_MSFT_s004
                                 + 1.06197 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.697952 w_k1_ADM_s004
                                 + 0.950042 w_k1_AXP_s004
                                 + 0.758295 w_k1_DIS_s004
                                 + 0.853302 w_k1_FDX_s004
                                 + 0.699862 w_k1_IBM_s004
                                 + 0.850064 w_k1_INTC_s004
                                 + 0.802513 w_k1_KO_s004
                                 + 1.03139 w_k1_MAT_s004
                                 + 0.887784 w_k1_MSFT_s004
                                 + 0.80059 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.22617 w_k1_ADM_s004
                                 + 1.32772 w_k1_AXP_s004
                                 + 1.41967 w_k1_DIS_s004
                                 + 1.20472 w_k1_FDX_s004
                                 + 1.62361 w_k1_IBM_s004
                                 + 1.22695 w_k1_INTC_s004 + 1.4711 w_k1_KO_s004
                                 + 2.26721 w_k1_MAT_s004
                                 + 1.34077 w_k1_MSFT_s004
                                 + 1.21099 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.965233 w_k1_ADM_s004
                                 + 0.85886 w_k1_AXP_s004
                                 + 0.803681 w_k1_DIS_s004
                                 + 0.845997 w_k1_FDX_s004
                                 + 1.14969 w_k1_IBM_s004
                                 + 0.933224 w_k1_INTC_s004
                                 + 1.08753 w_k1_KO_s004 + 1.29021 w_k1_MAT_s004
                                 + 0.952599 w_k1_MSFT_s004
                                 + 1.02404 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.683146 w_k1_ADM_s005
                                 + 0.824601 w_k1_AXP_s005
                                 + 0.645464 w_k1_DIS_s005
                                 + 0.776651 w_k1_FDX_s005
                                 + 0.941674 w_k1_IBM_s005
                                 + 1.04182 w_k1_INTC_s005
                                 + 0.90768 w_k1_KO_s005
                                 + 0.942123 w_k1_MAT_s005
                                 + 0.819306 w_k1_MSFT_s005
                                 + 0.980745 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.38694 w_k1_ADM_s005
                                 + 1.38307 w_k1_AXP_s005
                                 + 1.55349 w_k1_DIS_s005
                                 + 1.28655 w_k1_FDX_s005
                                 + 1.11694 w_k1_IBM_s005
                                 + 0.961257 w_k1_INTC_s005
                                 + 1.28421 w_k1_KO_s005 + 1.23596 w_k1_MAT_s005
                                 + 1.03492 w_k1_MSFT_s005
                                 + 1.30659 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06225 w_k1_ADM_s005
                                 + 1.19845 w_k1_AXP_s005
                                 + 1.16663 w_k1_DIS_s005
                                 + 1.01303 w_k1_FDX_s005
                                 + 1.45438 w_k1_IBM_s005
                                 + 0.836511 w_k1_INTC_s005
                                 + 1.05236 w_k1_KO_s005 + 1.64786 w_k1_MAT_s005
                                 + 0.940682 w_k1_MSFT_s005
                                 + 1.18884 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.896232 w_k1_ADM_s005
                                 + 1.22341 w_k1_AXP_s005
                                 + 0.893428 w_k1_DIS_s005
                                 + 0.995782 w_k1_FDX_s005
                                 + 1.03067 w_k1_IBM_s005
                                 + 1.29124 w_k1_INTC_s005
                                 + 1.06171 w_k1_KO_s005 + 1.26478 w_k1_MAT_s005
                                 + 1.12294 w_k1_MSFT_s005
                                 + 1.02057 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.931349 w_k1_ADM_s005
                                 + 0.780941 w_k1_AXP_s005
                                 + 0.949614 w_k1_DIS_s005
                                 + 0.889963 w_k1_FDX_s005
                                 + 0.908549 w_k1_IBM_s005
                                 + 0.792004 w_k1_INTC_s005
                                 + 0.917636 w_k1_KO_s005
                                 + 0.89981 w_k1_MAT_s005
                                 + 0.936594 w_k1_MSFT_s005
                                 + 1.0517 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.851965 w_k1_ADM_s005
                                 + 0.883454 w_k1_AXP_s005
                                 + 0.752208 w_k1_DIS_s005
                                 + 0.911349 w_k1_FDX_s005
                                 + 0.632158 w_k1_IBM_s005
                                 + 0.805704 w_k1_INTC_s005
                                 + 0.702984 w_k1_KO_s005
                                 + 0.740549 w_k1_MAT_s005
                                 + 0.860238 w_k1_MSFT_s005
                                 + 0.945649 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.985044 w_k1_ADM_s005
                                 + 1.09216 w_k1_AXP_s005
                                 + 1.06906 w_k1_DIS_s005
                                 + 0.959688 w_k1_FDX_s005
                                 + 1.02067 w_k1_IBM_s005
                                 + 0.916756 w_k1_INTC_s005
                                 + 1.1357 w_k1_KO_s005 + 1.22774 w_k1_MAT_s005
                                 + 1.00985 w_k1_MSFT_s005
                                 + 1.11166 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.10926 w_k1_ADM_s005
                                 + 1.18306 w_k1_AXP_s005
                                 + 1.41901 w_k1_DIS_s005
                                 + 1.27634 w_k1_FDX_s005
                                 + 0.746929 w_k1_IBM_s005
                                 + 1.01601 w_k1_INTC_s005
                                 + 1.09226 w_k1_KO_s005 + 1.03573 w_k1_MAT_s005
                                 + 1.21064 w_k1_MSFT_s005
                                 + 1.12412 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.81054 w_k1_ADM_s006
                                 + 0.9107 w_k1_AXP_s006
                                 + 0.834737 w_k1_DIS_s006
                                 + 0.872267 w_k1_FDX_s006
                                 + 1.27702 w_k1_IBM_s006
                                 + 1.01165 w_k1_INTC_s006 + 1.0264 w_k1_KO_s006
                                 + 1.33251 w_k1_MAT_s006
                                 + 0.821719 w_k1_MSFT_s006
                                 + 0.992389 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.921403 w_k1_ADM_s006
                                 + 0.918658 w_k1_AXP_s006
                                 + 1.20334 w_k1_DIS_s006
                                 + 1.13177 w_k1_FDX_s006
                                 + 1.19621 w_k1_IBM_s006
                                 + 1.32051 w_k1_INTC_s006
                                 + 1.14139 w_k1_KO_s006 + 1.08524 w_k1_MAT_s006
                                 + 1.35302 w_k1_MSFT_s006
                                 + 1.08649 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.840387 w_k1_ADM_s006
                                 + 0.775501 w_k1_AXP_s006
                                 + 0.70111 w_k1_DIS_s006
                                 + 0.873105 w_k1_FDX_s006
                                 + 0.908709 w_k1_IBM_s006
                                 + 1.02806 w_k1_INTC_s006
                                 + 0.916127 w_k1_KO_s006
                                 + 0.798911 w_k1_MAT_s006
                                 + 0.923735 w_k1_MSFT_s006
                                 + 0.88216 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.819493 w_k1_ADM_s006
                                 + 1.12262 w_k1_AXP_s006
                                 + 1.20578 w_k1_DIS_s006
                                 + 0.992975 w_k1_FDX_s006
                                 + 1.47391 w_k1_IBM_s006
                                 + 1.27496 w_k1_INTC_s006
                                 + 1.23856 w_k1_KO_s006 + 1.35142 w_k1_MAT_s006
                                 + 1.04019 w_k1_MSFT_s006
                                 + 1.09174 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.5133 w_k1_ADM_s006
                                 + 1.45145 w_k1_AXP_s006
                                 + 1.33849 w_k1_DIS_s006
                                 + 1.26218 w_k1_FDX_s006
                                 + 1.98621 w_k1_IBM_s006
                                 + 1.1182 w_k1_INTC_s006 + 1.36494 w_k1_KO_s006
                                 + 2.17234 w_k1_MAT_s006
                                 + 1.15651 w_k1_MSFT_s006 + 1.2051 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.27076 w_k1_ADM_s006
                                 + 1.20284 w_k1_AXP_s006 + 1.2177 w_k1_DIS_s006
                                 + 1.2126 w_k1_FDX_s006
                                 + 0.943082 w_k1_IBM_s006
                                 + 1.67918 w_k1_INTC_s006
                                 + 1.04035 w_k1_KO_s006 + 1.13454 w_k1_MAT_s006
                                 + 1.22844 w_k1_MSFT_s006
                                 + 1.14219 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.04643 w_k1_ADM_s006
                                 + 0.737031 w_k1_AXP_s006
                                 + 0.897573 w_k1_DIS_s006
                                 + 1.01125 w_k1_FDX_s006
                                 + 0.962764 w_k1_IBM_s006
                                 + 1.04503 w_k1_INTC_s006
                                 + 0.97641 w_k1_KO_s006
                                 + 0.841519 w_k1_MAT_s006
                                 + 0.93831 w_k1_MSFT_s006
                                 + 0.918893 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.889214 w_k1_ADM_s006
                                 + 0.562742 w_k1_AXP_s006
                                 + 0.636876 w_k1_DIS_s006
                                 + 0.812808 w_k1_FDX_s006
                                 + 0.679354 w_k1_IBM_s006
                                 + 0.874756 w_k1_INTC_s006
                                 + 0.683156 w_k1_KO_s006
                                 + 0.667737 w_k1_MAT_s006
                                 + 0.799564 w_k1_MSFT_s006
                                 + 0.865836 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.684949 w_k1_ADM_s007
                                 + 0.896874 w_k1_AXP_s007
                                 + 0.969236 w_k1_DIS_s007
                                 + 0.864866 w_k1_FDX_s007
                                 + 1.17779 w_k1_IBM_s007
                                 + 0.924223 w_k1_INTC_s007
                                 + 1.00376 w_k1_KO_s007 + 1.36149 w_k1_MAT_s007
                                 + 0.832132 w_k1_MSFT_s007
                                 + 1.0181 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.898384 w_k1_ADM_s007
                                 + 0.863464 w_k1_AXP_s007
                                 + 0.685419 w_k1_DIS_s007
                                 + 0.750335 w_k1_FDX_s007
                                 + 0.891703 w_k1_IBM_s007
                                 + 0.736923 w_k1_INTC_s007
                                 + 0.752922 w_k1_KO_s007
                                 + 1.27401 w_k1_MAT_s007
                                 + 0.678872 w_k1_MSFT_s007
                                 + 0.861601 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.04006 w_k1_ADM_s007
                                 + 1.37153 w_k1_AXP_s007
                                 + 1.85849 w_k1_DIS_s007
                                 + 1.24866 w_k1_FDX_s007
                                 + 1.27878 w_k1_IBM_s007
                                 + 1.45183 w_k1_INTC_s007
                                 + 1.42958 w_k1_KO_s007 + 1.17515 w_k1_MAT_s007
                                 + 1.19959 w_k1_MSFT_s007
                                 + 1.24036 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.12216 w_k1_ADM_s007
                                 + 1.15719 w_k1_AXP_s007
                                 + 1.18083 w_k1_DIS_s007
                                 + 1.04647 w_k1_FDX_s007
                                 + 1.04846 w_k1_IBM_s007
                                 + 1.02001 w_k1_INTC_s007
                                 + 1.18855 w_k1_KO_s007 + 0.98919 w_k1_MAT_s007
                                 + 1.01781 w_k1_MSFT_s007
                                 + 1.07732 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.25423 w_k1_ADM_s007
                                 + 1.17206 w_k1_AXP_s007
                                 + 1.21512 w_k1_DIS_s007
                                 + 1.13986 w_k1_FDX_s007
                                 + 0.817182 w_k1_IBM_s007
                                 + 1.24874 w_k1_INTC_s007
                                 + 1.06239 w_k1_KO_s007
                                 + 0.967803 w_k1_MAT_s007
                                 + 1.12527 w_k1_MSFT_s007
                                 + 1.00717 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.03809 w_k1_ADM_s007
                                 + 1.35493 w_k1_AXP_s007
                                 + 1.03193 w_k1_DIS_s007 + 1.101 w_k1_FDX_s007
                                 + 1.54283 w_k1_IBM_s007
                                 + 1.00526 w_k1_INTC_s007
                                 + 1.14626 w_k1_KO_s007 + 2.25064 w_k1_MAT_s007
                                 + 0.927474 w_k1_MSFT_s007
                                 + 1.23049 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.609567 w_k1_ADM_s007
                                 + 0.643914 w_k1_AXP_s007
                                 + 0.793267 w_k1_DIS_s007
                                 + 0.750045 w_k1_FDX_s007
                                 + 0.719129 w_k1_IBM_s007
                                 + 0.764837 w_k1_INTC_s007
                                 + 0.824299 w_k1_KO_s007
                                 + 0.663248 w_k1_MAT_s007
                                 + 0.689649 w_k1_MSFT_s007
                                 + 0.893915 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.01018 w_k1_ADM_s007
                                 + 0.891343 w_k1_AXP_s007
                                 + 0.795649 w_k1_DIS_s007
                                 + 0.919191 w_k1_FDX_s007
                                 + 0.74711 w_k1_IBM_s007
                                 + 1.00659 w_k1_INTC_s007
                                 + 0.724161 w_k1_KO_s007
                                 + 0.745431 w_k1_MAT_s007
                                 + 0.836538 w_k1_MSFT_s007
                                 + 0.969116 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.848342 w_k1_ADM_s010
                                 + 1.29554 w_k1_AXP_s010
                                 + 1.07745 w_k1_DIS_s010
                                 + 1.08764 w_k1_FDX_s010
                                 + 1.61742 w_k1_IBM_s010
                                 + 0.973577 w_k1_INTC_s010
                                 + 1.26088 w_k1_KO_s010 + 2.35182 w_k1_MAT_s010
                                 + 0.958868 w_k1_MSFT_s010
                                 + 0.929261 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.828231 w_k1_ADM_s010
                                 + 0.87449 w_k1_AXP_s010
                                 + 0.810788 w_k1_DIS_s010
                                 + 0.911145 w_k1_FDX_s010
                                 + 1.06841 w_k1_IBM_s010
                                 + 1.04944 w_k1_INTC_s010
                                 + 1.02946 w_k1_KO_s010 + 1.24216 w_k1_MAT_s010
                                 + 1.03955 w_k1_MSFT_s010
                                 + 0.950527 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.09918 w_k1_ADM_s010
                                 + 1.19189 w_k1_AXP_s010
                                 + 1.27146 w_k1_DIS_s010
                                 + 1.21648 w_k1_FDX_s010 + 1.2271 w_k1_IBM_s010
                                 + 1.06698 w_k1_INTC_s010
                                 + 0.958616 w_k1_KO_s010
                                 + 1.40744 w_k1_MAT_s010
                                 + 1.24782 w_k1_MSFT_s010
                                 + 1.06935 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.622679 w_k1_ADM_s010
                                 + 0.656419 w_k1_AXP_s010
                                 + 0.588588 w_k1_DIS_s010
                                 + 0.75572 w_k1_FDX_s010
                                 + 0.770008 w_k1_IBM_s010
                                 + 0.702635 w_k1_INTC_s010
                                 + 0.789381 w_k1_KO_s010
                                 + 0.740098 w_k1_MAT_s010
                                 + 0.758918 w_k1_MSFT_s010
                                 + 0.715715 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.866202 w_k1_ADM_s010
                                 + 0.824038 w_k1_AXP_s010
                                 + 0.857981 w_k1_DIS_s010
                                 + 0.966234 w_k1_FDX_s010
                                 + 0.774407 w_k1_IBM_s010
                                 + 0.931984 w_k1_INTC_s010
                                 + 0.974362 w_k1_KO_s010
                                 + 0.980859 w_k1_MAT_s010
                                 + 0.942286 w_k1_MSFT_s010
                                 + 0.961676 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.674956 w_k1_ADM_s010
                                 + 0.884699 w_k1_AXP_s010
                                 + 0.923849 w_k1_DIS_s010
                                 + 0.796085 w_k1_FDX_s010
                                 + 1.49537 w_k1_IBM_s010
                                 + 1.1049 w_k1_INTC_s010 + 1.20741 w_k1_KO_s010
                                 + 1.35589 w_k1_MAT_s010
                                 + 1.02136 w_k1_MSFT_s010
                                 + 1.02468 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.20814 w_k1_ADM_s010
                                 + 1.21836 w_k1_AXP_s010
                                 + 1.29948 w_k1_DIS_s010 + 1.2998 w_k1_FDX_s010
                                 + 0.967336 w_k1_IBM_s010
                                 + 1.13258 w_k1_INTC_s010
                                 + 1.17294 w_k1_KO_s010
                                 + 0.884168 w_k1_MAT_s010
                                 + 1.27952 w_k1_MSFT_s010
                                 + 1.10075 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.00457 w_k1_ADM_s010
                                 + 1.28787 w_k1_AXP_s010
                                 + 1.07759 w_k1_DIS_s010 + 1.2282 w_k1_FDX_s010
                                 + 0.83465 w_k1_IBM_s010
                                 + 1.09742 w_k1_INTC_s010
                                 + 0.97203 w_k1_KO_s010 + 1.07437 w_k1_MAT_s010
                                 + 1.12326 w_k1_MSFT_s010
                                 + 1.02247 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.86095 w_k1_ADM_s011
                                 + 0.925677 w_k1_AXP_s011
                                 + 0.922963 w_k1_DIS_s011
                                 + 1.00118 w_k1_FDX_s011
                                 + 0.74434 w_k1_IBM_s011
                                 + 1.02831 w_k1_INTC_s011
                                 + 0.875858 w_k1_KO_s011
                                 + 1.10054 w_k1_MAT_s011
                                 + 1.06263 w_k1_MSFT_s011
                                 + 0.938304 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.15682 w_k1_ADM_s011
                                 + 1.38449 w_k1_AXP_s011
                                 + 1.36238 w_k1_DIS_s011
                                 + 1.25178 w_k1_FDX_s011
                                 + 0.974265 w_k1_IBM_s011
                                 + 1.28523 w_k1_INTC_s011
                                 + 1.11017 w_k1_KO_s011 + 1.07904 w_k1_MAT_s011
                                 + 1.27714 w_k1_MSFT_s011
                                 + 1.14377 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.951613 w_k1_ADM_s011
                                 + 0.756646 w_k1_AXP_s011
                                 + 0.693676 w_k1_DIS_s011
                                 + 0.851384 w_k1_FDX_s011
                                 + 0.738291 w_k1_IBM_s011
                                 + 1.0167 w_k1_INTC_s011
                                 + 0.822279 w_k1_KO_s011
                                 + 0.783499 w_k1_MAT_s011
                                 + 1.00559 w_k1_MSFT_s011
                                 + 0.903536 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.02204 w_k1_ADM_s011
                                 + 1.12041 w_k1_AXP_s011
                                 + 1.02026 w_k1_DIS_s011
                                 + 0.986306 w_k1_FDX_s011
                                 + 1.5901 w_k1_IBM_s011
                                 + 1.29566 w_k1_INTC_s011
                                 + 1.44659 w_k1_KO_s011 + 1.81217 w_k1_MAT_s011
                                 + 1.04654 w_k1_MSFT_s011
                                 + 1.06455 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.28216 w_k1_ADM_s011
                                 + 1.37043 w_k1_AXP_s011
                                 + 1.03412 w_k1_DIS_s011
                                 + 1.21013 w_k1_FDX_s011
                                 + 1.01819 w_k1_IBM_s011
                                 + 1.14835 w_k1_INTC_s011
                                 + 1.01153 w_k1_KO_s011 + 1.50809 w_k1_MAT_s011
                                 + 1.13736 w_k1_MSFT_s011
                                 + 1.04223 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.813578 w_k1_ADM_s011
                                 + 0.852359 w_k1_AXP_s011
                                 + 0.857354 w_k1_DIS_s011
                                 + 0.773639 w_k1_FDX_s011
                                 + 1.1749 w_k1_IBM_s011
                                 + 0.979858 w_k1_INTC_s011
                                 + 0.980213 w_k1_KO_s011
                                 + 1.10019 w_k1_MAT_s011
                                 + 1.0587 w_k1_MSFT_s011 + 0.91897 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.989049 w_k1_ADM_s011
                                 + 1.07515 w_k1_AXP_s011
                                 + 0.840301 w_k1_DIS_s011
                                 + 0.974741 w_k1_FDX_s011
                                 + 1.01354 w_k1_IBM_s011
                                 + 0.977728 w_k1_INTC_s011
                                 + 1.00658 w_k1_KO_s011 + 1.03454 w_k1_MAT_s011
                                 + 0.955495 w_k1_MSFT_s011
                                 + 1.04139 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.17981 w_k1_ADM_s011
                                 + 1.14821 w_k1_AXP_s011 + 1.3743 w_k1_DIS_s011
                                 + 1.07111 w_k1_FDX_s011
                                 + 1.28825 w_k1_IBM_s011
                                 + 1.11525 w_k1_INTC_s011
                                 + 1.39316 w_k1_KO_s011 + 1.53213 w_k1_MAT_s011
                                 + 1.2933 w_k1_MSFT_s011 + 1.19604 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.999479 w_k1_ADM_s012
                                 + 1.01567 w_k1_AXP_s012
                                 + 1.01721 w_k1_DIS_s012
                                 + 1.07558 w_k1_FDX_s012
                                 + 0.941784 w_k1_IBM_s012
                                 + 0.991919 w_k1_INTC_s012
                                 + 1.05907 w_k1_KO_s012 + 1.08923 w_k1_MAT_s012
                                 + 1.04504 w_k1_MSFT_s012
                                 + 1.08547 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.966214 w_k1_ADM_s012
                                 + 1.28151 w_k1_AXP_s012
                                 + 1.15423 w_k1_DIS_s012
                                 + 1.00733 w_k1_FDX_s012
                                 + 1.71419 w_k1_IBM_s012
                                 + 0.972452 w_k1_INTC_s012
                                 + 1.3273 w_k1_KO_s012 + 1.54869 w_k1_MAT_s012
                                 + 0.90625 w_k1_MSFT_s012
                                 + 1.05952 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.951487 w_k1_ADM_s012
                                 + 0.679268 w_k1_AXP_s012
                                 + 0.811053 w_k1_DIS_s012
                                 + 0.908286 w_k1_FDX_s012
                                 + 0.604974 w_k1_IBM_s012
                                 + 0.887287 w_k1_INTC_s012
                                 + 0.837029 w_k1_KO_s012
                                 + 0.530859 w_k1_MAT_s012
                                 + 0.926512 w_k1_MSFT_s012
                                 + 1.00354 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.787776 w_k1_ADM_s012
                                 + 0.810672 w_k1_AXP_s012
                                 + 0.514362 w_k1_DIS_s012
                                 + 0.719359 w_k1_FDX_s012
                                 + 0.844862 w_k1_IBM_s012
                                 + 0.807871 w_k1_INTC_s012
                                 + 0.78471 w_k1_KO_s012
                                 + 0.760002 w_k1_MAT_s012
                                 + 0.719753 w_k1_MSFT_s012
                                 + 0.816132 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.770533 w_k1_ADM_s012
                                 + 0.810809 w_k1_AXP_s012
                                 + 0.778397 w_k1_DIS_s012
                                 + 0.823341 w_k1_FDX_s012
                                 + 0.939535 w_k1_IBM_s012
                                 + 0.855713 w_k1_INTC_s012
                                 + 0.950707 w_k1_KO_s012
                                 + 0.826365 w_k1_MAT_s012
                                 + 0.792829 w_k1_MSFT_s012
                                 + 1.00839 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.20975 w_k1_ADM_s012
                                 + 1.35236 w_k1_AXP_s012 + 1.1138 w_k1_DIS_s012
                                 + 1.27303 w_k1_FDX_s012
                                 + 0.855753 w_k1_IBM_s012
                                 + 1.17095 w_k1_INTC_s012
                                 + 0.943902 w_k1_KO_s012
                                 + 0.97842 w_k1_MAT_s012
                                 + 1.09746 w_k1_MSFT_s012
                                 + 1.15113 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.788836 w_k1_ADM_s012
                                 + 0.543278 w_k1_AXP_s012
                                 + 0.536747 w_k1_DIS_s012
                                 + 0.675236 w_k1_FDX_s012
                                 + 0.525921 w_k1_IBM_s012
                                 + 0.744191 w_k1_INTC_s012
                                 + 0.564164 w_k1_KO_s012
                                 + 0.397622 w_k1_MAT_s012
                                 + 0.623484 w_k1_MSFT_s012
                                 + 0.775709 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.14304 w_k1_ADM_s012
                                 + 1.12689 w_k1_AXP_s012 + 1.3553 w_k1_DIS_s012
                                 + 1.10548 w_k1_FDX_s012
                                 + 1.32418 w_k1_IBM_s012
                                 + 0.99225 w_k1_INTC_s012
                                 + 1.17907 w_k1_KO_s012 + 1.15307 w_k1_MAT_s012
                                 + 1.05525 w_k1_MSFT_s012
                                 + 1.08162 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.936186 w_k1_ADM_s013
                                 + 0.91873 w_k1_AXP_s013
                                 + 0.838118 w_k1_DIS_s013
                                 + 0.877442 w_k1_FDX_s013
                                 + 0.839968 w_k1_IBM_s013
                                 + 0.832978 w_k1_INTC_s013
                                 + 0.894454 w_k1_KO_s013
                                 + 1.09989 w_k1_MAT_s013
                                 + 0.849949 w_k1_MSFT_s013
                                 + 0.964193 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.860019 w_k1_ADM_s013
                                 + 0.558775 w_k1_AXP_s013
                                 + 0.705106 w_k1_DIS_s013
                                 + 0.760531 w_k1_FDX_s013
                                 + 0.733913 w_k1_IBM_s013
                                 + 0.61679 w_k1_INTC_s013
                                 + 0.744443 w_k1_KO_s013
                                 + 0.824806 w_k1_MAT_s013
                                 + 0.815928 w_k1_MSFT_s013
                                 + 0.834111 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.6139 w_k1_ADM_s013
                                 + 1.40301 w_k1_AXP_s013
                                 + 1.34523 w_k1_DIS_s013
                                 + 1.34404 w_k1_FDX_s013
                                 + 0.971992 w_k1_IBM_s013
                                 + 0.924168 w_k1_INTC_s013
                                 + 1.21572 w_k1_KO_s013 + 1.32912 w_k1_MAT_s013
                                 + 1.17168 w_k1_MSFT_s013
                                 + 1.12568 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.977002 w_k1_ADM_s013
                                 + 0.817626 w_k1_AXP_s013
                                 + 0.945849 w_k1_DIS_s013
                                 + 0.987641 w_k1_FDX_s013
                                 + 0.662695 w_k1_IBM_s013
                                 + 1.04709 w_k1_INTC_s013
                                 + 0.791962 w_k1_KO_s013
                                 + 0.925736 w_k1_MAT_s013
                                 + 1.07386 w_k1_MSFT_s013
                                 + 0.953829 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.07468 w_k1_ADM_s013
                                 + 1.05064 w_k1_AXP_s013
                                 + 1.11884 w_k1_DIS_s013
                                 + 1.05966 w_k1_FDX_s013
                                 + 0.799822 w_k1_IBM_s013
                                 + 1.00462 w_k1_INTC_s013
                                 + 0.872794 w_k1_KO_s013
                                 + 0.940455 w_k1_MAT_s013
                                 + 1.0866 w_k1_MSFT_s013
                                 + 0.949234 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.1708 w_k1_ADM_s013
                                 + 1.00309 w_k1_AXP_s013
                                 + 1.24844 w_k1_DIS_s013
                                 + 1.21252 w_k1_FDX_s013
                                 + 1.05237 w_k1_IBM_s013
                                 + 1.05986 w_k1_INTC_s013 + 1.121 w_k1_KO_s013
                                 + 1.37055 w_k1_MAT_s013
                                 + 1.10697 w_k1_MSFT_s013
                                 + 1.13888 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.3334 w_k1_ADM_s013
                                 + 1.48737 w_k1_AXP_s013
                                 + 1.71206 w_k1_DIS_s013
                                 + 1.39495 w_k1_FDX_s013
                                 + 1.25191 w_k1_IBM_s013
                                 + 1.05518 w_k1_INTC_s013
                                 + 1.41508 w_k1_KO_s013 + 1.7969 w_k1_MAT_s013
                                 + 1.2008 w_k1_MSFT_s013 + 1.28207 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.08748 w_k1_ADM_s013
                                 + 1.19609 w_k1_AXP_s013
                                 + 1.00752 w_k1_DIS_s013
                                 + 1.02845 w_k1_FDX_s013
                                 + 1.24566 w_k1_IBM_s013
                                 + 0.835442 w_k1_INTC_s013
                                 + 1.00593 w_k1_KO_s013 + 1.57131 w_k1_MAT_s013
                                 + 0.929597 w_k1_MSFT_s013
                                 + 1.04009 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.710251 w_k1_ADM_s014
                                 + 0.71399 w_k1_AXP_s014
                                 + 0.826585 w_k1_DIS_s014
                                 + 0.808346 w_k1_FDX_s014
                                 + 0.706265 w_k1_IBM_s014
                                 + 0.942133 w_k1_INTC_s014
                                 + 0.902012 w_k1_KO_s014
                                 + 0.748176 w_k1_MAT_s014
                                 + 1.01539 w_k1_MSFT_s014
                                 + 0.872787 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.672157 w_k1_ADM_s014
                                 + 0.793621 w_k1_AXP_s014
                                 + 0.78443 w_k1_DIS_s014
                                 + 0.783689 w_k1_FDX_s014
                                 + 1.18789 w_k1_IBM_s014
                                 + 0.798254 w_k1_INTC_s014
                                 + 0.968278 w_k1_KO_s014
                                 + 1.30446 w_k1_MAT_s014
                                 + 0.923875 w_k1_MSFT_s014
                                 + 0.878557 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.83041 w_k1_ADM_s014
                                 + 0.819804 w_k1_AXP_s014
                                 + 1.18599 w_k1_DIS_s014
                                 + 0.884124 w_k1_FDX_s014
                                 + 1.21454 w_k1_IBM_s014
                                 + 0.894881 w_k1_INTC_s014
                                 + 1.03462 w_k1_KO_s014 + 1.08569 w_k1_MAT_s014
                                 + 0.991087 w_k1_MSFT_s014
                                 + 0.975756 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.948442 w_k1_ADM_s014
                                 + 0.935673 w_k1_AXP_s014
                                 + 0.869679 w_k1_DIS_s014
                                 + 0.932612 w_k1_FDX_s014
                                 + 0.854533 w_k1_IBM_s014
                                 + 1.17361 w_k1_INTC_s014
                                 + 1.05307 w_k1_KO_s014
                                 + 0.666471 w_k1_MAT_s014
                                 + 1.0853 w_k1_MSFT_s014
                                 + 0.975992 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.920019 w_k1_ADM_s014
                                 + 1.27091 w_k1_AXP_s014
                                 + 1.31541 w_k1_DIS_s014
                                 + 1.35085 w_k1_FDX_s014
                                 + 0.806052 w_k1_IBM_s014
                                 + 1.39356 w_k1_INTC_s014
                                 + 1.00033 w_k1_KO_s014 + 1.10081 w_k1_MAT_s014
                                 + 1.31969 w_k1_MSFT_s014
                                 + 1.11029 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.43052 w_k1_ADM_s014
                                 + 1.62567 w_k1_AXP_s014 + 1.9583 w_k1_DIS_s014
                                 + 1.48848 w_k1_FDX_s014
                                 + 1.45844 w_k1_IBM_s014
                                 + 1.59881 w_k1_INTC_s014
                                 + 1.62188 w_k1_KO_s014 + 2.03856 w_k1_MAT_s014
                                 + 1.41799 w_k1_MSFT_s014
                                 + 1.34028 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.885495 w_k1_ADM_s014
                                 + 1.16108 w_k1_AXP_s014
                                 + 1.17032 w_k1_DIS_s014 + 1.0214 w_k1_FDX_s014
                                 + 1.54418 w_k1_IBM_s014
                                 + 1.07491 w_k1_INTC_s014
                                 + 1.32748 w_k1_KO_s014 + 1.36267 w_k1_MAT_s014
                                 + 1.01306 w_k1_MSFT_s014
                                 + 1.11707 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.987682 w_k1_ADM_s014
                                 + 1.45508 w_k1_AXP_s014
                                 + 1.16814 w_k1_DIS_s014
                                 + 1.15045 w_k1_FDX_s014
                                 + 1.14839 w_k1_IBM_s014
                                 + 1.23244 w_k1_INTC_s014 + 1.0632 w_k1_KO_s014
                                 + 1.04017 w_k1_MAT_s014
                                 + 1.05905 w_k1_MSFT_s014
                                 + 0.957149 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.02555 w_k1_ADM_s015
                                 + 0.958938 w_k1_AXP_s015
                                 + 0.917351 w_k1_DIS_s015
                                 + 1.06054 w_k1_FDX_s015
                                 + 0.785302 w_k1_IBM_s015
                                 + 0.927643 w_k1_INTC_s015
                                 + 0.819534 w_k1_KO_s015
                                 + 1.00048 w_k1_MAT_s015
                                 + 1.01871 w_k1_MSFT_s015
                                 + 0.991759 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.02409 w_k1_ADM_s015
                                 + 1.23114 w_k1_AXP_s015
                                 + 1.07383 w_k1_DIS_s015
                                 + 0.966301 w_k1_FDX_s015
                                 + 1.13812 w_k1_IBM_s015
                                 + 1.1127 w_k1_INTC_s015 + 1.16213 w_k1_KO_s015
                                 + 1.44327 w_k1_MAT_s015
                                 + 0.926804 w_k1_MSFT_s015
                                 + 0.962183 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.043 w_k1_ADM_s015
                                 + 1.49763 w_k1_AXP_s015
                                 + 1.71367 w_k1_DIS_s015
                                 + 1.30247 w_k1_FDX_s015
                                 + 1.45088 w_k1_IBM_s015
                                 + 1.19657 w_k1_INTC_s015
                                 + 1.46945 w_k1_KO_s015 + 1.74688 w_k1_MAT_s015
                                 + 1.17404 w_k1_MSFT_s015
                                 + 1.26574 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.885 w_k1_ADM_s015
                                 + 0.745288 w_k1_AXP_s015
                                 + 1.09912 w_k1_DIS_s015
                                 + 0.88848 w_k1_FDX_s015
                                 + 1.02288 w_k1_IBM_s015
                                 + 1.0354 w_k1_INTC_s015 + 1.22606 w_k1_KO_s015
                                 + 0.965728 w_k1_MAT_s015
                                 + 1.18582 w_k1_MSFT_s015
                                 + 0.991652 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.784788 w_k1_ADM_s015
                                 + 0.784027 w_k1_AXP_s015
                                 + 0.761135 w_k1_DIS_s015
                                 + 0.79714 w_k1_FDX_s015
                                 + 0.633811 w_k1_IBM_s015
                                 + 0.826564 w_k1_INTC_s015
                                 + 0.715606 w_k1_KO_s015
                                 + 1.04458 w_k1_MAT_s015
                                 + 0.806448 w_k1_MSFT_s015
                                 + 0.871393 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.19069 w_k1_ADM_s015
                                 + 1.16149 w_k1_AXP_s015
                                 + 1.41529 w_k1_DIS_s015
                                 + 1.27379 w_k1_FDX_s015
                                 + 0.963038 w_k1_IBM_s015
                                 + 1.29256 w_k1_INTC_s015
                                 + 1.10545 w_k1_KO_s015 + 1.34656 w_k1_MAT_s015
                                 + 1.34108 w_k1_MSFT_s015
                                 + 1.11433 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.741558 w_k1_ADM_s015
                                 + 0.803048 w_k1_AXP_s015
                                 + 0.674741 w_k1_DIS_s015
                                 + 0.724978 w_k1_FDX_s015
                                 + 1.06332 w_k1_IBM_s015
                                 + 0.76603 w_k1_INTC_s015
                                 + 0.787861 w_k1_KO_s015
                                 + 1.36867 w_k1_MAT_s015
                                 + 0.825472 w_k1_MSFT_s015
                                 + 0.915278 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.765207 w_k1_ADM_s015
                                 + 0.513657 w_k1_AXP_s015
                                 + 0.600217 w_k1_DIS_s015
                                 + 0.635617 w_k1_FDX_s015
                                 + 0.744868 w_k1_IBM_s015
                                 + 0.893122 w_k1_INTC_s015
                                 + 0.745657 w_k1_KO_s015
                                 + 0.563758 w_k1_MAT_s015
                                 + 0.717708 w_k1_MSFT_s015
                                 + 0.75311 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.948869 w_k1_ADM_s016
                                 + 0.672864 w_k1_AXP_s016
                                 + 0.733116 w_k1_DIS_s016
                                 + 0.802943 w_k1_FDX_s016
                                 + 0.739097 w_k1_IBM_s016
                                 + 0.966647 w_k1_INTC_s016
                                 + 0.781078 w_k1_KO_s016
                                 + 0.611061 w_k1_MAT_s016
                                 + 0.862389 w_k1_MSFT_s016
                                 + 0.813902 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.32787 w_k1_ADM_s016
                                 + 1.27021 w_k1_AXP_s016 + 1.3014 w_k1_DIS_s016
                                 + 1.15634 w_k1_FDX_s016
                                 + 1.62515 w_k1_IBM_s016
                                 + 1.29072 w_k1_INTC_s016
                                 + 1.24613 w_k1_KO_s016 + 1.32763 w_k1_MAT_s016
                                 + 1.16232 w_k1_MSFT_s016
                                 + 1.21207 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.06407 w_k1_ADM_s016
                                 + 1.47534 w_k1_AXP_s016
                                 + 1.09877 w_k1_DIS_s016
                                 + 1.10214 w_k1_FDX_s016
                                 + 1.26974 w_k1_IBM_s016
                                 + 1.43766 w_k1_INTC_s016
                                 + 1.18599 w_k1_KO_s016 + 1.46891 w_k1_MAT_s016
                                 + 1.15837 w_k1_MSFT_s016
                                 + 1.11654 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.14179 w_k1_ADM_s016
                                 + 1.33963 w_k1_AXP_s016
                                 + 1.81083 w_k1_DIS_s016
                                 + 1.35624 w_k1_FDX_s016
                                 + 1.35154 w_k1_IBM_s016
                                 + 1.60148 w_k1_INTC_s016
                                 + 1.34514 w_k1_KO_s016 + 1.38239 w_k1_MAT_s016
                                 + 1.40818 w_k1_MSFT_s016
                                 + 1.26537 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.979843 w_k1_ADM_s016
                                 + 0.831213 w_k1_AXP_s016
                                 + 0.981835 w_k1_DIS_s016
                                 + 0.888622 w_k1_FDX_s016
                                 + 1.08263 w_k1_IBM_s016
                                 + 1.10482 w_k1_INTC_s016
                                 + 1.02548 w_k1_KO_s016
                                 + 0.841569 w_k1_MAT_s016
                                 + 0.959111 w_k1_MSFT_s016
                                 + 1.00733 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.13783 w_k1_ADM_s016
                                 + 1.10028 w_k1_AXP_s016
                                 + 1.48746 w_k1_DIS_s016
                                 + 1.24593 w_k1_FDX_s016
                                 + 0.981632 w_k1_IBM_s016
                                 + 1.35567 w_k1_INTC_s016
                                 + 1.09515 w_k1_KO_s016
                                 + 0.789133 w_k1_MAT_s016
                                 + 1.34611 w_k1_MSFT_s016
                                 + 1.12035 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.07811 w_k1_ADM_s016
                                 + 0.739576 w_k1_AXP_s016
                                 + 0.639068 w_k1_DIS_s016
                                 + 0.757879 w_k1_FDX_s016
                                 + 1.30826 w_k1_IBM_s016
                                 + 1.11322 w_k1_INTC_s016
                                 + 0.989571 w_k1_KO_s016
                                 + 0.993072 w_k1_MAT_s016
                                 + 0.850927 w_k1_MSFT_s016
                                 + 1.04374 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.940186 w_k1_ADM_s016
                                 + 1.1331 w_k1_AXP_s016 + 1.02905 w_k1_DIS_s016
                                 + 1.08817 w_k1_FDX_s016
                                 + 0.83539 w_k1_IBM_s016
                                 + 1.21914 w_k1_INTC_s016
                                 + 0.92064 w_k1_KO_s016
                                 + 0.906902 w_k1_MAT_s016
                                 + 1.11342 w_k1_MSFT_s016
                                 + 0.999128 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.955662 w_k1_ADM_s017
                                 + 0.670757 w_k1_AXP_s017
                                 + 0.800396 w_k1_DIS_s017
                                 + 0.929231 w_k1_FDX_s017
                                 + 0.639739 w_k1_IBM_s017
                                 + 0.957465 w_k1_INTC_s017
                                 + 0.788426 w_k1_KO_s017
                                 + 0.581414 w_k1_MAT_s017
                                 + 0.879759 w_k1_MSFT_s017
                                 + 0.884149 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.996608 w_k1_ADM_s017
                                 + 1.12871 w_k1_AXP_s017
                                 + 0.827246 w_k1_DIS_s017
                                 + 0.887525 w_k1_FDX_s017
                                 + 1.38699 w_k1_IBM_s017
                                 + 1.00087 w_k1_INTC_s017
                                 + 0.961922 w_k1_KO_s017
                                 + 1.31016 w_k1_MAT_s017
                                 + 0.868935 w_k1_MSFT_s017
                                 + 0.976495 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.805591 w_k1_ADM_s017
                                 + 0.69623 w_k1_AXP_s017
                                 + 0.698058 w_k1_DIS_s017
                                 + 0.791485 w_k1_FDX_s017
                                 + 1.09307 w_k1_IBM_s017
                                 + 0.895083 w_k1_INTC_s017
                                 + 1.01444 w_k1_KO_s017 + 1.15328 w_k1_MAT_s017
                                 + 0.810748 w_k1_MSFT_s017
                                 + 0.908939 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.10807 w_k1_ADM_s017
                                 + 1.02738 w_k1_AXP_s017
                                 + 1.25597 w_k1_DIS_s017
                                 + 1.06975 w_k1_FDX_s017
                                 + 1.09231 w_k1_IBM_s017
                                 + 1.26586 w_k1_INTC_s017 + 1.3105 w_k1_KO_s017
                                 + 1.08504 w_k1_MAT_s017
                                 + 1.07196 w_k1_MSFT_s017
                                 + 1.09974 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.33755 w_k1_ADM_s017
                                 + 1.37776 w_k1_AXP_s017
                                 + 1.02049 w_k1_DIS_s017
                                 + 1.23745 w_k1_FDX_s017
                                 + 0.971896 w_k1_IBM_s017
                                 + 1.07675 w_k1_INTC_s017
                                 + 1.10237 w_k1_KO_s017 + 1.08091 w_k1_MAT_s017
                                 + 1.02282 w_k1_MSFT_s017
                                 + 1.07232 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.920141 w_k1_ADM_s017
                                 + 0.657295 w_k1_AXP_s017
                                 + 0.615063 w_k1_DIS_s017
                                 + 0.774836 w_k1_FDX_s017
                                 + 0.728801 w_k1_IBM_s017
                                 + 0.870352 w_k1_INTC_s017
                                 + 0.785436 w_k1_KO_s017
                                 + 0.845903 w_k1_MAT_s017
                                 + 0.741092 w_k1_MSFT_s017
                                 + 0.834456 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.1746 w_k1_ADM_s017
                                 + 1.40321 w_k1_AXP_s017
                                 + 1.49075 w_k1_DIS_s017
                                 + 1.32609 w_k1_FDX_s017
                                 + 1.47028 w_k1_IBM_s017
                                 + 1.38717 w_k1_INTC_s017
                                 + 1.47843 w_k1_KO_s017 + 1.65658 w_k1_MAT_s017
                                 + 1.2728 w_k1_MSFT_s017 + 1.26811 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.01562 w_k1_ADM_s017
                                 + 0.970235 w_k1_AXP_s017
                                 + 0.936183 w_k1_DIS_s017
                                 + 1.06039 w_k1_FDX_s017
                                 + 0.732151 w_k1_IBM_s017
                                 + 1.12357 w_k1_INTC_s017
                                 + 0.984371 w_k1_KO_s017
                                 + 0.813187 w_k1_MAT_s017
                                 + 0.992852 w_k1_MSFT_s017
                                 + 1.06518 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.01955 w_k1_ADM_s100
                                 + 0.867483 w_k1_AXP_s100
                                 + 0.936062 w_k1_DIS_s100
                                 + 0.903819 w_k1_FDX_s100
                                 + 0.702923 w_k1_IBM_s100
                                 + 0.868041 w_k1_INTC_s100
                                 + 0.984979 w_k1_KO_s100
                                 + 0.84628 w_k1_MAT_s100
                                 + 0.948078 w_k1_MSFT_s100
                                 + 1.00857 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.53104 w_k1_ADM_s100
                                 + 1.34153 w_k1_AXP_s100
                                 + 0.982045 w_k1_DIS_s100
                                 + 1.07275 w_k1_FDX_s100
                                 + 0.999668 w_k1_IBM_s100
                                 + 0.818724 w_k1_INTC_s100
                                 + 1.04477 w_k1_KO_s100 + 1.1138 w_k1_MAT_s100
                                 + 0.967841 w_k1_MSFT_s100
                                 + 1.03348 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.68101 w_k1_ADM_s100
                                 + 1.03144 w_k1_AXP_s100
                                 + 1.48676 w_k1_DIS_s100
                                 + 1.21029 w_k1_FDX_s100
                                 + 1.07454 w_k1_IBM_s100
                                 + 0.966391 w_k1_INTC_s100
                                 + 1.34449 w_k1_KO_s100
                                 + 0.914484 w_k1_MAT_s100
                                 + 1.25086 w_k1_MSFT_s100 + 1.2145 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.16339 w_k1_ADM_s100
                                 + 0.872422 w_k1_AXP_s100
                                 + 1.00297 w_k1_DIS_s100
                                 + 1.12245 w_k1_FDX_s100
                                 + 0.603042 w_k1_IBM_s100
                                 + 1.10315 w_k1_INTC_s100
                                 + 1.15073 w_k1_KO_s100
                                 + 0.945958 w_k1_MAT_s100
                                 + 1.22729 w_k1_MSFT_s100
                                 + 1.08757 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.00385 w_k1_ADM_s100
                                 + 0.79524 w_k1_AXP_s100
                                 + 0.93011 w_k1_DIS_s100
                                 + 0.954456 w_k1_FDX_s100
                                 + 0.983726 w_k1_IBM_s100
                                 + 0.938922 w_k1_INTC_s100
                                 + 1.13652 w_k1_KO_s100
                                 + 0.865785 w_k1_MAT_s100
                                 + 0.896883 w_k1_MSFT_s100
                                 + 0.854617 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.685671 w_k1_ADM_s100
                                 + 0.56572 w_k1_AXP_s100
                                 + 0.573546 w_k1_DIS_s100
                                 + 0.660215 w_k1_FDX_s100
                                 + 0.736967 w_k1_IBM_s100
                                 + 0.818724 w_k1_INTC_s100
                                 + 0.757393 w_k1_KO_s100
                                 + 0.79271 w_k1_MAT_s100
                                 + 0.801382 w_k1_MSFT_s100
                                 + 0.748751 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.24951 w_k1_ADM_s100
                                 + 1.0037 w_k1_AXP_s100 + 1.16508 w_k1_DIS_s100
                                 + 0.965259 w_k1_FDX_s100
                                 + 1.40133 w_k1_IBM_s100
                                 + 1.00338 w_k1_INTC_s100
                                 + 1.37305 w_k1_KO_s100 + 1.09797 w_k1_MAT_s100
                                 + 0.896297 w_k1_MSFT_s100
                                 + 1.07634 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.73226 w_k1_ADM_s100
                                 + 1.40277 w_k1_AXP_s100
                                 + 2.04385 w_k1_DIS_s100
                                 + 1.57098 w_k1_FDX_s100
                                 + 1.05629 w_k1_IBM_s100
                                 + 1.42076 w_k1_INTC_s100
                                 + 1.54239 w_k1_KO_s100
                                 + 0.919725 w_k1_MAT_s100
                                 + 1.4862 w_k1_MSFT_s100 + 1.17354 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.11618 w_k1_ADM_s101
                                 + 1.19711 w_k1_AXP_s101
                                 + 1.35711 w_k1_DIS_s101
                                 + 1.09589 w_k1_FDX_s101
                                 + 1.37037 w_k1_IBM_s101
                                 + 1.05885 w_k1_INTC_s101
                                 + 1.38981 w_k1_KO_s101 + 1.2394 w_k1_MAT_s101
                                 + 1.06698 w_k1_MSFT_s101
                                 + 1.07587 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.964727 w_k1_ADM_s101
                                 + 0.776727 w_k1_AXP_s101
                                 + 0.783825 w_k1_DIS_s101
                                 + 0.841568 w_k1_FDX_s101
                                 + 0.764777 w_k1_IBM_s101
                                 + 0.7764 w_k1_INTC_s101
                                 + 0.834074 w_k1_KO_s101
                                 + 0.787659 w_k1_MAT_s101
                                 + 0.816151 w_k1_MSFT_s101
                                 + 0.822061 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.12287 w_k1_ADM_s101
                                 + 1.26957 w_k1_AXP_s101
                                 + 2.00739 w_k1_DIS_s101
                                 + 1.43127 w_k1_FDX_s101
                                 + 1.41102 w_k1_IBM_s101
                                 + 1.30105 w_k1_INTC_s101
                                 + 1.67797 w_k1_KO_s101 + 1.46988 w_k1_MAT_s101
                                 + 1.43208 w_k1_MSFT_s101
                                 + 1.15284 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.869037 w_k1_ADM_s101
                                 + 0.763126 w_k1_AXP_s101
                                 + 0.686249 w_k1_DIS_s101
                                 + 0.788401 w_k1_FDX_s101
                                 + 1.14544 w_k1_IBM_s101
                                 + 0.635351 w_k1_INTC_s101
                                 + 0.881688 w_k1_KO_s101
                                 + 1.14538 w_k1_MAT_s101
                                 + 0.779092 w_k1_MSFT_s101
                                 + 0.78712 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.1178 w_k1_ADM_s101
                                 + 0.943738 w_k1_AXP_s101
                                 + 1.1722 w_k1_DIS_s101 + 1.10401 w_k1_FDX_s101
                                 + 0.628344 w_k1_IBM_s101
                                 + 1.15508 w_k1_INTC_s101
                                 + 0.905063 w_k1_KO_s101
                                 + 0.553319 w_k1_MAT_s101
                                 + 1.21092 w_k1_MSFT_s101
                                 + 1.05593 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.793953 w_k1_ADM_s101
                                 + 0.668182 w_k1_AXP_s101
                                 + 0.977212 w_k1_DIS_s101
                                 + 0.820224 w_k1_FDX_s101
                                 + 1.1682 w_k1_IBM_s101
                                 + 0.752369 w_k1_INTC_s101
                                 + 1.08671 w_k1_KO_s101 + 1.12259 w_k1_MAT_s101
                                 + 0.878506 w_k1_MSFT_s101
                                 + 1.02171 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.953342 w_k1_ADM_s101
                                 + 1.53865 w_k1_AXP_s101
                                 + 1.02799 w_k1_DIS_s101
                                 + 1.23186 w_k1_FDX_s101
                                 + 0.858683 w_k1_IBM_s101
                                 + 1.02136 w_k1_INTC_s101
                                 + 0.915914 w_k1_KO_s101
                                 + 1.35484 w_k1_MAT_s101
                                 + 1.11764 w_k1_MSFT_s101
                                 + 0.997997 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.00678 w_k1_ADM_s101
                                 + 0.809354 w_k1_AXP_s101
                                 + 1.16096 w_k1_DIS_s101 + 1.0072 w_k1_FDX_s101
                                 + 0.924546 w_k1_IBM_s101
                                 + 0.983494 w_k1_INTC_s101
                                 + 1.08058 w_k1_KO_s101
                                 + 0.773322 w_k1_MAT_s101
                                 + 1.04102 w_k1_MSFT_s101
                                 + 0.992082 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.23721 w_k1_ADM_s102
                                 + 0.736653 w_k1_AXP_s102
                                 + 0.832566 w_k1_DIS_s102
                                 + 0.981586 w_k1_FDX_s102
                                 + 0.91781 w_k1_IBM_s102
                                 + 0.729262 w_k1_INTC_s102
                                 + 0.828754 w_k1_KO_s102
                                 + 0.648122 w_k1_MAT_s102
                                 + 0.855065 w_k1_MSFT_s102
                                 + 0.918209 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.971468 w_k1_ADM_s102
                                 + 0.874135 w_k1_AXP_s102
                                 + 1.02004 w_k1_DIS_s102
                                 + 1.00259 w_k1_FDX_s102
                                 + 0.72815 w_k1_IBM_s102
                                 + 1.31142 w_k1_INTC_s102
                                 + 1.12158 w_k1_KO_s102
                                 + 0.521102 w_k1_MAT_s102
                                 + 1.20833 w_k1_MSFT_s102
                                 + 0.936811 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.74048 w_k1_ADM_s102
                                 + 1.05122 w_k1_AXP_s102
                                 + 1.37171 w_k1_DIS_s102
                                 + 1.35503 w_k1_FDX_s102
                                 + 0.888559 w_k1_IBM_s102
                                 + 1.05954 w_k1_INTC_s102
                                 + 1.09891 w_k1_KO_s102
                                 + 0.687399 w_k1_MAT_s102
                                 + 1.2894 w_k1_MSFT_s102 + 1.1946 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.22165 w_k1_ADM_s102
                                 + 1.22704 w_k1_AXP_s102
                                 + 1.72493 w_k1_DIS_s102
                                 + 1.49672 w_k1_FDX_s102
                                 + 1.28586 w_k1_IBM_s102
                                 + 1.42484 w_k1_INTC_s102 + 1.593 w_k1_KO_s102
                                 + 1.21686 w_k1_MAT_s102
                                 + 1.63372 w_k1_MSFT_s102 + 1.2491 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.12245 w_k1_ADM_s102
                                 + 1.06573 w_k1_AXP_s102
                                 + 1.04219 w_k1_DIS_s102
                                 + 1.02735 w_k1_FDX_s102
                                 + 1.46816 w_k1_IBM_s102
                                 + 0.839175 w_k1_INTC_s102
                                 + 1.23345 w_k1_KO_s102 + 1.51139 w_k1_MAT_s102
                                 + 0.78463 w_k1_MSFT_s102
                                 + 1.16567 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.13509 w_k1_ADM_s102
                                 + 0.620015 w_k1_AXP_s102
                                 + 0.72918 w_k1_DIS_s102
                                 + 0.924328 w_k1_FDX_s102
                                 + 0.699833 w_k1_IBM_s102
                                 + 0.792455 w_k1_INTC_s102
                                 + 0.735611 w_k1_KO_s102
                                 + 0.595343 w_k1_MAT_s102
                                 + 0.773358 w_k1_MSFT_s102
                                 + 0.862987 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.34701 w_k1_ADM_s102
                                 + 1.05895 w_k1_AXP_s102
                                 + 1.19954 w_k1_DIS_s102 + 1.2404 w_k1_FDX_s102
                                 + 1.0873 w_k1_IBM_s102
                                 + 1.11659 w_k1_INTC_s102
                                 + 1.08806 w_k1_KO_s102 + 1.04741 w_k1_MAT_s102
                                 + 1.11583 w_k1_MSFT_s102
                                 + 1.05342 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.963584 w_k1_ADM_s102
                                 + 0.788543 w_k1_AXP_s102
                                 + 0.872643 w_k1_DIS_s102
                                 + 0.783367 w_k1_FDX_s102
                                 + 1.05798 w_k1_IBM_s102
                                 + 0.93573 w_k1_INTC_s102
                                 + 1.05876 w_k1_KO_s102 + 0.91167 w_k1_MAT_s102
                                 + 0.879058 w_k1_MSFT_s102
                                 + 0.959734 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.878856 w_k1_ADM_s103
                                 + 1.18687 w_k1_AXP_s103
                                 + 0.777442 w_k1_DIS_s103
                                 + 0.944753 w_k1_FDX_s103
                                 + 1.20361 w_k1_IBM_s103
                                 + 1.02206 w_k1_INTC_s103
                                 + 0.913768 w_k1_KO_s103
                                 + 1.10994 w_k1_MAT_s103
                                 + 0.95139 w_k1_MSFT_s103
                                 + 1.00101 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.18269 w_k1_ADM_s103
                                 + 1.12014 w_k1_AXP_s103
                                 + 1.22545 w_k1_DIS_s103
                                 + 1.18168 w_k1_FDX_s103
                                 + 0.907288 w_k1_IBM_s103
                                 + 0.916363 w_k1_INTC_s103
                                 + 0.914355 w_k1_KO_s103
                                 + 0.967838 w_k1_MAT_s103
                                 + 1.18926 w_k1_MSFT_s103
                                 + 0.915994 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.02114 w_k1_ADM_s103
                                 + 0.92768 w_k1_AXP_s103
                                 + 0.857481 w_k1_DIS_s103
                                 + 0.956754 w_k1_FDX_s103
                                 + 0.906592 w_k1_IBM_s103
                                 + 0.966124 w_k1_INTC_s103
                                 + 0.892655 w_k1_KO_s103
                                 + 0.708602 w_k1_MAT_s103
                                 + 0.966167 w_k1_MSFT_s103
                                 + 0.848148 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.992761 w_k1_ADM_s103
                                 + 0.814468 w_k1_AXP_s103
                                 + 1.01708 w_k1_DIS_s103
                                 + 0.894381 w_k1_FDX_s103
                                 + 1.06027 w_k1_IBM_s103
                                 + 0.775905 w_k1_INTC_s103
                                 + 0.914753 w_k1_KO_s103
                                 + 0.708688 w_k1_MAT_s103
                                 + 0.868504 w_k1_MSFT_s103
                                 + 0.910682 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.5486 w_k1_ADM_s103
                                 + 1.45806 w_k1_AXP_s103
                                 + 1.54618 w_k1_DIS_s103
                                 + 1.25197 w_k1_FDX_s103
                                 + 1.16541 w_k1_IBM_s103
                                 + 1.10462 w_k1_INTC_s103
                                 + 1.23396 w_k1_KO_s103
                                 + 0.903537 w_k1_MAT_s103
                                 + 1.21182 w_k1_MSFT_s103
                                 + 1.23613 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.958674 w_k1_ADM_s103
                                 + 0.885174 w_k1_AXP_s103
                                 + 0.564759 w_k1_DIS_s103
                                 + 0.773314 w_k1_FDX_s103
                                 + 0.910748 w_k1_IBM_s103
                                 + 0.650853 w_k1_INTC_s103
                                 + 0.653777 w_k1_KO_s103
                                 + 0.89081 w_k1_MAT_s103
                                 + 0.789533 w_k1_MSFT_s103
                                 + 0.897257 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.30033 w_k1_ADM_s103
                                 + 0.969342 w_k1_AXP_s103
                                 + 0.767985 w_k1_DIS_s103
                                 + 0.986808 w_k1_FDX_s103
                                 + 0.746032 w_k1_IBM_s103
                                 + 0.853675 w_k1_INTC_s103
                                 + 0.740731 w_k1_KO_s103
                                 + 0.783356 w_k1_MAT_s103
                                 + 0.905697 w_k1_MSFT_s103
                                 + 0.957857 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.774486 w_k1_ADM_s103
                                 + 0.627992 w_k1_AXP_s103
                                 + 0.643308 w_k1_DIS_s103
                                 + 0.748877 w_k1_FDX_s103
                                 + 0.607245 w_k1_IBM_s103
                                 + 0.708333 w_k1_INTC_s103
                                 + 0.618562 w_k1_KO_s103
                                 + 0.507858 w_k1_MAT_s103
                                 + 0.821466 w_k1_MSFT_s103
                                 + 0.751139 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.852599 w_k1_ADM_s104
                                 + 0.755498 w_k1_AXP_s104
                                 + 0.648993 w_k1_DIS_s104
                                 + 0.695972 w_k1_FDX_s104
                                 + 0.976399 w_k1_IBM_s104
                                 + 0.941751 w_k1_INTC_s104
                                 + 0.772291 w_k1_KO_s104
                                 + 0.764633 w_k1_MAT_s104
                                 + 0.718604 w_k1_MSFT_s104
                                 + 0.927875 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.881821 w_k1_ADM_s104
                                 + 0.795603 w_k1_AXP_s104
                                 + 0.732229 w_k1_DIS_s104
                                 + 0.92217 w_k1_FDX_s104
                                 + 0.659156 w_k1_IBM_s104
                                 + 1.17545 w_k1_INTC_s104
                                 + 0.756587 w_k1_KO_s104
                                 + 0.664654 w_k1_MAT_s104
                                 + 0.836789 w_k1_MSFT_s104
                                 + 0.848697 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.816933 w_k1_ADM_s104
                                 + 1.1598 w_k1_AXP_s104
                                 + 0.886378 w_k1_DIS_s104
                                 + 0.888522 w_k1_FDX_s104
                                 + 1.13246 w_k1_IBM_s104
                                 + 1.01404 w_k1_INTC_s104
                                 + 0.943885 w_k1_KO_s104
                                 + 1.38241 w_k1_MAT_s104
                                 + 0.814017 w_k1_MSFT_s104
                                 + 1.06795 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.07331 w_k1_ADM_s104
                                 + 0.786024 w_k1_AXP_s104
                                 + 0.878257 w_k1_DIS_s104
                                 + 0.979003 w_k1_FDX_s104
                                 + 0.969815 w_k1_IBM_s104
                                 + 1.10116 w_k1_INTC_s104
                                 + 0.91446 w_k1_KO_s104
                                 + 0.683033 w_k1_MAT_s104
                                 + 0.870502 w_k1_MSFT_s104
                                 + 1.03315 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.5264 w_k1_ADM_s104
                                 + 1.38044 w_k1_AXP_s104
                                 + 1.17492 w_k1_DIS_s104
                                 + 1.37243 w_k1_FDX_s104
                                 + 0.77422 w_k1_IBM_s104
                                 + 1.41553 w_k1_INTC_s104
                                 + 0.902598 w_k1_KO_s104
                                 + 0.893805 w_k1_MAT_s104
                                 + 1.1134 w_k1_MSFT_s104 + 1.18692 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.21631 w_k1_ADM_s104
                                 + 0.776853 w_k1_AXP_s104
                                 + 1.32359 w_k1_DIS_s104
                                 + 1.12207 w_k1_FDX_s104
                                 + 0.826348 w_k1_IBM_s104
                                 + 1.20841 w_k1_INTC_s104
                                 + 0.962965 w_k1_KO_s104
                                 + 0.746717 w_k1_MAT_s104
                                 + 1.08437 w_k1_MSFT_s104
                                 + 1.14874 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.00495 w_k1_ADM_s104
                                 + 1.35548 w_k1_AXP_s104
                                 + 1.04253 w_k1_DIS_s104
                                 + 1.13576 w_k1_FDX_s104
                                 + 1.00924 w_k1_IBM_s104
                                 + 1.13773 w_k1_INTC_s104
                                 + 0.966139 w_k1_KO_s104
                                 + 1.54586 w_k1_MAT_s104
                                 + 1.01987 w_k1_MSFT_s104
                                 + 1.23938 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.09297 w_k1_ADM_s104
                                 + 1.09548 w_k1_AXP_s104
                                 + 1.31875 w_k1_DIS_s104
                                 + 1.07158 w_k1_FDX_s104
                                 + 1.33474 w_k1_IBM_s104
                                 + 1.30633 w_k1_INTC_s104
                                 + 1.14509 w_k1_KO_s104 + 1.39655 w_k1_MAT_s104
                                 + 1.08131 w_k1_MSFT_s104
                                 + 1.19455 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.32777 w_k1_ADM_s105
                                 + 0.935564 w_k1_AXP_s105
                                 + 0.932385 w_k1_DIS_s105
                                 + 1.02888 w_k1_FDX_s105
                                 + 0.886567 w_k1_IBM_s105
                                 + 0.885323 w_k1_INTC_s105
                                 + 0.86223 w_k1_KO_s105
                                 + 0.898397 w_k1_MAT_s105
                                 + 1.03769 w_k1_MSFT_s105
                                 + 1.00153 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.31825 w_k1_ADM_s105
                                 + 1.43151 w_k1_AXP_s105
                                 + 1.34971 w_k1_DIS_s105
                                 + 1.04549 w_k1_FDX_s105
                                 + 1.44239 w_k1_IBM_s105
                                 + 1.09919 w_k1_INTC_s105
                                 + 1.15441 w_k1_KO_s105 + 1.85763 w_k1_MAT_s105
                                 + 1.21847 w_k1_MSFT_s105
                                 + 1.12658 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.04491 w_k1_ADM_s105
                                 + 1.0031 w_k1_AXP_s105
                                 + 0.951388 w_k1_DIS_s105
                                 + 0.873476 w_k1_FDX_s105
                                 + 1.15318 w_k1_IBM_s105
                                 + 0.839959 w_k1_INTC_s105
                                 + 0.957448 w_k1_KO_s105
                                 + 1.18102 w_k1_MAT_s105
                                 + 0.826319 w_k1_MSFT_s105
                                 + 0.812346 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.35075 w_k1_ADM_s105
                                 + 1.28236 w_k1_AXP_s105
                                 + 1.50471 w_k1_DIS_s105
                                 + 1.28577 w_k1_FDX_s105
                                 + 0.931001 w_k1_IBM_s105
                                 + 0.953846 w_k1_INTC_s105
                                 + 1.1132 w_k1_KO_s105 + 0.895842 w_k1_MAT_s105
                                 + 1.11135 w_k1_MSFT_s105
                                 + 1.07407 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.05855 w_k1_ADM_s105
                                 + 0.686032 w_k1_AXP_s105
                                 + 0.698261 w_k1_DIS_s105
                                 + 0.805853 w_k1_FDX_s105
                                 + 0.579179 w_k1_IBM_s105
                                 + 0.73901 w_k1_INTC_s105
                                 + 0.63826 w_k1_KO_s105
                                 + 0.548555 w_k1_MAT_s105
                                 + 0.884634 w_k1_MSFT_s105
                                 + 0.768251 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.22676 w_k1_ADM_s105
                                 + 1.48154 w_k1_AXP_s105
                                 + 1.39566 w_k1_DIS_s105
                                 + 1.26694 w_k1_FDX_s105
                                 + 1.19544 w_k1_IBM_s105
                                 + 0.907736 w_k1_INTC_s105
                                 + 1.08535 w_k1_KO_s105 + 1.24609 w_k1_MAT_s105
                                 + 1.17468 w_k1_MSFT_s105
                                 + 1.04027 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.691204 w_k1_ADM_s105
                                 + 0.620494 w_k1_AXP_s105
                                 + 0.653812 w_k1_DIS_s105
                                 + 0.627739 w_k1_FDX_s105
                                 + 1.02575 w_k1_IBM_s105
                                 + 0.785093 w_k1_INTC_s105
                                 + 0.747539 w_k1_KO_s105
                                 + 0.647604 w_k1_MAT_s105
                                 + 0.731485 w_k1_MSFT_s105
                                 + 0.787136 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.25045 w_k1_ADM_s105
                                 + 1.01197 w_k1_AXP_s105
                                 + 0.890262 w_k1_DIS_s105
                                 + 1.02987 w_k1_FDX_s105
                                 + 0.641697 w_k1_IBM_s105
                                 + 0.823025 w_k1_INTC_s105
                                 + 0.737145 w_k1_KO_s105
                                 + 0.821885 w_k1_MAT_s105
                                 + 0.913496 w_k1_MSFT_s105
                                 + 1.06798 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.818478 w_k1_ADM_s106
                                 + 0.7962 w_k1_AXP_s106 + 1.08516 w_k1_DIS_s106
                                 + 0.872235 w_k1_FDX_s106
                                 + 1.08681 w_k1_IBM_s106
                                 + 0.801137 w_k1_INTC_s106
                                 + 0.984106 w_k1_KO_s106
                                 + 1.04019 w_k1_MAT_s106
                                 + 0.96028 w_k1_MSFT_s106
                                 + 1.03743 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.03341 w_k1_ADM_s106
                                 + 1.39281 w_k1_AXP_s106
                                 + 1.38866 w_k1_DIS_s106
                                 + 1.22854 w_k1_FDX_s106
                                 + 1.55236 w_k1_IBM_s106
                                 + 1.04206 w_k1_INTC_s106
                                 + 1.32887 w_k1_KO_s106 + 1.71885 w_k1_MAT_s106
                                 + 1.0945 w_k1_MSFT_s106 + 1.18787 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.793761 w_k1_ADM_s106
                                 + 0.647999 w_k1_AXP_s106
                                 + 0.609049 w_k1_DIS_s106
                                 + 0.678316 w_k1_FDX_s106
                                 + 0.780387 w_k1_IBM_s106
                                 + 0.785895 w_k1_INTC_s106
                                 + 0.722114 w_k1_KO_s106
                                 + 0.601636 w_k1_MAT_s106
                                 + 0.695933 w_k1_MSFT_s106
                                 + 0.934748 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.23234 w_k1_ADM_s106
                                 + 1.0451 w_k1_AXP_s106 + 1.02827 w_k1_DIS_s106
                                 + 1.12747 w_k1_FDX_s106 + 1.0477 w_k1_IBM_s106
                                 + 1.00639 w_k1_INTC_s106
                                 + 1.07893 w_k1_KO_s106
                                 + 0.911189 w_k1_MAT_s106
                                 + 1.00263 w_k1_MSFT_s106
                                 + 1.01613 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.977106 w_k1_ADM_s106
                                 + 0.767637 w_k1_AXP_s106
                                 + 0.762133 w_k1_DIS_s106
                                 + 0.908755 w_k1_FDX_s106
                                 + 0.920149 w_k1_IBM_s106
                                 + 0.701244 w_k1_INTC_s106
                                 + 0.957824 w_k1_KO_s106
                                 + 0.766333 w_k1_MAT_s106
                                 + 0.743217 w_k1_MSFT_s106
                                 + 0.996136 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.707269 w_k1_ADM_s106
                                 + 0.892062 w_k1_AXP_s106
                                 + 0.821422 w_k1_DIS_s106
                                 + 0.782421 w_k1_FDX_s106
                                 + 1.27755 w_k1_IBM_s106
                                 + 0.750413 w_k1_INTC_s106
                                 + 1.02664 w_k1_KO_s106 + 1.07527 w_k1_MAT_s106
                                 + 0.867521 w_k1_MSFT_s106
                                 + 1.00024 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.833831 w_k1_ADM_s106
                                 + 0.950088 w_k1_AXP_s106
                                 + 1.07399 w_k1_DIS_s106
                                 + 1.09692 w_k1_FDX_s106
                                 + 0.753786 w_k1_IBM_s106
                                 + 1.06146 w_k1_INTC_s106
                                 + 0.981196 w_k1_KO_s106
                                 + 0.75974 w_k1_MAT_s106
                                 + 1.16843 w_k1_MSFT_s106
                                 + 1.08226 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.23165 w_k1_ADM_s106
                                 + 1.53238 w_k1_AXP_s106
                                 + 1.30986 w_k1_DIS_s106
                                 + 1.22211 w_k1_FDX_s106
                                 + 1.25732 w_k1_IBM_s106
                                 + 1.04986 w_k1_INTC_s106
                                 + 1.17642 w_k1_KO_s106 + 1.1251 w_k1_MAT_s106
                                 + 1.08562 w_k1_MSFT_s106
                                 + 1.09991 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.968017 w_k1_ADM_s107
                                 + 0.791489 w_k1_AXP_s107
                                 + 0.897931 w_k1_DIS_s107
                                 + 0.882509 w_k1_FDX_s107
                                 + 0.786941 w_k1_IBM_s107
                                 + 0.932169 w_k1_INTC_s107
                                 + 0.957444 w_k1_KO_s107
                                 + 0.724628 w_k1_MAT_s107
                                 + 0.869271 w_k1_MSFT_s107
                                 + 0.973437 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.56262 w_k1_ADM_s107
                                 + 1.49995 w_k1_AXP_s107
                                 + 0.998882 w_k1_DIS_s107
                                 + 1.14295 w_k1_FDX_s107
                                 + 0.891984 w_k1_IBM_s107
                                 + 0.931539 w_k1_INTC_s107
                                 + 1.12588 w_k1_KO_s107 + 1.2253 w_k1_MAT_s107
                                 + 1.02397 w_k1_MSFT_s107
                                 + 1.10369 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.747438 w_k1_ADM_s107
                                 + 0.568445 w_k1_AXP_s107
                                 + 0.549125 w_k1_DIS_s107
                                 + 0.646889 w_k1_FDX_s107
                                 + 0.90589 w_k1_IBM_s107
                                 + 0.712685 w_k1_INTC_s107
                                 + 0.817488 w_k1_KO_s107
                                 + 0.928713 w_k1_MAT_s107
                                 + 0.635807 w_k1_MSFT_s107
                                 + 0.841034 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.49239 w_k1_ADM_s107
                                 + 1.46783 w_k1_AXP_s107
                                 + 1.71098 w_k1_DIS_s107
                                 + 1.38912 w_k1_FDX_s107
                                 + 1.36215 w_k1_IBM_s107
                                 + 1.28974 w_k1_INTC_s107
                                 + 1.62532 w_k1_KO_s107 + 1.45939 w_k1_MAT_s107
                                 + 1.43795 w_k1_MSFT_s107
                                 + 1.56513 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.03447 w_k1_ADM_s107
                                 + 1.34571 w_k1_AXP_s107
                                 + 1.40719 w_k1_DIS_s107
                                 + 1.13578 w_k1_FDX_s107
                                 + 1.25795 w_k1_IBM_s107
                                 + 1.12192 w_k1_INTC_s107
                                 + 1.27565 w_k1_KO_s107 + 1.30509 w_k1_MAT_s107
                                 + 1.0864 w_k1_MSFT_s107 + 1.27898 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.32664 w_k1_ADM_s107
                                 + 1.04094 w_k1_AXP_s107 + 1.0706 w_k1_DIS_s107
                                 + 1.00796 w_k1_FDX_s107
                                 + 1.21013 w_k1_IBM_s107
                                 + 1.11253 w_k1_INTC_s107
                                 + 1.25054 w_k1_KO_s107 + 1.22059 w_k1_MAT_s107
                                 + 0.980243 w_k1_MSFT_s107
                                 + 1.20881 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.578789 w_k1_ADM_s107
                                 + 0.892129 w_k1_AXP_s107
                                 + 0.733684 w_k1_DIS_s107
                                 + 0.750592 w_k1_FDX_s107
                                 + 1.15375 w_k1_IBM_s107
                                 + 1.01692 w_k1_INTC_s107 + 1.0166 w_k1_KO_s107
                                 + 1.08846 w_k1_MAT_s107
                                 + 0.88226 w_k1_MSFT_s107
                                 + 0.783878 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.20256 w_k1_ADM_s107
                                 + 1.00974 w_k1_AXP_s107
                                 + 1.12211 w_k1_DIS_s107
                                 + 1.10344 w_k1_FDX_s107
                                 + 0.730913 w_k1_IBM_s107
                                 + 0.955162 w_k1_INTC_s107
                                 + 1.06495 w_k1_KO_s107
                                 + 0.815576 w_k1_MAT_s107
                                 + 0.890643 w_k1_MSFT_s107
                                 + 1.10812 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.12594 w_k1_ADM_s110
                                 + 1.40572 w_k1_AXP_s110
                                 + 1.31487 w_k1_DIS_s110
                                 + 1.28864 w_k1_FDX_s110
                                 + 0.982949 w_k1_IBM_s110
                                 + 1.37185 w_k1_INTC_s110
                                 + 1.00506 w_k1_KO_s110
                                 + 0.984449 w_k1_MAT_s110
                                 + 1.27406 w_k1_MSFT_s110
                                 + 1.10254 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.03087 w_k1_ADM_s110
                                 + 0.976055 w_k1_AXP_s110
                                 + 1.20409 w_k1_DIS_s110
                                 + 0.897043 w_k1_FDX_s110
                                 + 1.6556 w_k1_IBM_s110
                                 + 0.997135 w_k1_INTC_s110
                                 + 1.43618 w_k1_KO_s110 + 1.54024 w_k1_MAT_s110
                                 + 1.11916 w_k1_MSFT_s110
                                 + 1.13686 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.816112 w_k1_ADM_s110
                                 + 0.482002 w_k1_AXP_s110
                                 + 0.564242 w_k1_DIS_s110
                                 + 0.688978 w_k1_FDX_s110
                                 + 0.626163 w_k1_IBM_s110
                                 + 0.691018 w_k1_INTC_s110
                                 + 0.722856 w_k1_KO_s110
                                 + 0.50329 w_k1_MAT_s110
                                 + 0.819836 w_k1_MSFT_s110
                                 + 0.763478 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.13651 w_k1_ADM_s110
                                 + 1.16448 w_k1_AXP_s110
                                 + 1.20296 w_k1_DIS_s110
                                 + 1.17462 w_k1_FDX_s110
                                 + 1.35068 w_k1_IBM_s110
                                 + 1.25544 w_k1_INTC_s110 + 1.186 w_k1_KO_s110
                                 + 1.02366 w_k1_MAT_s110
                                 + 1.35465 w_k1_MSFT_s110
                                 + 1.20806 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.836879 w_k1_ADM_s110
                                 + 0.767551 w_k1_AXP_s110
                                 + 0.645589 w_k1_DIS_s110
                                 + 0.81881 w_k1_FDX_s110
                                 + 0.876633 w_k1_IBM_s110
                                 + 0.772195 w_k1_INTC_s110
                                 + 0.75225 w_k1_KO_s110
                                 + 0.702206 w_k1_MAT_s110
                                 + 0.938888 w_k1_MSFT_s110
                                 + 0.841478 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.27387 w_k1_ADM_s110
                                 + 1.19043 w_k1_AXP_s110
                                 + 1.29446 w_k1_DIS_s110
                                 + 1.10894 w_k1_FDX_s110 + 1.364 w_k1_IBM_s110
                                 + 1.1932 w_k1_INTC_s110 + 1.30457 w_k1_KO_s110
                                 + 0.968676 w_k1_MAT_s110
                                 + 1.1225 w_k1_MSFT_s110 + 1.14935 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.08919 w_k1_ADM_s110
                                 + 0.805967 w_k1_AXP_s110
                                 + 0.897711 w_k1_DIS_s110
                                 + 0.962685 w_k1_FDX_s110
                                 + 0.76988 w_k1_IBM_s110
                                 + 0.900188 w_k1_INTC_s110
                                 + 0.83215 w_k1_KO_s110
                                 + 0.648043 w_k1_MAT_s110
                                 + 1.15167 w_k1_MSFT_s110
                                 + 0.868894 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.884719 w_k1_ADM_s110
                                 + 0.709997 w_k1_AXP_s110
                                 + 0.921952 w_k1_DIS_s110
                                 + 0.873318 w_k1_FDX_s110
                                 + 1.14718 w_k1_IBM_s110
                                 + 0.949549 w_k1_INTC_s110
                                 + 0.984437 w_k1_KO_s110
                                 + 0.910048 w_k1_MAT_s110
                                 + 1.01247 w_k1_MSFT_s110
                                 + 0.97104 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.23219 w_k1_ADM_s111
                                 + 1.59935 w_k1_AXP_s111
                                 + 1.67693 w_k1_DIS_s111
                                 + 1.43196 w_k1_FDX_s111
                                 + 1.29002 w_k1_IBM_s111
                                 + 1.33506 w_k1_INTC_s111
                                 + 1.28592 w_k1_KO_s111 + 1.21502 w_k1_MAT_s111
                                 + 1.30782 w_k1_MSFT_s111
                                 + 1.27753 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.949671 w_k1_ADM_s111
                                 + 0.81722 w_k1_AXP_s111
                                 + 1.06942 w_k1_DIS_s111
                                 + 0.998068 w_k1_FDX_s111
                                 + 0.751126 w_k1_IBM_s111
                                 + 0.899019 w_k1_INTC_s111
                                 + 1.06893 w_k1_KO_s111 + 0.65127 w_k1_MAT_s111
                                 + 0.987864 w_k1_MSFT_s111
                                 + 0.977358 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.09471 w_k1_ADM_s111
                                 + 1.16448 w_k1_AXP_s111
                                 + 1.24628 w_k1_DIS_s111
                                 + 1.09439 w_k1_FDX_s111
                                 + 1.57826 w_k1_IBM_s111
                                 + 1.1012 w_k1_INTC_s111 + 1.233 w_k1_KO_s111
                                 + 1.29819 w_k1_MAT_s111
                                 + 1.04254 w_k1_MSFT_s111
                                 + 1.05074 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.764359 w_k1_ADM_s111
                                 + 0.689668 w_k1_AXP_s111
                                 + 0.599146 w_k1_DIS_s111
                                 + 0.741015 w_k1_FDX_s111
                                 + 0.627075 w_k1_IBM_s111
                                 + 0.762269 w_k1_INTC_s111
                                 + 0.643937 w_k1_KO_s111
                                 + 0.481024 w_k1_MAT_s111
                                 + 0.737696 w_k1_MSFT_s111
                                 + 0.77055 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.805955 w_k1_ADM_s111
                                 + 0.809387 w_k1_AXP_s111
                                 + 0.770971 w_k1_DIS_s111
                                 + 0.857873 w_k1_FDX_s111
                                 + 0.724301 w_k1_IBM_s111
                                 + 0.937918 w_k1_INTC_s111
                                 + 0.76834 w_k1_KO_s111
                                 + 0.742676 w_k1_MAT_s111
                                 + 0.840356 w_k1_MSFT_s111
                                 + 0.845461 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.83443 w_k1_ADM_s111
                                 + 0.753566 w_k1_AXP_s111
                                 + 0.89263 w_k1_DIS_s111
                                 + 0.843929 w_k1_FDX_s111
                                 + 1.01065 w_k1_IBM_s111
                                 + 0.972315 w_k1_INTC_s111
                                 + 0.990306 w_k1_KO_s111
                                 + 0.947693 w_k1_MAT_s111
                                 + 0.951258 w_k1_MSFT_s111
                                 + 1.06909 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.951776 w_k1_ADM_s111
                                 + 1.05209 w_k1_AXP_s111
                                 + 1.24874 w_k1_DIS_s111
                                 + 1.14954 w_k1_FDX_s111
                                 + 0.930717 w_k1_IBM_s111
                                 + 1.10027 w_k1_INTC_s111
                                 + 1.03052 w_k1_KO_s111 + 0.92934 w_k1_MAT_s111
                                 + 1.22179 w_k1_MSFT_s111
                                 + 1.13318 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.1017 w_k1_ADM_s111
                                 + 1.11292 w_k1_AXP_s111
                                 + 1.01669 w_k1_DIS_s111
                                 + 1.01294 w_k1_FDX_s111
                                 + 1.15766 w_k1_IBM_s111
                                 + 1.09616 w_k1_INTC_s111 + 1.0527 w_k1_KO_s111
                                 + 1.14115 w_k1_MAT_s111
                                 + 0.977349 w_k1_MSFT_s111
                                 + 1.07196 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.888032 w_k1_ADM_s112
                                 + 1.0825 w_k1_AXP_s112 + 1.27818 w_k1_DIS_s112
                                 + 1.01114 w_k1_FDX_s112
                                 + 1.16204 w_k1_IBM_s112
                                 + 1.27259 w_k1_INTC_s112
                                 + 1.30047 w_k1_KO_s112
                                 + 0.850619 w_k1_MAT_s112
                                 + 1.20821 w_k1_MSFT_s112
                                 + 0.878703 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.04262 w_k1_ADM_s112
                                 + 1.20548 w_k1_AXP_s112 + 1.6047 w_k1_DIS_s112
                                 + 1.31705 w_k1_FDX_s112
                                 + 0.838449 w_k1_IBM_s112
                                 + 1.4208 w_k1_INTC_s112 + 1.36337 w_k1_KO_s112
                                 + 0.730567 w_k1_MAT_s112
                                 + 1.34765 w_k1_MSFT_s112
                                 + 1.18431 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.2006 w_k1_ADM_s112
                                 + 1.44295 w_k1_AXP_s112
                                 + 1.18053 w_k1_DIS_s112
                                 + 1.14419 w_k1_FDX_s112
                                 + 1.64227 w_k1_IBM_s112
                                 + 1.32853 w_k1_INTC_s112
                                 + 1.67604 w_k1_KO_s112 + 1.66306 w_k1_MAT_s112
                                 + 1.25039 w_k1_MSFT_s112
                                 + 1.22093 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.772252 w_k1_ADM_s112
                                 + 0.823747 w_k1_AXP_s112
                                 + 0.836441 w_k1_DIS_s112
                                 + 0.883036 w_k1_FDX_s112
                                 + 1.00309 w_k1_IBM_s112
                                 + 0.949176 w_k1_INTC_s112
                                 + 1.09097 w_k1_KO_s112
                                 + 0.714282 w_k1_MAT_s112
                                 + 0.871841 w_k1_MSFT_s112
                                 + 0.945315 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.907472 w_k1_ADM_s112
                                 + 1.03547 w_k1_AXP_s112
                                 + 0.783432 w_k1_DIS_s112
                                 + 0.812236 w_k1_FDX_s112
                                 + 1.43958 w_k1_IBM_s112
                                 + 0.859838 w_k1_INTC_s112
                                 + 1.15187 w_k1_KO_s112 + 1.40402 w_k1_MAT_s112
                                 + 0.794852 w_k1_MSFT_s112
                                 + 0.945559 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.06578 w_k1_ADM_s112
                                 + 1.16926 w_k1_AXP_s112
                                 + 0.823618 w_k1_DIS_s112
                                 + 1.04924 w_k1_FDX_s112
                                 + 0.81316 w_k1_IBM_s112
                                 + 0.869522 w_k1_INTC_s112
                                 + 0.96181 w_k1_KO_s112 + 0.99686 w_k1_MAT_s112
                                 + 0.912467 w_k1_MSFT_s112
                                 + 0.91825 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.28746 w_k1_ADM_s112
                                 + 1.1159 w_k1_AXP_s112 + 1.19338 w_k1_DIS_s112
                                 + 1.25945 w_k1_FDX_s112
                                 + 0.830427 w_k1_IBM_s112
                                 + 1.18782 w_k1_INTC_s112
                                 + 1.13767 w_k1_KO_s112
                                 + 0.916874 w_k1_MAT_s112
                                 + 1.1607 w_k1_MSFT_s112 + 1.0928 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.708784 w_k1_ADM_s112
                                 + 0.694208 w_k1_AXP_s112
                                 + 0.601935 w_k1_DIS_s112
                                 + 0.68925 w_k1_FDX_s112
                                 + 0.73429 w_k1_IBM_s112
                                 + 0.836345 w_k1_INTC_s112
                                 + 0.809857 w_k1_KO_s112
                                 + 0.54226 w_k1_MAT_s112
                                 + 0.695528 w_k1_MSFT_s112
                                 + 0.730214 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.79952 w_k1_ADM_s113
                                 + 0.760083 w_k1_AXP_s113
                                 + 0.66608 w_k1_DIS_s113
                                 + 0.727845 w_k1_FDX_s113
                                 + 0.975412 w_k1_IBM_s113
                                 + 0.801814 w_k1_INTC_s113
                                 + 0.868634 w_k1_KO_s113
                                 + 1.14888 w_k1_MAT_s113
                                 + 0.755068 w_k1_MSFT_s113
                                 + 0.822102 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.12425 w_k1_ADM_s113
                                 + 0.991632 w_k1_AXP_s113
                                 + 1.44906 w_k1_DIS_s113
                                 + 1.24069 w_k1_FDX_s113
                                 + 0.843343 w_k1_IBM_s113
                                 + 1.41093 w_k1_INTC_s113
                                 + 1.07823 w_k1_KO_s113
                                 + 0.855206 w_k1_MAT_s113
                                 + 1.24017 w_k1_MSFT_s113
                                 + 1.00094 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.11831 w_k1_ADM_s113
                                 + 0.823513 w_k1_AXP_s113
                                 + 0.864383 w_k1_DIS_s113
                                 + 1.01707 w_k1_FDX_s113
                                 + 0.671307 w_k1_IBM_s113
                                 + 0.939774 w_k1_INTC_s113
                                 + 0.883972 w_k1_KO_s113
                                 + 0.807461 w_k1_MAT_s113
                                 + 0.917514 w_k1_MSFT_s113
                                 + 0.883007 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.868716 w_k1_ADM_s113
                                 + 1.04295 w_k1_AXP_s113
                                 + 0.993852 w_k1_DIS_s113
                                 + 0.975928 w_k1_FDX_s113
                                 + 1.36757 w_k1_IBM_s113
                                 + 1.19586 w_k1_INTC_s113
                                 + 1.11937 w_k1_KO_s113 + 1.42544 w_k1_MAT_s113
                                 + 0.943153 w_k1_MSFT_s113
                                 + 1.04425 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.29306 w_k1_ADM_s113
                                 + 1.11523 w_k1_AXP_s113
                                 + 0.858408 w_k1_DIS_s113
                                 + 1.08373 w_k1_FDX_s113
                                 + 0.836537 w_k1_IBM_s113
                                 + 1.2911 w_k1_INTC_s113 + 1.04377 w_k1_KO_s113
                                 + 1.08193 w_k1_MAT_s113
                                 + 1.05693 w_k1_MSFT_s113
                                 + 1.08739 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.09458 w_k1_ADM_s113
                                 + 1.60934 w_k1_AXP_s113
                                 + 1.26545 w_k1_DIS_s113 + 1.2538 w_k1_FDX_s113
                                 + 1.62721 w_k1_IBM_s113
                                 + 1.44532 w_k1_INTC_s113
                                 + 1.53738 w_k1_KO_s113 + 1.92679 w_k1_MAT_s113
                                 + 1.35114 w_k1_MSFT_s113
                                 + 1.19017 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.08972 w_k1_ADM_s113
                                 + 1.27366 w_k1_AXP_s113
                                 + 1.61754 w_k1_DIS_s113
                                 + 1.23148 w_k1_FDX_s113
                                 + 1.32318 w_k1_IBM_s113
                                 + 1.19021 w_k1_INTC_s113
                                 + 1.41537 w_k1_KO_s113 + 1.73115 w_k1_MAT_s113
                                 + 1.35323 w_k1_MSFT_s113
                                 + 1.29653 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.983059 w_k1_ADM_s113
                                 + 0.936714 w_k1_AXP_s113
                                 + 1.05359 w_k1_DIS_s113
                                 + 1.02993 w_k1_FDX_s113
                                 + 1.02832 w_k1_IBM_s113
                                 + 1.13101 w_k1_INTC_s113
                                 + 0.99978 w_k1_KO_s113 + 1.40391 w_k1_MAT_s113
                                 + 0.945136 w_k1_MSFT_s113
                                 + 0.950281 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.12548 w_k1_ADM_s114
                                 + 0.965149 w_k1_AXP_s114
                                 + 0.821422 w_k1_DIS_s114
                                 + 0.926341 w_k1_FDX_s114
                                 + 0.992712 w_k1_IBM_s114
                                 + 1.00203 w_k1_INTC_s114
                                 + 0.992503 w_k1_KO_s114
                                 + 0.913026 w_k1_MAT_s114
                                 + 0.839408 w_k1_MSFT_s114
                                 + 0.914291 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.05487 w_k1_ADM_s114
                                 + 0.816154 w_k1_AXP_s114
                                 + 0.966185 w_k1_DIS_s114
                                 + 0.903861 w_k1_FDX_s114
                                 + 1.15117 w_k1_IBM_s114
                                 + 1.01875 w_k1_INTC_s114
                                 + 1.15822 w_k1_KO_s114
                                 + 0.803238 w_k1_MAT_s114
                                 + 0.893743 w_k1_MSFT_s114
                                 + 1.07223 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.36325 w_k1_ADM_s114
                                 + 1.23942 w_k1_AXP_s114
                                 + 1.82348 w_k1_DIS_s114
                                 + 1.38893 w_k1_FDX_s114
                                 + 1.54179 w_k1_IBM_s114
                                 + 1.12308 w_k1_INTC_s114
                                 + 1.54673 w_k1_KO_s114 + 1.39521 w_k1_MAT_s114
                                 + 1.21099 w_k1_MSFT_s114
                                 + 1.20001 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.948209 w_k1_ADM_s114
                                 + 0.555433 w_k1_AXP_s114
                                 + 0.788396 w_k1_DIS_s114
                                 + 0.785185 w_k1_FDX_s114
                                 + 0.866139 w_k1_IBM_s114
                                 + 0.881815 w_k1_INTC_s114
                                 + 0.855499 w_k1_KO_s114
                                 + 0.472042 w_k1_MAT_s114
                                 + 0.809352 w_k1_MSFT_s114
                                 + 0.859412 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.83315 w_k1_ADM_s114
                                 + 0.608125 w_k1_AXP_s114
                                 + 0.661764 w_k1_DIS_s114
                                 + 0.759695 w_k1_FDX_s114
                                 + 0.633321 w_k1_IBM_s114
                                 + 0.719539 w_k1_INTC_s114
                                 + 0.80384 w_k1_KO_s114
                                 + 0.589545 w_k1_MAT_s114
                                 + 0.781984 w_k1_MSFT_s114
                                 + 0.81269 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.01144 w_k1_ADM_s114
                                 + 1.58092 w_k1_AXP_s114
                                 + 1.29009 w_k1_DIS_s114
                                 + 1.34851 w_k1_FDX_s114
                                 + 1.15853 w_k1_IBM_s114
                                 + 1.25374 w_k1_INTC_s114
                                 + 1.15853 w_k1_KO_s114 + 1.20391 w_k1_MAT_s114
                                 + 1.11794 w_k1_MSFT_s114
                                 + 1.13898 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.942708 w_k1_ADM_s114
                                 + 1.01233 w_k1_AXP_s114
                                 + 0.885158 w_k1_DIS_s114
                                 + 1.07134 w_k1_FDX_s114
                                 + 0.786276 w_k1_IBM_s114
                                 + 1.1251 w_k1_INTC_s114
                                 + 0.763239 w_k1_KO_s114
                                 + 0.815202 w_k1_MAT_s114
                                 + 0.921889 w_k1_MSFT_s114
                                 + 0.934701 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.1924 w_k1_ADM_s114
                                 + 1.00972 w_k1_AXP_s114
                                 + 1.00822 w_k1_DIS_s114
                                 + 1.01517 w_k1_FDX_s114
                                 + 1.40418 w_k1_IBM_s114
                                 + 1.12064 w_k1_INTC_s114
                                 + 1.28499 w_k1_KO_s114 + 1.03963 w_k1_MAT_s114
                                 + 0.957142 w_k1_MSFT_s114
                                 + 0.992737 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.13352 w_k1_ADM_s115
                                 + 0.831645 w_k1_AXP_s115
                                 + 1.03092 w_k1_DIS_s115 + 1.0491 w_k1_FDX_s115
                                 + 0.785607 w_k1_IBM_s115
                                 + 0.936528 w_k1_INTC_s115
                                 + 0.964851 w_k1_KO_s115
                                 + 0.649343 w_k1_MAT_s115
                                 + 0.97508 w_k1_MSFT_s115
                                 + 0.924629 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.988319 w_k1_ADM_s115
                                 + 1.35943 w_k1_AXP_s115 + 1.0731 w_k1_DIS_s115
                                 + 1.06507 w_k1_FDX_s115
                                 + 1.23805 w_k1_IBM_s115
                                 + 0.965288 w_k1_INTC_s115
                                 + 1.11647 w_k1_KO_s115 + 1.18759 w_k1_MAT_s115
                                 + 0.942604 w_k1_MSFT_s115
                                 + 1.02359 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.09564 w_k1_ADM_s115
                                 + 1.76478 w_k1_AXP_s115
                                 + 1.74211 w_k1_DIS_s115
                                 + 1.31863 w_k1_FDX_s115
                                 + 1.49566 w_k1_IBM_s115
                                 + 1.30413 w_k1_INTC_s115
                                 + 1.36515 w_k1_KO_s115 + 1.34676 w_k1_MAT_s115
                                 + 1.12185 w_k1_MSFT_s115
                                 + 1.14862 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.16627 w_k1_ADM_s115
                                 + 1.25843 w_k1_AXP_s115
                                 + 1.35483 w_k1_DIS_s115
                                 + 1.20059 w_k1_FDX_s115 + 1.0131 w_k1_IBM_s115
                                 + 0.985492 w_k1_INTC_s115
                                 + 1.12626 w_k1_KO_s115 + 1.13081 w_k1_MAT_s115
                                 + 1.27158 w_k1_MSFT_s115
                                 + 1.20346 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.900865 w_k1_ADM_s115
                                 + 0.831561 w_k1_AXP_s115
                                 + 0.628309 w_k1_DIS_s115
                                 + 0.89088 w_k1_FDX_s115
                                 + 0.63019 w_k1_IBM_s115
                                 + 0.664058 w_k1_INTC_s115
                                 + 0.622402 w_k1_KO_s115
                                 + 0.641378 w_k1_MAT_s115
                                 + 0.856169 w_k1_MSFT_s115
                                 + 0.893344 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.17671 w_k1_ADM_s115
                                 + 1.30404 w_k1_AXP_s115
                                 + 1.66001 w_k1_DIS_s115
                                 + 1.18347 w_k1_FDX_s115
                                 + 1.34165 w_k1_IBM_s115
                                 + 1.13605 w_k1_INTC_s115
                                 + 1.47834 w_k1_KO_s115 + 1.37473 w_k1_MAT_s115
                                 + 1.38138 w_k1_MSFT_s115
                                 + 1.26443 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.805421 w_k1_ADM_s115
                                 + 0.551973 w_k1_AXP_s115
                                 + 0.626232 w_k1_DIS_s115
                                 + 0.656594 w_k1_FDX_s115
                                 + 0.835828 w_k1_IBM_s115
                                 + 0.693581 w_k1_INTC_s115
                                 + 0.799412 w_k1_KO_s115
                                 + 0.611305 w_k1_MAT_s115
                                 + 0.81228 w_k1_MSFT_s115
                                 + 0.783746 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.847781 w_k1_ADM_s115
                                 + 0.888832 w_k1_AXP_s115
                                 + 0.976021 w_k1_DIS_s115
                                 + 0.863824 w_k1_FDX_s115
                                 + 1.28783 w_k1_IBM_s115
                                 + 0.879292 w_k1_INTC_s115
                                 + 1.04167 w_k1_KO_s115
                                 + 0.855042 w_k1_MAT_s115
                                 + 0.894269 w_k1_MSFT_s115
                                 + 0.975438 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.878559 w_k1_ADM_s116
                                 + 0.809576 w_k1_AXP_s116
                                 + 0.793919 w_k1_DIS_s116
                                 + 0.827489 w_k1_FDX_s116
                                 + 0.71201 w_k1_IBM_s116
                                 + 0.823042 w_k1_INTC_s116
                                 + 0.85831 w_k1_KO_s116
                                 + 0.894159 w_k1_MAT_s116
                                 + 0.884033 w_k1_MSFT_s116
                                 + 0.858823 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.08089 w_k1_ADM_s116
                                 + 0.989477 w_k1_AXP_s116
                                 + 1.33553 w_k1_DIS_s116
                                 + 1.14911 w_k1_FDX_s116
                                 + 1.06012 w_k1_IBM_s116
                                 + 1.18219 w_k1_INTC_s116
                                 + 1.09515 w_k1_KO_s116
                                 + 0.864591 w_k1_MAT_s116
                                 + 1.12134 w_k1_MSFT_s116
                                 + 1.08079 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.26312 w_k1_ADM_s116
                                 + 1.39566 w_k1_AXP_s116
                                 + 1.46568 w_k1_DIS_s116 + 1.1932 w_k1_FDX_s116
                                 + 1.47406 w_k1_IBM_s116
                                 + 1.56274 w_k1_INTC_s116
                                 + 1.35535 w_k1_KO_s116 + 1.14654 w_k1_MAT_s116
                                 + 1.29983 w_k1_MSFT_s116
                                 + 1.19765 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.01416 w_k1_ADM_s116
                                 + 0.820691 w_k1_AXP_s116
                                 + 0.660166 w_k1_DIS_s116
                                 + 0.874212 w_k1_FDX_s116
                                 + 0.605176 w_k1_IBM_s116
                                 + 0.943657 w_k1_INTC_s116
                                 + 0.714096 w_k1_KO_s116
                                 + 0.593132 w_k1_MAT_s116
                                 + 0.852645 w_k1_MSFT_s116
                                 + 0.813914 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.04641 w_k1_ADM_s116
                                 + 1.11375 w_k1_AXP_s116
                                 + 0.960682 w_k1_DIS_s116
                                 + 0.96633 w_k1_FDX_s116
                                 + 1.26512 w_k1_IBM_s116
                                 + 1.07584 w_k1_INTC_s116
                                 + 1.02257 w_k1_KO_s116
                                 + 0.847602 w_k1_MAT_s116
                                 + 1.00715 w_k1_MSFT_s116
                                 + 1.08635 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.909928 w_k1_ADM_s116
                                 + 1.18451 w_k1_AXP_s116
                                 + 0.821254 w_k1_DIS_s116
                                 + 1.05837 w_k1_FDX_s116
                                 + 0.844748 w_k1_IBM_s116
                                 + 1.15884 w_k1_INTC_s116
                                 + 0.789405 w_k1_KO_s116
                                 + 1.20437 w_k1_MAT_s116
                                 + 1.00879 w_k1_MSFT_s116
                                 + 1.02177 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.850494 w_k1_ADM_s116
                                 + 0.944435 w_k1_AXP_s116
                                 + 0.744261 w_k1_DIS_s116
                                 + 0.790222 w_k1_FDX_s116
                                 + 1.14593 w_k1_IBM_s116
                                 + 0.859752 w_k1_INTC_s116
                                 + 1.00083 w_k1_KO_s116 + 0.89497 w_k1_MAT_s116
                                 + 0.784532 w_k1_MSFT_s116
                                 + 0.953586 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.09497 w_k1_ADM_s116
                                 + 1.46818 w_k1_AXP_s116
                                 + 1.12391 w_k1_DIS_s116
                                 + 1.21333 w_k1_FDX_s116
                                 + 0.849597 w_k1_IBM_s116
                                 + 1.19124 w_k1_INTC_s116
                                 + 1.11875 w_k1_KO_s116 + 1.20474 w_k1_MAT_s116
                                 + 1.08965 w_k1_MSFT_s116
                                 + 1.12595 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.991622 w_k1_ADM_s117
                                 + 0.896586 w_k1_AXP_s117
                                 + 0.928889 w_k1_DIS_s117
                                 + 0.96565 w_k1_FDX_s117
                                 + 0.731292 w_k1_IBM_s117
                                 + 0.838002 w_k1_INTC_s117
                                 + 0.849584 w_k1_KO_s117
                                 + 0.625615 w_k1_MAT_s117
                                 + 0.86968 w_k1_MSFT_s117
                                 + 0.855286 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.30378 w_k1_ADM_s117
                                 + 1.30697 w_k1_AXP_s117
                                 + 1.88915 w_k1_DIS_s117
                                 + 1.36174 w_k1_FDX_s117
                                 + 1.44152 w_k1_IBM_s117 + 1.138 w_k1_INTC_s117
                                 + 1.5505 w_k1_KO_s117 + 1.195 w_k1_MAT_s117
                                 + 1.49109 w_k1_MSFT_s117
                                 + 1.24705 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.21398 w_k1_ADM_s117
                                 + 1.32275 w_k1_AXP_s117
                                 + 0.989475 w_k1_DIS_s117
                                 + 1.08815 w_k1_FDX_s117
                                 + 1.04915 w_k1_IBM_s117
                                 + 0.842665 w_k1_INTC_s117
                                 + 0.999721 w_k1_KO_s117
                                 + 1.04193 w_k1_MAT_s117
                                 + 0.904905 w_k1_MSFT_s117
                                 + 1.05603 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.30962 w_k1_ADM_s117
                                 + 1.55643 w_k1_AXP_s117
                                 + 1.27784 w_k1_DIS_s117
                                 + 1.26332 w_k1_FDX_s117 + 1.3356 w_k1_IBM_s117
                                 + 1.01719 w_k1_INTC_s117 + 1.141 w_k1_KO_s117
                                 + 1.08816 w_k1_MAT_s117
                                 + 0.895109 w_k1_MSFT_s117
                                 + 1.08533 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.882895 w_k1_ADM_s117
                                 + 0.871225 w_k1_AXP_s117
                                 + 0.938211 w_k1_DIS_s117
                                 + 0.866752 w_k1_FDX_s117
                                 + 1.00473 w_k1_IBM_s117
                                 + 0.946127 w_k1_INTC_s117
                                 + 1.0375 w_k1_KO_s117 + 0.994726 w_k1_MAT_s117
                                 + 0.943687 w_k1_MSFT_s117
                                 + 0.95348 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.755407 w_k1_ADM_s117
                                 + 0.81176 w_k1_AXP_s117
                                 + 0.541586 w_k1_DIS_s117 + 0.758 w_k1_FDX_s117
                                 + 0.957177 w_k1_IBM_s117
                                 + 0.723766 w_k1_INTC_s117
                                 + 0.78627 w_k1_KO_s117
                                 + 0.998416 w_k1_MAT_s117
                                 + 0.710433 w_k1_MSFT_s117
                                 + 0.817556 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.876655 w_k1_ADM_s117
                                 + 0.969345 w_k1_AXP_s117
                                 + 0.985816 w_k1_DIS_s117
                                 + 0.905692 w_k1_FDX_s117
                                 + 1.46524 w_k1_IBM_s117
                                 + 0.880416 w_k1_INTC_s117
                                 + 1.06368 w_k1_KO_s117 + 1.5357 w_k1_MAT_s117
                                 + 0.910082 w_k1_MSFT_s117
                                 + 0.982117 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.00969 w_k1_ADM_s117
                                 + 1.20738 w_k1_AXP_s117
                                 + 1.39842 w_k1_DIS_s117
                                 + 1.26457 w_k1_FDX_s117
                                 + 1.02808 w_k1_IBM_s117
                                 + 1.1414 w_k1_INTC_s117 + 1.10417 w_k1_KO_s117
                                 + 1.05186 w_k1_MAT_s117
                                 + 1.21665 w_k1_MSFT_s117
                                 + 1.05816 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0161168401126684 w_k1_ADM_s000
                                 + 0.0207899363892884 w_k1_ADM_s001
                                 + 0.00788371698178571 w_k1_ADM_s002
                                 + 0.0109324981436637 w_k1_ADM_s003
                                 + 0.00922645089032374 w_k1_ADM_s004
                                 + 0.0151206778767379 w_k1_ADM_s005
                                 + 0.0153037341908998 w_k1_ADM_s006
                                 + 0.0223914119399287 w_k1_ADM_s007
                                 + 0.0207892886633954 w_k1_ADM_s010
                                 + 0.0380531014743716 w_k1_ADM_s011
                                 + 0.0794855853563268 w_k1_ADM_s012
                                 + 0.0286994006559937 w_k1_ADM_s013
                                 + 0.0321214135775512 w_k1_ADM_s014
                                 + 0.0658339951790895 w_k1_ADM_s015
                                 + 0.0301964935050057 w_k1_ADM_s016
                                 + 0.0936570672931017 w_k1_ADM_s017
                                 + 0.0105528251389302 w_k1_ADM_s100
                                 + 0.0266526951607971 w_k1_ADM_s101
                                 + 0.00715771696251585 w_k1_ADM_s102
                                 + 0.00902321378540138 w_k1_ADM_s103
                                 + 0.0221938800929505 w_k1_ADM_s104
                                 + 0.0321264730066618 w_k1_ADM_s105
                                 + 0.0219687183422835 w_k1_ADM_s106
                                 + 0.00684634557665177 w_k1_ADM_s107
                                 + 0.0231863660487913 w_k1_ADM_s110
                                 + 0.0764508612459919 w_k1_ADM_s111
                                 + 0.030732351420502 w_k1_ADM_s112
                                 + 0.0370073294369043 w_k1_ADM_s113
                                 + 0.0389349275823745 w_k1_ADM_s114
                                 + 0.0360809236218041 w_k1_ADM_s115
                                 + 0.0421841874021922 w_k1_ADM_s116
                                 + 0.0834625622191618 w_k1_ADM_s117
                                 + 0.0155825765719138 w_k1_AXP_s000
                                 + 0.0194596526349825 w_k1_AXP_s001
                                 + 0.00953628569600453 w_k1_AXP_s002
                                 + 0.010334999169037 w_k1_AXP_s003
                                 + 0.0101284940926547 w_k1_AXP_s004
                                 + 0.0163974697585987 w_k1_AXP_s005
                                 + 0.0141243746307341 w_k1_AXP_s006
                                 + 0.0244970347477016 w_k1_AXP_s007
                                 + 0.0232410109713189 w_k1_AXP_s010
                                 + 0.039065706634537 w_k1_AXP_s011
                                 + 0.0827031270406499 w_k1_AXP_s012
                                 + 0.0264245135520513 w_k1_AXP_s013
                                 + 0.0365312579746992 w_k1_AXP_s014
                                 + 0.0676922905702088 w_k1_AXP_s015
                                 + 0.0301814021015519 w_k1_AXP_s016
                                 + 0.0902377197387561 w_k1_AXP_s017
                                 + 0.00810106316207056 w_k1_AXP_s100
                                 + 0.0259506667065009 w_k1_AXP_s101
                                 + 0.00518811532202802 w_k1_AXP_s102
                                 + 0.0083567327409274 w_k1_AXP_s103
                                 + 0.021928473926506 w_k1_AXP_s104
                                 + 0.0286576594229808 w_k1_AXP_s105
                                 + 0.0229820840634258 w_k1_AXP_s106
                                 + 0.00659483798922153 w_k1_AXP_s107
                                 + 0.0216356369904051 w_k1_AXP_s110
                                 + 0.0784656279891417 w_k1_AXP_s111
                                 + 0.0329188662830892 w_k1_AXP_s112
                                 + 0.0364714558875671 w_k1_AXP_s113
                                 + 0.0364532717233998 w_k1_AXP_s114
                                 + 0.0361443391131831 w_k1_AXP_s115
                                 + 0.0454854691696766 w_k1_AXP_s116
                                 + 0.087650034849268 w_k1_AXP_s117
                                 + 0.0154283742539639 w_k1_DIS_s000
                                 + 0.0192199897603552 w_k1_DIS_s001
                                 + 0.00907332954654978 w_k1_DIS_s002
                                 + 0.0128949589504306 w_k1_DIS_s003
                                 + 0.00946876708558117 w_k1_DIS_s004
                                 + 0.0161336759159996 w_k1_DIS_s005
                                 + 0.014777407913674 w_k1_DIS_s006
                                 + 0.0265140295495966 w_k1_DIS_s007
                                 + 0.0224757220692446 w_k1_DIS_s010
                                 + 0.0365450784744622 w_k1_DIS_s011
                                 + 0.0791069259663035 w_k1_DIS_s012
                                 + 0.0274900521390793 w_k1_DIS_s013
                                 + 0.038424525870571 w_k1_DIS_s014
                                 + 0.0717625007797603 w_k1_DIS_s015
                                 + 0.0321767421792862 w_k1_DIS_s016
                                 + 0.0894817228616874 w_k1_DIS_s017
                                 + 0.00949002238618832 w_k1_DIS_s100
                                 + 0.0293354543844011 w_k1_DIS_s101
                                 + 0.00601735128475059 w_k1_DIS_s102
                                 + 0.0081621752530323 w_k1_DIS_s103
                                 + 0.0216140997454587 w_k1_DIS_s104
                                 + 0.0283341505891912 w_k1_DIS_s105
                                 + 0.0232751742877209 w_k1_DIS_s106
                                 + 0.00653337118225165 w_k1_DIS_s107
                                 + 0.0231990117916934 w_k1_DIS_s110
                                 + 0.0850418022745724 w_k1_DIS_s111
                                 + 0.030826766063505 w_k1_DIS_s112
                                 + 0.0381649391879322 w_k1_DIS_s113
                                 + 0.0369215348618394 w_k1_DIS_s114
                                 + 0.0385436614202373 w_k1_DIS_s115
                                 + 0.041320229105057 w_k1_DIS_s116
                                 + 0.0889005083967787 w_k1_DIS_s117
                                 + 0.0160080121547454 w_k1_FDX_s000
                                 + 0.0184669887328037 w_k1_FDX_s001
                                 + 0.00892794936534027 w_k1_FDX_s002
                                 + 0.0116705153469742 w_k1_FDX_s003
                                 + 0.00927436877286754 w_k1_FDX_s004
                                 + 0.0153800801647328 w_k1_FDX_s005
                                 + 0.0154473734416929 w_k1_FDX_s006
                                 + 0.0232648824451174 w_k1_FDX_s007
                                 + 0.0237800511047001 w_k1_FDX_s010
                                 + 0.0372514691337491 w_k1_FDX_s011
                                 + 0.0808352480730234 w_k1_FDX_s012
                                 + 0.0272828620620704 w_k1_FDX_s013
                                 + 0.0356249090011183 w_k1_FDX_s014
                                 + 0.0673182311747553 w_k1_FDX_s015
                                 + 0.0301111183105407 w_k1_FDX_s016
                                 + 0.0916534124601005 w_k1_FDX_s017
                                 + 0.00861321017251699 w_k1_FDX_s100
                                 + 0.0267783025983216 w_k1_FDX_s101
                                 + 0.0063445518092842 w_k1_FDX_s102
                                 + 0.00818662543959254 w_k1_FDX_s103
                                 + 0.0216624009938226 w_k1_FDX_s104
                                 + 0.0271324211837424 w_k1_FDX_s105
                                 + 0.0232697832504684 w_k1_FDX_s106
                                 + 0.00619611960843064 w_k1_FDX_s107
                                 + 0.0225807349899194 w_k1_FDX_s110
                                 + 0.0807019789255029 w_k1_FDX_s111
                                 + 0.0318131616753729 w_k1_FDX_s112
                                 + 0.0375992943354401 w_k1_FDX_s113
                                 + 0.0378765281359779 w_k1_FDX_s114
                                 + 0.0358432213051358 w_k1_FDX_s115
                                 + 0.0415091695731683 w_k1_FDX_s116
                                 + 0.0855103281187846 w_k1_FDX_s117
                                 + 0.0158701257439805 w_k1_IBM_s000
                                 + 0.0192878076302986 w_k1_IBM_s001
                                 + 0.00968912188160701 w_k1_IBM_s002
                                 + 0.0110502709236896 w_k1_IBM_s003
                                 + 0.00974683821043921 w_k1_IBM_s004
                                 + 0.0156577477803999 w_k1_IBM_s005
                                 + 0.0162933775969668 w_k1_IBM_s006
                                 + 0.0237777107805692 w_k1_IBM_s007
                                 + 0.0242465011483385 w_k1_IBM_s010
                                 + 0.0367604778030006 w_k1_IBM_s011
                                 + 0.0832101537378107 w_k1_IBM_s012
                                 + 0.0247893599551055 w_k1_IBM_s013
                                 + 0.0387596190894753 w_k1_IBM_s014
                                 + 0.066130257482478 w_k1_IBM_s015
                                 + 0.0304159768878795 w_k1_IBM_s016
                                 + 0.0920557917174295 w_k1_IBM_s017
                                 + 0.00848379087134961 w_k1_IBM_s100
                                 + 0.0278182183104987 w_k1_IBM_s101
                                 + 0.00589187568282379 w_k1_IBM_s102
                                 + 0.00764703791850905 w_k1_IBM_s103
                                 + 0.0215407806798979 w_k1_IBM_s104
                                 + 0.0273046983776664 w_k1_IBM_s105
                                 + 0.023466315504812 w_k1_IBM_s106
                                 + 0.00652584371437015 w_k1_IBM_s107
                                 + 0.0246939819603189 w_k1_IBM_s110
                                 + 0.082378922375018 w_k1_IBM_s111
                                 + 0.0321796532721673 w_k1_IBM_s112
                                 + 0.0384748601829615 w_k1_IBM_s113
                                 + 0.0382608131633887 w_k1_IBM_s114
                                 + 0.0358261099390197 w_k1_IBM_s115
                                 + 0.0427000849695184 w_k1_IBM_s116
                                 + 0.0864732588994666 w_k1_IBM_s117
                                 + 0.0156164407941811 w_k1_INTC_s000
                                 + 0.0172105092997063 w_k1_INTC_s001
                                 + 0.010278414007484 w_k1_INTC_s002
                                 + 0.0127511168872491 w_k1_INTC_s003
                                 + 0.00993861459374136 w_k1_INTC_s004
                                 + 0.0145155116120655 w_k1_INTC_s005
                                 + 0.0182781543739889 w_k1_INTC_s006
                                 + 0.0246497328947927 w_k1_INTC_s007
                                 + 0.0230613496818021 w_k1_INTC_s010
                                 + 0.0397815607497223 w_k1_INTC_s011
                                 + 0.0776231506376339 w_k1_INTC_s012
                                 + 0.0234888810325834 w_k1_INTC_s013
                                 + 0.0393374772047502 w_k1_INTC_s014
                                 + 0.0698923313617895 w_k1_INTC_s015
                                 + 0.0356722043750676 w_k1_INTC_s016
                                 + 0.0992698995969488 w_k1_INTC_s017
                                 + 0.00807391228337576 w_k1_INTC_s100
                                 + 0.0248491801793134 w_k1_INTC_s101
                                 + 0.00558749064082395 w_k1_INTC_s102
                                 + 0.00724997683448321 w_k1_INTC_s103
                                 + 0.0252942846361069 w_k1_INTC_s104
                                 + 0.0242286439718815 w_k1_INTC_s105
                                 + 0.0209610301497813 w_k1_INTC_s106
                                 + 0.00639515573816123 w_k1_INTC_s107
                                 + 0.0236008138609482 w_k1_INTC_s110
                                 + 0.0825561791438803 w_k1_INTC_s111
                                 + 0.0331306720394762 w_k1_INTC_s112
                                 + 0.0413256113843384 w_k1_INTC_s113
                                 + 0.0386244794026115 w_k1_INTC_s114
                                 + 0.0324800437916511 w_k1_INTC_s115
                                 + 0.0445706276912403 w_k1_INTC_s116
                                 + 0.0766763686303244 w_k1_INTC_s117
                                 + 0.015499167942846 w_k1_KO_s000
                                 + 0.0196246886786084 w_k1_KO_s001
                                 + 0.00878190677222004 w_k1_KO_s002
                                 + 0.0115758989814961 w_k1_KO_s003
                                 + 0.0102842107934211 w_k1_KO_s004
                                 + 0.0156464565442769 w_k1_KO_s005
                                 + 0.0153435443888 w_k1_KO_s006
                                 + 0.0247654435668906 w_k1_KO_s007
                                 + 0.0233650122142721 w_k1_KO_s010
                                 + 0.0383256449079636 w_k1_KO_s011
                                 + 0.0815833856033866 w_k1_KO_s012
                                 + 0.0255212423157864 w_k1_KO_s013
                                 + 0.0394210026576959 w_k1_KO_s014
                                 + 0.0691270918626984 w_k1_KO_s015
                                 + 0.0298642291226023 w_k1_KO_s016
                                 + 0.0968493358425982 w_k1_KO_s017
                                 + 0.00991354470851681 w_k1_KO_s100
                                 + 0.0290473943428247 w_k1_KO_s101
                                 + 0.00599306328773803 w_k1_KO_s102
                                 + 0.00723733304165 w_k1_KO_s103
                                 + 0.0201356546724837 w_k1_KO_s104
                                 + 0.0250793506970528 w_k1_KO_s105
                                 + 0.0234001458466475 w_k1_KO_s106
                                 + 0.00710168021506068 w_k1_KO_s107
                                 + 0.0229420917715292 w_k1_KO_s110
                                 + 0.0813408365171363 w_k1_KO_s111
                                 + 0.0363294794351982 w_k1_KO_s112
                                 + 0.0388599555602285 w_k1_KO_s113
                                 + 0.0381895229235081 w_k1_KO_s114
                                 + 0.0362411199713023 w_k1_KO_s115
                                 + 0.0427268685995333 w_k1_KO_s116
                                 + 0.083988083488458 w_k1_KO_s117
                                 + 0.0155864154778339 w_k1_MAT_s000
                                 + 0.0200895095452805 w_k1_MAT_s001
                                 + 0.00827279476067942 w_k1_MAT_s002
                                 + 0.0119481451762444 w_k1_MAT_s003
                                 + 0.0124549275491562 w_k1_MAT_s004
                                 + 0.0177294137759046 w_k1_MAT_s005
                                 + 0.0162101549481445 w_k1_MAT_s006
                                 + 0.0256788817035132 w_k1_MAT_s007
                                 + 0.028091747641554 w_k1_MAT_s010
                                 + 0.0428741890200165 w_k1_MAT_s011
                                 + 0.0801877796859683 w_k1_MAT_s012
                                 + 0.0316190612015367 w_k1_MAT_s013
                                 + 0.0401082025517598 w_k1_MAT_s014
                                 + 0.0822354192331819 w_k1_MAT_s015
                                 + 0.0277580436552077 w_k1_MAT_s016
                                 + 0.096239059177672 w_k1_MAT_s017
                                 + 0.00789439413486254 w_k1_MAT_s100
                                 + 0.0275841441030671 w_k1_MAT_s101
                                 + 0.00519860415242514 w_k1_MAT_s102
                                 + 0.00673199498450039 w_k1_MAT_s103
                                 + 0.0228493195320617 w_k1_MAT_s104
                                 + 0.0282814277056526 w_k1_MAT_s105
                                 + 0.0220325294193427 w_k1_MAT_s106
                                 + 0.00674938146265117 w_k1_MAT_s107
                                 + 0.0200179112070958 w_k1_MAT_s110
                                 + 0.0764528954597972 w_k1_MAT_s111
                                 + 0.0297339287269036 w_k1_MAT_s112
                                 + 0.0460196139749594 w_k1_MAT_s113
                                 + 0.0331943807663066 w_k1_MAT_s114
                                 + 0.0320126008068603 w_k1_MAT_s115
                                 + 0.0401990476674537 w_k1_MAT_s116
                                 + 0.0829395915062895 w_k1_MAT_s117
                                 + 0.0156249216635976 w_k1_MSFT_s000
                                 + 0.0167262469203271 w_k1_MSFT_s001
                                 + 0.0100645937041831 w_k1_MSFT_s002
                                 + 0.0124840583026615 w_k1_MSFT_s003
                                 + 0.00964968589044406 w_k1_MSFT_s004
                                 + 0.0149640246726527 w_k1_MSFT_s005
                                 + 0.0157859009608805 w_k1_MSFT_s006
                                 + 0.0221678964313119 w_k1_MSFT_s007
                                 + 0.0241856253315955 w_k1_MSFT_s010
                                 + 0.0402508135119414 w_k1_MSFT_s011
                                 + 0.0758301737636009 w_k1_MSFT_s012
                                 + 0.0263126112963602 w_k1_MSFT_s013
                                 + 0.0387400161064574 w_k1_MSFT_s014
                                 + 0.0704117957735581 w_k1_MSFT_s015
                                 + 0.0318375579375583 w_k1_MSFT_s016
                                 + 0.0877797168269428 w_k1_MSFT_s017
                                 + 0.00855032580399497 w_k1_MSFT_s100
                                 + 0.0266133235047777 w_k1_MSFT_s101
                                 + 0.00580420827689943 w_k1_MSFT_s102
                                 + 0.00819646901194227 w_k1_MSFT_s103
                                 + 0.0203056689162218 w_k1_MSFT_s104
                                 + 0.0269973373382352 w_k1_MSFT_s105
                                 + 0.0221545699007809 w_k1_MSFT_s106
                                 + 0.00603403880540512 w_k1_MSFT_s107
                                 + 0.025215439941205 w_k1_MSFT_s110
                                 + 0.0817132315188788 w_k1_MSFT_s111
                                 + 0.0313577952347613 w_k1_MSFT_s112
                                 + 0.0370269879460116 w_k1_MSFT_s113
                                 + 0.0344551298137762 w_k1_MSFT_s114
                                 + 0.0362173113680497 w_k1_MSFT_s115
                                 + 0.0416079995700951 w_k1_MSFT_s116
                                 + 0.080458282246346 w_k1_MSFT_s117
                                 + 0.0156685032949767 w_k1_PG_s000
                                 + 0.017591858368621 w_k1_PG_s001
                                 + 0.00885553722398527 w_k1_PG_s002
                                 + 0.010395625765022 w_k1_PG_s003
                                 + 0.00926356956996227 w_k1_PG_s004
                                 + 0.0168231596004473 w_k1_PG_s005
                                 + 0.0153539137930446 w_k1_PG_s006
                                 + 0.0243422071935163 w_k1_PG_s007
                                 + 0.0223108458663458 w_k1_PG_s010
                                 + 0.0377428815106174 w_k1_PG_s011
                                 + 0.0834991512422951 w_k1_PG_s012
                                 + 0.0267423164126689 w_k1_PG_s013
                                 + 0.0360073919102009 w_k1_PG_s014
                                 + 0.0675845475248932 w_k1_PG_s015
                                 + 0.0300495343407456 w_k1_PG_s016
                                 + 0.0926746159649517 w_k1_PG_s017
                                 + 0.00857085007059511 w_k1_PG_s100
                                 + 0.0258137766168467 w_k1_PG_s101
                                 + 0.00595146684888488 w_k1_PG_s102
                                 + 0.00768357660034595 w_k1_PG_s103
                                 + 0.0231792777642895 w_k1_PG_s104
                                 + 0.0264440223538611 w_k1_PG_s105
                                 + 0.0234895902194339 w_k1_PG_s106
                                 + 0.00682207101648893 w_k1_PG_s107
                                 + 0.02282690566869 w_k1_PG_s110
                                 + 0.0831593890357197 w_k1_PG_s111
                                 + 0.0310984105532049 w_k1_PG_s112
                                 + 0.0364700185665002 w_k1_PG_s113
                                 + 0.0363147718884192 w_k1_PG_s114
                                 + 0.0357718095376377 w_k1_PG_s115
                                 + 0.0428059789496984 w_k1_PG_s116
                                 + 0.0805648523999978 w_k1_PG_s117 >= 1.041
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
