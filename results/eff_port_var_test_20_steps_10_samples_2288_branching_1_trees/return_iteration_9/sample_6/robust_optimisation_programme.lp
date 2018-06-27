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
 w_k1_ADM_s0_return_constraint:  1.12559 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.955561 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.16912 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.957688 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.21976 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.904289 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.08624 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.821771 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.11995 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.07711 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.08356 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.54933 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.8674 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.23492 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.20395 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.85175 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.13246 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.944813 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.658683 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.01092 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.830111 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.17861 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.21337 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.734395 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.822018 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.14414 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.93811 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.879105 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.38033 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.07274 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.808353 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.03124 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.36728 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.54019 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.670964 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.86273 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.15466 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.18027 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19661 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.937015 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.40057 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.901777 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.42014 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.953274 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.708476 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.754501 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.11259 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.18627 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.01447 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.41508 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.964519 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.04758 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.16913 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.777996 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.14852 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.732757 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.590867 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.14512 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.840307 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.52102 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.16995 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.603737 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.765591 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.21045 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.898605 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.626697 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.33303 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.961574 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.88051 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.855213 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.37624 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.19102 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.650454 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.654232 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.13299 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.18523 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.27921 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.945622 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.49196 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.895747 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.4288 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.856612 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.705425 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.753867 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.10866 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.0376 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.19755 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.60557 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.653849 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.954753 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.17971 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.672006 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.0512 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.835281 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.477938 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.46674 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.761779 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.46965 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.49785 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.538329 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.686325 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.27727 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.761757 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.922549 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.30305 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.15709 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.890367 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.967189 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.50867 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.39135 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.626689 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.591103 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.1646 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.57797 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.22337 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.940715 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.23102 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.924307 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.22475 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.921043 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.917224 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.818724 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.04563 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.964235 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.13659 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.40387 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.756319 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.15715 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.01699 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.724448 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.07696 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.836964 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.616594 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.1744 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.905044 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.40758 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.41683 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.676809 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.876198 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.08154 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.866328 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.872191 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.21707 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.11411 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.841088 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.05809 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.19523 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.37118 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.745997 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.780537 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.18805 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.22497 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.977351 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.05284 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.28681 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.918035 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.40722 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.868122 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.63817 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.874889 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.00022 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.49866 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.03478 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.26173 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.07459 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.706907 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.51738 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.06499 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.839956 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.930938 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.577515 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.12951 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.566903 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.995738 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.772039 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.831142 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.581307 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.64142 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.890945 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.815444 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.19763 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.921733 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.19582 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.769295 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.60458 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.806172 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.775359 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.447711 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.968346 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.27025 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.0925 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.980546 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.23057 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.969831 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13642 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.886036 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.904515 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.818252 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.935515 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.06396 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.12827 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.06096 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.874137 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.13033 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.926632 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.798586 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.11854 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.08228 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.645295 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.2015 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.900283 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.37389 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.20028 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.776108 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.888342 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.06363 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.00106 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.02006 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.981734 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.1438 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.954736 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.10223 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.22218 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.12893 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.8384 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.931862 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.11497 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.29234 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.07526 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.98089 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.28948 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.934236 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.22331 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.922446 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.740731 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.883192 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.06298 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.12462 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.10296 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.42671 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.915185 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.800083 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.36039 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.936813 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.979832 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.94049 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.585439 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.2706 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.723967 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.16187 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.05491 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.783758 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.758194 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.38607 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.933009 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.05296 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.31912 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.20926 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.911711 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.899598 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.3895 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.15568 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.700453 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.574466 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.04585 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.16745 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.03306 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.07106 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.39027 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.795165 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.34139 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.819873 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.566253 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.765784 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.26292 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.50765 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.926279 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.67911 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.17778 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.607781 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 2.01258 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.988877 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.900744 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.966883 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.683049 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.40157 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.637843 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.19023 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.767969 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.827093 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.630775 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.56404 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.890396 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.648843 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.6086 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.987315 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.25067 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.67845 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.83903 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.976247 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.727743 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.384902 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.15887 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.21239 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19831 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.968374 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.20274 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.890862 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.1683 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.85576 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.915093 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.813755 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.900328 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.03655 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.21239 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.27189 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.741255 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.01632 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.02992 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.749751 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.05966 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.946252 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.708445 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.20999 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.89912 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.28884 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.27659 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.764378 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.851706 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.19987 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.931381 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.02592 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.19347 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.05253 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.940319 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.11879 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.24209 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.39225 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.834046 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.820488 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.24337 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.40264 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08883 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.0034 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.11469 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.939746 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.16732 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.893941 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.947331 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.904691 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.10016 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.16721 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.0673 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.28197 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.96727 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.98473 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.14087 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.83395 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.08409 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.943401 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.74494 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.22101 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.842085 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.27854 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.11387 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.750533 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.844195 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.15125 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.907812 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.944346 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.17276 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.07566 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.984687 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.996073 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.22288 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.17665 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.793147 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.749781 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.11438 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.18189 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.121000812937453 z_k1_s0000
                                 - 0.0660001657833424 z_k1_s0001
                                 - 0.0570001061582449 z_k1_s0002
                                 - 0.110999954907506 z_k1_s0003
                                 - 0.106000543895182 z_k1_s0004
                                 - 0.148999957812826 z_k1_s0005
                                 - 0.28100028937951 z_k1_s0006
                                 - 0.126999902551711 z_k1_s0007
                                 - 0.190999183010723 z_k1_s0010
                                 - 0.179999329502112 z_k1_s0011
                                 - 0.296998966728382 z_k1_s0012
                                 - 0.0689998647256381 z_k1_s0013
                                 - 0.289000002993537 z_k1_s0014
                                 - 0.168999475993501 z_k1_s0015
                                 - 0.148999875159476 z_k1_s0016
                                 - 0.116999783326064 z_k1_s0017
                                 - 0.0879997542255022 z_k1_s0020
                                 - 0.0889998471256867 z_k1_s0021
                                 - 0.028999967308351 z_k1_s0022
                                 - 0.0309999526089406 z_k1_s0023
                                 - 0.0689999941197906 z_k1_s0024
                                 - 0.0359999760103485 z_k1_s0025
                                 - 0.0229999713066263 z_k1_s0026
                                 - 0.0359999760103485 z_k1_s0027
                                 - 0.224999416878222 z_k1_s0030
                                 - 0.137999757485093 z_k1_s0031
                                 - 0.078999896437052 z_k1_s0032
                                 - 0.144999746321291 z_k1_s0033
                                 - 0.334999690017636 z_k1_s0034
                                 - 0.218000370040252 z_k1_s0035
                                 - 0.111999821379172 z_k1_s0036
                                 - 0.317999761986784 z_k1_s0037
                                 - 0.0700000968283948 z_k1_s0040
                                 - 0.0419999536971908 z_k1_s0041
                                 - 0.0569997059893853 z_k1_s0042
                                 - 0.0449999354755835 z_k1_s0043
                                 - 0.0560000774627159 z_k1_s0044
                                 - 0.0819998151422736 z_k1_s0045
                                 - 0.0590000070411855 z_k1_s0046
                                 - 0.111000004399532 z_k1_s0047
                                 - 0.169998940996517 z_k1_s0050
                                 - 0.126999592178384 z_k1_s0051
                                 - 0.215998637019166 z_k1_s0052
                                 - 0.235999470721222 z_k1_s0053
                                 - 0.224998641936411 z_k1_s0054
                                 - 0.1139995686202 z_k1_s0055
                                 - 0.184999936521739 z_k1_s0056
                                 - 0.207999784533064 z_k1_s0057
                                 - 0.0920000071228655 z_k1_s0060
                                 - 0.0459999497615175 z_k1_s0061
                                 - 0.0850001001459784 z_k1_s0062
                                 - 0.0839999597209509 z_k1_s0063
                                 - 0.0499999196625595 z_k1_s0064
                                 - 0.0289999297323218 z_k1_s0065
                                 - 0.0639998412161643 z_k1_s0066
                                 - 0.0879999834219082 z_k1_s0067
                                 - 0.358000227667848 z_k1_s0070
                                 - 0.168000556335954 z_k1_s0071
                                 - 0.0870000965310321 z_k1_s0072
                                 - 0.186999852968839 z_k1_s0073
                                 - 0.114000070999258 z_k1_s0074
                                 - 0.126999999106155 z_k1_s0075
                                 - 0.165000737972899 z_k1_s0076
                                 - 0.134999872341131 z_k1_s0077
                                 - 0.722002500409888 z_k1_s0100
                                 - 0.271000548686887 z_k1_s0101
                                 - 0.601002661019196 z_k1_s0102
                                 - 0.540001449704938 z_k1_s0103
                                 - 0.286000205705995 z_k1_s0104
                                 - 0.677999214081871 z_k1_s0105
                                 - 0.935000849385203 z_k1_s0106
                                 - 0.565999628803211 z_k1_s0107
                                 - 0.386000869613443 z_k1_s0110
                                 - 0.313000943734069 z_k1_s0111
                                 - 0.205000695000236 z_k1_s0112
                                 - 0.213001217297211 z_k1_s0113
                                 - 0.169000542322031 z_k1_s0114
                                 - 0.326000615149769 z_k1_s0115
                                 - 0.209000851498334 z_k1_s0116
                                 - 0.272001014034801 z_k1_s0117
                                 - 0.310000048443155 z_k1_s0120
                                 - 0.267999814007637 z_k1_s0121
                                 - 0.628999355058228 z_k1_s0122
                                 - 0.85099912743172 z_k1_s0123
                                 - 0.25900014513573 z_k1_s0124
                                 - 0.69200073346148 z_k1_s0125
                                 - 0.766000657585977 z_k1_s0126
                                 - 0.332000014785574 z_k1_s0127
                                 - 0.179000394780445 z_k1_s0130
                                 - 0.474001224308419 z_k1_s0131
                                 - 0.603000664288971 z_k1_s0132
                                 - 0.507000828581915 z_k1_s0133
                                 - 0.16800052668928 z_k1_s0134
                                 - 0.463001356217254 z_k1_s0135
                                 - 0.19700049967584 z_k1_s0136
                                 - 0.123000400151114 z_k1_s0137
                                 - 0.11199961539439 z_k1_s0140
                                 - 0.169999798080578 z_k1_s0141
                                 - 0.230999571250745 z_k1_s0142
                                 - 0.0629999537592584 z_k1_s0143
                                 - 0.0769999542835326 z_k1_s0144
                                 - 0.169999798080578 z_k1_s0145
                                 - 0.0629999537592584 z_k1_s0146
                                 - 0.0859999893348342 z_k1_s0147
                                 - 0.218000919181342 z_k1_s0150
                                 - 0.202000883608415 z_k1_s0151
                                 - 0.100000048530714 z_k1_s0152
                                 - 0.214000041287699 z_k1_s0153
                                 - 0.227000287440806 z_k1_s0154
                                 - 0.251999691273196 z_k1_s0155
                                 - 0.230999427333626 z_k1_s0156
                                 - 0.294000219152337 z_k1_s0157
                                 - 0.464001114931049 z_k1_s0160
                                 - 0.380002068739155 z_k1_s0161
                                 - 0.410000664162511 z_k1_s0162
                                 - 0.309000427893051 z_k1_s0163
                                 - 0.147000267190037 z_k1_s0164
                                 - 0.418002275613071 z_k1_s0165
                                 - 0.656001062660018 z_k1_s0166
                                 - 0.195000403072604 z_k1_s0167
                                 - 0.102000076249801 z_k1_s0170
                                 - 0.144000097799712 z_k1_s0171
                                 - 0.104000005533077 z_k1_s0172
                                 - 0.242999694224164 z_k1_s0173
                                 - 0.379999955832348 z_k1_s0174
                                 - 0.185000573208295 z_k1_s0175
                                 - 0.217000780941702 z_k1_s0176
                                 - 0.299000057757627 z_k1_s0177
                                 - 0.315999342419678 z_k1_s1000
                                 - 0.0929999545969082 z_k1_s1001
                                 - 0.174999483650161 z_k1_s1002
                                 - 0.173999211800773 z_k1_s1003
                                 - 0.291999230033398 z_k1_s1004
                                 - 0.209999380380193 z_k1_s1005
                                 - 0.118000018232625 z_k1_s1006
                                 - 0.225000252121498 z_k1_s1007
                                 - 0.281999031707021 z_k1_s1010
                                 - 0.206999168823778 z_k1_s1011
                                 - 0.315998881515819 z_k1_s1012
                                 - 0.417998430942212 z_k1_s1013
                                 - 0.114999381036136 z_k1_s1014
                                 - 0.327998689801898 z_k1_s1015
                                 - 0.19999900555813 z_k1_s1016
                                 - 0.491996855708124 z_k1_s1017
                                 - 0.142999959440536 z_k1_s1020
                                 - 0.0949999561374313 z_k1_s1021
                                 - 0.165999622623521 z_k1_s1022
                                 - 0.159000075741486 z_k1_s1023
                                 - 0.124999972601861 z_k1_s1024
                                 - 0.137000347827363 z_k1_s1025
                                 - 0.116999338451807 z_k1_s1026
                                 - 0.210000230450924 z_k1_s1027
                                 - 1.05600161840406 z_k1_s1030
                                 - 0.310000317557097 z_k1_s1031
                                 - 0.314000291535868 z_k1_s1032
                                 - 0.43200143371152 z_k1_s1033
                                 - 0.381000386180815 z_k1_s1034
                                 - 0.825000468627382 z_k1_s1035
                                 - 0.255000144848464 z_k1_s1036
                                 - 0.671002531445743 z_k1_s1037
                                 - 0.399000682062539 z_k1_s1040
                                 - 0.290000271048938 z_k1_s1041
                                 - 0.971999995052558 z_k1_s1042
                                 - 0.455001054413762 z_k1_s1043
                                 - 0.81200021833636 z_k1_s1044
                                 - 0.838003110306343 z_k1_s1045
                                 - 0.53700116523109 z_k1_s1046
                                 - 0.202000393855029 z_k1_s1047
                                 - 0.155000305158609 z_k1_s1050
                                 - 0.231999695463551 z_k1_s1051
                                 - 0.118999996379942 z_k1_s1052
                                 - 0.138999954868186 z_k1_s1053
                                 - 0.118999996379942 z_k1_s1054
                                 - 0.138999954868186 z_k1_s1055
                                 - 0.252999261001384 z_k1_s1056
                                 - 0.318999581262383 z_k1_s1057
                                 - 0.309998394062047 z_k1_s1060
                                 - 0.432997943834651 z_k1_s1061
                                 - 0.566998277975392 z_k1_s1062
                                 - 0.148999770318981 z_k1_s1063
                                 - 0.289999339023979 z_k1_s1064
                                 - 0.345000362870885 z_k1_s1065
                                 - 0.594998693824952 z_k1_s1066
                                 - 0.208999543627583 z_k1_s1067
                                 - 0.697002607894832 z_k1_s1070
                                 - 0.224000530395996 z_k1_s1071
                                 - 0.370000781919133 z_k1_s1072
                                 - 0.680999305054336 z_k1_s1073
                                 - 0.695003450791719 z_k1_s1074
                                 - 1.07800277228732 z_k1_s1075
                                 - 0.806001877084652 z_k1_s1076
                                 - 0.472000489744161 z_k1_s1077
                                 - 2.37900504550542 z_k1_s1100
                                 - 0.672999486021722 z_k1_s1101
                                 - 1.44199378189723 z_k1_s1102
                                 - 1.36700558982056 z_k1_s1103
                                 - 0.803999950006386 z_k1_s1104
                                 - 0.750999610776016 z_k1_s1105
                                 - 2.23400306416808 z_k1_s1106
                                 - 2.75500043042585 z_k1_s1107
                                 - 1.78200079227527 z_k1_s1110
                                 - 1.20799769959296 z_k1_s1111
                                 - 0.542999994794897 z_k1_s1112
                                 - 1.13300239980535 z_k1_s1113
                                 - 0.829996815534301 z_k1_s1114
                                 - 0.557000377582385 z_k1_s1115
                                 - 0.682999884668311 z_k1_s1116
                                 - 1.14000416639967 z_k1_s1117
                                 - 0.403000845444105 z_k1_s1120
                                 - 0.858003780949403 z_k1_s1121
                                 - 0.494000818443959 z_k1_s1122
                                 - 0.502000714844847 z_k1_s1123
                                 - 0.725001741914767 z_k1_s1124
                                 - 1.50000360396159 z_k1_s1125
                                 - 0.906001931152317 z_k1_s1126
                                 - 0.752998615862449 z_k1_s1127
                                 - 0.52699880733141 z_k1_s1130
                                 - 0.217000143442664 z_k1_s1131
                                 - 0.421001030165176 z_k1_s1132
                                 - 0.606999419458018 z_k1_s1133
                                 - 0.31100018849109 z_k1_s1134
                                 - 0.173000143873198 z_k1_s1135
                                 - 0.492001067777287 z_k1_s1136
                                 - 0.623001564484355 z_k1_s1137
                                 - 0.770994879584972 z_k1_s1140
                                 - 1.11199785433935 z_k1_s1141
                                 - 0.593998856339181 z_k1_s1142
                                 - 0.944995503432495 z_k1_s1143
                                 - 0.451999048084896 z_k1_s1144
                                 - 0.7909988538223 z_k1_s1145
                                 - 1.18999741720378 z_k1_s1146
                                 - 1.07399931096647 z_k1_s1147
                                 - 0.285001367521518 z_k1_s1150
                                 - 0.195000265403666 z_k1_s1151
                                 - 0.144000048047206 z_k1_s1152
                                 - 0.408999755641414 z_k1_s1153
                                 - 0.521001150268093 z_k1_s1154
                                 - 0.323001376590917 z_k1_s1155
                                 - 0.520000534476306 z_k1_s1156
                                 - 0.202000157642679 z_k1_s1157
                                 - 0.354999872230424 z_k1_s1160
                                 - 0.617999927274308 z_k1_s1161
                                 - 0.719000056663739 z_k1_s1162
                                 - 0.283999897784339 z_k1_s1163
                                 - 0.19500005813215 z_k1_s1164
                                 - 0.368000027945836 z_k1_s1165
                                 - 0.861000764755589 z_k1_s1166
                                 - 0.395998552933643 z_k1_s1167
                                 - 0.269000252490334 z_k1_s1170
                                 - 0.530000431791342 z_k1_s1171
                                 - 0.75999811898294 z_k1_s1172
                                 - 0.625999543262645 z_k1_s1173
                                 - 0.754998073239611 z_k1_s1174
                                 - 0.308000167588147 z_k1_s1175
                                 - 0.637002294098893 z_k1_s1176
                                 - 0.532001333488982 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.01212 w_k1_ADM_s000
                                 + 1.05395 w_k1_AXP_s000
                                 + 1.23592 w_k1_DIS_s000
                                 + 0.987969 w_k1_FDX_s000
                                 + 1.34777 w_k1_IBM_s000
                                 + 1.13451 w_k1_INTC_s000
                                 + 1.27354 w_k1_KO_s000 + 1.16807 w_k1_MAT_s000
                                 + 0.90408 w_k1_MSFT_s000
                                 + 1.09562 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.26833 w_k1_ADM_s000
                                 + 1.32703 w_k1_AXP_s000
                                 + 1.97608 w_k1_DIS_s000
                                 + 1.29141 w_k1_FDX_s000
                                 + 1.53453 w_k1_IBM_s000
                                 + 1.12793 w_k1_INTC_s000
                                 + 1.46549 w_k1_KO_s000 + 1.44958 w_k1_MAT_s000
                                 + 1.22104 w_k1_MSFT_s000
                                 + 1.25334 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.881463 w_k1_ADM_s000
                                 + 1.6221 w_k1_AXP_s000 + 1.31362 w_k1_DIS_s000
                                 + 1.0974 w_k1_FDX_s000 + 1.57363 w_k1_IBM_s000
                                 + 1.22859 w_k1_INTC_s000
                                 + 1.41431 w_k1_KO_s000 + 1.63846 w_k1_MAT_s000
                                 + 1.05651 w_k1_MSFT_s000
                                 + 1.20954 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.984969 w_k1_ADM_s000
                                 + 0.906805 w_k1_AXP_s000
                                 + 0.813511 w_k1_DIS_s000
                                 + 0.884292 w_k1_FDX_s000
                                 + 1.12154 w_k1_IBM_s000
                                 + 0.74895 w_k1_INTC_s000
                                 + 0.949957 w_k1_KO_s000
                                 + 1.05262 w_k1_MAT_s000
                                 + 0.732097 w_k1_MSFT_s000
                                 + 0.932706 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.742502 w_k1_ADM_s000
                                 + 0.770903 w_k1_AXP_s000
                                 + 0.732402 w_k1_DIS_s000
                                 + 0.783925 w_k1_FDX_s000
                                 + 0.646961 w_k1_IBM_s000
                                 + 0.722871 w_k1_INTC_s000
                                 + 0.709229 w_k1_KO_s000
                                 + 0.739617 w_k1_MAT_s000
                                 + 0.723842 w_k1_MSFT_s000
                                 + 0.730229 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.990879 w_k1_ADM_s000
                                 + 0.957355 w_k1_AXP_s000
                                 + 1.40108 w_k1_DIS_s000
                                 + 1.17911 w_k1_FDX_s000
                                 + 1.02492 w_k1_IBM_s000
                                 + 1.15586 w_k1_INTC_s000
                                 + 1.31662 w_k1_KO_s000 + 1.12434 w_k1_MAT_s000
                                 + 1.33697 w_k1_MSFT_s000
                                 + 1.16947 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.944035 w_k1_ADM_s000
                                 + 0.949859 w_k1_AXP_s000
                                 + 0.906854 w_k1_DIS_s000
                                 + 0.904898 w_k1_FDX_s000
                                 + 0.846868 w_k1_IBM_s000
                                 + 0.766862 w_k1_INTC_s000
                                 + 0.881899 w_k1_KO_s000
                                 + 0.806423 w_k1_MAT_s000
                                 + 0.81048 w_k1_MSFT_s000
                                 + 0.859153 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.08039 w_k1_ADM_s000
                                 + 1.39914 w_k1_AXP_s000
                                 + 1.28002 w_k1_DIS_s000
                                 + 1.25883 w_k1_FDX_s000
                                 + 0.815092 w_k1_IBM_s000
                                 + 1.20378 w_k1_INTC_s000
                                 + 1.05045 w_k1_KO_s000 + 1.15749 w_k1_MAT_s000
                                 + 1.23988 w_k1_MSFT_s000
                                 + 1.10399 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.991432 w_k1_ADM_s001
                                 + 0.865688 w_k1_AXP_s001
                                 + 0.774022 w_k1_DIS_s001
                                 + 0.888325 w_k1_FDX_s001
                                 + 0.905239 w_k1_IBM_s001
                                 + 0.916776 w_k1_INTC_s001
                                 + 0.848977 w_k1_KO_s001
                                 + 0.818829 w_k1_MAT_s001
                                 + 0.829525 w_k1_MSFT_s001
                                 + 0.869948 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.45927 w_k1_ADM_s001
                                 + 1.31579 w_k1_AXP_s001
                                 + 1.28333 w_k1_DIS_s001
                                 + 1.26851 w_k1_FDX_s001
                                 + 0.97524 w_k1_IBM_s001
                                 + 1.4209 w_k1_INTC_s001 + 1.02762 w_k1_KO_s001
                                 + 1.04851 w_k1_MAT_s001
                                 + 1.27834 w_k1_MSFT_s001
                                 + 1.23419 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.970863 w_k1_ADM_s001
                                 + 1.19696 w_k1_AXP_s001
                                 + 0.942096 w_k1_DIS_s001
                                 + 1.02018 w_k1_FDX_s001
                                 + 1.13882 w_k1_IBM_s001
                                 + 1.12833 w_k1_INTC_s001
                                 + 1.05421 w_k1_KO_s001 + 1.40601 w_k1_MAT_s001
                                 + 1.01605 w_k1_MSFT_s001
                                 + 1.05644 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.08728 w_k1_ADM_s001
                                 + 0.850332 w_k1_AXP_s001
                                 + 0.633815 w_k1_DIS_s001
                                 + 0.832054 w_k1_FDX_s001
                                 + 0.777803 w_k1_IBM_s001
                                 + 0.648227 w_k1_INTC_s001
                                 + 0.698341 w_k1_KO_s001
                                 + 0.861771 w_k1_MAT_s001
                                 + 0.669393 w_k1_MSFT_s001
                                 + 0.936725 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.14927 w_k1_ADM_s001
                                 + 1.29466 w_k1_AXP_s001
                                 + 1.47355 w_k1_DIS_s001
                                 + 1.25141 w_k1_FDX_s001
                                 + 1.35441 w_k1_IBM_s001
                                 + 1.19094 w_k1_INTC_s001
                                 + 1.49436 w_k1_KO_s001 + 2.00456 w_k1_MAT_s001
                                 + 1.34454 w_k1_MSFT_s001
                                 + 1.24656 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.908899 w_k1_ADM_s001
                                 + 0.711028 w_k1_AXP_s001
                                 + 0.992443 w_k1_DIS_s001
                                 + 0.896712 w_k1_FDX_s001
                                 + 1.04259 w_k1_IBM_s001
                                 + 1.07263 w_k1_INTC_s001
                                 + 0.927307 w_k1_KO_s001
                                 + 0.859616 w_k1_MAT_s001
                                 + 1.13772 w_k1_MSFT_s001
                                 + 1.04712 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.06662 w_k1_ADM_s001
                                 + 1.05859 w_k1_AXP_s001
                                 + 0.905763 w_k1_DIS_s001
                                 + 1.06947 w_k1_FDX_s001
                                 + 0.778697 w_k1_IBM_s001
                                 + 1.09458 w_k1_INTC_s001
                                 + 0.743052 w_k1_KO_s001
                                 + 0.640658 w_k1_MAT_s001
                                 + 1.04473 w_k1_MSFT_s001
                                 + 1.00769 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.83467 w_k1_ADM_s001
                                 + 0.644336 w_k1_AXP_s001
                                 + 0.530633 w_k1_DIS_s001
                                 + 0.774101 w_k1_FDX_s001
                                 + 0.542528 w_k1_IBM_s001
                                 + 0.879644 w_k1_INTC_s001
                                 + 0.53711 w_k1_KO_s001
                                 + 0.495437 w_k1_MAT_s001
                                 + 0.781132 w_k1_MSFT_s001
                                 + 0.776575 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.985619 w_k1_ADM_s002
                                 + 0.854853 w_k1_AXP_s002
                                 + 0.926122 w_k1_DIS_s002
                                 + 0.898173 w_k1_FDX_s002
                                 + 1.26311 w_k1_IBM_s002
                                 + 1.02123 w_k1_INTC_s002
                                 + 0.911253 w_k1_KO_s002
                                 + 1.09595 w_k1_MAT_s002
                                 + 1.01737 w_k1_MSFT_s002 + 0.9063 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.843351 w_k1_ADM_s002
                                 + 0.721 w_k1_AXP_s002 + 0.751678 w_k1_DIS_s002
                                 + 0.893443 w_k1_FDX_s002
                                 + 0.694343 w_k1_IBM_s002
                                 + 1.09783 w_k1_INTC_s002
                                 + 0.832913 w_k1_KO_s002 + 0.5133 w_k1_MAT_s002
                                 + 0.967548 w_k1_MSFT_s002
                                 + 0.893025 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.51774 w_k1_ADM_s002
                                 + 1.0344 w_k1_AXP_s002 + 1.16469 w_k1_DIS_s002
                                 + 1.18465 w_k1_FDX_s002
                                 + 0.860815 w_k1_IBM_s002
                                 + 1.29629 w_k1_INTC_s002
                                 + 1.05857 w_k1_KO_s002
                                 + 0.828653 w_k1_MAT_s002
                                 + 1.35237 w_k1_MSFT_s002
                                 + 1.17703 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.728121 w_k1_ADM_s002
                                 + 0.619329 w_k1_AXP_s002
                                 + 0.534713 w_k1_DIS_s002
                                 + 0.725569 w_k1_FDX_s002
                                 + 0.633421 w_k1_IBM_s002
                                 + 0.911575 w_k1_INTC_s002
                                 + 0.721013 w_k1_KO_s002
                                 + 0.55931 w_k1_MAT_s002
                                 + 0.869044 w_k1_MSFT_s002
                                 + 0.919047 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.313 w_k1_ADM_s002
                                 + 1.10224 w_k1_AXP_s002
                                 + 1.16889 w_k1_DIS_s002 + 1.1101 w_k1_FDX_s002
                                 + 1.23547 w_k1_IBM_s002
                                 + 1.14608 w_k1_INTC_s002
                                 + 1.11693 w_k1_KO_s002 + 1.23974 w_k1_MAT_s002
                                 + 1.35288 w_k1_MSFT_s002
                                 + 1.09361 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.842753 w_k1_ADM_s002
                                 + 0.663839 w_k1_AXP_s002
                                 + 1.02923 w_k1_DIS_s002
                                 + 0.826677 w_k1_FDX_s002
                                 + 0.938255 w_k1_IBM_s002
                                 + 1.20394 w_k1_INTC_s002
                                 + 0.969763 w_k1_KO_s002
                                 + 0.387268 w_k1_MAT_s002
                                 + 1.17315 w_k1_MSFT_s002
                                 + 0.963743 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.13763 w_k1_ADM_s002
                                 + 1.15992 w_k1_AXP_s002
                                 + 1.67122 w_k1_DIS_s002
                                 + 1.23878 w_k1_FDX_s002
                                 + 1.17418 w_k1_IBM_s002
                                 + 1.20119 w_k1_INTC_s002
                                 + 1.34849 w_k1_KO_s002
                                 + 0.762908 w_k1_MAT_s002
                                 + 1.51662 w_k1_MSFT_s002
                                 + 1.00802 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.19037 w_k1_ADM_s002
                                 + 1.36771 w_k1_AXP_s002
                                 + 1.36983 w_k1_DIS_s002 + 1.3596 w_k1_FDX_s002
                                 + 1.07968 w_k1_IBM_s002
                                 + 1.33058 w_k1_INTC_s002
                                 + 1.17766 w_k1_KO_s002 + 1.08336 w_k1_MAT_s002
                                 + 1.37022 w_k1_MSFT_s002
                                 + 1.14115 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.72094 w_k1_ADM_s003
                                 + 0.7702 w_k1_AXP_s003 + 1.0247 w_k1_DIS_s003
                                 + 1.13324 w_k1_FDX_s003
                                 + 0.707853 w_k1_IBM_s003
                                 + 0.997259 w_k1_INTC_s003
                                 + 0.885195 w_k1_KO_s003
                                 + 0.694553 w_k1_MAT_s003
                                 + 0.891224 w_k1_MSFT_s003
                                 + 0.847435 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.804206 w_k1_ADM_s003
                                 + 0.733826 w_k1_AXP_s003
                                 + 0.839699 w_k1_DIS_s003
                                 + 0.885846 w_k1_FDX_s003
                                 + 1.21163 w_k1_IBM_s003
                                 + 1.08913 w_k1_INTC_s003
                                 + 1.11361 w_k1_KO_s003 + 1.13841 w_k1_MAT_s003
                                 + 0.959662 w_k1_MSFT_s003
                                 + 0.922095 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.3648 w_k1_ADM_s003
                                 + 1.7086 w_k1_AXP_s003 + 1.87697 w_k1_DIS_s003
                                 + 1.36382 w_k1_FDX_s003 + 1.8655 w_k1_IBM_s003
                                 + 1.00317 w_k1_INTC_s003
                                 + 1.66523 w_k1_KO_s003 + 1.80685 w_k1_MAT_s003
                                 + 1.15407 w_k1_MSFT_s003
                                 + 1.26425 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.748451 w_k1_ADM_s003
                                 + 0.617308 w_k1_AXP_s003
                                 + 0.53444 w_k1_DIS_s003
                                 + 0.667857 w_k1_FDX_s003
                                 + 0.742717 w_k1_IBM_s003
                                 + 0.690306 w_k1_INTC_s003
                                 + 0.756138 w_k1_KO_s003
                                 + 0.694313 w_k1_MAT_s003
                                 + 0.639321 w_k1_MSFT_s003
                                 + 0.760184 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.05399 w_k1_ADM_s003
                                 + 1.02502 w_k1_AXP_s003
                                 + 1.10972 w_k1_DIS_s003
                                 + 1.05911 w_k1_FDX_s003
                                 + 1.00434 w_k1_IBM_s003
                                 + 0.986277 w_k1_INTC_s003
                                 + 1.23402 w_k1_KO_s003 + 1.12998 w_k1_MAT_s003
                                 + 1.01278 w_k1_MSFT_s003
                                 + 1.05569 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.760532 w_k1_ADM_s003
                                 + 0.78618 w_k1_AXP_s003
                                 + 0.845971 w_k1_DIS_s003
                                 + 0.884979 w_k1_FDX_s003
                                 + 0.861148 w_k1_IBM_s003
                                 + 0.906956 w_k1_INTC_s003
                                 + 0.98456 w_k1_KO_s003 + 0.81403 w_k1_MAT_s003
                                 + 0.809664 w_k1_MSFT_s003
                                 + 0.908541 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.05605 w_k1_ADM_s003
                                 + 0.915108 w_k1_AXP_s003
                                 + 0.902164 w_k1_DIS_s003
                                 + 0.893515 w_k1_FDX_s003
                                 + 1.71687 w_k1_IBM_s003
                                 + 0.906952 w_k1_INTC_s003
                                 + 1.19866 w_k1_KO_s003 + 1.96107 w_k1_MAT_s003
                                 + 0.825225 w_k1_MSFT_s003
                                 + 0.979541 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.04333 w_k1_ADM_s003
                                 + 1.27262 w_k1_AXP_s003
                                 + 1.52029 w_k1_DIS_s003
                                 + 1.25989 w_k1_FDX_s003
                                 + 1.22527 w_k1_IBM_s003
                                 + 1.28931 w_k1_INTC_s003
                                 + 1.43319 w_k1_KO_s003 + 1.39495 w_k1_MAT_s003
                                 + 1.21713 w_k1_MSFT_s003
                                 + 1.14285 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.969862 w_k1_ADM_s004
                                 + 0.937959 w_k1_AXP_s004
                                 + 1.02472 w_k1_DIS_s004
                                 + 0.89346 w_k1_FDX_s004
                                 + 0.982413 w_k1_IBM_s004
                                 + 1.00171 w_k1_INTC_s004
                                 + 1.00396 w_k1_KO_s004
                                 + 0.629459 w_k1_MAT_s004
                                 + 0.863842 w_k1_MSFT_s004
                                 + 1.01076 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.34909 w_k1_ADM_s004
                                 + 0.937813 w_k1_AXP_s004
                                 + 1.38032 w_k1_DIS_s004
                                 + 1.11321 w_k1_FDX_s004
                                 + 1.19435 w_k1_IBM_s004
                                 + 1.28152 w_k1_INTC_s004
                                 + 1.24608 w_k1_KO_s004
                                 + 0.809348 w_k1_MAT_s004
                                 + 1.2369 w_k1_MSFT_s004 + 1.19786 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.761498 w_k1_ADM_s004
                                 + 0.918807 w_k1_AXP_s004
                                 + 0.925298 w_k1_DIS_s004
                                 + 0.878581 w_k1_FDX_s004
                                 + 1.44461 w_k1_IBM_s004
                                 + 0.975648 w_k1_INTC_s004 + 1.237 w_k1_KO_s004
                                 + 1.76272 w_k1_MAT_s004
                                 + 0.941074 w_k1_MSFT_s004
                                 + 1.10271 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.41978 w_k1_ADM_s004
                                 + 1.23368 w_k1_AXP_s004
                                 + 1.62564 w_k1_DIS_s004
                                 + 1.36783 w_k1_FDX_s004
                                 + 1.05659 w_k1_IBM_s004
                                 + 1.30864 w_k1_INTC_s004
                                 + 1.39581 w_k1_KO_s004 + 1.08192 w_k1_MAT_s004
                                 + 1.17546 w_k1_MSFT_s004
                                 + 1.26619 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.03969 w_k1_ADM_s004
                                 + 1.08233 w_k1_AXP_s004
                                 + 1.15949 w_k1_DIS_s004
                                 + 1.04123 w_k1_FDX_s004
                                 + 0.84304 w_k1_IBM_s004
                                 + 0.928735 w_k1_INTC_s004
                                 + 0.869794 w_k1_KO_s004 + 1.3966 w_k1_MAT_s004
                                 + 0.840154 w_k1_MSFT_s004
                                 + 1.06893 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.07672 w_k1_ADM_s004
                                 + 0.873017 w_k1_AXP_s004
                                 + 0.993679 w_k1_DIS_s004
                                 + 1.05352 w_k1_FDX_s004
                                 + 0.68174 w_k1_IBM_s004
                                 + 1.03104 w_k1_INTC_s004
                                 + 0.822085 w_k1_KO_s004
                                 + 0.48077 w_k1_MAT_s004
                                 + 0.990252 w_k1_MSFT_s004
                                 + 0.845229 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.24003 w_k1_ADM_s004
                                 + 1.3393 w_k1_AXP_s004 + 1.19188 w_k1_DIS_s004
                                 + 1.14753 w_k1_FDX_s004
                                 + 1.19422 w_k1_IBM_s004
                                 + 1.0419 w_k1_INTC_s004 + 1.22342 w_k1_KO_s004
                                 + 1.15844 w_k1_MAT_s004
                                 + 0.868689 w_k1_MSFT_s004 + 1.096 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.980486 w_k1_ADM_s004
                                 + 0.855039 w_k1_AXP_s004
                                 + 0.831763 w_k1_DIS_s004
                                 + 0.91601 w_k1_FDX_s004
                                 + 0.927648 w_k1_IBM_s004
                                 + 0.825742 w_k1_INTC_s004
                                 + 0.99937 w_k1_KO_s004
                                 + 0.939341 w_k1_MAT_s004
                                 + 0.697586 w_k1_MSFT_s004
                                 + 0.975665 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.643039 w_k1_ADM_s005
                                 + 0.748536 w_k1_AXP_s005
                                 + 0.653091 w_k1_DIS_s005
                                 + 0.69787 w_k1_FDX_s005
                                 + 0.864411 w_k1_IBM_s005
                                 + 1.03543 w_k1_INTC_s005
                                 + 0.62743 w_k1_KO_s005
                                 + 0.492044 w_k1_MAT_s005
                                 + 0.790611 w_k1_MSFT_s005
                                 + 0.755287 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.18515 w_k1_ADM_s005
                                 + 0.938488 w_k1_AXP_s005
                                 + 1.13851 w_k1_DIS_s005
                                 + 1.14633 w_k1_FDX_s005
                                 + 0.667588 w_k1_IBM_s005
                                 + 1.12741 w_k1_INTC_s005
                                 + 0.90103 w_k1_KO_s005
                                 + 0.543783 w_k1_MAT_s005
                                 + 0.935703 w_k1_MSFT_s005
                                 + 1.01993 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.899661 w_k1_ADM_s005
                                 + 1.51954 w_k1_AXP_s005
                                 + 1.57771 w_k1_DIS_s005
                                 + 1.34871 w_k1_FDX_s005
                                 + 1.24488 w_k1_IBM_s005
                                 + 1.85265 w_k1_INTC_s005
                                 + 1.30472 w_k1_KO_s005
                                 + 0.873297 w_k1_MAT_s005
                                 + 1.40496 w_k1_MSFT_s005
                                 + 1.11568 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.768114 w_k1_ADM_s005
                                 + 1.07799 w_k1_AXP_s005
                                 + 0.849184 w_k1_DIS_s005
                                 + 0.879604 w_k1_FDX_s005
                                 + 1.11589 w_k1_IBM_s005
                                 + 1.15512 w_k1_INTC_s005
                                 + 0.88117 w_k1_KO_s005
                                 + 0.969985 w_k1_MAT_s005
                                 + 0.88245 w_k1_MSFT_s005
                                 + 0.886053 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.769612 w_k1_ADM_s005
                                 + 0.730923 w_k1_AXP_s005
                                 + 0.819828 w_k1_DIS_s005
                                 + 0.885541 w_k1_FDX_s005
                                 + 0.667506 w_k1_IBM_s005
                                 + 1.15771 w_k1_INTC_s005
                                 + 0.754852 w_k1_KO_s005
                                 + 0.339331 w_k1_MAT_s005
                                 + 0.932935 w_k1_MSFT_s005
                                 + 0.834812 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.982947 w_k1_ADM_s005
                                 + 1.33387 w_k1_AXP_s005
                                 + 1.48733 w_k1_DIS_s005
                                 + 1.07121 w_k1_FDX_s005
                                 + 1.63107 w_k1_IBM_s005
                                 + 1.27476 w_k1_INTC_s005
                                 + 1.37174 w_k1_KO_s005 + 1.39423 w_k1_MAT_s005
                                 + 1.25794 w_k1_MSFT_s005
                                 + 1.12496 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.960029 w_k1_ADM_s005
                                 + 1.06899 w_k1_AXP_s005
                                 + 0.888923 w_k1_DIS_s005
                                 + 1.02976 w_k1_FDX_s005
                                 + 0.77266 w_k1_IBM_s005
                                 + 1.25949 w_k1_INTC_s005
                                 + 0.926181 w_k1_KO_s005
                                 + 0.903651 w_k1_MAT_s005
                                 + 1.07602 w_k1_MSFT_s005
                                 + 1.03832 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.773185 w_k1_ADM_s005
                                 + 0.978002 w_k1_AXP_s005
                                 + 1.13518 w_k1_DIS_s005
                                 + 0.978163 w_k1_FDX_s005
                                 + 1.12197 w_k1_IBM_s005
                                 + 1.10796 w_k1_INTC_s005
                                 + 0.924444 w_k1_KO_s005
                                 + 0.790086 w_k1_MAT_s005
                                 + 0.945488 w_k1_MSFT_s005
                                 + 1.05113 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.10293 w_k1_ADM_s006
                                 + 0.953449 w_k1_AXP_s006
                                 + 1.04814 w_k1_DIS_s006
                                 + 1.03966 w_k1_FDX_s006 + 1.2658 w_k1_IBM_s006
                                 + 0.93468 w_k1_INTC_s006
                                 + 1.16844 w_k1_KO_s006 + 1.25613 w_k1_MAT_s006
                                 + 0.995772 w_k1_MSFT_s006
                                 + 0.983633 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.62174 w_k1_ADM_s006
                                 + 1.64895 w_k1_AXP_s006
                                 + 1.56585 w_k1_DIS_s006
                                 + 1.60908 w_k1_FDX_s006
                                 + 1.27005 w_k1_IBM_s006
                                 + 1.26775 w_k1_INTC_s006
                                 + 1.17737 w_k1_KO_s006 + 1.23906 w_k1_MAT_s006
                                 + 1.41803 w_k1_MSFT_s006
                                 + 1.27783 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.917749 w_k1_ADM_s006
                                 + 0.574671 w_k1_AXP_s006
                                 + 0.740342 w_k1_DIS_s006
                                 + 0.783885 w_k1_FDX_s006
                                 + 0.851593 w_k1_IBM_s006
                                 + 0.864921 w_k1_INTC_s006
                                 + 0.839538 w_k1_KO_s006
                                 + 0.728727 w_k1_MAT_s006
                                 + 0.823452 w_k1_MSFT_s006
                                 + 0.919623 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.21497 w_k1_ADM_s006
                                 + 0.994027 w_k1_AXP_s006
                                 + 0.988034 w_k1_DIS_s006
                                 + 1.06642 w_k1_FDX_s006
                                 + 0.741213 w_k1_IBM_s006
                                 + 0.862945 w_k1_INTC_s006
                                 + 0.7787 w_k1_KO_s006 + 0.94194 w_k1_MAT_s006
                                 + 0.879413 w_k1_MSFT_s006
                                 + 0.857051 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.968558 w_k1_ADM_s006
                                 + 1.23621 w_k1_AXP_s006 + 1.0808 w_k1_DIS_s006
                                 + 1.23071 w_k1_FDX_s006 + 1.0223 w_k1_IBM_s006
                                 + 1.13741 w_k1_INTC_s006
                                 + 1.08342 w_k1_KO_s006 + 1.45072 w_k1_MAT_s006
                                 + 1.19932 w_k1_MSFT_s006
                                 + 1.08061 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.988947 w_k1_ADM_s006
                                 + 0.61606 w_k1_AXP_s006
                                 + 0.601418 w_k1_DIS_s006
                                 + 0.775151 w_k1_FDX_s006
                                 + 0.563935 w_k1_IBM_s006
                                 + 0.880112 w_k1_INTC_s006
                                 + 0.672246 w_k1_KO_s006
                                 + 0.396531 w_k1_MAT_s006
                                 + 0.918608 w_k1_MSFT_s006
                                 + 0.828597 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.10943 w_k1_ADM_s006
                                 + 0.898981 w_k1_AXP_s006
                                 + 1.29306 w_k1_DIS_s006
                                 + 1.12165 w_k1_FDX_s006
                                 + 1.46862 w_k1_IBM_s006
                                 + 1.36042 w_k1_INTC_s006
                                 + 1.39756 w_k1_KO_s006 + 1.11697 w_k1_MAT_s006
                                 + 1.48579 w_k1_MSFT_s006
                                 + 1.17959 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.857964 w_k1_ADM_s006
                                 + 0.799142 w_k1_AXP_s006
                                 + 0.950368 w_k1_DIS_s006
                                 + 0.96972 w_k1_FDX_s006
                                 + 0.997718 w_k1_IBM_s006
                                 + 1.05707 w_k1_INTC_s006
                                 + 0.848417 w_k1_KO_s006
                                 + 0.862218 w_k1_MAT_s006
                                 + 0.917546 w_k1_MSFT_s006
                                 + 0.856057 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.866693 w_k1_ADM_s007
                                 + 0.79312 w_k1_AXP_s007
                                 + 0.848807 w_k1_DIS_s007
                                 + 0.890724 w_k1_FDX_s007
                                 + 0.744942 w_k1_IBM_s007
                                 + 0.97961 w_k1_INTC_s007
                                 + 0.990178 w_k1_KO_s007
                                 + 0.906993 w_k1_MAT_s007
                                 + 1.00742 w_k1_MSFT_s007
                                 + 0.918149 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.912615 w_k1_ADM_s007
                                 + 0.967659 w_k1_AXP_s007
                                 + 1.18239 w_k1_DIS_s007
                                 + 1.02275 w_k1_FDX_s007
                                 + 1.46394 w_k1_IBM_s007
                                 + 1.05916 w_k1_INTC_s007 + 1.3828 w_k1_KO_s007
                                 + 1.6852 w_k1_MAT_s007
                                 + 1.21135 w_k1_MSFT_s007
                                 + 1.23576 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.43438 w_k1_ADM_s007
                                 + 1.42167 w_k1_AXP_s007
                                 + 1.71854 w_k1_DIS_s007
                                 + 1.33188 w_k1_FDX_s007
                                 + 1.51767 w_k1_IBM_s007
                                 + 1.07133 w_k1_INTC_s007
                                 + 1.74121 w_k1_KO_s007 + 1.90456 w_k1_MAT_s007
                                 + 1.46443 w_k1_MSFT_s007
                                 + 1.50685 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.964152 w_k1_ADM_s007
                                 + 1.19919 w_k1_AXP_s007
                                 + 1.39122 w_k1_DIS_s007
                                 + 1.23322 w_k1_FDX_s007
                                 + 1.19471 w_k1_IBM_s007
                                 + 1.18098 w_k1_INTC_s007
                                 + 1.27064 w_k1_KO_s007 + 1.24772 w_k1_MAT_s007
                                 + 1.21462 w_k1_MSFT_s007
                                 + 1.19748 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.09431 w_k1_ADM_s007
                                 + 0.918923 w_k1_AXP_s007
                                 + 1.27312 w_k1_DIS_s007
                                 + 1.20897 w_k1_FDX_s007
                                 + 0.769878 w_k1_IBM_s007
                                 + 1.00155 w_k1_INTC_s007
                                 + 1.17582 w_k1_KO_s007
                                 + 0.972276 w_k1_MAT_s007
                                 + 1.15664 w_k1_MSFT_s007
                                 + 1.05491 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.92428 w_k1_ADM_s007
                                 + 0.875092 w_k1_AXP_s007
                                 + 0.97827 w_k1_DIS_s007
                                 + 0.839447 w_k1_FDX_s007
                                 + 1.37046 w_k1_IBM_s007
                                 + 0.698333 w_k1_INTC_s007
                                 + 1.26505 w_k1_KO_s007 + 1.99294 w_k1_MAT_s007
                                 + 0.845439 w_k1_MSFT_s007
                                 + 1.08908 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.883804 w_k1_ADM_s007
                                 + 0.949795 w_k1_AXP_s007
                                 + 0.756883 w_k1_DIS_s007
                                 + 0.760073 w_k1_FDX_s007
                                 + 1.14093 w_k1_IBM_s007
                                 + 0.703691 w_k1_INTC_s007
                                 + 1.03271 w_k1_KO_s007 + 1.14486 w_k1_MAT_s007
                                 + 0.810985 w_k1_MSFT_s007
                                 + 0.986617 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.981554 w_k1_ADM_s007
                                 + 1.27441 w_k1_AXP_s007
                                 + 0.907537 w_k1_DIS_s007
                                 + 0.987378 w_k1_FDX_s007
                                 + 1.29813 w_k1_IBM_s007
                                 + 0.826248 w_k1_INTC_s007
                                 + 1.10041 w_k1_KO_s007 + 1.95894 w_k1_MAT_s007
                                 + 0.854928 w_k1_MSFT_s007
                                 + 1.01781 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.623363 w_k1_ADM_s010
                                 + 0.733325 w_k1_AXP_s010
                                 + 0.780587 w_k1_DIS_s010
                                 + 0.737736 w_k1_FDX_s010
                                 + 1.16712 w_k1_IBM_s010
                                 + 0.863133 w_k1_INTC_s010
                                 + 0.840164 w_k1_KO_s010 + 0.9879 w_k1_MAT_s010
                                 + 0.81151 w_k1_MSFT_s010
                                 + 0.921878 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.0265 w_k1_ADM_s010
                                 + 1.44023 w_k1_AXP_s010
                                 + 1.58782 w_k1_DIS_s010 + 1.2238 w_k1_FDX_s010
                                 + 1.82054 w_k1_IBM_s010
                                 + 1.19493 w_k1_INTC_s010
                                 + 1.39518 w_k1_KO_s010 + 1.69594 w_k1_MAT_s010
                                 + 1.24256 w_k1_MSFT_s010
                                 + 1.15643 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.708855 w_k1_ADM_s010
                                 + 0.683847 w_k1_AXP_s010
                                 + 0.628791 w_k1_DIS_s010
                                 + 0.752154 w_k1_FDX_s010
                                 + 0.658191 w_k1_IBM_s010
                                 + 0.972275 w_k1_INTC_s010
                                 + 0.707055 w_k1_KO_s010
                                 + 0.434957 w_k1_MAT_s010
                                 + 0.771123 w_k1_MSFT_s010
                                 + 0.800435 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.986275 w_k1_ADM_s010
                                 + 1.17377 w_k1_AXP_s010
                                 + 1.24485 w_k1_DIS_s010
                                 + 1.26819 w_k1_FDX_s010
                                 + 0.78935 w_k1_IBM_s010
                                 + 1.30824 w_k1_INTC_s010
                                 + 0.952675 w_k1_KO_s010
                                 + 0.871561 w_k1_MAT_s010
                                 + 1.30609 w_k1_MSFT_s010
                                 + 1.01975 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.50312 w_k1_ADM_s010
                                 + 1.98916 w_k1_AXP_s010
                                 + 1.84057 w_k1_DIS_s010
                                 + 1.51355 w_k1_FDX_s010 + 1.3101 w_k1_IBM_s010
                                 + 1.382 w_k1_INTC_s010 + 1.18444 w_k1_KO_s010
                                 + 1.43883 w_k1_MAT_s010
                                 + 1.28926 w_k1_MSFT_s010
                                 + 1.26967 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.871316 w_k1_ADM_s010
                                 + 0.977714 w_k1_AXP_s010
                                 + 1.03054 w_k1_DIS_s010
                                 + 1.01053 w_k1_FDX_s010
                                 + 0.973023 w_k1_IBM_s010
                                 + 1.36385 w_k1_INTC_s010
                                 + 0.997221 w_k1_KO_s010
                                 + 0.737098 w_k1_MAT_s010
                                 + 1.08283 w_k1_MSFT_s010
                                 + 0.933621 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.793001 w_k1_ADM_s010
                                 + 1.23368 w_k1_AXP_s010
                                 + 1.10563 w_k1_DIS_s010
                                 + 1.04757 w_k1_FDX_s010
                                 + 1.34196 w_k1_IBM_s010
                                 + 1.35733 w_k1_INTC_s010
                                 + 1.04714 w_k1_KO_s010 + 1.17618 w_k1_MAT_s010
                                 + 1.09234 w_k1_MSFT_s010
                                 + 1.11095 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.847612 w_k1_ADM_s010
                                 + 1.04815 w_k1_AXP_s010
                                 + 0.821784 w_k1_DIS_s010
                                 + 1.02278 w_k1_FDX_s010
                                 + 0.718722 w_k1_IBM_s010
                                 + 1.13521 w_k1_INTC_s010
                                 + 0.759432 w_k1_KO_s010
                                 + 0.77314 w_k1_MAT_s010
                                 + 1.01356 w_k1_MSFT_s010
                                 + 0.943636 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.903872 w_k1_ADM_s011
                                 + 1.13832 w_k1_AXP_s011
                                 + 1.01277 w_k1_DIS_s011
                                 + 1.01059 w_k1_FDX_s011
                                 + 1.10557 w_k1_IBM_s011
                                 + 1.04701 w_k1_INTC_s011
                                 + 1.01306 w_k1_KO_s011 + 1.18823 w_k1_MAT_s011
                                 + 1.09905 w_k1_MSFT_s011
                                 + 1.17405 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.09673 w_k1_ADM_s011
                                 + 1.04915 w_k1_AXP_s011
                                 + 0.848086 w_k1_DIS_s011
                                 + 0.972378 w_k1_FDX_s011
                                 + 0.798778 w_k1_IBM_s011
                                 + 1.03287 w_k1_INTC_s011
                                 + 0.817223 w_k1_KO_s011
                                 + 0.949756 w_k1_MAT_s011
                                 + 1.03181 w_k1_MSFT_s011
                                 + 1.04533 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.733649 w_k1_ADM_s011
                                 + 0.725764 w_k1_AXP_s011
                                 + 0.69653 w_k1_DIS_s011
                                 + 0.879009 w_k1_FDX_s011
                                 + 0.536232 w_k1_IBM_s011
                                 + 0.939076 w_k1_INTC_s011
                                 + 0.639482 w_k1_KO_s011
                                 + 0.653467 w_k1_MAT_s011
                                 + 0.93241 w_k1_MSFT_s011
                                 + 0.816456 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.815159 w_k1_ADM_s011
                                 + 0.785376 w_k1_AXP_s011
                                 + 0.897151 w_k1_DIS_s011
                                 + 0.794922 w_k1_FDX_s011
                                 + 1.3359 w_k1_IBM_s011 + 1.0408 w_k1_INTC_s011
                                 + 1.155 w_k1_KO_s011 + 1.45908 w_k1_MAT_s011
                                 + 0.957941 w_k1_MSFT_s011
                                 + 1.06718 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.780926 w_k1_ADM_s011
                                 + 0.822741 w_k1_AXP_s011
                                 + 0.647361 w_k1_DIS_s011
                                 + 0.786588 w_k1_FDX_s011
                                 + 0.957915 w_k1_IBM_s011
                                 + 1.03207 w_k1_INTC_s011
                                 + 0.756259 w_k1_KO_s011
                                 + 0.970042 w_k1_MAT_s011
                                 + 0.881988 w_k1_MSFT_s011
                                 + 0.949527 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.27452 w_k1_ADM_s011
                                 + 1.56967 w_k1_AXP_s011
                                 + 1.44147 w_k1_DIS_s011
                                 + 1.25867 w_k1_FDX_s011
                                 + 1.45411 w_k1_IBM_s011
                                 + 1.34603 w_k1_INTC_s011
                                 + 1.15974 w_k1_KO_s011 + 1.45669 w_k1_MAT_s011
                                 + 1.26701 w_k1_MSFT_s011
                                 + 1.24968 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.666453 w_k1_ADM_s011
                                 + 0.6415 w_k1_AXP_s011
                                 + 0.591229 w_k1_DIS_s011
                                 + 0.775885 w_k1_FDX_s011
                                 + 0.725095 w_k1_IBM_s011
                                 + 0.785457 w_k1_INTC_s011
                                 + 0.665915 w_k1_KO_s011
                                 + 0.953072 w_k1_MAT_s011
                                 + 0.918686 w_k1_MSFT_s011
                                 + 0.851267 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.09954 w_k1_ADM_s011
                                 + 1.16905 w_k1_AXP_s011
                                 + 1.11052 w_k1_DIS_s011
                                 + 1.17502 w_k1_FDX_s011
                                 + 0.787634 w_k1_IBM_s011
                                 + 1.2837 w_k1_INTC_s011
                                 + 0.835605 w_k1_KO_s011
                                 + 0.820701 w_k1_MAT_s011
                                 + 1.23298 w_k1_MSFT_s011
                                 + 1.11597 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.02129 w_k1_ADM_s012
                                 + 1.15401 w_k1_AXP_s012
                                 + 0.944222 w_k1_DIS_s012
                                 + 0.966245 w_k1_FDX_s012
                                 + 1.34081 w_k1_IBM_s012
                                 + 0.816525 w_k1_INTC_s012
                                 + 1.0114 w_k1_KO_s012 + 2.04796 w_k1_MAT_s012
                                 + 0.955573 w_k1_MSFT_s012
                                 + 1.04689 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.2174 w_k1_ADM_s012
                                 + 1.44875 w_k1_AXP_s012 + 1.5909 w_k1_DIS_s012
                                 + 1.43052 w_k1_FDX_s012
                                 + 1.39325 w_k1_IBM_s012
                                 + 1.05387 w_k1_INTC_s012
                                 + 1.34973 w_k1_KO_s012 + 1.7777 w_k1_MAT_s012
                                 + 1.24492 w_k1_MSFT_s012
                                 + 1.13008 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.10387 w_k1_ADM_s012
                                 + 1.1019 w_k1_AXP_s012 + 1.27797 w_k1_DIS_s012
                                 + 1.23882 w_k1_FDX_s012
                                 + 0.925334 w_k1_IBM_s012
                                 + 1.15897 w_k1_INTC_s012
                                 + 1.02771 w_k1_KO_s012 + 1.25009 w_k1_MAT_s012
                                 + 1.3228 w_k1_MSFT_s012 + 1.0915 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.836159 w_k1_ADM_s012
                                 + 0.670649 w_k1_AXP_s012
                                 + 0.724199 w_k1_DIS_s012
                                 + 0.80541 w_k1_FDX_s012
                                 + 0.875769 w_k1_IBM_s012
                                 + 0.772287 w_k1_INTC_s012
                                 + 0.791705 w_k1_KO_s012
                                 + 0.594891 w_k1_MAT_s012
                                 + 0.827793 w_k1_MSFT_s012
                                 + 0.8555 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.998609 w_k1_ADM_s012
                                 + 1.02056 w_k1_AXP_s012
                                 + 1.01584 w_k1_DIS_s012
                                 + 1.08615 w_k1_FDX_s012
                                 + 1.52784 w_k1_IBM_s012
                                 + 0.943129 w_k1_INTC_s012
                                 + 1.13408 w_k1_KO_s012 + 1.51089 w_k1_MAT_s012
                                 + 0.950315 w_k1_MSFT_s012
                                 + 1.00069 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.938292 w_k1_ADM_s012
                                 + 0.961159 w_k1_AXP_s012
                                 + 0.857429 w_k1_DIS_s012
                                 + 0.90703 w_k1_FDX_s012
                                 + 1.08612 w_k1_IBM_s012
                                 + 0.906496 w_k1_INTC_s012
                                 + 0.87956 w_k1_KO_s012 + 1.48499 w_k1_MAT_s012
                                 + 0.848449 w_k1_MSFT_s012
                                 + 0.849912 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.999974 w_k1_ADM_s012
                                 + 0.845391 w_k1_AXP_s012 + 1.016 w_k1_DIS_s012
                                 + 1.02945 w_k1_FDX_s012
                                 + 0.767382 w_k1_IBM_s012
                                 + 0.961712 w_k1_INTC_s012
                                 + 0.839994 w_k1_KO_s012
                                 + 0.79138 w_k1_MAT_s012
                                 + 1.05517 w_k1_MSFT_s012
                                 + 0.907187 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.941144 w_k1_ADM_s012
                                 + 0.949653 w_k1_AXP_s012
                                 + 1.17923 w_k1_DIS_s012
                                 + 0.947786 w_k1_FDX_s012
                                 + 1.45599 w_k1_IBM_s012
                                 + 1.00799 w_k1_INTC_s012
                                 + 1.22701 w_k1_KO_s012 + 1.9003 w_k1_MAT_s012
                                 + 1.04084 w_k1_MSFT_s012
                                 + 0.95206 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.700795 w_k1_ADM_s013
                                 + 0.706257 w_k1_AXP_s013
                                 + 0.595813 w_k1_DIS_s013
                                 + 0.72668 w_k1_FDX_s013
                                 + 0.734441 w_k1_IBM_s013
                                 + 0.733751 w_k1_INTC_s013
                                 + 0.661054 w_k1_KO_s013
                                 + 0.539823 w_k1_MAT_s013
                                 + 0.75187 w_k1_MSFT_s013
                                 + 0.771246 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.924409 w_k1_ADM_s013
                                 + 0.911149 w_k1_AXP_s013
                                 + 0.793397 w_k1_DIS_s013
                                 + 0.947331 w_k1_FDX_s013
                                 + 0.67542 w_k1_IBM_s013
                                 + 0.817819 w_k1_INTC_s013
                                 + 0.7089 w_k1_KO_s013 + 0.761478 w_k1_MAT_s013
                                 + 0.882026 w_k1_MSFT_s013
                                 + 0.895448 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.26734 w_k1_ADM_s013
                                 + 1.38515 w_k1_AXP_s013
                                 + 1.28277 w_k1_DIS_s013
                                 + 1.24105 w_k1_FDX_s013
                                 + 0.962153 w_k1_IBM_s013
                                 + 0.926985 w_k1_INTC_s013
                                 + 1.17069 w_k1_KO_s013 + 1.1725 w_k1_MAT_s013
                                 + 1.06076 w_k1_MSFT_s013
                                 + 1.14209 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.882978 w_k1_ADM_s013
                                 + 0.924247 w_k1_AXP_s013
                                 + 0.927185 w_k1_DIS_s013
                                 + 0.903836 w_k1_FDX_s013
                                 + 1.19615 w_k1_IBM_s013
                                 + 0.759549 w_k1_INTC_s013
                                 + 1.09838 w_k1_KO_s013 + 1.40369 w_k1_MAT_s013
                                 + 0.945304 w_k1_MSFT_s013
                                 + 0.981339 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.961476 w_k1_ADM_s013
                                 + 1.00118 w_k1_AXP_s013
                                 + 1.35598 w_k1_DIS_s013
                                 + 1.07193 w_k1_FDX_s013
                                 + 1.36351 w_k1_IBM_s013
                                 + 0.813589 w_k1_INTC_s013
                                 + 1.29028 w_k1_KO_s013 + 1.05815 w_k1_MAT_s013
                                 + 1.18479 w_k1_MSFT_s013 + 1.1021 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.953263 w_k1_ADM_s013
                                 + 1.23047 w_k1_AXP_s013
                                 + 0.973742 w_k1_DIS_s013
                                 + 0.976005 w_k1_FDX_s013
                                 + 1.02141 w_k1_IBM_s013
                                 + 0.731004 w_k1_INTC_s013
                                 + 1.05401 w_k1_KO_s013 + 1.32009 w_k1_MAT_s013
                                 + 0.929434 w_k1_MSFT_s013
                                 + 1.08596 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.05106 w_k1_ADM_s013
                                 + 1.56187 w_k1_AXP_s013
                                 + 1.50121 w_k1_DIS_s013
                                 + 1.28476 w_k1_FDX_s013
                                 + 1.56766 w_k1_IBM_s013
                                 + 0.994552 w_k1_INTC_s013
                                 + 1.3718 w_k1_KO_s013 + 2.05915 w_k1_MAT_s013
                                 + 1.30716 w_k1_MSFT_s013
                                 + 1.11155 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.886813 w_k1_ADM_s013
                                 + 0.829446 w_k1_AXP_s013
                                 + 0.631397 w_k1_DIS_s013
                                 + 0.740822 w_k1_FDX_s013
                                 + 1.02006 w_k1_IBM_s013
                                 + 0.609231 w_k1_INTC_s013
                                 + 0.807696 w_k1_KO_s013
                                 + 0.899937 w_k1_MAT_s013
                                 + 0.755 w_k1_MSFT_s013 + 0.891995 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.76463 w_k1_ADM_s014
                                 + 0.63915 w_k1_AXP_s014
                                 + 0.804606 w_k1_DIS_s014
                                 + 0.721633 w_k1_FDX_s014
                                 + 1.19655 w_k1_IBM_s014
                                 + 0.835088 w_k1_INTC_s014
                                 + 0.931763 w_k1_KO_s014
                                 + 0.873976 w_k1_MAT_s014
                                 + 0.795316 w_k1_MSFT_s014
                                 + 0.832223 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.740751 w_k1_ADM_s014
                                 + 0.695173 w_k1_AXP_s014
                                 + 0.787119 w_k1_DIS_s014
                                 + 0.847618 w_k1_FDX_s014
                                 + 0.744715 w_k1_IBM_s014
                                 + 1.00627 w_k1_INTC_s014
                                 + 0.73182 w_k1_KO_s014 + 0.69857 w_k1_MAT_s014
                                 + 0.79134 w_k1_MSFT_s014
                                 + 0.834353 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.927509 w_k1_ADM_s014
                                 + 0.957663 w_k1_AXP_s014
                                 + 0.942895 w_k1_DIS_s014
                                 + 0.958547 w_k1_FDX_s014
                                 + 1.43533 w_k1_IBM_s014
                                 + 0.995462 w_k1_INTC_s014
                                 + 1.14503 w_k1_KO_s014 + 1.34417 w_k1_MAT_s014
                                 + 0.834738 w_k1_MSFT_s014
                                 + 0.918752 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.1863 w_k1_ADM_s014
                                 + 1.46194 w_k1_AXP_s014
                                 + 1.80105 w_k1_DIS_s014
                                 + 1.39615 w_k1_FDX_s014
                                 + 1.47491 w_k1_IBM_s014
                                 + 1.44289 w_k1_INTC_s014
                                 + 1.22932 w_k1_KO_s014 + 1.24302 w_k1_MAT_s014
                                 + 1.15339 w_k1_MSFT_s014 + 1.168 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.839211 w_k1_ADM_s014
                                 + 0.896205 w_k1_AXP_s014
                                 + 0.801927 w_k1_DIS_s014
                                 + 1.09459 w_k1_FDX_s014
                                 + 0.64485 w_k1_IBM_s014
                                 + 1.22212 w_k1_INTC_s014
                                 + 0.752962 w_k1_KO_s014
                                 + 0.806999 w_k1_MAT_s014
                                 + 0.994463 w_k1_MSFT_s014
                                 + 0.941569 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.21936 w_k1_ADM_s014
                                 + 1.17638 w_k1_AXP_s014
                                 + 1.22965 w_k1_DIS_s014
                                 + 1.12861 w_k1_FDX_s014
                                 + 0.976014 w_k1_IBM_s014
                                 + 1.39493 w_k1_INTC_s014
                                 + 1.12706 w_k1_KO_s014
                                 + 0.823609 w_k1_MAT_s014
                                 + 1.07143 w_k1_MSFT_s014
                                 + 1.06201 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.1057 w_k1_ADM_s014
                                 + 0.961309 w_k1_AXP_s014
                                 + 0.635646 w_k1_DIS_s014
                                 + 0.749919 w_k1_FDX_s014
                                 + 0.955559 w_k1_IBM_s014
                                 + 0.925926 w_k1_INTC_s014
                                 + 0.819833 w_k1_KO_s014
                                 + 0.852561 w_k1_MAT_s014
                                 + 0.738145 w_k1_MSFT_s014
                                 + 0.888758 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.829386 w_k1_ADM_s014
                                 + 0.593325 w_k1_AXP_s014
                                 + 0.795696 w_k1_DIS_s014
                                 + 0.91545 w_k1_FDX_s014
                                 + 0.592298 w_k1_IBM_s014
                                 + 0.971609 w_k1_INTC_s014
                                 + 0.682318 w_k1_KO_s014
                                 + 0.382411 w_k1_MAT_s014
                                 + 0.869756 w_k1_MSFT_s014
                                 + 0.748422 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.03367 w_k1_ADM_s015
                                 + 0.910736 w_k1_AXP_s015
                                 + 1.14582 w_k1_DIS_s015
                                 + 0.955569 w_k1_FDX_s015
                                 + 1.20129 w_k1_IBM_s015
                                 + 0.955704 w_k1_INTC_s015
                                 + 1.04513 w_k1_KO_s015
                                 + 0.935793 w_k1_MAT_s015
                                 + 1.02296 w_k1_MSFT_s015
                                 + 0.907628 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.829271 w_k1_ADM_s015
                                 + 0.585996 w_k1_AXP_s015
                                 + 0.654898 w_k1_DIS_s015
                                 + 0.840847 w_k1_FDX_s015
                                 + 0.683853 w_k1_IBM_s015
                                 + 0.90739 w_k1_INTC_s015
                                 + 0.645283 w_k1_KO_s015
                                 + 0.377067 w_k1_MAT_s015
                                 + 0.90039 w_k1_MSFT_s015
                                 + 0.770428 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.76148 w_k1_ADM_s015
                                 + 1.38621 w_k1_AXP_s015
                                 + 1.73608 w_k1_DIS_s015
                                 + 1.37837 w_k1_FDX_s015
                                 + 1.81413 w_k1_IBM_s015
                                 + 1.22551 w_k1_INTC_s015
                                 + 1.77591 w_k1_KO_s015 + 1.57655 w_k1_MAT_s015
                                 + 1.42909 w_k1_MSFT_s015
                                 + 1.33361 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.97524 w_k1_ADM_s015
                                 + 0.983476 w_k1_AXP_s015
                                 + 0.794218 w_k1_DIS_s015
                                 + 0.973617 w_k1_FDX_s015
                                 + 0.986038 w_k1_IBM_s015
                                 + 1.00143 w_k1_INTC_s015
                                 + 0.831939 w_k1_KO_s015
                                 + 0.756152 w_k1_MAT_s015
                                 + 0.83114 w_k1_MSFT_s015
                                 + 0.83756 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.4124 w_k1_ADM_s015
                                 + 1.21905 w_k1_AXP_s015
                                 + 1.27636 w_k1_DIS_s015
                                 + 1.12676 w_k1_FDX_s015
                                 + 1.32615 w_k1_IBM_s015
                                 + 1.12479 w_k1_INTC_s015
                                 + 1.30001 w_k1_KO_s015 + 1.23101 w_k1_MAT_s015
                                 + 1.19291 w_k1_MSFT_s015
                                 + 0.953133 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.26178 w_k1_ADM_s015
                                 + 1.31177 w_k1_AXP_s015
                                 + 1.57655 w_k1_DIS_s015
                                 + 1.31343 w_k1_FDX_s015 + 1.0821 w_k1_IBM_s015
                                 + 1.13869 w_k1_INTC_s015
                                 + 1.26423 w_k1_KO_s015 + 1.05512 w_k1_MAT_s015
                                 + 1.11078 w_k1_MSFT_s015
                                 + 1.10184 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.14151 w_k1_ADM_s015
                                 + 0.69044 w_k1_AXP_s015
                                 + 0.776064 w_k1_DIS_s015
                                 + 0.835673 w_k1_FDX_s015
                                 + 0.952095 w_k1_IBM_s015
                                 + 0.925014 w_k1_INTC_s015
                                 + 0.870853 w_k1_KO_s015
                                 + 0.579361 w_k1_MAT_s015
                                 + 0.944377 w_k1_MSFT_s015
                                 + 0.830659 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.06313 w_k1_ADM_s015
                                 + 0.900421 w_k1_AXP_s015
                                 + 1.10084 w_k1_DIS_s015
                                 + 1.10755 w_k1_FDX_s015
                                 + 0.675551 w_k1_IBM_s015
                                 + 1.09516 w_k1_INTC_s015
                                 + 0.967884 w_k1_KO_s015
                                 + 0.646453 w_k1_MAT_s015
                                 + 1.21947 w_k1_MSFT_s015
                                 + 0.963556 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.03615 w_k1_ADM_s016
                                 + 1.06061 w_k1_AXP_s016
                                 + 0.903272 w_k1_DIS_s016
                                 + 0.968913 w_k1_FDX_s016
                                 + 1.19697 w_k1_IBM_s016
                                 + 0.946152 w_k1_INTC_s016
                                 + 1.02362 w_k1_KO_s016 + 1.15835 w_k1_MAT_s016
                                 + 0.98219 w_k1_MSFT_s016
                                 + 0.952754 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.895802 w_k1_ADM_s016
                                 + 0.786818 w_k1_AXP_s016
                                 + 0.903711 w_k1_DIS_s016
                                 + 0.921247 w_k1_FDX_s016
                                 + 1.18448 w_k1_IBM_s016
                                 + 1.05256 w_k1_INTC_s016
                                 + 1.15142 w_k1_KO_s016 + 1.46401 w_k1_MAT_s016
                                 + 1.09147 w_k1_MSFT_s016
                                 + 1.02045 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.08842 w_k1_ADM_s016
                                 + 0.977877 w_k1_AXP_s016
                                 + 1.03128 w_k1_DIS_s016
                                 + 1.15088 w_k1_FDX_s016
                                 + 0.744102 w_k1_IBM_s016
                                 + 1.04153 w_k1_INTC_s016
                                 + 0.795116 w_k1_KO_s016
                                 + 0.836504 w_k1_MAT_s016
                                 + 1.09695 w_k1_MSFT_s016
                                 + 0.930393 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.31354 w_k1_ADM_s016
                                 + 0.974993 w_k1_AXP_s016
                                 + 1.11274 w_k1_DIS_s016
                                 + 1.08583 w_k1_FDX_s016
                                 + 1.03032 w_k1_IBM_s016
                                 + 1.17859 w_k1_INTC_s016 + 1.129 w_k1_KO_s016
                                 + 1.30147 w_k1_MAT_s016
                                 + 1.22192 w_k1_MSFT_s016
                                 + 1.12566 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.04429 w_k1_ADM_s016
                                 + 1.76098 w_k1_AXP_s016
                                 + 1.07939 w_k1_DIS_s016
                                 + 1.09593 w_k1_FDX_s016 + 1.5382 w_k1_IBM_s016
                                 + 1.03682 w_k1_INTC_s016
                                 + 1.27888 w_k1_KO_s016 + 2.32506 w_k1_MAT_s016
                                 + 1.01382 w_k1_MSFT_s016
                                 + 1.08829 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.5744 w_k1_ADM_s016
                                 + 1.37296 w_k1_AXP_s016 + 1.5001 w_k1_DIS_s016
                                 + 1.30291 w_k1_FDX_s016
                                 + 1.38339 w_k1_IBM_s016
                                 + 1.25014 w_k1_INTC_s016
                                 + 1.37106 w_k1_KO_s016 + 1.64652 w_k1_MAT_s016
                                 + 1.27028 w_k1_MSFT_s016 + 1.3344 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.867225 w_k1_ADM_s016
                                 + 0.595282 w_k1_AXP_s016
                                 + 0.607228 w_k1_DIS_s016
                                 + 0.75018 w_k1_FDX_s016
                                 + 0.789049 w_k1_IBM_s016
                                 + 0.785864 w_k1_INTC_s016
                                 + 0.756944 w_k1_KO_s016
                                 + 0.562957 w_k1_MAT_s016
                                 + 0.802724 w_k1_MSFT_s016
                                 + 0.849876 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.890489 w_k1_ADM_s016
                                 + 0.963757 w_k1_AXP_s016
                                 + 0.769187 w_k1_DIS_s016
                                 + 0.792826 w_k1_FDX_s016
                                 + 1.57501 w_k1_IBM_s016
                                 + 0.818886 w_k1_INTC_s016
                                 + 1.00913 w_k1_KO_s016 + 1.47928 w_k1_MAT_s016
                                 + 0.743666 w_k1_MSFT_s016
                                 + 0.890406 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.873686 w_k1_ADM_s017
                                 + 0.70159 w_k1_AXP_s017
                                 + 0.542005 w_k1_DIS_s017
                                 + 0.720433 w_k1_FDX_s017
                                 + 0.76706 w_k1_IBM_s017
                                 + 0.833262 w_k1_INTC_s017
                                 + 0.674525 w_k1_KO_s017
                                 + 0.593654 w_k1_MAT_s017
                                 + 0.748241 w_k1_MSFT_s017
                                 + 0.785721 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.32691 w_k1_ADM_s017
                                 + 1.4113 w_k1_AXP_s017 + 1.71406 w_k1_DIS_s017
                                 + 1.33044 w_k1_FDX_s017
                                 + 1.74653 w_k1_IBM_s017
                                 + 1.35336 w_k1_INTC_s017
                                 + 1.63994 w_k1_KO_s017 + 1.94226 w_k1_MAT_s017
                                 + 1.49636 w_k1_MSFT_s017
                                 + 1.30973 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.09924 w_k1_ADM_s017
                                 + 0.857281 w_k1_AXP_s017
                                 + 1.25003 w_k1_DIS_s017
                                 + 1.02636 w_k1_FDX_s017
                                 + 1.02376 w_k1_IBM_s017
                                 + 1.10165 w_k1_INTC_s017
                                 + 1.10736 w_k1_KO_s017
                                 + 0.856221 w_k1_MAT_s017
                                 + 1.16727 w_k1_MSFT_s017
                                 + 0.974415 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.14876 w_k1_ADM_s017
                                 + 0.928159 w_k1_AXP_s017
                                 + 0.717825 w_k1_DIS_s017
                                 + 0.875727 w_k1_FDX_s017
                                 + 1.25023 w_k1_IBM_s017
                                 + 0.953115 w_k1_INTC_s017
                                 + 0.957724 w_k1_KO_s017
                                 + 1.60072 w_k1_MAT_s017
                                 + 0.887199 w_k1_MSFT_s017
                                 + 0.922102 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.899077 w_k1_ADM_s017
                                 + 0.818178 w_k1_AXP_s017
                                 + 0.769744 w_k1_DIS_s017
                                 + 0.87704 w_k1_FDX_s017
                                 + 0.778225 w_k1_IBM_s017
                                 + 0.87242 w_k1_INTC_s017
                                 + 0.862662 w_k1_KO_s017
                                 + 0.752687 w_k1_MAT_s017
                                 + 0.882147 w_k1_MSFT_s017
                                 + 0.785137 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.59051 w_k1_ADM_s017
                                 + 1.19596 w_k1_AXP_s017
                                 + 1.41619 w_k1_DIS_s017
                                 + 1.33187 w_k1_FDX_s017
                                 + 0.893646 w_k1_IBM_s017
                                 + 1.08947 w_k1_INTC_s017
                                 + 1.13353 w_k1_KO_s017
                                 + 0.869196 w_k1_MAT_s017
                                 + 1.22453 w_k1_MSFT_s017
                                 + 0.909808 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.16765 w_k1_ADM_s017
                                 + 0.803191 w_k1_AXP_s017
                                 + 1.01743 w_k1_DIS_s017
                                 + 1.05227 w_k1_FDX_s017
                                 + 0.715991 w_k1_IBM_s017
                                 + 0.898097 w_k1_INTC_s017
                                 + 0.938753 w_k1_KO_s017
                                 + 0.548466 w_k1_MAT_s017
                                 + 1.04553 w_k1_MSFT_s017
                                 + 0.974006 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.23571 w_k1_ADM_s017
                                 + 1.23527 w_k1_AXP_s017
                                 + 1.19652 w_k1_DIS_s017
                                 + 1.08025 w_k1_FDX_s017
                                 + 1.38478 w_k1_IBM_s017
                                 + 0.890786 w_k1_INTC_s017
                                 + 1.27583 w_k1_KO_s017 + 1.83379 w_k1_MAT_s017
                                 + 0.992926 w_k1_MSFT_s017
                                 + 1.06183 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.12736 w_k1_ADM_s100
                                 + 0.954819 w_k1_AXP_s100
                                 + 1.20544 w_k1_DIS_s100
                                 + 1.06348 w_k1_FDX_s100
                                 + 1.05907 w_k1_IBM_s100
                                 + 1.2543 w_k1_INTC_s100 + 1.30282 w_k1_KO_s100
                                 + 0.868289 w_k1_MAT_s100
                                 + 1.24273 w_k1_MSFT_s100
                                 + 0.920134 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.65848 w_k1_ADM_s100
                                 + 1.68449 w_k1_AXP_s100
                                 + 1.95968 w_k1_DIS_s100 + 1.3454 w_k1_FDX_s100
                                 + 1.68672 w_k1_IBM_s100
                                 + 1.19433 w_k1_INTC_s100
                                 + 1.91426 w_k1_KO_s100 + 2.23959 w_k1_MAT_s100
                                 + 1.39933 w_k1_MSFT_s100
                                 + 1.62674 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.21266 w_k1_ADM_s100
                                 + 1.34155 w_k1_AXP_s100
                                 + 0.969584 w_k1_DIS_s100
                                 + 1.07725 w_k1_FDX_s100
                                 + 1.01063 w_k1_IBM_s100
                                 + 1.07058 w_k1_INTC_s100
                                 + 1.02574 w_k1_KO_s100 + 1.03665 w_k1_MAT_s100
                                 + 1.02026 w_k1_MSFT_s100
                                 + 1.03374 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.57162 w_k1_ADM_s100
                                 + 1.31148 w_k1_AXP_s100
                                 + 1.33875 w_k1_DIS_s100
                                 + 1.28977 w_k1_FDX_s100
                                 + 1.06241 w_k1_IBM_s100
                                 + 1.13476 w_k1_INTC_s100 + 1.2761 w_k1_KO_s100
                                 + 1.53376 w_k1_MAT_s100
                                 + 1.32354 w_k1_MSFT_s100
                                 + 1.10257 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.04837 w_k1_ADM_s100
                                 + 0.989067 w_k1_AXP_s100
                                 + 0.832853 w_k1_DIS_s100
                                 + 0.865068 w_k1_FDX_s100
                                 + 1.07607 w_k1_IBM_s100
                                 + 0.961349 w_k1_INTC_s100
                                 + 1.06867 w_k1_KO_s100 + 1.51654 w_k1_MAT_s100
                                 + 0.9364 w_k1_MSFT_s100
                                 + 0.978989 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.950802 w_k1_ADM_s100
                                 + 0.664904 w_k1_AXP_s100
                                 + 0.766139 w_k1_DIS_s100
                                 + 0.788401 w_k1_FDX_s100
                                 + 0.684722 w_k1_IBM_s100
                                 + 0.790654 w_k1_INTC_s100
                                 + 0.929432 w_k1_KO_s100
                                 + 0.811659 w_k1_MAT_s100
                                 + 0.853854 w_k1_MSFT_s100
                                 + 0.865597 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.01305 w_k1_ADM_s100
                                 + 1.10112 w_k1_AXP_s100
                                 + 1.63092 w_k1_DIS_s100
                                 + 1.16903 w_k1_FDX_s100
                                 + 1.42985 w_k1_IBM_s100
                                 + 1.09735 w_k1_INTC_s100
                                 + 1.66725 w_k1_KO_s100 + 1.83048 w_k1_MAT_s100
                                 + 1.25575 w_k1_MSFT_s100
                                 + 1.26855 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.904965 w_k1_ADM_s100
                                 + 1.03055 w_k1_AXP_s100
                                 + 0.738196 w_k1_DIS_s100
                                 + 0.936526 w_k1_FDX_s100
                                 + 0.70685 w_k1_IBM_s100
                                 + 1.04587 w_k1_INTC_s100
                                 + 0.862901 w_k1_KO_s100
                                 + 0.830457 w_k1_MAT_s100
                                 + 0.954372 w_k1_MSFT_s100
                                 + 0.892377 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.72691 w_k1_ADM_s101
                                 + 0.740431 w_k1_AXP_s101
                                 + 0.638293 w_k1_DIS_s101
                                 + 0.758185 w_k1_FDX_s101
                                 + 0.789984 w_k1_IBM_s101
                                 + 0.757319 w_k1_INTC_s101
                                 + 0.769064 w_k1_KO_s101
                                 + 0.763249 w_k1_MAT_s101
                                 + 0.705713 w_k1_MSFT_s101
                                 + 0.795796 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.31213 w_k1_ADM_s101
                                 + 1.56635 w_k1_AXP_s101
                                 + 1.40814 w_k1_DIS_s101 + 1.2006 w_k1_FDX_s101
                                 + 1.72935 w_k1_IBM_s101
                                 + 1.13767 w_k1_INTC_s101
                                 + 1.60243 w_k1_KO_s101 + 1.64145 w_k1_MAT_s101
                                 + 1.05939 w_k1_MSFT_s101
                                 + 1.37988 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.744315 w_k1_ADM_s101
                                 + 1.03456 w_k1_AXP_s101
                                 + 0.948394 w_k1_DIS_s101
                                 + 0.840754 w_k1_FDX_s101
                                 + 1.13241 w_k1_IBM_s101
                                 + 0.841173 w_k1_INTC_s101
                                 + 0.999975 w_k1_KO_s101
                                 + 1.12223 w_k1_MAT_s101
                                 + 0.870769 w_k1_MSFT_s101
                                 + 0.932119 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.945937 w_k1_ADM_s101
                                 + 1.13555 w_k1_AXP_s101
                                 + 1.11408 w_k1_DIS_s101
                                 + 1.16087 w_k1_FDX_s101
                                 + 0.822323 w_k1_IBM_s101
                                 + 1.17056 w_k1_INTC_s101
                                 + 0.977404 w_k1_KO_s101
                                 + 0.98671 w_k1_MAT_s101
                                 + 1.1384 w_k1_MSFT_s101 + 1.12659 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.755081 w_k1_ADM_s101
                                 + 0.758985 w_k1_AXP_s101
                                 + 0.800931 w_k1_DIS_s101
                                 + 0.797877 w_k1_FDX_s101
                                 + 0.987203 w_k1_IBM_s101
                                 + 0.833186 w_k1_INTC_s101
                                 + 1.01042 w_k1_KO_s101 + 0.83583 w_k1_MAT_s101
                                 + 0.779418 w_k1_MSFT_s101
                                 + 0.966926 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.00794 w_k1_ADM_s101
                                 + 1.2769 w_k1_AXP_s101 + 1.4809 w_k1_DIS_s101
                                 + 1.08974 w_k1_FDX_s101
                                 + 1.25606 w_k1_IBM_s101
                                 + 1.29098 w_k1_INTC_s101 + 1.4031 w_k1_KO_s101
                                 + 1.36675 w_k1_MAT_s101 + 1.109 w_k1_MSFT_s101
                                 + 1.2065 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.867983 w_k1_ADM_s101
                                 + 0.995194 w_k1_AXP_s101
                                 + 1.06352 w_k1_DIS_s101
                                 + 0.885559 w_k1_FDX_s101
                                 + 1.51214 w_k1_IBM_s101
                                 + 1.08552 w_k1_INTC_s101
                                 + 1.22681 w_k1_KO_s101 + 1.26167 w_k1_MAT_s101
                                 + 0.936814 w_k1_MSFT_s101
                                 + 0.977498 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.795619 w_k1_ADM_s101
                                 + 0.795176 w_k1_AXP_s101
                                 + 0.986957 w_k1_DIS_s101
                                 + 0.949711 w_k1_FDX_s101
                                 + 0.72336 w_k1_IBM_s101
                                 + 0.947902 w_k1_INTC_s101
                                 + 0.944369 w_k1_KO_s101
                                 + 0.750497 w_k1_MAT_s101
                                 + 0.886834 w_k1_MSFT_s101
                                 + 0.978136 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.11152 w_k1_ADM_s102
                                 + 1.10884 w_k1_AXP_s102
                                 + 1.05085 w_k1_DIS_s102
                                 + 1.08742 w_k1_FDX_s102
                                 + 0.815212 w_k1_IBM_s102
                                 + 1.04181 w_k1_INTC_s102
                                 + 0.926801 w_k1_KO_s102
                                 + 0.999102 w_k1_MAT_s102
                                 + 1.10104 w_k1_MSFT_s102
                                 + 0.880588 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.709966 w_k1_ADM_s102
                                 + 0.546584 w_k1_AXP_s102
                                 + 0.720493 w_k1_DIS_s102
                                 + 0.790203 w_k1_FDX_s102
                                 + 0.828783 w_k1_IBM_s102
                                 + 0.732179 w_k1_INTC_s102
                                 + 0.80019 w_k1_KO_s102
                                 + 0.796179 w_k1_MAT_s102
                                 + 0.782553 w_k1_MSFT_s102
                                 + 0.841953 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.22565 w_k1_ADM_s102
                                 + 1.17629 w_k1_AXP_s102
                                 + 1.12935 w_k1_DIS_s102
                                 + 1.10175 w_k1_FDX_s102
                                 + 1.41965 w_k1_IBM_s102
                                 + 1.13557 w_k1_INTC_s102
                                 + 1.17784 w_k1_KO_s102 + 1.8473 w_k1_MAT_s102
                                 + 1.07008 w_k1_MSFT_s102
                                 + 1.08485 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.996562 w_k1_ADM_s102
                                 + 0.94535 w_k1_AXP_s102
                                 + 0.976364 w_k1_DIS_s102
                                 + 0.815858 w_k1_FDX_s102
                                 + 1.26694 w_k1_IBM_s102
                                 + 0.921358 w_k1_INTC_s102
                                 + 1.17655 w_k1_KO_s102 + 1.15351 w_k1_MAT_s102
                                 + 0.884205 w_k1_MSFT_s102
                                 + 1.00261 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.920548 w_k1_ADM_s102
                                 + 0.851184 w_k1_AXP_s102
                                 + 0.822806 w_k1_DIS_s102
                                 + 0.938018 w_k1_FDX_s102
                                 + 0.900773 w_k1_IBM_s102
                                 + 0.882428 w_k1_INTC_s102
                                 + 1.05712 w_k1_KO_s102 + 1.22847 w_k1_MAT_s102
                                 + 1.01142 w_k1_MSFT_s102
                                 + 1.00036 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.88733 w_k1_ADM_s102
                                 + 1.08863 w_k1_AXP_s102
                                 + 0.981051 w_k1_DIS_s102
                                 + 0.985845 w_k1_FDX_s102
                                 + 1.08068 w_k1_IBM_s102
                                 + 1.07616 w_k1_INTC_s102
                                 + 0.932109 w_k1_KO_s102
                                 + 0.939898 w_k1_MAT_s102
                                 + 0.89038 w_k1_MSFT_s102
                                 + 0.86482 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.86203 w_k1_ADM_s102
                                 + 0.752481 w_k1_AXP_s102
                                 + 0.697386 w_k1_DIS_s102
                                 + 0.815346 w_k1_FDX_s102
                                 + 0.660301 w_k1_IBM_s102
                                 + 0.820475 w_k1_INTC_s102
                                 + 0.654387 w_k1_KO_s102
                                 + 0.57003 w_k1_MAT_s102
                                 + 0.800844 w_k1_MSFT_s102
                                 + 0.752555 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.43242 w_k1_ADM_s102
                                 + 1.05042 w_k1_AXP_s102
                                 + 1.50836 w_k1_DIS_s102
                                 + 1.34155 w_k1_FDX_s102
                                 + 0.934536 w_k1_IBM_s102
                                 + 1.12816 w_k1_INTC_s102
                                 + 1.19834 w_k1_KO_s102
                                 + 0.683079 w_k1_MAT_s102
                                 + 1.22269 w_k1_MSFT_s102 + 1.0722 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.06223 w_k1_ADM_s103
                                 + 0.899931 w_k1_AXP_s103
                                 + 0.997424 w_k1_DIS_s103
                                 + 0.998175 w_k1_FDX_s103
                                 + 0.807822 w_k1_IBM_s103
                                 + 1.02336 w_k1_INTC_s103
                                 + 0.967098 w_k1_KO_s103
                                 + 0.993657 w_k1_MAT_s103
                                 + 0.800555 w_k1_MSFT_s103
                                 + 0.966622 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.905994 w_k1_ADM_s103
                                 + 0.888599 w_k1_AXP_s103
                                 + 1.20464 w_k1_DIS_s103
                                 + 0.844835 w_k1_FDX_s103
                                 + 1.38583 w_k1_IBM_s103
                                 + 0.915277 w_k1_INTC_s103
                                 + 1.33846 w_k1_KO_s103 + 1.24295 w_k1_MAT_s103
                                 + 0.918193 w_k1_MSFT_s103
                                 + 0.950439 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.679384 w_k1_ADM_s103
                                 + 0.758747 w_k1_AXP_s103
                                 + 0.876453 w_k1_DIS_s103
                                 + 0.873236 w_k1_FDX_s103
                                 + 1.17049 w_k1_IBM_s103
                                 + 0.970066 w_k1_INTC_s103
                                 + 1.01063 w_k1_KO_s103 + 1.05118 w_k1_MAT_s103
                                 + 0.738539 w_k1_MSFT_s103
                                 + 0.882012 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.66218 w_k1_ADM_s103
                                 + 0.719234 w_k1_AXP_s103
                                 + 0.603768 w_k1_DIS_s103
                                 + 0.716726 w_k1_FDX_s103
                                 + 0.986422 w_k1_IBM_s103
                                 + 0.863364 w_k1_INTC_s103
                                 + 0.888522 w_k1_KO_s103
                                 + 1.05578 w_k1_MAT_s103
                                 + 0.810243 w_k1_MSFT_s103
                                 + 0.809554 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.989106 w_k1_ADM_s103
                                 + 1.26671 w_k1_AXP_s103 + 1.189 w_k1_DIS_s103
                                 + 1.01687 w_k1_FDX_s103
                                 + 1.16797 w_k1_IBM_s103
                                 + 1.17191 w_k1_INTC_s103
                                 + 1.34885 w_k1_KO_s103 + 1.37526 w_k1_MAT_s103
                                 + 0.939376 w_k1_MSFT_s103
                                 + 1.12126 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.2885 w_k1_ADM_s103
                                 + 1.53975 w_k1_AXP_s103
                                 + 1.77077 w_k1_DIS_s103
                                 + 1.47464 w_k1_FDX_s103
                                 + 1.06527 w_k1_IBM_s103
                                 + 1.32484 w_k1_INTC_s103
                                 + 1.38636 w_k1_KO_s103 + 1.3641 w_k1_MAT_s103
                                 + 1.25604 w_k1_MSFT_s103
                                 + 1.19669 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.16893 w_k1_ADM_s103
                                 + 1.20751 w_k1_AXP_s103
                                 + 1.15238 w_k1_DIS_s103
                                 + 1.00709 w_k1_FDX_s103
                                 + 1.85533 w_k1_IBM_s103
                                 + 1.08522 w_k1_INTC_s103
                                 + 1.35203 w_k1_KO_s103 + 1.94328 w_k1_MAT_s103
                                 + 0.934608 w_k1_MSFT_s103
                                 + 0.967771 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.698712 w_k1_ADM_s103
                                 + 0.565689 w_k1_AXP_s103
                                 + 0.604261 w_k1_DIS_s103
                                 + 0.640147 w_k1_FDX_s103
                                 + 0.69032 w_k1_IBM_s103
                                 + 0.706718 w_k1_INTC_s103
                                 + 0.766483 w_k1_KO_s103
                                 + 0.677627 w_k1_MAT_s103
                                 + 0.595648 w_k1_MSFT_s103
                                 + 0.791556 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.1305 w_k1_ADM_s104
                                 + 0.890024 w_k1_AXP_s104
                                 + 1.15216 w_k1_DIS_s104
                                 + 1.04658 w_k1_FDX_s104
                                 + 0.928708 w_k1_IBM_s104
                                 + 0.954699 w_k1_INTC_s104
                                 + 1.05038 w_k1_KO_s104
                                 + 0.759561 w_k1_MAT_s104
                                 + 1.11408 w_k1_MSFT_s104
                                 + 0.948975 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.966363 w_k1_ADM_s104
                                 + 1.22149 w_k1_AXP_s104
                                 + 1.22178 w_k1_DIS_s104
                                 + 1.23287 w_k1_FDX_s104 + 0.7281 w_k1_IBM_s104
                                 + 1.06742 w_k1_INTC_s104
                                 + 1.01488 w_k1_KO_s104 + 1.00366 w_k1_MAT_s104
                                 + 1.2145 w_k1_MSFT_s104 + 1.04067 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.889663 w_k1_ADM_s104
                                 + 0.894637 w_k1_AXP_s104
                                 + 0.911708 w_k1_DIS_s104
                                 + 0.916537 w_k1_FDX_s104
                                 + 1.16286 w_k1_IBM_s104
                                 + 0.946638 w_k1_INTC_s104
                                 + 1.10208 w_k1_KO_s104 + 1.14676 w_k1_MAT_s104
                                 + 0.948214 w_k1_MSFT_s104
                                 + 0.950245 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.04772 w_k1_ADM_s104
                                 + 1.34495 w_k1_AXP_s104
                                 + 1.24342 w_k1_DIS_s104
                                 + 1.24482 w_k1_FDX_s104
                                 + 1.12153 w_k1_IBM_s104
                                 + 1.02731 w_k1_INTC_s104
                                 + 1.15822 w_k1_KO_s104 + 1.33348 w_k1_MAT_s104
                                 + 1.08842 w_k1_MSFT_s104
                                 + 1.06796 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.40845 w_k1_ADM_s104
                                 + 1.37796 w_k1_AXP_s104
                                 + 1.55957 w_k1_DIS_s104
                                 + 1.26688 w_k1_FDX_s104
                                 + 1.30093 w_k1_IBM_s104
                                 + 1.16211 w_k1_INTC_s104 + 1.479 w_k1_KO_s104
                                 + 1.36364 w_k1_MAT_s104
                                 + 1.25991 w_k1_MSFT_s104
                                 + 1.12025 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.823616 w_k1_ADM_s104
                                 + 0.717284 w_k1_AXP_s104
                                 + 0.795057 w_k1_DIS_s104
                                 + 0.847153 w_k1_FDX_s104
                                 + 0.70463 w_k1_IBM_s104
                                 + 0.741564 w_k1_INTC_s104
                                 + 0.848332 w_k1_KO_s104
                                 + 0.617451 w_k1_MAT_s104
                                 + 0.740048 w_k1_MSFT_s104
                                 + 0.853523 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.779423 w_k1_ADM_s104
                                 + 0.849026 w_k1_AXP_s104
                                 + 0.81075 w_k1_DIS_s104
                                 + 0.916902 w_k1_FDX_s104
                                 + 1.03177 w_k1_IBM_s104
                                 + 0.893543 w_k1_INTC_s104
                                 + 0.973495 w_k1_KO_s104
                                 + 1.12121 w_k1_MAT_s104
                                 + 0.86976 w_k1_MSFT_s104
                                 + 0.891144 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.44531 w_k1_ADM_s104
                                 + 1.28236 w_k1_AXP_s104
                                 + 1.25879 w_k1_DIS_s104
                                 + 1.44951 w_k1_FDX_s104
                                 + 0.599049 w_k1_IBM_s104
                                 + 1.23888 w_k1_INTC_s104
                                 + 0.958968 w_k1_KO_s104
                                 + 0.990768 w_k1_MAT_s104
                                 + 1.14339 w_k1_MSFT_s104
                                 + 1.06245 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.0724 w_k1_ADM_s105
                                 + 1.09251 w_k1_AXP_s105
                                 + 1.14366 w_k1_DIS_s105
                                 + 1.16479 w_k1_FDX_s105
                                 + 0.709246 w_k1_IBM_s105
                                 + 1.09247 w_k1_INTC_s105
                                 + 0.969188 w_k1_KO_s105
                                 + 0.938821 w_k1_MAT_s105
                                 + 1.10316 w_k1_MSFT_s105 + 1.0688 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.695331 w_k1_ADM_s105
                                 + 0.817793 w_k1_AXP_s105
                                 + 0.77332 w_k1_DIS_s105
                                 + 0.764994 w_k1_FDX_s105
                                 + 0.947577 w_k1_IBM_s105
                                 + 0.679896 w_k1_INTC_s105
                                 + 0.862943 w_k1_KO_s105
                                 + 1.08241 w_k1_MAT_s105
                                 + 0.75113 w_k1_MSFT_s105
                                 + 0.777996 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.906727 w_k1_ADM_s105
                                 + 0.811169 w_k1_AXP_s105
                                 + 0.895457 w_k1_DIS_s105
                                 + 0.973946 w_k1_FDX_s105
                                 + 0.579186 w_k1_IBM_s105
                                 + 0.837379 w_k1_INTC_s105
                                 + 0.824614 w_k1_KO_s105
                                 + 0.820054 w_k1_MAT_s105
                                 + 0.93211 w_k1_MSFT_s105
                                 + 0.871508 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.20527 w_k1_ADM_s105
                                 + 1.4918 w_k1_AXP_s105 + 1.57356 w_k1_DIS_s105
                                 + 1.39993 w_k1_FDX_s105
                                 + 0.951882 w_k1_IBM_s105
                                 + 1.1303 w_k1_INTC_s105 + 1.07587 w_k1_KO_s105
                                 + 1.38686 w_k1_MAT_s105
                                 + 1.30554 w_k1_MSFT_s105 + 1.2671 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.949867 w_k1_ADM_s105
                                 + 1.46319 w_k1_AXP_s105 + 1.2767 w_k1_DIS_s105
                                 + 1.05776 w_k1_FDX_s105
                                 + 1.80007 w_k1_IBM_s105
                                 + 1.04749 w_k1_INTC_s105
                                 + 1.49936 w_k1_KO_s105 + 2.10516 w_k1_MAT_s105
                                 + 1.02469 w_k1_MSFT_s105
                                 + 1.09177 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.955364 w_k1_ADM_s105
                                 + 1.02307 w_k1_AXP_s105
                                 + 0.801528 w_k1_DIS_s105
                                 + 0.91996 w_k1_FDX_s105
                                 + 0.870168 w_k1_IBM_s105
                                 + 0.799404 w_k1_INTC_s105
                                 + 0.82949 w_k1_KO_s105 + 1.31318 w_k1_MAT_s105
                                 + 0.91593 w_k1_MSFT_s105
                                 + 0.893758 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.03136 w_k1_ADM_s105
                                 + 1.08205 w_k1_AXP_s105
                                 + 1.13865 w_k1_DIS_s105
                                 + 1.04387 w_k1_FDX_s105
                                 + 1.02931 w_k1_IBM_s105
                                 + 0.976817 w_k1_INTC_s105
                                 + 1.10103 w_k1_KO_s105 + 1.42908 w_k1_MAT_s105
                                 + 1.13715 w_k1_MSFT_s105
                                 + 0.989525 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.2106 w_k1_ADM_s105
                                 + 1.23071 w_k1_AXP_s105 + 1.2754 w_k1_DIS_s105
                                 + 0.993822 w_k1_FDX_s105
                                 + 1.35674 w_k1_IBM_s105
                                 + 1.02322 w_k1_INTC_s105
                                 + 1.23673 w_k1_KO_s105 + 1.06629 w_k1_MAT_s105
                                 + 0.932496 w_k1_MSFT_s105
                                 + 0.97625 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.66844 w_k1_ADM_s106
                                 + 0.697654 w_k1_AXP_s106
                                 + 0.495689 w_k1_DIS_s106
                                 + 0.645709 w_k1_FDX_s106
                                 + 0.768468 w_k1_IBM_s106
                                 + 0.919238 w_k1_INTC_s106
                                 + 0.694926 w_k1_KO_s106
                                 + 0.55023 w_k1_MAT_s106
                                 + 0.62261 w_k1_MSFT_s106
                                 + 0.704815 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.833167 w_k1_ADM_s106
                                 + 0.793776 w_k1_AXP_s106
                                 + 0.775846 w_k1_DIS_s106
                                 + 0.822413 w_k1_FDX_s106
                                 + 0.992879 w_k1_IBM_s106
                                 + 1.04466 w_k1_INTC_s106
                                 + 0.863708 w_k1_KO_s106
                                 + 0.612132 w_k1_MAT_s106
                                 + 0.843937 w_k1_MSFT_s106
                                 + 0.856108 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.08759 w_k1_ADM_s106
                                 + 1.54512 w_k1_AXP_s106
                                 + 1.63481 w_k1_DIS_s106
                                 + 1.38548 w_k1_FDX_s106
                                 + 1.10565 w_k1_IBM_s106
                                 + 1.45755 w_k1_INTC_s106
                                 + 1.14845 w_k1_KO_s106 + 1.2386 w_k1_MAT_s106
                                 + 1.39989 w_k1_MSFT_s106
                                 + 1.06061 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.776468 w_k1_ADM_s106
                                 + 1.32462 w_k1_AXP_s106
                                 + 1.25046 w_k1_DIS_s106
                                 + 1.09193 w_k1_FDX_s106
                                 + 1.73176 w_k1_IBM_s106
                                 + 1.33416 w_k1_INTC_s106
                                 + 1.17089 w_k1_KO_s106 + 1.30888 w_k1_MAT_s106
                                 + 1.00836 w_k1_MSFT_s106
                                 + 1.00117 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.783655 w_k1_ADM_s106
                                 + 1.07376 w_k1_AXP_s106
                                 + 1.06744 w_k1_DIS_s106
                                 + 0.943021 w_k1_FDX_s106
                                 + 1.39343 w_k1_IBM_s106
                                 + 1.27559 w_k1_INTC_s106
                                 + 1.04572 w_k1_KO_s106 + 1.10111 w_k1_MAT_s106
                                 + 1.10635 w_k1_MSFT_s106
                                 + 1.07232 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.950036 w_k1_ADM_s106
                                 + 1.07026 w_k1_AXP_s106
                                 + 1.03781 w_k1_DIS_s106
                                 + 1.07321 w_k1_FDX_s106
                                 + 0.788971 w_k1_IBM_s106
                                 + 1.28005 w_k1_INTC_s106
                                 + 0.847251 w_k1_KO_s106
                                 + 0.646589 w_k1_MAT_s106
                                 + 1.02309 w_k1_MSFT_s106
                                 + 0.923045 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.755372 w_k1_ADM_s106
                                 + 1.03259 w_k1_AXP_s106
                                 + 0.779623 w_k1_DIS_s106
                                 + 0.889314 w_k1_FDX_s106
                                 + 1.1101 w_k1_IBM_s106
                                 + 1.04759 w_k1_INTC_s106
                                 + 0.893352 w_k1_KO_s106
                                 + 0.946934 w_k1_MAT_s106
                                 + 0.844138 w_k1_MSFT_s106
                                 + 0.875881 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.672975 w_k1_ADM_s106
                                 + 0.974307 w_k1_AXP_s106
                                 + 0.748404 w_k1_DIS_s106
                                 + 0.923739 w_k1_FDX_s106
                                 + 0.69781 w_k1_IBM_s106
                                 + 1.0365 w_k1_INTC_s106
                                 + 0.706565 w_k1_KO_s106
                                 + 0.557388 w_k1_MAT_s106
                                 + 0.911674 w_k1_MSFT_s106
                                 + 0.91675 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.791639 w_k1_ADM_s107
                                 + 1.1578 w_k1_AXP_s107
                                 + 0.956683 w_k1_DIS_s107
                                 + 0.880552 w_k1_FDX_s107
                                 + 1.44817 w_k1_IBM_s107
                                 + 1.11261 w_k1_INTC_s107
                                 + 1.08792 w_k1_KO_s107 + 1.74875 w_k1_MAT_s107
                                 + 1.01464 w_k1_MSFT_s107
                                 + 0.986803 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.610509 w_k1_ADM_s107
                                 + 0.669384 w_k1_AXP_s107
                                 + 0.537137 w_k1_DIS_s107
                                 + 0.690595 w_k1_FDX_s107
                                 + 0.708531 w_k1_IBM_s107
                                 + 0.658721 w_k1_INTC_s107
                                 + 0.585863 w_k1_KO_s107
                                 + 0.506039 w_k1_MAT_s107
                                 + 0.626973 w_k1_MSFT_s107
                                 + 0.760501 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.30085 w_k1_ADM_s107
                                 + 1.55276 w_k1_AXP_s107
                                 + 1.53998 w_k1_DIS_s107
                                 + 1.39087 w_k1_FDX_s107
                                 + 1.10848 w_k1_IBM_s107
                                 + 1.0704 w_k1_INTC_s107
                                 + 0.992478 w_k1_KO_s107
                                 + 1.30247 w_k1_MAT_s107
                                 + 1.26418 w_k1_MSFT_s107
                                 + 1.02969 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.75848 w_k1_ADM_s107
                                 + 0.794693 w_k1_AXP_s107
                                 + 0.771006 w_k1_DIS_s107
                                 + 0.776109 w_k1_FDX_s107
                                 + 0.885685 w_k1_IBM_s107
                                 + 0.737141 w_k1_INTC_s107
                                 + 0.823156 w_k1_KO_s107 + 1.0456 w_k1_MAT_s107
                                 + 0.863319 w_k1_MSFT_s107
                                 + 0.941345 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.0123 w_k1_ADM_s107
                                 + 0.908819 w_k1_AXP_s107
                                 + 0.873812 w_k1_DIS_s107
                                 + 0.960825 w_k1_FDX_s107
                                 + 0.649949 w_k1_IBM_s107
                                 + 0.861291 w_k1_INTC_s107
                                 + 0.797845 w_k1_KO_s107
                                 + 0.585233 w_k1_MAT_s107
                                 + 0.918191 w_k1_MSFT_s107
                                 + 0.899472 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.02328 w_k1_ADM_s107
                                 + 1.06504 w_k1_AXP_s107
                                 + 1.03038 w_k1_DIS_s107
                                 + 0.982618 w_k1_FDX_s107
                                 + 1.04248 w_k1_IBM_s107
                                 + 0.884517 w_k1_INTC_s107
                                 + 0.979889 w_k1_KO_s107 + 1.0078 w_k1_MAT_s107
                                 + 1.0312 w_k1_MSFT_s107
                                 + 0.832722 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.10556 w_k1_ADM_s107
                                 + 1.40509 w_k1_AXP_s107
                                 + 1.07193 w_k1_DIS_s107
                                 + 1.14912 w_k1_FDX_s107
                                 + 0.802446 w_k1_IBM_s107
                                 + 1.04628 w_k1_INTC_s107
                                 + 0.855582 w_k1_KO_s107
                                 + 0.880911 w_k1_MAT_s107
                                 + 1.19823 w_k1_MSFT_s107
                                 + 1.04838 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.82385 w_k1_ADM_s107
                                 + 1.33771 w_k1_AXP_s107
                                 + 1.41516 w_k1_DIS_s107
                                 + 1.14648 w_k1_FDX_s107
                                 + 1.39374 w_k1_IBM_s107
                                 + 1.31448 w_k1_INTC_s107
                                 + 1.29146 w_k1_KO_s107 + 1.52665 w_k1_MAT_s107
                                 + 1.41875 w_k1_MSFT_s107
                                 + 1.01433 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.24368 w_k1_ADM_s110
                                 + 0.81082 w_k1_AXP_s110
                                 + 0.944591 w_k1_DIS_s110
                                 + 0.99352 w_k1_FDX_s110
                                 + 0.791111 w_k1_IBM_s110
                                 + 0.976752 w_k1_INTC_s110
                                 + 0.945855 w_k1_KO_s110
                                 + 0.616692 w_k1_MAT_s110
                                 + 0.920284 w_k1_MSFT_s110
                                 + 0.897603 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.50266 w_k1_ADM_s110
                                 + 1.17748 w_k1_AXP_s110
                                 + 0.974204 w_k1_DIS_s110
                                 + 1.17945 w_k1_FDX_s110
                                 + 0.736156 w_k1_IBM_s110
                                 + 0.860227 w_k1_INTC_s110
                                 + 0.956131 w_k1_KO_s110
                                 + 0.823895 w_k1_MAT_s110
                                 + 1.02516 w_k1_MSFT_s110
                                 + 1.07493 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.914387 w_k1_ADM_s110
                                 + 0.836066 w_k1_AXP_s110
                                 + 0.870366 w_k1_DIS_s110
                                 + 0.840466 w_k1_FDX_s110
                                 + 1.18374 w_k1_IBM_s110
                                 + 0.833005 w_k1_INTC_s110
                                 + 0.972625 w_k1_KO_s110
                                 + 0.77605 w_k1_MAT_s110
                                 + 0.891872 w_k1_MSFT_s110
                                 + 0.858928 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.46163 w_k1_ADM_s110
                                 + 1.17504 w_k1_AXP_s110
                                 + 1.37574 w_k1_DIS_s110
                                 + 1.33618 w_k1_FDX_s110
                                 + 0.892674 w_k1_IBM_s110
                                 + 1.14459 w_k1_INTC_s110
                                 + 1.19365 w_k1_KO_s110
                                 + 0.826314 w_k1_MAT_s110
                                 + 1.14132 w_k1_MSFT_s110
                                 + 1.06002 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.13189 w_k1_ADM_s110
                                 + 0.982616 w_k1_AXP_s110
                                 + 0.92668 w_k1_DIS_s110
                                 + 1.02605 w_k1_FDX_s110
                                 + 1.29128 w_k1_IBM_s110
                                 + 0.98593 w_k1_INTC_s110
                                 + 1.26865 w_k1_KO_s110 + 1.22145 w_k1_MAT_s110
                                 + 0.981541 w_k1_MSFT_s110
                                 + 1.0313 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.872778 w_k1_ADM_s110
                                 + 0.850024 w_k1_AXP_s110
                                 + 0.683995 w_k1_DIS_s110
                                 + 0.685029 w_k1_FDX_s110
                                 + 1.48483 w_k1_IBM_s110
                                 + 0.71856 w_k1_INTC_s110
                                 + 1.04431 w_k1_KO_s110 + 1.51761 w_k1_MAT_s110
                                 + 0.768557 w_k1_MSFT_s110
                                 + 0.940294 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.10005 w_k1_ADM_s110
                                 + 1.1725 w_k1_AXP_s110 + 1.40817 w_k1_DIS_s110
                                 + 1.23179 w_k1_FDX_s110
                                 + 1.28734 w_k1_IBM_s110
                                 + 1.04288 w_k1_INTC_s110
                                 + 1.39131 w_k1_KO_s110 + 1.35311 w_k1_MAT_s110
                                 + 1.17189 w_k1_MSFT_s110
                                 + 1.21355 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.998534 w_k1_ADM_s110
                                 + 0.7531 w_k1_AXP_s110
                                 + 0.687431 w_k1_DIS_s110
                                 + 0.84661 w_k1_FDX_s110
                                 + 0.764961 w_k1_IBM_s110
                                 + 0.831744 w_k1_INTC_s110
                                 + 0.803737 w_k1_KO_s110
                                 + 0.628107 w_k1_MAT_s110
                                 + 0.79571 w_k1_MSFT_s110
                                 + 0.887369 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04397 w_k1_ADM_s111
                                 + 1.02974 w_k1_AXP_s111
                                 + 1.04524 w_k1_DIS_s111
                                 + 1.03519 w_k1_FDX_s111
                                 + 0.839644 w_k1_IBM_s111
                                 + 0.952766 w_k1_INTC_s111
                                 + 1.00128 w_k1_KO_s111
                                 + 0.907451 w_k1_MAT_s111
                                 + 0.953138 w_k1_MSFT_s111
                                 + 0.917422 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.801633 w_k1_ADM_s111
                                 + 0.797716 w_k1_AXP_s111
                                 + 0.670122 w_k1_DIS_s111
                                 + 0.78094 w_k1_FDX_s111
                                 + 0.638025 w_k1_IBM_s111
                                 + 0.829363 w_k1_INTC_s111
                                 + 0.685605 w_k1_KO_s111
                                 + 0.75186 w_k1_MAT_s111
                                 + 0.805206 w_k1_MSFT_s111
                                 + 0.77505 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.27693 w_k1_ADM_s111
                                 + 1.29826 w_k1_AXP_s111
                                 + 0.787078 w_k1_DIS_s111
                                 + 1.00079 w_k1_FDX_s111
                                 + 0.78696 w_k1_IBM_s111
                                 + 0.77003 w_k1_INTC_s111
                                 + 0.76044 w_k1_KO_s111 + 1.11101 w_k1_MAT_s111
                                 + 0.807115 w_k1_MSFT_s111
                                 + 0.882052 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.08294 w_k1_ADM_s111
                                 + 1.46616 w_k1_AXP_s111 + 1.1691 w_k1_DIS_s111
                                 + 1.06392 w_k1_FDX_s111 + 1.3946 w_k1_IBM_s111
                                 + 0.962437 w_k1_INTC_s111
                                 + 1.08643 w_k1_KO_s111 + 1.88642 w_k1_MAT_s111
                                 + 1.00614 w_k1_MSFT_s111
                                 + 1.12422 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.28647 w_k1_ADM_s111
                                 + 1.4476 w_k1_AXP_s111 + 1.28606 w_k1_DIS_s111
                                 + 1.2737 w_k1_FDX_s111
                                 + 0.926911 w_k1_IBM_s111
                                 + 0.959173 w_k1_INTC_s111
                                 + 0.996643 w_k1_KO_s111
                                 + 1.22708 w_k1_MAT_s111
                                 + 1.08711 w_k1_MSFT_s111
                                 + 1.18013 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.00018 w_k1_ADM_s111
                                 + 0.905969 w_k1_AXP_s111 + 1.272 w_k1_DIS_s111
                                 + 1.02024 w_k1_FDX_s111
                                 + 1.41359 w_k1_IBM_s111
                                 + 0.982414 w_k1_INTC_s111
                                 + 1.15113 w_k1_KO_s111 + 1.61579 w_k1_MAT_s111
                                 + 1.14437 w_k1_MSFT_s111
                                 + 1.04205 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.17076 w_k1_ADM_s111
                                 + 1.22299 w_k1_AXP_s111
                                 + 1.79862 w_k1_DIS_s111
                                 + 1.27594 w_k1_FDX_s111
                                 + 1.05782 w_k1_IBM_s111
                                 + 1.10703 w_k1_INTC_s111
                                 + 1.22515 w_k1_KO_s111 + 1.51119 w_k1_MAT_s111
                                 + 1.35066 w_k1_MSFT_s111 + 1.1726 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.843982 w_k1_ADM_s111
                                 + 0.774569 w_k1_AXP_s111
                                 + 0.840812 w_k1_DIS_s111
                                 + 0.804287 w_k1_FDX_s111
                                 + 1.02605 w_k1_IBM_s111
                                 + 0.84678 w_k1_INTC_s111
                                 + 0.86653 w_k1_KO_s111
                                 + 0.801411 w_k1_MAT_s111
                                 + 0.854332 w_k1_MSFT_s111
                                 + 0.796814 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.568303 w_k1_ADM_s112
                                 + 0.594907 w_k1_AXP_s112
                                 + 0.595559 w_k1_DIS_s112
                                 + 0.832686 w_k1_FDX_s112
                                 + 0.608309 w_k1_IBM_s112
                                 + 0.889217 w_k1_INTC_s112
                                 + 0.702487 w_k1_KO_s112
                                 + 0.619556 w_k1_MAT_s112
                                 + 0.880424 w_k1_MSFT_s112
                                 + 0.895206 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.976728 w_k1_ADM_s112
                                 + 1.31234 w_k1_AXP_s112
                                 + 1.09699 w_k1_DIS_s112
                                 + 0.984749 w_k1_FDX_s112
                                 + 1.58092 w_k1_IBM_s112
                                 + 0.882225 w_k1_INTC_s112
                                 + 1.26258 w_k1_KO_s112 + 1.82307 w_k1_MAT_s112
                                 + 1.08414 w_k1_MSFT_s112
                                 + 1.12044 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.76711 w_k1_ADM_s112
                                 + 1.82919 w_k1_AXP_s112
                                 + 1.84277 w_k1_DIS_s112 + 1.6823 w_k1_FDX_s112
                                 + 1.18247 w_k1_IBM_s112
                                 + 1.59735 w_k1_INTC_s112
                                 + 1.25636 w_k1_KO_s112 + 1.24918 w_k1_MAT_s112
                                 + 1.4909 w_k1_MSFT_s112 + 1.34131 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.06836 w_k1_ADM_s112
                                 + 1.46512 w_k1_AXP_s112
                                 + 1.28085 w_k1_DIS_s112
                                 + 1.37054 w_k1_FDX_s112
                                 + 0.830282 w_k1_IBM_s112
                                 + 1.10804 w_k1_INTC_s112
                                 + 0.95537 w_k1_KO_s112 + 1.02841 w_k1_MAT_s112
                                 + 1.168 w_k1_MSFT_s112 + 1.09813 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.736881 w_k1_ADM_s112
                                 + 0.706247 w_k1_AXP_s112
                                 + 0.609452 w_k1_DIS_s112
                                 + 0.792882 w_k1_FDX_s112
                                 + 0.950527 w_k1_IBM_s112
                                 + 0.82193 w_k1_INTC_s112
                                 + 0.774896 w_k1_KO_s112
                                 + 0.844241 w_k1_MAT_s112
                                 + 0.774959 w_k1_MSFT_s112
                                 + 0.827201 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.91642 w_k1_ADM_s112
                                 + 1.01773 w_k1_AXP_s112
                                 + 0.734533 w_k1_DIS_s112
                                 + 0.987425 w_k1_FDX_s112
                                 + 0.743998 w_k1_IBM_s112
                                 + 1.06632 w_k1_INTC_s112
                                 + 0.831214 w_k1_KO_s112
                                 + 0.738319 w_k1_MAT_s112
                                 + 1.00595 w_k1_MSFT_s112
                                 + 0.903982 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.02249 w_k1_ADM_s112
                                 + 1.05263 w_k1_AXP_s112
                                 + 1.21329 w_k1_DIS_s112
                                 + 1.14476 w_k1_FDX_s112
                                 + 1.10219 w_k1_IBM_s112
                                 + 0.969084 w_k1_INTC_s112
                                 + 1.22029 w_k1_KO_s112 + 1.03947 w_k1_MAT_s112
                                 + 1.23312 w_k1_MSFT_s112
                                 + 1.20262 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.752214 w_k1_ADM_s112
                                 + 0.810996 w_k1_AXP_s112
                                 + 0.797639 w_k1_DIS_s112
                                 + 0.83097 w_k1_FDX_s112
                                 + 1.23545 w_k1_IBM_s112
                                 + 0.827225 w_k1_INTC_s112
                                 + 0.906468 w_k1_KO_s112
                                 + 0.834293 w_k1_MAT_s112
                                 + 0.847761 w_k1_MSFT_s112
                                 + 0.951394 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.898573 w_k1_ADM_s113
                                 + 0.97586 w_k1_AXP_s113
                                 + 0.930534 w_k1_DIS_s113
                                 + 1.00307 w_k1_FDX_s113
                                 + 0.691077 w_k1_IBM_s113
                                 + 1.26753 w_k1_INTC_s113
                                 + 0.71273 w_k1_KO_s113
                                 + 0.420212 w_k1_MAT_s113
                                 + 0.920118 w_k1_MSFT_s113
                                 + 0.94983 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.634953 w_k1_ADM_s113
                                 + 0.852375 w_k1_AXP_s113
                                 + 0.913334 w_k1_DIS_s113
                                 + 0.972835 w_k1_FDX_s113
                                 + 0.702898 w_k1_IBM_s113
                                 + 1.06321 w_k1_INTC_s113
                                 + 0.704654 w_k1_KO_s113
                                 + 0.646938 w_k1_MAT_s113
                                 + 0.985447 w_k1_MSFT_s113
                                 + 0.833582 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.03193 w_k1_ADM_s113
                                 + 1.49894 w_k1_AXP_s113
                                 + 1.71251 w_k1_DIS_s113
                                 + 1.43633 w_k1_FDX_s113
                                 + 1.08685 w_k1_IBM_s113
                                 + 1.64933 w_k1_INTC_s113
                                 + 1.11486 w_k1_KO_s113
                                 + 0.867664 w_k1_MAT_s113
                                 + 1.30265 w_k1_MSFT_s113
                                 + 1.23416 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.679599 w_k1_ADM_s113
                                 + 0.718892 w_k1_AXP_s113
                                 + 0.656556 w_k1_DIS_s113
                                 + 0.784581 w_k1_FDX_s113
                                 + 0.8869 w_k1_IBM_s113
                                 + 0.823585 w_k1_INTC_s113
                                 + 0.760112 w_k1_KO_s113
                                 + 0.696145 w_k1_MAT_s113
                                 + 0.670823 w_k1_MSFT_s113
                                 + 0.851638 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.784612 w_k1_ADM_s113
                                 + 1.39635 w_k1_AXP_s113
                                 + 0.939981 w_k1_DIS_s113
                                 + 1.02789 w_k1_FDX_s113
                                 + 0.921351 w_k1_IBM_s113
                                 + 1.06577 w_k1_INTC_s113
                                 + 0.912225 w_k1_KO_s113
                                 + 1.22805 w_k1_MAT_s113
                                 + 0.91606 w_k1_MSFT_s113
                                 + 1.03472 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.973288 w_k1_ADM_s113
                                 + 0.818996 w_k1_AXP_s113
                                 + 0.889068 w_k1_DIS_s113
                                 + 1.02677 w_k1_FDX_s113
                                 + 0.828317 w_k1_IBM_s113
                                 + 1.14495 w_k1_INTC_s113
                                 + 0.822937 w_k1_KO_s113
                                 + 0.929544 w_k1_MAT_s113
                                 + 1.03446 w_k1_MSFT_s113
                                 + 0.919528 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.07381 w_k1_ADM_s113
                                 + 1.44223 w_k1_AXP_s113
                                 + 1.23702 w_k1_DIS_s113
                                 + 1.20271 w_k1_FDX_s113
                                 + 1.32388 w_k1_IBM_s113
                                 + 1.50565 w_k1_INTC_s113
                                 + 1.02334 w_k1_KO_s113 + 1.039 w_k1_MAT_s113
                                 + 1.27753 w_k1_MSFT_s113
                                 + 1.09964 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.886925 w_k1_ADM_s113
                                 + 0.910586 w_k1_AXP_s113
                                 + 1.1221 w_k1_DIS_s113
                                 + 0.951828 w_k1_FDX_s113
                                 + 1.19212 w_k1_IBM_s113
                                 + 1.04336 w_k1_INTC_s113
                                 + 0.948196 w_k1_KO_s113
                                 + 1.00006 w_k1_MAT_s113
                                 + 0.910599 w_k1_MSFT_s113
                                 + 1.0678 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.866041 w_k1_ADM_s114
                                 + 0.71468 w_k1_AXP_s114
                                 + 0.485886 w_k1_DIS_s114
                                 + 0.670511 w_k1_FDX_s114
                                 + 0.82056 w_k1_IBM_s114
                                 + 0.755524 w_k1_INTC_s114
                                 + 0.738459 w_k1_KO_s114
                                 + 0.59759 w_k1_MAT_s114
                                 + 0.655106 w_k1_MSFT_s114
                                 + 0.790925 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.32194 w_k1_ADM_s114
                                 + 1.35082 w_k1_AXP_s114
                                 + 1.45915 w_k1_DIS_s114
                                 + 1.22023 w_k1_FDX_s114
                                 + 1.34754 w_k1_IBM_s114
                                 + 1.25769 w_k1_INTC_s114
                                 + 1.23679 w_k1_KO_s114 + 1.3367 w_k1_MAT_s114
                                 + 1.26686 w_k1_MSFT_s114
                                 + 1.31845 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.2569 w_k1_ADM_s114
                                 + 0.967852 w_k1_AXP_s114
                                 + 1.06423 w_k1_DIS_s114
                                 + 1.04395 w_k1_FDX_s114
                                 + 0.865559 w_k1_IBM_s114
                                 + 0.957672 w_k1_INTC_s114
                                 + 1.06586 w_k1_KO_s114
                                 + 0.919703 w_k1_MAT_s114
                                 + 1.11901 w_k1_MSFT_s114
                                 + 1.02234 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.00836 w_k1_ADM_s114
                                 + 0.950655 w_k1_AXP_s114
                                 + 1.02891 w_k1_DIS_s114
                                 + 0.890985 w_k1_FDX_s114
                                 + 1.26465 w_k1_IBM_s114
                                 + 1.04629 w_k1_INTC_s114
                                 + 1.14573 w_k1_KO_s114
                                 + 0.908876 w_k1_MAT_s114
                                 + 1.07077 w_k1_MSFT_s114 + 1.035 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.2911 w_k1_ADM_s114
                                 + 0.942493 w_k1_AXP_s114
                                 + 1.04224 w_k1_DIS_s114
                                 + 1.12932 w_k1_FDX_s114
                                 + 0.739295 w_k1_IBM_s114
                                 + 1.08004 w_k1_INTC_s114
                                 + 0.977517 w_k1_KO_s114
                                 + 0.766334 w_k1_MAT_s114
                                 + 1.12902 w_k1_MSFT_s114
                                 + 1.12363 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.917062 w_k1_ADM_s114
                                 + 0.777005 w_k1_AXP_s114
                                 + 0.791803 w_k1_DIS_s114
                                 + 0.820694 w_k1_FDX_s114
                                 + 0.987135 w_k1_IBM_s114
                                 + 0.856992 w_k1_INTC_s114
                                 + 0.810569 w_k1_KO_s114
                                 + 0.862003 w_k1_MAT_s114
                                 + 0.842063 w_k1_MSFT_s114
                                 + 1.0508 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.54365 w_k1_ADM_s114
                                 + 1.26318 w_k1_AXP_s114
                                 + 1.55867 w_k1_DIS_s114 + 1.2586 w_k1_FDX_s114
                                 + 0.949703 w_k1_IBM_s114
                                 + 1.08031 w_k1_INTC_s114
                                 + 1.20767 w_k1_KO_s114
                                 + 0.924864 w_k1_MAT_s114
                                 + 1.23684 w_k1_MSFT_s114
                                 + 1.16145 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.10505 w_k1_ADM_s114
                                 + 0.782022 w_k1_AXP_s114
                                 + 0.859335 w_k1_DIS_s114
                                 + 0.925557 w_k1_FDX_s114
                                 + 0.705947 w_k1_IBM_s114
                                 + 0.795967 w_k1_INTC_s114
                                 + 0.862867 w_k1_KO_s114
                                 + 0.852252 w_k1_MAT_s114
                                 + 0.886551 w_k1_MSFT_s114
                                 + 0.934286 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.805939 w_k1_ADM_s115
                                 + 0.729512 w_k1_AXP_s115
                                 + 0.782409 w_k1_DIS_s115
                                 + 0.831621 w_k1_FDX_s115
                                 + 1.39088 w_k1_IBM_s115
                                 + 0.771205 w_k1_INTC_s115
                                 + 1.05128 w_k1_KO_s115 + 1.71145 w_k1_MAT_s115
                                 + 0.844603 w_k1_MSFT_s115
                                 + 1.03939 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.526006 w_k1_ADM_s115
                                 + 0.451156 w_k1_AXP_s115
                                 + 0.383259 w_k1_DIS_s115
                                 + 0.633245 w_k1_FDX_s115
                                 + 0.777559 w_k1_IBM_s115
                                 + 0.658835 w_k1_INTC_s115
                                 + 0.632936 w_k1_KO_s115 + 0.8117 w_k1_MAT_s115
                                 + 0.724056 w_k1_MSFT_s115
                                 + 0.76126 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.08496 w_k1_ADM_s115
                                 + 1.30693 w_k1_AXP_s115
                                 + 1.14217 w_k1_DIS_s115
                                 + 1.25122 w_k1_FDX_s115
                                 + 0.964948 w_k1_IBM_s115
                                 + 1.0027 w_k1_INTC_s115
                                 + 0.938033 w_k1_KO_s115
                                 + 0.983786 w_k1_MAT_s115
                                 + 1.10657 w_k1_MSFT_s115
                                 + 1.08014 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.15017 w_k1_ADM_s115
                                 + 1.0344 w_k1_AXP_s115
                                 + 0.951147 w_k1_DIS_s115
                                 + 0.996361 w_k1_FDX_s115
                                 + 1.09353 w_k1_IBM_s115
                                 + 0.912214 w_k1_INTC_s115
                                 + 0.932113 w_k1_KO_s115
                                 + 1.46309 w_k1_MAT_s115
                                 + 1.03304 w_k1_MSFT_s115
                                 + 1.02907 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.92341 w_k1_ADM_s115
                                 + 0.723812 w_k1_AXP_s115
                                 + 0.756736 w_k1_DIS_s115
                                 + 0.838635 w_k1_FDX_s115
                                 + 0.90552 w_k1_IBM_s115
                                 + 0.691129 w_k1_INTC_s115
                                 + 0.883821 w_k1_KO_s115
                                 + 1.03543 w_k1_MAT_s115
                                 + 0.928756 w_k1_MSFT_s115
                                 + 1.01703 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.987326 w_k1_ADM_s115
                                 + 1.0208 w_k1_AXP_s115 + 1.30207 w_k1_DIS_s115
                                 + 1.10684 w_k1_FDX_s115
                                 + 1.27945 w_k1_IBM_s115
                                 + 0.934934 w_k1_INTC_s115
                                 + 1.28882 w_k1_KO_s115 + 1.56619 w_k1_MAT_s115
                                 + 1.09299 w_k1_MSFT_s115
                                 + 1.09455 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.23079 w_k1_ADM_s115
                                 + 1.2081 w_k1_AXP_s115 + 1.58854 w_k1_DIS_s115
                                 + 1.41188 w_k1_FDX_s115
                                 + 1.23327 w_k1_IBM_s115
                                 + 1.09621 w_k1_INTC_s115
                                 + 1.30349 w_k1_KO_s115 + 2.28607 w_k1_MAT_s115
                                 + 1.42144 w_k1_MSFT_s115
                                 + 1.32452 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.915401 w_k1_ADM_s115
                                 + 0.675159 w_k1_AXP_s115
                                 + 0.917653 w_k1_DIS_s115
                                 + 1.04946 w_k1_FDX_s115
                                 + 0.709861 w_k1_IBM_s115
                                 + 0.849005 w_k1_INTC_s115
                                 + 0.855458 w_k1_KO_s115
                                 + 0.892904 w_k1_MAT_s115
                                 + 0.999986 w_k1_MSFT_s115
                                 + 1.00714 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.49203 w_k1_ADM_s116
                                 + 1.6477 w_k1_AXP_s116 + 1.63696 w_k1_DIS_s116
                                 + 1.32818 w_k1_FDX_s116 + 1.4476 w_k1_IBM_s116
                                 + 1.4228 w_k1_INTC_s116 + 1.51263 w_k1_KO_s116
                                 + 1.49298 w_k1_MAT_s116
                                 + 1.26138 w_k1_MSFT_s116
                                 + 1.25186 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.891099 w_k1_ADM_s116
                                 + 0.837798 w_k1_AXP_s116
                                 + 0.860963 w_k1_DIS_s116
                                 + 0.89876 w_k1_FDX_s116
                                 + 0.833232 w_k1_IBM_s116
                                 + 0.89916 w_k1_INTC_s116
                                 + 0.781435 w_k1_KO_s116
                                 + 0.82677 w_k1_MAT_s116
                                 + 0.894307 w_k1_MSFT_s116
                                 + 0.918082 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.15249 w_k1_ADM_s116
                                 + 0.977624 w_k1_AXP_s116
                                 + 0.949996 w_k1_DIS_s116
                                 + 0.850929 w_k1_FDX_s116
                                 + 1.29236 w_k1_IBM_s116
                                 + 0.969553 w_k1_INTC_s116
                                 + 1.14272 w_k1_KO_s116
                                 + 0.988365 w_k1_MAT_s116
                                 + 0.891594 w_k1_MSFT_s116
                                 + 1.00517 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.768096 w_k1_ADM_s116
                                 + 0.723183 w_k1_AXP_s116
                                 + 0.66055 w_k1_DIS_s116
                                 + 0.685312 w_k1_FDX_s116
                                 + 0.868562 w_k1_IBM_s116
                                 + 0.887252 w_k1_INTC_s116
                                 + 0.70635 w_k1_KO_s116
                                 + 0.598084 w_k1_MAT_s116
                                 + 0.645829 w_k1_MSFT_s116
                                 + 0.730137 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.13819 w_k1_ADM_s116
                                 + 1.83036 w_k1_AXP_s116
                                 + 1.09409 w_k1_DIS_s116 + 1.1509 w_k1_FDX_s116
                                 + 1.25457 w_k1_IBM_s116 + 1.181 w_k1_INTC_s116
                                 + 1.08885 w_k1_KO_s116 + 1.57576 w_k1_MAT_s116
                                 + 1.04172 w_k1_MSFT_s116
                                 + 1.06921 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.12896 w_k1_ADM_s116
                                 + 1.25183 w_k1_AXP_s116 + 1.1932 w_k1_DIS_s116
                                 + 1.20278 w_k1_FDX_s116
                                 + 1.07176 w_k1_IBM_s116
                                 + 1.17211 w_k1_INTC_s116
                                 + 1.06406 w_k1_KO_s116
                                 + 0.926324 w_k1_MAT_s116
                                 + 1.06692 w_k1_MSFT_s116
                                 + 1.11615 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.00472 w_k1_ADM_s116
                                 + 1.0556 w_k1_AXP_s116 + 1.02678 w_k1_DIS_s116
                                 + 1.08311 w_k1_FDX_s116
                                 + 0.942913 w_k1_IBM_s116
                                 + 1.0131 w_k1_INTC_s116
                                 + 0.929189 w_k1_KO_s116
                                 + 0.847752 w_k1_MAT_s116
                                 + 0.972085 w_k1_MSFT_s116
                                 + 1.01996 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.20285 w_k1_ADM_s116
                                 + 1.05238 w_k1_AXP_s116
                                 + 1.37661 w_k1_DIS_s116
                                 + 1.03019 w_k1_FDX_s116 + 1.3134 w_k1_IBM_s116
                                 + 1.23529 w_k1_INTC_s116
                                 + 1.29609 w_k1_KO_s116 + 1.00678 w_k1_MAT_s116
                                 + 1.2357 w_k1_MSFT_s116 + 1.11591 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.67623 w_k1_ADM_s117
                                 + 1.47861 w_k1_AXP_s117
                                 + 1.83579 w_k1_DIS_s117
                                 + 1.57347 w_k1_FDX_s117
                                 + 0.969683 w_k1_IBM_s117
                                 + 1.22269 w_k1_INTC_s117
                                 + 1.45501 w_k1_KO_s117 + 1.27119 w_k1_MAT_s117
                                 + 1.34882 w_k1_MSFT_s117 + 1.3216 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.995919 w_k1_ADM_s117
                                 + 1.08585 w_k1_AXP_s117
                                 + 0.990636 w_k1_DIS_s117
                                 + 1.00918 w_k1_FDX_s117
                                 + 1.24772 w_k1_IBM_s117
                                 + 0.873208 w_k1_INTC_s117
                                 + 1.02643 w_k1_KO_s117 + 1.45898 w_k1_MAT_s117
                                 + 0.88797 w_k1_MSFT_s117
                                 + 1.01471 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.843243 w_k1_ADM_s117
                                 + 0.845843 w_k1_AXP_s117
                                 + 0.853491 w_k1_DIS_s117
                                 + 0.975138 w_k1_FDX_s117
                                 + 0.89811 w_k1_IBM_s117
                                 + 1.03758 w_k1_INTC_s117
                                 + 0.940152 w_k1_KO_s117
                                 + 0.630999 w_k1_MAT_s117
                                 + 0.870091 w_k1_MSFT_s117
                                 + 0.922122 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.11391 w_k1_ADM_s117
                                 + 1.13868 w_k1_AXP_s117
                                 + 1.25053 w_k1_DIS_s117 + 1.2207 w_k1_FDX_s117
                                 + 1.21887 w_k1_IBM_s117
                                 + 1.02926 w_k1_INTC_s117
                                 + 1.26051 w_k1_KO_s117 + 1.4328 w_k1_MAT_s117
                                 + 1.09458 w_k1_MSFT_s117
                                 + 1.10093 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.8041 w_k1_ADM_s117
                                 + 0.677755 w_k1_AXP_s117
                                 + 0.623399 w_k1_DIS_s117
                                 + 0.765505 w_k1_FDX_s117
                                 + 0.75486 w_k1_IBM_s117
                                 + 0.825045 w_k1_INTC_s117
                                 + 0.850541 w_k1_KO_s117
                                 + 0.723268 w_k1_MAT_s117
                                 + 0.76535 w_k1_MSFT_s117
                                 + 0.870041 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.61274 w_k1_ADM_s117
                                 + 1.45175 w_k1_AXP_s117
                                 + 1.82475 w_k1_DIS_s117
                                 + 1.37943 w_k1_FDX_s117
                                 + 1.34696 w_k1_IBM_s117
                                 + 1.26534 w_k1_INTC_s117
                                 + 1.75827 w_k1_KO_s117 + 1.65175 w_k1_MAT_s117
                                 + 1.3795 w_k1_MSFT_s117 + 1.39089 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.04334 w_k1_ADM_s117
                                 + 0.975988 w_k1_AXP_s117
                                 + 1.10333 w_k1_DIS_s117
                                 + 1.20716 w_k1_FDX_s117
                                 + 0.763113 w_k1_IBM_s117
                                 + 0.983565 w_k1_INTC_s117
                                 + 0.944225 w_k1_KO_s117
                                 + 0.777421 w_k1_MAT_s117
                                 + 1.08835 w_k1_MSFT_s117
                                 + 0.926792 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.739633 w_k1_ADM_s117
                                 + 0.861741 w_k1_AXP_s117
                                 + 0.927732 w_k1_DIS_s117
                                 + 0.803883 w_k1_FDX_s117
                                 + 1.09148 w_k1_IBM_s117
                                 + 0.821734 w_k1_INTC_s117
                                 + 1.1078 w_k1_KO_s117 + 1.00646 w_k1_MAT_s117
                                 + 0.897617 w_k1_MSFT_s117
                                 + 0.887362 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00994582410161202 w_k1_ADM_s000
                                 + 0.0155772274848971 w_k1_ADM_s001
                                 + 0.00418333295572362 w_k1_ADM_s002
                                 + 0.0145847426011374 w_k1_ADM_s003
                                 + 0.00560356612486713 w_k1_ADM_s004
                                 + 0.0125907436918024 w_k1_ADM_s005
                                 + 0.00579747091667864 w_k1_ADM_s006
                                 + 0.0128915655352882 w_k1_ADM_s007
                                 + 0.0392871425732482 w_k1_ADM_s010
                                 + 0.0200203777820722 w_k1_ADM_s011
                                 + 0.0403514621212884 w_k1_ADM_s012
                                 + 0.026945192542008 w_k1_ADM_s013
                                 + 0.00913453211270659 w_k1_ADM_s014
                                 + 0.0199253595635143 w_k1_ADM_s015
                                 + 0.0322747864060504 w_k1_ADM_s016
                                 + 0.0193241330843059 w_k1_ADM_s017
                                 + 0.0182510743374972 w_k1_ADM_s100
                                 + 0.020896752550066 w_k1_ADM_s101
                                 + 0.012266033879705 w_k1_ADM_s102
                                 + 0.0410874222213145 w_k1_ADM_s103
                                 + 0.0461714066499664 w_k1_ADM_s104
                                 + 0.0149591582616678 w_k1_ADM_s105
                                 + 0.0244544915318897 w_k1_ADM_s106
                                 + 0.0477296036614797 w_k1_ADM_s107
                                 + 0.140605792633962 w_k1_ADM_s110
                                 + 0.0813571137149943 w_k1_ADM_s111
                                 + 0.0587800150316238 w_k1_ADM_s112
                                 + 0.029515563559948 w_k1_ADM_s113
                                 + 0.0816993386196644 w_k1_ADM_s114
                                 + 0.0258384650126565 w_k1_ADM_s115
                                 + 0.0410594511282776 w_k1_ADM_s116
                                 + 0.0447882789512798 w_k1_ADM_s117
                                 + 0.0107719148253235 w_k1_AXP_s000
                                 + 0.0154379274822143 w_k1_AXP_s001
                                 + 0.00364460875010082 w_k1_AXP_s002
                                 + 0.0152100479019271 w_k1_AXP_s003
                                 + 0.0051905841130451 w_k1_AXP_s004
                                 + 0.0154676613327174 w_k1_AXP_s005
                                 + 0.00503449473868117 w_k1_AXP_s006
                                 + 0.0133909420552228 w_k1_AXP_s007
                                 + 0.0494313082377716 w_k1_AXP_s010
                                 + 0.0218665951092541 w_k1_AXP_s011
                                 + 0.0390212678265032 w_k1_AXP_s012
                                 + 0.0300976579472593 w_k1_AXP_s013
                                 + 0.0088366619132706 w_k1_AXP_s014
                                 + 0.0169750351816551 w_k1_AXP_s015
                                 + 0.0290452372677866 w_k1_AXP_s016
                                 + 0.0166528891844444 w_k1_AXP_s017
                                 + 0.0171158913359598 w_k1_AXP_s100
                                 + 0.0243097812854616 w_k1_AXP_s101
                                 + 0.0112023283535176 w_k1_AXP_s102
                                 + 0.0421515690967404 w_k1_AXP_s103
                                 + 0.046258486620504 w_k1_AXP_s104
                                 + 0.0164563646462908 w_k1_AXP_s105
                                 + 0.0313207264874797 w_k1_AXP_s106
                                 + 0.0561629315167404 w_k1_AXP_s107
                                 + 0.116558268522325 w_k1_AXP_s110
                                 + 0.0858919694186698 w_k1_AXP_s111
                                 + 0.066078460016155 w_k1_AXP_s112
                                 + 0.0361949160580727 w_k1_AXP_s113
                                 + 0.0691007927342666 w_k1_AXP_s114
                                 + 0.0237857634732818 w_k1_AXP_s115
                                 + 0.0415419514781834 w_k1_AXP_s116
                                 + 0.0436789659963229 w_k1_AXP_s117
                                 + 0.0114892961902668 w_k1_DIS_s000
                                 + 0.0149299161829657 w_k1_DIS_s001
                                 + 0.00404207302675449 w_k1_DIS_s002
                                 + 0.0171288117293029 w_k1_DIS_s003
                                 + 0.00564658933548013 w_k1_DIS_s004
                                 + 0.0155138832721809 w_k1_DIS_s005
                                 + 0.0055225086498377 w_k1_DIS_s006
                                 + 0.0142896605061203 w_k1_DIS_s007
                                 + 0.0476801486265198 w_k1_DIS_s010
                                 + 0.0199521960560849 w_k1_DIS_s011
                                 + 0.041654100425853 w_k1_DIS_s012
                                 + 0.0277836706151505 w_k1_DIS_s013
                                 + 0.0093446415123153 w_k1_DIS_s014
                                 + 0.0191559229791499 w_k1_DIS_s015
                                 + 0.0286324370606411 w_k1_DIS_s016
                                 + 0.0173958504282106 w_k1_DIS_s017
                                 + 0.0172841056837173 w_k1_DIS_s100
                                 + 0.0251297946480389 w_k1_DIS_s101
                                 + 0.0119703663847809 w_k1_DIS_s102
                                 + 0.0457596854649837 w_k1_DIS_s103
                                 + 0.0488824792944943 w_k1_DIS_s104
                                 + 0.0164023112791948 w_k1_DIS_s105
                                 + 0.0289074770646032 w_k1_DIS_s106
                                 + 0.0513196653017884 w_k1_DIS_s107
                                 + 0.123369916245361 w_k1_DIS_s110
                                 + 0.0838701741867185 w_k1_DIS_s111
                                 + 0.0597806990683639 w_k1_DIS_s112
                                 + 0.0356190987866309 w_k1_DIS_s113
                                 + 0.0747679923903133 w_k1_DIS_s114
                                 + 0.0267745309570191 w_k1_DIS_s115
                                 + 0.0406547815268046 w_k1_DIS_s116
                                 + 0.0467941499660689 w_k1_DIS_s117
                                 + 0.0103841831869222 w_k1_FDX_s000
                                 + 0.0152152621565863 w_k1_FDX_s001
                                 + 0.0039919742703343 w_k1_FDX_s002
                                 + 0.0163025111839147 w_k1_FDX_s003
                                 + 0.00535006973412154 w_k1_FDX_s004
                                 + 0.0147845243178497 w_k1_FDX_s005
                                 + 0.00567011358600019 w_k1_FDX_s006
                                 + 0.0134032855497915 w_k1_FDX_s007
                                 + 0.0467755479941321 w_k1_FDX_s010
                                 + 0.0206898891008721 w_k1_FDX_s011
                                 + 0.0415973805487636 w_k1_FDX_s012
                                 + 0.0276190784367551 w_k1_FDX_s013
                                 + 0.00936419243400737 w_k1_FDX_s014
                                 + 0.0182977662795847 w_k1_FDX_s015
                                 + 0.0295947759157473 w_k1_FDX_s016
                                 + 0.0171562064880863 w_k1_FDX_s017
                                 + 0.0164094533560806 w_k1_FDX_s100
                                 + 0.0230680419866461 w_k1_FDX_s101
                                 + 0.0117261591822894 w_k1_FDX_s102
                                 + 0.0419015145901697 w_k1_FDX_s103
                                 + 0.0475618985883278 w_k1_FDX_s104
                                 + 0.0150338593285226 w_k1_FDX_s105
                                 + 0.0287046898834169 w_k1_FDX_s106
                                 + 0.050059669810539 w_k1_FDX_s107
                                 + 0.126195048213901 w_k1_FDX_s110
                                 + 0.0795073441466273 w_k1_FDX_s111
                                 + 0.0641841708417113 w_k1_FDX_s112
                                 + 0.0350268892821013 w_k1_FDX_s113
                                 + 0.0698733352208195 w_k1_FDX_s114
                                 + 0.0268879329435048 w_k1_FDX_s115
                                 + 0.0384094481018138 w_k1_FDX_s116
                                 + 0.046628403397956 w_k1_FDX_s117
                                 + 0.0104132647881041 w_k1_IBM_s000
                                 + 0.0148746219257182 w_k1_IBM_s001
                                 + 0.00402448558887046 w_k1_IBM_s002
                                 + 0.0168764811482154 w_k1_IBM_s003
                                 + 0.0052536125668639 w_k1_IBM_s004
                                 + 0.0147641498004365 w_k1_IBM_s005
                                 + 0.00558782805150252 w_k1_IBM_s006
                                 + 0.0149339641257785 w_k1_IBM_s007
                                 + 0.0485378340780097 w_k1_IBM_s010
                                 + 0.0207295713622057 w_k1_IBM_s011
                                 + 0.0433486548638306 w_k1_IBM_s012
                                 + 0.0277452528002116 w_k1_IBM_s013
                                 + 0.0101180781377098 w_k1_IBM_s014
                                 + 0.0178471733414603 w_k1_IBM_s015
                                 + 0.0325807244637688 w_k1_IBM_s016
                                 + 0.0177048761667449 w_k1_IBM_s017
                                 + 0.0163901509924145 w_k1_IBM_s100
                                 + 0.0246614986289375 w_k1_IBM_s101
                                 + 0.0116657224355461 w_k1_IBM_s102
                                 + 0.0433650015545859 w_k1_IBM_s103
                                 + 0.0454420897143558 w_k1_IBM_s104
                                 + 0.0155938140623858 w_k1_IBM_s105
                                 + 0.030357109814804 w_k1_IBM_s106
                                 + 0.0506151099544365 w_k1_IBM_s107
                                 + 0.124414109338551 w_k1_IBM_s110
                                 + 0.0772327898314892 w_k1_IBM_s111
                                 + 0.063365395045193 w_k1_IBM_s112
                                 + 0.0333652076985323 w_k1_IBM_s113
                                 + 0.0684365276505002 w_k1_IBM_s114
                                 + 0.0280396714050295 w_k1_IBM_s115
                                 + 0.0417571580929582 w_k1_IBM_s116
                                 + 0.0441926638296002 w_k1_IBM_s117
                                 + 0.00982099572942046 w_k1_INTC_s000
                                 + 0.016021709459897 w_k1_INTC_s001
                                 + 0.00451375348591705 w_k1_INTC_s002
                                 + 0.0159372485834754 w_k1_INTC_s003
                                 + 0.0052812773668161 w_k1_INTC_s004
                                 + 0.0186124647635842 w_k1_INTC_s005
                                 + 0.00552795121070575 w_k1_INTC_s006
                                 + 0.0127320694236233 w_k1_INTC_s007
                                 + 0.054693768825541 w_k1_INTC_s010
                                 + 0.022681948361257 w_k1_INTC_s011
                                 + 0.0386465786327562 w_k1_INTC_s012
                                 + 0.0220905559071462 w_k1_INTC_s013
                                 + 0.0105858119139878 w_k1_INTC_s014
                                 + 0.0180642721911322 w_k1_INTC_s015
                                 + 0.0298038826746164 w_k1_INTC_s016
                                 + 0.0162035997132184 w_k1_INTC_s017
                                 + 0.0170378725534234 w_k1_INTC_s100
                                 + 0.0240688528645543 w_k1_INTC_s101
                                 + 0.0114418260621988 w_k1_INTC_s102
                                 + 0.0433241407712282 w_k1_INTC_s103
                                 + 0.0437319129835913 w_k1_INTC_s104
                                 + 0.0139313775208202 w_k1_INTC_s105
                                 + 0.0341399648939147 w_k1_INTC_s106
                                 + 0.048369381136323 w_k1_INTC_s107
                                 + 0.116220573680043 w_k1_INTC_s110
                                 + 0.0727301767447081 w_k1_INTC_s111
                                 + 0.0615691244801346 w_k1_INTC_s112
                                 + 0.0401331627235816 w_k1_INTC_s113
                                 + 0.0684513999582202 w_k1_INTC_s114
                                 + 0.0226934453350925 w_k1_INTC_s115
                                 + 0.0403294728070182 w_k1_INTC_s116
                                 + 0.0430090873651447 w_k1_INTC_s117
                                 + 0.0108945886860764 w_k1_KO_s000
                                 + 0.0147055039004274 w_k1_KO_s001
                                 + 0.00392759605153841 w_k1_KO_s002
                                 + 0.0181207363106284 w_k1_KO_s003
                                 + 0.00555163808178804 w_k1_KO_s004
                                 + 0.0140071229553958 w_k1_KO_s005
                                 + 0.0053619733163299 w_k1_KO_s006
                                 + 0.0158954808397666 w_k1_KO_s007
                                 + 0.0434786302929477 w_k1_KO_s010
                                 + 0.0189629061553798 w_k1_KO_s011
                                 + 0.0394861697355539 w_k1_KO_s012
                                 + 0.0279152280835666 w_k1_KO_s013
                                 + 0.00930621685909837 w_k1_KO_s014
                                 + 0.0181322596915656 w_k1_KO_s015
                                 + 0.0304180110126369 w_k1_KO_s016
                                 + 0.0177554380654363 w_k1_KO_s017
                                 + 0.0188938015513572 w_k1_KO_s100
                                 + 0.0255952129624401 w_k1_KO_s101
                                 + 0.0117919680543551 w_k1_KO_s102
                                 + 0.0465410087044777 w_k1_KO_s103
                                 + 0.0493996392783052 w_k1_KO_s104
                                 + 0.0156490092902818 w_k1_KO_s105
                                 + 0.0268981629239114 w_k1_KO_s106
                                 + 0.0472730153712479 w_k1_KO_s107
                                 + 0.130546724375551 w_k1_KO_s110
                                 + 0.0754935320926382 w_k1_KO_s111
                                 + 0.0606341975122174 w_k1_KO_s112
                                 + 0.0297954496072167 w_k1_KO_s113
                                 + 0.071073239009381 w_k1_KO_s114
                                 + 0.0266673025128182 w_k1_KO_s115
                                 + 0.0395931142955975 w_k1_KO_s116
                                 + 0.0481352767567672 w_k1_KO_s117
                                 + 0.0106677573676311 w_k1_MAT_s000
                                 + 0.0170018884986399 w_k1_MAT_s001
                                 + 0.0033952793311654 w_k1_MAT_s002
                                 + 0.017760255443 w_k1_MAT_s003
                                 + 0.00517463204005603 w_k1_MAT_s004
                                 + 0.0113705781460234 w_k1_MAT_s005
                                 + 0.00545021783676228 w_k1_MAT_s006
                                 + 0.01824140904475 w_k1_MAT_s007
                                 + 0.0435350525135026 w_k1_MAT_s010
                                 + 0.0226192316682787 w_k1_MAT_s011
                                 + 0.0505988062121725 w_k1_MAT_s012
                                 + 0.0318157971400986 w_k1_MAT_s013
                                 + 0.00894205526321698 w_k1_MAT_s014
                                 + 0.0146886215806372 w_k1_MAT_s015
                                 + 0.035267186371642 w_k1_MAT_s016
                                 + 0.0193240349371578 w_k1_MAT_s017
                                 + 0.019470728743088 w_k1_MAT_s100
                                 + 0.0248806797271472 w_k1_MAT_s101
                                 + 0.0120103274537382 w_k1_MAT_s102
                                 + 0.0482037143919749 w_k1_MAT_s103
                                 + 0.0474244098835689 w_k1_MAT_s104
                                 + 0.0182174447282499 w_k1_MAT_s105
                                 + 0.0255524214277853 w_k1_MAT_s106
                                 + 0.0544994972238897 w_k1_MAT_s107
                                 + 0.111452866222021 w_k1_MAT_s110
                                 + 0.0913014296983494 w_k1_MAT_s111
                                 + 0.0623678170034062 w_k1_MAT_s112
                                 + 0.0282664699153247 w_k1_MAT_s113
                                 + 0.0639646208087706 w_k1_MAT_s114
                                 + 0.0380058210725747 w_k1_MAT_s115
                                 + 0.036982008879209 w_k1_MAT_s116
                                 + 0.0457716290543852 w_k1_MAT_s117
                                 + 0.00992613046537064 w_k1_MSFT_s000
                                 + 0.0156439582893203 w_k1_MSFT_s001
                                 + 0.0046159113016032 w_k1_MSFT_s002
                                 + 0.0151209093744345 w_k1_MSFT_s003
                                 + 0.00475889208051403 w_k1_MSFT_s004
                                 + 0.0151400276748444 w_k1_MSFT_s005
                                 + 0.00563144326867214 w_k1_MSFT_s006
                                 + 0.0140715946476135 w_k1_MSFT_s007
                                 + 0.0478928408508224 w_k1_MSFT_s010
                                 + 0.0223185951289364 w_k1_MSFT_s011
                                 + 0.041534363347942 w_k1_MSFT_s012
                                 + 0.0265132634840938 w_k1_MSFT_s013
                                 + 0.00869108959888859 w_k1_MSFT_s014
                                 + 0.0185304098294875 w_k1_MSFT_s015
                                 + 0.0304943685276746 w_k1_MSFT_s016
                                 + 0.017143021283668 w_k1_MSFT_s017
                                 + 0.0174732873269875 w_k1_MSFT_s100
                                 + 0.0224637921845326 w_k1_MSFT_s101
                                 + 0.0114888595612624 w_k1_MSFT_s102
                                 + 0.0374409779477636 w_k1_MSFT_s103
                                 + 0.0455485734467641 w_k1_MSFT_s104
                                 + 0.0147205943372618 w_k1_MSFT_s105
                                 + 0.0286901936093067 w_k1_MSFT_s106
                                 + 0.0528852493441738 w_k1_MSFT_s107
                                 + 0.119020892965607 w_k1_MSFT_s110
                                 + 0.0768555716612927 w_k1_MSFT_s111
                                 + 0.0643419770182705 w_k1_MSFT_s112
                                 + 0.0331405667969868 w_k1_MSFT_s113
                                 + 0.0719077087052221 w_k1_MSFT_s114
                                 + 0.0274183047120027 w_k1_MSFT_s115
                                 + 0.0374700629694307 w_k1_MSFT_s116
                                 + 0.0435347040167688 w_k1_MSFT_s117
                                 + 0.0102105030666877 w_k1_PG_s000
                                 + 0.0154493158240551 w_k1_PG_s001
                                 + 0.00396277122850527 w_k1_PG_s002
                                 + 0.0155287591542009 w_k1_PG_s003
                                 + 0.00543027588062088 w_k1_PG_s004
                                 + 0.0143482255117212 w_k1_PG_s005
                                 + 0.00528320897856746 w_k1_PG_s006
                                 + 0.0145010032501013 w_k1_PG_s007
                                 + 0.0457967843367926 w_k1_PG_s010
                                 + 0.0222438581017017 w_k1_PG_s011
                                 + 0.0390028808533609 w_k1_PG_s012
                                 + 0.0276536348218432 w_k1_PG_s013
                                 + 0.00894262013962341 w_k1_PG_s014
                                 + 0.0163528204761021 w_k1_PG_s015
                                 + 0.0300805344427997 w_k1_PG_s016
                                 + 0.0158966884023981 w_k1_PG_s017
                                 + 0.016329116032492 w_k1_PG_s100
                                 + 0.0245917390341143 w_k1_PG_s101
                                 + 0.0110214560053404 w_k1_PG_s102
                                 + 0.0413445943078067 w_k1_PG_s103
                                 + 0.0440805617385252 w_k1_PG_s104
                                 + 0.0144192083554456 w_k1_PG_s105
                                 + 0.026818939992 w_k1_PG_s106
                                 + 0.047267715608912 w_k1_PG_s107
                                 + 0.122375534757135 w_k1_PG_s110
                                 + 0.0759298686722215 w_k1_PG_s111
                                 + 0.0629765364536551 w_k1_PG_s112
                                 + 0.0340511480977381 w_k1_PG_s113
                                 + 0.0738585813948496 w_k1_PG_s114
                                 + 0.0279671156716256 w_k1_PG_s115
                                 + 0.03881185198179 w_k1_PG_s116
                                 + 0.0443102016364725 w_k1_PG_s117 >= 1.12
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
