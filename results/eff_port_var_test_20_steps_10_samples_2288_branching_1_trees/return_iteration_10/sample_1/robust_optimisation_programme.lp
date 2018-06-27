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
 w_k1_ADM_s0_return_constraint:  1.10874 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.956781 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.940214 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.06365 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.04222 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.968303 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.928669 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.45775 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.665858 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.0121 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.918211 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.853911 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.21844 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.08048 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.804765 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.980733 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.794512 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.32034 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.3306 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.17986 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.887979 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.873521 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.20051 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.901486 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.647813 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.07229 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.821306 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.989477 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.721981 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.991089 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.36015 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.851654 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.01077 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.49997 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.907278 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.29618 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.12381 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.883744 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.2939 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.895502 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.86099 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.16213 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.12725 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.994688 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.633876 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.5622 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.580517 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.903411 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.953164 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.86206 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.21757 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.921783 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.631564 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.929522 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.698948 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.49085 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.1053 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.900453 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.03627 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.853678 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.53256 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.945004 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.728551 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.1461 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.821219 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.909591 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.610827 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.46845 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.36815 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.670147 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.21384 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.65773 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.87334 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.29533 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.18116 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.042 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.34698 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.873935 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.876502 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.15783 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.06248 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.0276 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.640944 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.93926 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.498847 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.770413 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.06273 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.623228 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.44895 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.992704 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.560851 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.93871 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.613451 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.62337 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.41721 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.0123 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.04236 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.887719 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.74516 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.908203 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.775257 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.24484 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.541573 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.856533 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.541703 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.40344 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.53933 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.582857 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.2857 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 2.33879 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.864073 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.23924 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.09845 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.788992 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20401 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.904552 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.94076 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.0749 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.924715 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.02711 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.789115 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.62309 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.633761 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.832932 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.890901 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.753211 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.20203 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.08037 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.645223 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.870865 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.765619 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.47062 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.28776 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.04805 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.893347 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.897609 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.52603 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.945016 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.749764 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.24762 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.683243 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.08081 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.734279 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.2246 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.49082 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.7336 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.1776 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.50754 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.02162 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.11763 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.23459 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.913101 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17653 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.95402 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.751608 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.24788 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.5157 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.924555 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.723089 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.12101 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.72794 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.979254 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.47693 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.31167 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.24965 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.864291 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.844048 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.16184 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.626033 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.12833 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.874683 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.732302 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.60083 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.95251 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.38016 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.906334 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.11372 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.14057 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.916604 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.675711 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.676664 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.76032 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.924468 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.744451 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.12788 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.75918 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.6948 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.54553 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.728766 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.13473 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.07462 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.957116 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.03387 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.17352 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.907125 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.971009 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.779513 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.24864 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.692576 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.785883 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.01996 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.759305 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.16052 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.998195 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.694065 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.855215 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.930816 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.18636 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.17794 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.873607 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.05522 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.896363 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.52635 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.971703 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.982799 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.08134 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.689698 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.15903 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.856044 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.40776 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.27253 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.804116 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.01767 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.36655 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.986323 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.00218 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.31891 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.09932 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.2003 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.952642 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.791369 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.14437 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.14599 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.921953 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.813262 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.43716 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.682099 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.02302 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.31903 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.974685 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.36428 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.993244 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.764536 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.04728 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.691847 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.25462 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.0864 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.869854 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.30803 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.856989 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.47344 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.869382 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.946731 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.0607 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.771499 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.879457 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.676547 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.40924 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.12982 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.766922 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.1318 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.94745 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.83585 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.26776 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.921645 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.00064 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.29346 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.87591 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.630173 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.4153 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.70588 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.03361 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.744386 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.32412 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.738952 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.17758 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.41834 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.6687 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.67194 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.11192 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.66324 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.867626 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.515505 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.20897 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.562882 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.568737 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.42705 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.696085 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.31185 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.839281 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.890693 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.17806 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.915387 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.613081 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.495444 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.42388 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.917253 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.684942 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.1382 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.90344 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.627078 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.55631 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.647265 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.21751 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11664 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.947252 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.965016 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.17533 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.930345 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.978661 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.813172 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.53133 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.714786 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.834377 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.943638 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.829695 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.24246 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.012 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.689487 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.887529 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.760144 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.34331 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.17388 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.929298 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.897635 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.85569 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.34359 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.888834 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.886516 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.20936 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.634565 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.12185 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.770565 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.17468 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.39296 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.786905 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.17536 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.59248 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.01212 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.12527 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.10482 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.903988 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13178 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.957245 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.893859 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.04087 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.07561 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.934231 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.825226 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.2698 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.783499 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.941476 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.11939 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.90843 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.24187 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.04784 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.772863 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.951472 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.799304 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.26332 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.15952 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.995835 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.0815 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.912279 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.33132 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.872528 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.869564 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.02123 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.784731 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.963033 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.770443 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.1076 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.20729 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.854405 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.18769 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.44358 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.969924 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.16806 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.13018 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.04517 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.169000525038344 z_k1_s0000
                                 - 0.160000755847677 z_k1_s0001
                                 - 0.294000502917628 z_k1_s0002
                                 - 0.117000317087955 z_k1_s0003
                                 - 0.335001629681207 z_k1_s0004
                                 - 0.244001114967603 z_k1_s0005
                                 - 0.0890002979149276 z_k1_s0006
                                 - 0.100000283904271 z_k1_s0007
                                 - 0.26500007671763 z_k1_s0010
                                 - 0.114999324727303 z_k1_s0011
                                 - 0.163000111603603 z_k1_s0012
                                 - 0.0879998494083136 z_k1_s0013
                                 - 0.171999405643854 z_k1_s0014
                                 - 0.0999999323275145 z_k1_s0015
                                 - 0.122999455873353 z_k1_s0016
                                 - 0.111999901446841 z_k1_s0017
                                 - 0.116999936908306 z_k1_s0020
                                 - 0.141000140120205 z_k1_s0021
                                 - 0.145999971706077 z_k1_s0022
                                 - 0.153000241726154 z_k1_s0023
                                 - 0.0659999687328893 z_k1_s0024
                                 - 0.0479999849239173 z_k1_s0025
                                 - 0.111000307605212 z_k1_s0026
                                 - 0.0609999685470651 z_k1_s0027
                                 - 0.173000019456606 z_k1_s0030
                                 - 0.537000686560131 z_k1_s0031
                                 - 0.61800044756639 z_k1_s0032
                                 - 0.140000116824426 z_k1_s0033
                                 - 0.196000027294189 z_k1_s0034
                                 - 0.201000088241493 z_k1_s0035
                                 - 0.190000044997429 z_k1_s0036
                                 - 0.216000043983393 z_k1_s0037
                                 - 0.17499929450781 z_k1_s0040
                                 - 0.0769998725632655 z_k1_s0041
                                 - 0.188000011361957 z_k1_s0042
                                 - 0.289999503069457 z_k1_s0043
                                 - 0.183999288099612 z_k1_s0044
                                 - 0.0609999856110775 z_k1_s0045
                                 - 0.0589998853796609 z_k1_s0046
                                 - 0.272999369951744 z_k1_s0047
                                 - 0.151000263506609 z_k1_s0050
                                 - 0.0869998877106654 z_k1_s0051
                                 - 0.0829998928733934 z_k1_s0052
                                 - 0.257999896203838 z_k1_s0053
                                 - 0.184999761223829 z_k1_s0054
                                 - 0.228999246034229 z_k1_s0055
                                 - 0.0689999109429415 z_k1_s0056
                                 - 0.0839998915827114 z_k1_s0057
                                 - 0.418999399363343 z_k1_s0060
                                 - 0.276999865743353 z_k1_s0061
                                 - 0.501998711951765 z_k1_s0062
                                 - 0.172999619370524 z_k1_s0063
                                 - 0.555999869199427 z_k1_s0064
                                 - 0.181999726878608 z_k1_s0065
                                 - 0.248999474362741 z_k1_s0066
                                 - 0.202999636264699 z_k1_s0067
                                 - 0.60500216100089 z_k1_s0070
                                 - 0.354000004386848 z_k1_s0071
                                 - 0.518999635295839 z_k1_s0072
                                 - 0.666999078135339 z_k1_s0073
                                 - 0.467999882087653 z_k1_s0074
                                 - 0.866002329184472 z_k1_s0075
                                 - 0.254999982541366 z_k1_s0076
                                 - 0.320999997105021 z_k1_s0077
                                 - 0.142000330864023 z_k1_s0100
                                 - 0.187000142306825 z_k1_s0101
                                 - 0.273999956962922 z_k1_s0102
                                 - 0.190999886612834 z_k1_s0103
                                 - 0.25999950789179 z_k1_s0104
                                 - 0.0830000703500887 z_k1_s0105
                                 - 0.0860000129796054 z_k1_s0106
                                 - 0.121000060457355 z_k1_s0107
                                 - 0.13600012400611 z_k1_s0110
                                 - 0.156000244671837 z_k1_s0111
                                 - 0.317999749872469 z_k1_s0112
                                 - 0.199000749468439 z_k1_s0113
                                 - 0.246000075316758 z_k1_s0114
                                 - 0.0710000088678295 z_k1_s0115
                                 - 0.206000783786434 z_k1_s0116
                                 - 0.251000778258989 z_k1_s0117
                                 - 0.228999558290757 z_k1_s0120
                                 - 0.0830000900942574 z_k1_s0121
                                 - 0.341000590163354 z_k1_s0122
                                 - 0.110000082452597 z_k1_s0123
                                 - 0.113000112581329 z_k1_s0124
                                 - 0.116000142710062 z_k1_s0125
                                 - 0.117000059819555 z_k1_s0126
                                 - 0.28500077122769 z_k1_s0127
                                 - 0.358000796025063 z_k1_s0130
                                 - 0.401002350181494 z_k1_s0131
                                 - 0.526002485558708 z_k1_s0132
                                 - 0.52100075270056 z_k1_s0133
                                 - 0.324000398009821 z_k1_s0134
                                 - 0.95500119132219 z_k1_s0135
                                 - 0.614000005403705 z_k1_s0136
                                 - 0.227000544221548 z_k1_s0137
                                 - 0.191999813959902 z_k1_s0140
                                 - 0.099999982304118 z_k1_s0141
                                 - 0.127000055142232 z_k1_s0142
                                 - 0.153999969580342 z_k1_s0143
                                 - 0.389999820106057 z_k1_s0144
                                 - 0.370999739516272 z_k1_s0145
                                 - 0.149000073425139 z_k1_s0146
                                 - 0.100999961535159 z_k1_s0147
                                 - 0.372999037549187 z_k1_s0150
                                 - 0.71499964831271 z_k1_s0151
                                 - 0.187999504606044 z_k1_s0152
                                 - 0.226999456323154 z_k1_s0153
                                 - 0.402000044021525 z_k1_s0154
                                 - 0.301999388094476 z_k1_s0155
                                 - 0.347999273691686 z_k1_s0156
                                 - 0.645999019668371 z_k1_s0157
                                 - 0.260999306999461 z_k1_s0160
                                 - 0.181999564571174 z_k1_s0161
                                 - 0.603998612691029 z_k1_s0162
                                 - 0.291999546797205 z_k1_s0163
                                 - 0.752999955950767 z_k1_s0164
                                 - 0.214999427879252 z_k1_s0165
                                 - 0.343998296448415 z_k1_s0166
                                 - 0.632999557377769 z_k1_s0167
                                 - 0.282999796919628 z_k1_s0170
                                 - 1.26699736079427 z_k1_s0171
                                 - 0.527997678928709 z_k1_s0172
                                 - 0.429998372165205 z_k1_s0173
                                 - 0.389999597730331 z_k1_s0174
                                 - 0.299999660954126 z_k1_s0175
                                 - 0.291999906067151 z_k1_s0176
                                 - 0.35899977774395 z_k1_s0177
                                 - 0.0859998619854737 z_k1_s1000
                                 - 0.198999799077836 z_k1_s1001
                                 - 0.060999874964156 z_k1_s1002
                                 - 0.226000086384405 z_k1_s1003
                                 - 0.102999814964027 z_k1_s1004
                                 - 0.0659999360283236 z_k1_s1005
                                 - 0.147999727942495 z_k1_s1006
                                 - 0.171999511771267 z_k1_s1007
                                 - 0.194000179081954 z_k1_s1010
                                 - 0.634001327206914 z_k1_s1011
                                 - 0.284001177244846 z_k1_s1012
                                 - 0.415001137413384 z_k1_s1013
                                 - 0.239000198362791 z_k1_s1014
                                 - 0.327000907788392 z_k1_s1015
                                 - 0.138000251048145 z_k1_s1016
                                 - 0.168000263902036 z_k1_s1017
                                 - 0.117999832164069 z_k1_s1020
                                 - 0.0440000076170808 z_k1_s1021
                                 - 0.036000018995788 z_k1_s1022
                                 - 0.0480000019277272 z_k1_s1023
                                 - 0.109999843542776 z_k1_s1024
                                 - 0.175999749668442 z_k1_s1025
                                 - 0.115999835008746 z_k1_s1026
                                 - 0.0539999933936968 z_k1_s1027
                                 - 0.440001443082023 z_k1_s1030
                                 - 0.143000087376632 z_k1_s1031
                                 - 0.203999947447071 z_k1_s1032
                                 - 0.625001043730078 z_k1_s1033
                                 - 0.47000022313298 z_k1_s1034
                                 - 0.269000153691005 z_k1_s1035
                                 - 0.242999888013419 z_k1_s1036
                                 - 0.658999254054468 z_k1_s1037
                                 - 0.055999969378667 z_k1_s1040
                                 - 0.145999477766229 z_k1_s1041
                                 - 0.0959999251063075 z_k1_s1042
                                 - 0.0639999762041803 z_k1_s1043
                                 - 0.129999620915054 z_k1_s1044
                                 - 0.0990002902655317 z_k1_s1045
                                 - 0.0869996136278927 z_k1_s1046
                                 - 0.105999835638292 z_k1_s1047
                                 - 0.0869998504642304 z_k1_s1050
                                 - 0.151999320154253 z_k1_s1051
                                 - 0.196999641738576 z_k1_s1052
                                 - 0.11599966648585 z_k1_s1053
                                 - 0.162999912940727 z_k1_s1054
                                 - 0.16899968755239 z_k1_s1055
                                 - 0.0399999058088814 z_k1_s1056
                                 - 0.0819998370881596 z_k1_s1057
                                 - 0.328000370134633 z_k1_s1060
                                 - 0.251000980057812 z_k1_s1061
                                 - 0.128000037081728 z_k1_s1062
                                 - 0.257999847245815 z_k1_s1063
                                 - 0.196000448287937 z_k1_s1064
                                 - 0.235000425197188 z_k1_s1065
                                 - 0.139000143435703 z_k1_s1066
                                 - 0.158000188619739 z_k1_s1067
                                 - 0.205999942890595 z_k1_s1070
                                 - 0.175000165795112 z_k1_s1071
                                 - 0.0819999289099153 z_k1_s1072
                                 - 0.0699999531182013 z_k1_s1073
                                 - 0.142999702084605 z_k1_s1074
                                 - 0.244999552914096 z_k1_s1075
                                 - 0.139999906236403 z_k1_s1076
                                 - 0.0709998473343211 z_k1_s1077
                                 - 0.738000293620023 z_k1_s1100
                                 - 0.218000202734131 z_k1_s1101
                                 - 1.05100053025158 z_k1_s1102
                                 - 0.220000135294058 z_k1_s1103
                                 - 0.258999899312932 z_k1_s1104
                                 - 0.263000124132886 z_k1_s1105
                                 - 0.401000866269343 z_k1_s1106
                                 - 0.446999315147659 z_k1_s1107
                                 - 3.67399155443425 z_k1_s1110
                                 - 1.81699150842886 z_k1_s1111
                                 - 0.917998672075257 z_k1_s1112
                                 - 1.59099810660367 z_k1_s1113
                                 - 1.69799139043242 z_k1_s1114
                                 - 1.50199370955284 z_k1_s1115
                                 - 1.16399919128519 z_k1_s1116
                                 - 1.87899121083921 z_k1_s1117
                                 - 0.386998969858327 z_k1_s1120
                                 - 0.745998213562128 z_k1_s1121
                                 - 1.19499519965134 z_k1_s1122
                                 - 0.404998793814807 z_k1_s1123
                                 - 0.523999007368537 z_k1_s1124
                                 - 1.92999489819261 z_k1_s1125
                                 - 0.983998640669588 z_k1_s1126
                                 - 0.715998506967994 z_k1_s1127
                                 - 0.272000069853966 z_k1_s1130
                                 - 0.171999928204704 z_k1_s1131
                                 - 0.713000733956214 z_k1_s1132
                                 - 0.585999914098597 z_k1_s1133
                                 - 0.249999899173116 z_k1_s1134
                                 - 0.268000149112002 z_k1_s1135
                                 - 0.240000097318208 z_k1_s1136
                                 - 0.532000377482042 z_k1_s1137
                                 - 1.09599872381703 z_k1_s1140
                                 - 0.640995593352623 z_k1_s1141
                                 - 1.0439941682334 z_k1_s1142
                                 - 0.495998639113653 z_k1_s1143
                                 - 0.906997521746913 z_k1_s1144
                                 - 0.376998852281476 z_k1_s1145
                                 - 0.580998395594048 z_k1_s1146
                                 - 1.24599491220411 z_k1_s1147
                                 - 0.820001467531697 z_k1_s1150
                                 - 0.622002069521138 z_k1_s1151
                                 - 0.573001953136831 z_k1_s1152
                                 - 0.748002256766125 z_k1_s1153
                                 - 1.43800802471403 z_k1_s1154
                                 - 0.983005823713152 z_k1_s1155
                                 - 1.41800301592836 z_k1_s1156
                                 - 1.24000470569038 z_k1_s1157
                                 - 0.986002415244319 z_k1_s1160
                                 - 0.451000139818033 z_k1_s1161
                                 - 0.391001787528449 z_k1_s1162
                                 - 0.304000289606072 z_k1_s1163
                                 - 0.269999989307853 z_k1_s1164
                                 - 0.258000318849936 z_k1_s1165
                                 - 0.244000086315306 z_k1_s1166
                                 - 0.797999927265407 z_k1_s1167
                                 - 0.774001945316827 z_k1_s1170
                                 - 0.973001294829735 z_k1_s1171
                                 - 0.671999623607492 z_k1_s1172
                                 - 1.11899901240393 z_k1_s1173
                                 - 0.771000864491153 z_k1_s1174
                                 - 1.91900034622049 z_k1_s1175
                                 - 0.819998560664941 z_k1_s1176
                                 - 0.43599941723141 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.19211 w_k1_ADM_s000
                                 + 1.14947 w_k1_AXP_s000
                                 + 1.15112 w_k1_DIS_s000
                                 + 1.16176 w_k1_FDX_s000
                                 + 0.779275 w_k1_IBM_s000
                                 + 1.08098 w_k1_INTC_s000
                                 + 0.938186 w_k1_KO_s000
                                 + 0.85808 w_k1_MAT_s000
                                 + 1.18065 w_k1_MSFT_s000 + 1.0195 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.07943 w_k1_ADM_s000
                                 + 1.56221 w_k1_AXP_s000
                                 + 1.18472 w_k1_DIS_s000
                                 + 1.19796 w_k1_FDX_s000
                                 + 1.24222 w_k1_IBM_s000
                                 + 1.26363 w_k1_INTC_s000
                                 + 1.11324 w_k1_KO_s000 + 1.45273 w_k1_MAT_s000
                                 + 1.11005 w_k1_MSFT_s000
                                 + 1.25839 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.888846 w_k1_ADM_s000
                                 + 1.0233 w_k1_AXP_s000 + 1.19613 w_k1_DIS_s000
                                 + 1.02885 w_k1_FDX_s000
                                 + 1.06342 w_k1_IBM_s000
                                 + 1.12936 w_k1_INTC_s000
                                 + 1.00089 w_k1_KO_s000
                                 + 0.932356 w_k1_MAT_s000
                                 + 1.05786 w_k1_MSFT_s000
                                 + 1.03024 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.917374 w_k1_ADM_s000
                                 + 1.37647 w_k1_AXP_s000
                                 + 1.88039 w_k1_DIS_s000
                                 + 1.14527 w_k1_FDX_s000
                                 + 1.51544 w_k1_IBM_s000
                                 + 1.11125 w_k1_INTC_s000
                                 + 1.26555 w_k1_KO_s000 + 1.44526 w_k1_MAT_s000
                                 + 1.25387 w_k1_MSFT_s000
                                 + 1.14147 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.751277 w_k1_ADM_s000
                                 + 0.854568 w_k1_AXP_s000
                                 + 0.702115 w_k1_DIS_s000
                                 + 0.814853 w_k1_FDX_s000
                                 + 0.876076 w_k1_IBM_s000
                                 + 0.977124 w_k1_INTC_s000
                                 + 0.784053 w_k1_KO_s000
                                 + 0.662324 w_k1_MAT_s000
                                 + 0.918007 w_k1_MSFT_s000
                                 + 1.03804 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.830314 w_k1_ADM_s000
                                 + 0.900433 w_k1_AXP_s000
                                 + 0.897698 w_k1_DIS_s000
                                 + 0.854847 w_k1_FDX_s000
                                 + 1.2068 w_k1_IBM_s000
                                 + 1.01726 w_k1_INTC_s000
                                 + 1.05924 w_k1_KO_s000 + 1.03164 w_k1_MAT_s000
                                 + 0.967876 w_k1_MSFT_s000
                                 + 1.01424 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.3092 w_k1_ADM_s000
                                 + 1.43211 w_k1_AXP_s000
                                 + 1.61927 w_k1_DIS_s000
                                 + 1.40182 w_k1_FDX_s000
                                 + 0.978329 w_k1_IBM_s000
                                 + 1.42903 w_k1_INTC_s000
                                 + 1.17417 w_k1_KO_s000
                                 + 0.862717 w_k1_MAT_s000
                                 + 1.31037 w_k1_MSFT_s000
                                 + 1.24354 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.01306 w_k1_ADM_s000
                                 + 0.916315 w_k1_AXP_s000
                                 + 0.873991 w_k1_DIS_s000
                                 + 1.02053 w_k1_FDX_s000
                                 + 0.603018 w_k1_IBM_s000
                                 + 0.862457 w_k1_INTC_s000
                                 + 0.760531 w_k1_KO_s000
                                 + 0.625882 w_k1_MAT_s000
                                 + 1.06504 w_k1_MSFT_s000
                                 + 0.946622 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.891195 w_k1_ADM_s001
                                 + 0.918485 w_k1_AXP_s001
                                 + 1.02152 w_k1_DIS_s001
                                 + 0.990491 w_k1_FDX_s001
                                 + 1.03155 w_k1_IBM_s001
                                 + 0.990664 w_k1_INTC_s001
                                 + 1.11665 w_k1_KO_s001 + 1.12433 w_k1_MAT_s001
                                 + 1.04575 w_k1_MSFT_s001
                                 + 1.06909 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.858358 w_k1_ADM_s001
                                 + 0.775627 w_k1_AXP_s001
                                 + 0.666411 w_k1_DIS_s001
                                 + 0.895643 w_k1_FDX_s001
                                 + 0.525591 w_k1_IBM_s001
                                 + 1.10416 w_k1_INTC_s001
                                 + 0.730193 w_k1_KO_s001
                                 + 0.761553 w_k1_MAT_s001
                                 + 0.985091 w_k1_MSFT_s001
                                 + 0.919346 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.894368 w_k1_ADM_s001
                                 + 0.709627 w_k1_AXP_s001
                                 + 0.7113 w_k1_DIS_s001
                                 + 0.752779 w_k1_FDX_s001
                                 + 0.801029 w_k1_IBM_s001
                                 + 0.882449 w_k1_INTC_s001
                                 + 0.870826 w_k1_KO_s001
                                 + 0.737945 w_k1_MAT_s001
                                 + 0.874082 w_k1_MSFT_s001
                                 + 0.896467 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.02594 w_k1_ADM_s001
                                 + 0.95297 w_k1_AXP_s001
                                 + 0.708559 w_k1_DIS_s001
                                 + 0.84971 w_k1_FDX_s001
                                 + 0.865157 w_k1_IBM_s001
                                 + 0.842766 w_k1_INTC_s001
                                 + 0.806481 w_k1_KO_s001
                                 + 1.20505 w_k1_MAT_s001
                                 + 0.746581 w_k1_MSFT_s001
                                 + 0.904859 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.46172 w_k1_ADM_s001
                                 + 1.11517 w_k1_AXP_s001
                                 + 1.38514 w_k1_DIS_s001
                                 + 1.21993 w_k1_FDX_s001
                                 + 1.01614 w_k1_IBM_s001
                                 + 1.07874 w_k1_INTC_s001
                                 + 1.10271 w_k1_KO_s001 + 1.28952 w_k1_MAT_s001
                                 + 1.23564 w_k1_MSFT_s001 + 1.1515 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.42969 w_k1_ADM_s001
                                 + 1.20243 w_k1_AXP_s001
                                 + 0.936365 w_k1_DIS_s001
                                 + 1.18345 w_k1_FDX_s001
                                 + 0.695929 w_k1_IBM_s001
                                 + 1.0988 w_k1_INTC_s001
                                 + 0.822087 w_k1_KO_s001
                                 + 0.522002 w_k1_MAT_s001
                                 + 1.04725 w_k1_MSFT_s001
                                 + 0.939847 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.00064 w_k1_ADM_s001
                                 + 0.861883 w_k1_AXP_s001
                                 + 0.851864 w_k1_DIS_s001
                                 + 0.774909 w_k1_FDX_s001
                                 + 1.32594 w_k1_IBM_s001
                                 + 1.03205 w_k1_INTC_s001
                                 + 1.09734 w_k1_KO_s001 + 1.22508 w_k1_MAT_s001
                                 + 0.93731 w_k1_MSFT_s001
                                 + 0.931413 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.36141 w_k1_ADM_s001
                                 + 1.56212 w_k1_AXP_s001
                                 + 1.63128 w_k1_DIS_s001
                                 + 1.28785 w_k1_FDX_s001
                                 + 1.58825 w_k1_IBM_s001
                                 + 1.74909 w_k1_INTC_s001
                                 + 1.70711 w_k1_KO_s001 + 1.61642 w_k1_MAT_s001
                                 + 1.41691 w_k1_MSFT_s001
                                 + 1.27244 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.620678 w_k1_ADM_s002
                                 + 0.816961 w_k1_AXP_s002
                                 + 0.772349 w_k1_DIS_s002
                                 + 0.804632 w_k1_FDX_s002
                                 + 1.00372 w_k1_IBM_s002
                                 + 0.94001 w_k1_INTC_s002 + 0.9188 w_k1_KO_s002
                                 + 1.02247 w_k1_MAT_s002
                                 + 0.846378 w_k1_MSFT_s002
                                 + 0.893302 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.4225 w_k1_ADM_s002
                                 + 1.41958 w_k1_AXP_s002
                                 + 0.978831 w_k1_DIS_s002
                                 + 1.21155 w_k1_FDX_s002
                                 + 1.07686 w_k1_IBM_s002
                                 + 1.21641 w_k1_INTC_s002
                                 + 1.06113 w_k1_KO_s002 + 1.13555 w_k1_MAT_s002
                                 + 1.03479 w_k1_MSFT_s002
                                 + 1.15991 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.15244 w_k1_ADM_s002
                                 + 0.880002 w_k1_AXP_s002
                                 + 0.989768 w_k1_DIS_s002
                                 + 1.09541 w_k1_FDX_s002
                                 + 0.847107 w_k1_IBM_s002
                                 + 0.988301 w_k1_INTC_s002
                                 + 0.965053 w_k1_KO_s002
                                 + 0.784815 w_k1_MAT_s002
                                 + 1.07513 w_k1_MSFT_s002
                                 + 0.993778 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.925993 w_k1_ADM_s002
                                 + 0.842801 w_k1_AXP_s002
                                 + 0.653708 w_k1_DIS_s002
                                 + 0.933731 w_k1_FDX_s002
                                 + 0.682233 w_k1_IBM_s002
                                 + 0.872149 w_k1_INTC_s002
                                 + 0.730341 w_k1_KO_s002
                                 + 0.871008 w_k1_MAT_s002
                                 + 0.809804 w_k1_MSFT_s002
                                 + 0.834683 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.23788 w_k1_ADM_s002
                                 + 1.45641 w_k1_AXP_s002
                                 + 1.34864 w_k1_DIS_s002
                                 + 1.23314 w_k1_FDX_s002
                                 + 1.45238 w_k1_IBM_s002
                                 + 1.08367 w_k1_INTC_s002
                                 + 1.26934 w_k1_KO_s002 + 1.12154 w_k1_MAT_s002
                                 + 1.01214 w_k1_MSFT_s002 + 1.1714 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.3829 w_k1_ADM_s002
                                 + 0.882999 w_k1_AXP_s002
                                 + 0.649634 w_k1_DIS_s002
                                 + 0.901278 w_k1_FDX_s002
                                 + 0.653072 w_k1_IBM_s002
                                 + 0.835997 w_k1_INTC_s002
                                 + 0.812925 w_k1_KO_s002
                                 + 0.751578 w_k1_MAT_s002
                                 + 0.8256 w_k1_MSFT_s002
                                 + 0.864239 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.869263 w_k1_ADM_s002
                                 + 0.753467 w_k1_AXP_s002
                                 + 1.04524 w_k1_DIS_s002
                                 + 0.832461 w_k1_FDX_s002
                                 + 1.51626 w_k1_IBM_s002
                                 + 1.07489 w_k1_INTC_s002
                                 + 1.24279 w_k1_KO_s002 + 1.43547 w_k1_MAT_s002
                                 + 1.01786 w_k1_MSFT_s002 + 1.0574 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 2.07083 w_k1_ADM_s002
                                 + 1.47112 w_k1_AXP_s002
                                 + 1.41566 w_k1_DIS_s002
                                 + 1.57558 w_k1_FDX_s002
                                 + 0.772895 w_k1_IBM_s002
                                 + 1.33975 w_k1_INTC_s002
                                 + 1.05587 w_k1_KO_s002
                                 + 0.624858 w_k1_MAT_s002
                                 + 1.22358 w_k1_MSFT_s002
                                 + 1.01944 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.809529 w_k1_ADM_s003
                                 + 0.961568 w_k1_AXP_s003
                                 + 1.15767 w_k1_DIS_s003
                                 + 1.05495 w_k1_FDX_s003
                                 + 1.24014 w_k1_IBM_s003
                                 + 1.23427 w_k1_INTC_s003
                                 + 1.09374 w_k1_KO_s003 + 1.14571 w_k1_MAT_s003
                                 + 1.15252 w_k1_MSFT_s003
                                 + 1.19752 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.872797 w_k1_ADM_s003
                                 + 0.763757 w_k1_AXP_s003
                                 + 0.777599 w_k1_DIS_s003
                                 + 0.854355 w_k1_FDX_s003
                                 + 1.02571 w_k1_IBM_s003
                                 + 0.884143 w_k1_INTC_s003
                                 + 0.88706 w_k1_KO_s003
                                 + 0.825578 w_k1_MAT_s003
                                 + 0.908688 w_k1_MSFT_s003
                                 + 0.898926 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.997082 w_k1_ADM_s003
                                 + 0.838266 w_k1_AXP_s003
                                 + 0.855319 w_k1_DIS_s003
                                 + 0.966274 w_k1_FDX_s003
                                 + 0.796856 w_k1_IBM_s003
                                 + 1.2295 w_k1_INTC_s003
                                 + 0.729612 w_k1_KO_s003
                                 + 0.585044 w_k1_MAT_s003
                                 + 1.0435 w_k1_MSFT_s003
                                 + 0.931843 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.876155 w_k1_ADM_s003
                                 + 0.978466 w_k1_AXP_s003
                                 + 1.20602 w_k1_DIS_s003
                                 + 0.974191 w_k1_FDX_s003
                                 + 1.56689 w_k1_IBM_s003
                                 + 1.05216 w_k1_INTC_s003
                                 + 1.32861 w_k1_KO_s003 + 1.02359 w_k1_MAT_s003
                                 + 1.04716 w_k1_MSFT_s003
                                 + 1.12568 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.0578 w_k1_ADM_s003
                                 + 1.24662 w_k1_AXP_s003
                                 + 0.910096 w_k1_DIS_s003
                                 + 1.00183 w_k1_FDX_s003
                                 + 1.19187 w_k1_IBM_s003
                                 + 1.06563 w_k1_INTC_s003
                                 + 0.925804 w_k1_KO_s003
                                 + 1.21108 w_k1_MAT_s003
                                 + 0.948816 w_k1_MSFT_s003
                                 + 1.09577 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.972008 w_k1_ADM_s003
                                 + 1.00953 w_k1_AXP_s003
                                 + 1.25911 w_k1_DIS_s003
                                 + 1.12119 w_k1_FDX_s003
                                 + 0.99722 w_k1_IBM_s003
                                 + 1.1438 w_k1_INTC_s003 + 1.08264 w_k1_KO_s003
                                 + 0.912285 w_k1_MAT_s003
                                 + 1.1981 w_k1_MSFT_s003 + 1.14341 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.34965 w_k1_ADM_s003
                                 + 1.34889 w_k1_AXP_s003
                                 + 1.51842 w_k1_DIS_s003
                                 + 1.38934 w_k1_FDX_s003
                                 + 1.09074 w_k1_IBM_s003
                                 + 1.09849 w_k1_INTC_s003
                                 + 1.09863 w_k1_KO_s003 + 1.01723 w_k1_MAT_s003
                                 + 1.2292 w_k1_MSFT_s003 + 1.12039 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.01741 w_k1_ADM_s003
                                 + 1.49845 w_k1_AXP_s003 + 1.6667 w_k1_DIS_s003
                                 + 1.35046 w_k1_FDX_s003
                                 + 1.43531 w_k1_IBM_s003
                                 + 1.30345 w_k1_INTC_s003
                                 + 1.34996 w_k1_KO_s003 + 1.77625 w_k1_MAT_s003
                                 + 1.29839 w_k1_MSFT_s003
                                 + 1.15274 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.809604 w_k1_ADM_s004
                                 + 0.776065 w_k1_AXP_s004
                                 + 0.721712 w_k1_DIS_s004
                                 + 0.771269 w_k1_FDX_s004
                                 + 1.00091 w_k1_IBM_s004
                                 + 0.754695 w_k1_INTC_s004
                                 + 0.898437 w_k1_KO_s004
                                 + 1.08364 w_k1_MAT_s004
                                 + 0.746467 w_k1_MSFT_s004
                                 + 0.815634 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.12586 w_k1_ADM_s004
                                 + 1.01295 w_k1_AXP_s004
                                 + 0.906167 w_k1_DIS_s004
                                 + 1.06041 w_k1_FDX_s004
                                 + 0.735536 w_k1_IBM_s004
                                 + 0.922639 w_k1_INTC_s004
                                 + 0.863345 w_k1_KO_s004
                                 + 0.831335 w_k1_MAT_s004
                                 + 1.04066 w_k1_MSFT_s004
                                 + 0.878356 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.65535 w_k1_ADM_s004
                                 + 1.36782 w_k1_AXP_s004
                                 + 1.55378 w_k1_DIS_s004
                                 + 1.39288 w_k1_FDX_s004
                                 + 0.994903 w_k1_IBM_s004
                                 + 1.20603 w_k1_INTC_s004
                                 + 1.10536 w_k1_KO_s004 + 1.19168 w_k1_MAT_s004
                                 + 1.40831 w_k1_MSFT_s004 + 1.1274 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.90519 w_k1_ADM_s004
                                 + 0.954475 w_k1_AXP_s004
                                 + 1.00958 w_k1_DIS_s004
                                 + 0.927649 w_k1_FDX_s004
                                 + 1.23165 w_k1_IBM_s004
                                 + 1.00979 w_k1_INTC_s004
                                 + 0.994436 w_k1_KO_s004
                                 + 0.993486 w_k1_MAT_s004
                                 + 0.963403 w_k1_MSFT_s004
                                 + 0.88268 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.07616 w_k1_ADM_s004
                                 + 1.11707 w_k1_AXP_s004
                                 + 1.32746 w_k1_DIS_s004
                                 + 1.21958 w_k1_FDX_s004
                                 + 1.18049 w_k1_IBM_s004
                                 + 1.00634 w_k1_INTC_s004 + 1.3229 w_k1_KO_s004
                                 + 1.48319 w_k1_MAT_s004
                                 + 1.31831 w_k1_MSFT_s004
                                 + 1.16943 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.865249 w_k1_ADM_s004
                                 + 0.622951 w_k1_AXP_s004
                                 + 0.729065 w_k1_DIS_s004
                                 + 0.875177 w_k1_FDX_s004
                                 + 0.545871 w_k1_IBM_s004
                                 + 0.868499 w_k1_INTC_s004
                                 + 0.664074 w_k1_KO_s004
                                 + 0.572709 w_k1_MAT_s004
                                 + 0.953916 w_k1_MSFT_s004
                                 + 0.804623 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.859103 w_k1_ADM_s004
                                 + 0.573055 w_k1_AXP_s004
                                 + 0.612148 w_k1_DIS_s004
                                 + 0.741006 w_k1_FDX_s004
                                 + 0.748788 w_k1_IBM_s004
                                 + 0.855363 w_k1_INTC_s004
                                 + 0.749126 w_k1_KO_s004
                                 + 0.695797 w_k1_MAT_s004
                                 + 0.925346 w_k1_MSFT_s004
                                 + 0.729105 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.26024 w_k1_ADM_s004
                                 + 1.18397 w_k1_AXP_s004
                                 + 0.97896 w_k1_DIS_s004
                                 + 1.04582 w_k1_FDX_s004
                                 + 1.09635 w_k1_IBM_s004
                                 + 0.965165 w_k1_INTC_s004
                                 + 0.994291 w_k1_KO_s004
                                 + 1.15295 w_k1_MAT_s004
                                 + 1.1558 w_k1_MSFT_s004
                                 + 0.964838 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.976113 w_k1_ADM_s005
                                 + 0.862123 w_k1_AXP_s005
                                 + 0.797802 w_k1_DIS_s005
                                 + 0.976601 w_k1_FDX_s005
                                 + 0.721858 w_k1_IBM_s005
                                 + 0.871393 w_k1_INTC_s005
                                 + 0.859113 w_k1_KO_s005
                                 + 0.734365 w_k1_MAT_s005
                                 + 0.888547 w_k1_MSFT_s005
                                 + 1.00027 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.896203 w_k1_ADM_s005
                                 + 1.21949 w_k1_AXP_s005
                                 + 1.21849 w_k1_DIS_s005
                                 + 1.08557 w_k1_FDX_s005
                                 + 1.44639 w_k1_IBM_s005
                                 + 1.23272 w_k1_INTC_s005
                                 + 1.38114 w_k1_KO_s005 + 1.3591 w_k1_MAT_s005
                                 + 1.1282 w_k1_MSFT_s005 + 1.28902 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.46293 w_k1_ADM_s005
                                 + 1.48278 w_k1_AXP_s005
                                 + 1.71108 w_k1_DIS_s005
                                 + 1.49283 w_k1_FDX_s005
                                 + 1.40496 w_k1_IBM_s005
                                 + 1.0847 w_k1_INTC_s005 + 1.24915 w_k1_KO_s005
                                 + 1.34165 w_k1_MAT_s005
                                 + 1.11252 w_k1_MSFT_s005
                                 + 1.21103 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.842542 w_k1_ADM_s005
                                 + 0.937847 w_k1_AXP_s005
                                 + 0.946843 w_k1_DIS_s005
                                 + 0.878637 w_k1_FDX_s005
                                 + 1.10021 w_k1_IBM_s005
                                 + 1.35066 w_k1_INTC_s005
                                 + 1.07382 w_k1_KO_s005
                                 + 0.680199 w_k1_MAT_s005
                                 + 0.915374 w_k1_MSFT_s005
                                 + 1.04336 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.942238 w_k1_ADM_s005
                                 + 0.668359 w_k1_AXP_s005
                                 + 0.679249 w_k1_DIS_s005
                                 + 0.797393 w_k1_FDX_s005
                                 + 0.775537 w_k1_IBM_s005
                                 + 0.836909 w_k1_INTC_s005
                                 + 0.715186 w_k1_KO_s005
                                 + 0.524012 w_k1_MAT_s005
                                 + 0.787115 w_k1_MSFT_s005
                                 + 0.771372 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.15704 w_k1_ADM_s005
                                 + 1.16097 w_k1_AXP_s005
                                 + 1.21075 w_k1_DIS_s005
                                 + 1.17842 w_k1_FDX_s005
                                 + 0.874397 w_k1_IBM_s005
                                 + 1.16911 w_k1_INTC_s005 + 1.0426 w_k1_KO_s005
                                 + 0.974924 w_k1_MAT_s005
                                 + 1.16912 w_k1_MSFT_s005
                                 + 1.09398 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.5343 w_k1_ADM_s005
                                 + 1.69591 w_k1_AXP_s005
                                 + 1.43394 w_k1_DIS_s005
                                 + 1.29463 w_k1_FDX_s005
                                 + 1.17086 w_k1_IBM_s005
                                 + 1.34356 w_k1_INTC_s005
                                 + 1.19875 w_k1_KO_s005 + 1.54552 w_k1_MAT_s005
                                 + 1.16624 w_k1_MSFT_s005
                                 + 1.19164 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.0792 w_k1_ADM_s005
                                 + 1.2092 w_k1_AXP_s005 + 1.32524 w_k1_DIS_s005
                                 + 1.13857 w_k1_FDX_s005
                                 + 1.13585 w_k1_IBM_s005
                                 + 1.13437 w_k1_INTC_s005
                                 + 1.16503 w_k1_KO_s005 + 1.02492 w_k1_MAT_s005
                                 + 0.968137 w_k1_MSFT_s005
                                 + 1.09866 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.987202 w_k1_ADM_s006
                                 + 1.03555 w_k1_AXP_s006
                                 + 1.19958 w_k1_DIS_s006
                                 + 1.21784 w_k1_FDX_s006
                                 + 0.655772 w_k1_IBM_s006
                                 + 0.980232 w_k1_INTC_s006
                                 + 0.967974 w_k1_KO_s006
                                 + 0.816509 w_k1_MAT_s006
                                 + 1.04503 w_k1_MSFT_s006
                                 + 1.00258 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.981358 w_k1_ADM_s006
                                 + 0.862838 w_k1_AXP_s006
                                 + 1.30212 w_k1_DIS_s006 + 1.1106 w_k1_FDX_s006
                                 + 1.01053 w_k1_IBM_s006
                                 + 1.18374 w_k1_INTC_s006
                                 + 1.25335 w_k1_KO_s006
                                 + 0.745838 w_k1_MAT_s006
                                 + 1.19916 w_k1_MSFT_s006
                                 + 1.00795 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.905094 w_k1_ADM_s006
                                 + 0.798342 w_k1_AXP_s006
                                 + 0.663708 w_k1_DIS_s006
                                 + 0.813855 w_k1_FDX_s006
                                 + 0.819496 w_k1_IBM_s006
                                 + 0.712507 w_k1_INTC_s006
                                 + 0.843917 w_k1_KO_s006
                                 + 0.821632 w_k1_MAT_s006
                                 + 0.828773 w_k1_MSFT_s006
                                 + 0.852092 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.04387 w_k1_ADM_s006
                                 + 1.10559 w_k1_AXP_s006
                                 + 0.876397 w_k1_DIS_s006
                                 + 0.983549 w_k1_FDX_s006
                                 + 1.10853 w_k1_IBM_s006
                                 + 0.789166 w_k1_INTC_s006
                                 + 0.942201 w_k1_KO_s006
                                 + 0.92808 w_k1_MAT_s006
                                 + 0.914878 w_k1_MSFT_s006
                                 + 0.972084 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.78547 w_k1_ADM_s006
                                 + 0.844366 w_k1_AXP_s006
                                 + 0.867455 w_k1_DIS_s006
                                 + 0.990406 w_k1_FDX_s006
                                 + 0.844278 w_k1_IBM_s006
                                 + 0.986804 w_k1_INTC_s006
                                 + 0.796089 w_k1_KO_s006
                                 + 0.946279 w_k1_MAT_s006
                                 + 1.00522 w_k1_MSFT_s006
                                 + 0.841382 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.21566 w_k1_ADM_s006
                                 + 1.31881 w_k1_AXP_s006 + 1.1341 w_k1_DIS_s006
                                 + 1.218 w_k1_FDX_s006 + 1.22094 w_k1_IBM_s006
                                 + 1.09626 w_k1_INTC_s006
                                 + 1.12863 w_k1_KO_s006 + 1.02773 w_k1_MAT_s006
                                 + 1.02136 w_k1_MSFT_s006
                                 + 0.997871 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.59989 w_k1_ADM_s006
                                 + 1.56013 w_k1_AXP_s006
                                 + 1.51779 w_k1_DIS_s006
                                 + 1.42048 w_k1_FDX_s006
                                 + 1.25079 w_k1_IBM_s006
                                 + 1.13321 w_k1_INTC_s006
                                 + 1.33351 w_k1_KO_s006
                                 + 0.874997 w_k1_MAT_s006
                                 + 1.18722 w_k1_MSFT_s006
                                 + 1.11145 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.872066 w_k1_ADM_s006
                                 + 1.15072 w_k1_AXP_s006
                                 + 1.22259 w_k1_DIS_s006
                                 + 1.02106 w_k1_FDX_s006
                                 + 1.68624 w_k1_IBM_s006
                                 + 1.08038 w_k1_INTC_s006
                                 + 1.22025 w_k1_KO_s006 + 1.52544 w_k1_MAT_s006
                                 + 1.10782 w_k1_MSFT_s006
                                 + 1.03837 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.22196 w_k1_ADM_s007
                                 + 1.2935 w_k1_AXP_s007 + 1.15732 w_k1_DIS_s007
                                 + 1.12933 w_k1_FDX_s007
                                 + 1.14205 w_k1_IBM_s007
                                 + 1.10524 w_k1_INTC_s007 + 1.1689 w_k1_KO_s007
                                 + 1.08603 w_k1_MAT_s007
                                 + 1.11324 w_k1_MSFT_s007 + 1.1101 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.13635 w_k1_ADM_s007
                                 + 0.968151 w_k1_AXP_s007
                                 + 1.24915 w_k1_DIS_s007
                                 + 1.03689 w_k1_FDX_s007
                                 + 0.917242 w_k1_IBM_s007
                                 + 1.1778 w_k1_INTC_s007 + 1.16916 w_k1_KO_s007
                                 + 0.744751 w_k1_MAT_s007
                                 + 1.30743 w_k1_MSFT_s007
                                 + 0.952803 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.868879 w_k1_ADM_s007
                                 + 1.18215 w_k1_AXP_s007
                                 + 1.05431 w_k1_DIS_s007
                                 + 0.980242 w_k1_FDX_s007
                                 + 1.53944 w_k1_IBM_s007
                                 + 1.21314 w_k1_INTC_s007
                                 + 1.35473 w_k1_KO_s007 + 1.30187 w_k1_MAT_s007
                                 + 1.21543 w_k1_MSFT_s007
                                 + 1.03693 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.09388 w_k1_ADM_s007
                                 + 0.957895 w_k1_AXP_s007
                                 + 0.978339 w_k1_DIS_s007
                                 + 1.00736 w_k1_FDX_s007
                                 + 0.948525 w_k1_IBM_s007
                                 + 0.966124 w_k1_INTC_s007
                                 + 0.974172 w_k1_KO_s007
                                 + 0.775234 w_k1_MAT_s007
                                 + 1.14072 w_k1_MSFT_s007
                                 + 0.952924 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.10245 w_k1_ADM_s007
                                 + 1.61133 w_k1_AXP_s007
                                 + 1.67814 w_k1_DIS_s007
                                 + 1.35781 w_k1_FDX_s007
                                 + 1.23266 w_k1_IBM_s007 + 1.307 w_k1_INTC_s007
                                 + 1.32144 w_k1_KO_s007
                                 + 0.906915 w_k1_MAT_s007
                                 + 1.3441 w_k1_MSFT_s007 + 1.09121 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.749797 w_k1_ADM_s007
                                 + 0.841305 w_k1_AXP_s007
                                 + 0.813423 w_k1_DIS_s007
                                 + 0.787615 w_k1_FDX_s007
                                 + 0.937866 w_k1_IBM_s007
                                 + 0.747076 w_k1_INTC_s007
                                 + 0.840306 w_k1_KO_s007
                                 + 0.835003 w_k1_MAT_s007
                                 + 0.839271 w_k1_MSFT_s007
                                 + 0.843681 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.959671 w_k1_ADM_s007
                                 + 0.775941 w_k1_AXP_s007
                                 + 0.680494 w_k1_DIS_s007
                                 + 0.82828 w_k1_FDX_s007
                                 + 0.599571 w_k1_IBM_s007
                                 + 0.877738 w_k1_INTC_s007
                                 + 0.646799 w_k1_KO_s007
                                 + 0.363817 w_k1_MAT_s007
                                 + 0.805689 w_k1_MSFT_s007
                                 + 0.762564 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.863758 w_k1_ADM_s007
                                 + 0.989945 w_k1_AXP_s007
                                 + 0.974101 w_k1_DIS_s007
                                 + 0.925587 w_k1_FDX_s007
                                 + 0.783572 w_k1_IBM_s007
                                 + 0.999503 w_k1_INTC_s007
                                 + 0.816749 w_k1_KO_s007
                                 + 0.734476 w_k1_MAT_s007
                                 + 0.962899 w_k1_MSFT_s007
                                 + 0.849699 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.2132 w_k1_ADM_s010
                                 + 1.21747 w_k1_AXP_s010
                                 + 1.34783 w_k1_DIS_s010
                                 + 1.24299 w_k1_FDX_s010
                                 + 0.897002 w_k1_IBM_s010
                                 + 1.0032 w_k1_INTC_s010
                                 + 0.832496 w_k1_KO_s010
                                 + 0.877552 w_k1_MAT_s010
                                 + 0.984745 w_k1_MSFT_s010
                                 + 1.13461 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.848921 w_k1_ADM_s010
                                 + 0.725383 w_k1_AXP_s010
                                 + 0.683804 w_k1_DIS_s010
                                 + 0.761813 w_k1_FDX_s010
                                 + 1.19083 w_k1_IBM_s010
                                 + 0.948762 w_k1_INTC_s010
                                 + 1.01803 w_k1_KO_s010
                                 + 0.894543 w_k1_MAT_s010
                                 + 0.804129 w_k1_MSFT_s010
                                 + 0.985707 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.918015 w_k1_ADM_s010
                                 + 1.04027 w_k1_AXP_s010
                                 + 0.841879 w_k1_DIS_s010
                                 + 1.01513 w_k1_FDX_s010
                                 + 0.820535 w_k1_IBM_s010
                                 + 0.922263 w_k1_INTC_s010
                                 + 0.839966 w_k1_KO_s010
                                 + 0.906556 w_k1_MAT_s010
                                 + 0.94991 w_k1_MSFT_s010
                                 + 0.997208 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.770039 w_k1_ADM_s010
                                 + 0.673954 w_k1_AXP_s010
                                 + 0.623276 w_k1_DIS_s010
                                 + 0.776546 w_k1_FDX_s010
                                 + 0.733389 w_k1_IBM_s010
                                 + 0.705888 w_k1_INTC_s010
                                 + 0.542008 w_k1_KO_s010
                                 + 0.628734 w_k1_MAT_s010
                                 + 0.680112 w_k1_MSFT_s010
                                 + 0.861051 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.907229 w_k1_ADM_s010
                                 + 1.13184 w_k1_AXP_s010
                                 + 1.07349 w_k1_DIS_s010
                                 + 1.03571 w_k1_FDX_s010
                                 + 1.12394 w_k1_IBM_s010
                                 + 1.03539 w_k1_INTC_s010
                                 + 1.07482 w_k1_KO_s010 + 1.22839 w_k1_MAT_s010
                                 + 0.964665 w_k1_MSFT_s010
                                 + 1.09934 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.761715 w_k1_ADM_s010
                                 + 1.38492 w_k1_AXP_s010
                                 + 1.41704 w_k1_DIS_s010
                                 + 1.15308 w_k1_FDX_s010
                                 + 1.30609 w_k1_IBM_s010
                                 + 1.24861 w_k1_INTC_s010
                                 + 1.32432 w_k1_KO_s010 + 1.39244 w_k1_MAT_s010
                                 + 1.17761 w_k1_MSFT_s010
                                 + 1.13984 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.6858 w_k1_ADM_s010
                                 + 1.90701 w_k1_AXP_s010
                                 + 1.64964 w_k1_DIS_s010
                                 + 1.47209 w_k1_FDX_s010 + 1.1842 w_k1_IBM_s010
                                 + 1.37276 w_k1_INTC_s010 + 1.19 w_k1_KO_s010
                                 + 1.14907 w_k1_MAT_s010
                                 + 1.26834 w_k1_MSFT_s010
                                 + 1.25684 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.10827 w_k1_ADM_s010
                                 + 1.50504 w_k1_AXP_s010
                                 + 1.13966 w_k1_DIS_s010
                                 + 1.00868 w_k1_FDX_s010
                                 + 1.65459 w_k1_IBM_s010
                                 + 1.15416 w_k1_INTC_s010
                                 + 1.17611 w_k1_KO_s010 + 1.95165 w_k1_MAT_s010
                                 + 1.11075 w_k1_MSFT_s010
                                 + 1.15214 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.735529 w_k1_ADM_s011
                                 + 0.622395 w_k1_AXP_s011
                                 + 0.519156 w_k1_DIS_s011
                                 + 0.677541 w_k1_FDX_s011
                                 + 0.709477 w_k1_IBM_s011
                                 + 0.852808 w_k1_INTC_s011
                                 + 0.745853 w_k1_KO_s011
                                 + 0.831096 w_k1_MAT_s011
                                 + 0.750317 w_k1_MSFT_s011
                                 + 0.899237 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.785672 w_k1_ADM_s011
                                 + 0.64534 w_k1_AXP_s011
                                 + 0.678079 w_k1_DIS_s011
                                 + 0.737439 w_k1_FDX_s011
                                 + 1.05064 w_k1_IBM_s011
                                 + 0.799609 w_k1_INTC_s011
                                 + 0.872217 w_k1_KO_s011
                                 + 1.00133 w_k1_MAT_s011
                                 + 0.806074 w_k1_MSFT_s011
                                 + 0.872783 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.13306 w_k1_ADM_s011
                                 + 1.05598 w_k1_AXP_s011
                                 + 1.04728 w_k1_DIS_s011
                                 + 1.10958 w_k1_FDX_s011
                                 + 0.976952 w_k1_IBM_s011
                                 + 1.09337 w_k1_INTC_s011
                                 + 0.937351 w_k1_KO_s011
                                 + 1.02642 w_k1_MAT_s011
                                 + 0.944064 w_k1_MSFT_s011
                                 + 1.18383 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.20416 w_k1_ADM_s011
                                 + 1.03484 w_k1_AXP_s011
                                 + 0.779353 w_k1_DIS_s011
                                 + 0.966748 w_k1_FDX_s011
                                 + 1.2319 w_k1_IBM_s011
                                 + 0.916683 w_k1_INTC_s011
                                 + 1.0884 w_k1_KO_s011 + 1.53616 w_k1_MAT_s011
                                 + 0.838398 w_k1_MSFT_s011
                                 + 1.09371 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.38088 w_k1_ADM_s011
                                 + 1.47988 w_k1_AXP_s011
                                 + 1.41332 w_k1_DIS_s011
                                 + 1.30413 w_k1_FDX_s011
                                 + 1.27763 w_k1_IBM_s011
                                 + 1.04707 w_k1_INTC_s011
                                 + 1.30095 w_k1_KO_s011 + 1.63281 w_k1_MAT_s011
                                 + 1.16223 w_k1_MSFT_s011
                                 + 1.28118 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.946862 w_k1_ADM_s011
                                 + 0.863886 w_k1_AXP_s011
                                 + 0.831858 w_k1_DIS_s011
                                 + 0.9904 w_k1_FDX_s011
                                 + 0.887915 w_k1_IBM_s011
                                 + 0.981719 w_k1_INTC_s011
                                 + 0.921242 w_k1_KO_s011
                                 + 0.904342 w_k1_MAT_s011
                                 + 0.985046 w_k1_MSFT_s011
                                 + 0.976373 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.844334 w_k1_ADM_s011
                                 + 0.993095 w_k1_AXP_s011
                                 + 1.11706 w_k1_DIS_s011
                                 + 1.07199 w_k1_FDX_s011
                                 + 0.974433 w_k1_IBM_s011
                                 + 1.07891 w_k1_INTC_s011 + 0.9854 w_k1_KO_s011
                                 + 1.4303 w_k1_MAT_s011
                                 + 1.00225 w_k1_MSFT_s011
                                 + 1.31856 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.983693 w_k1_ADM_s011
                                 + 0.67485 w_k1_AXP_s011
                                 + 0.940067 w_k1_DIS_s011
                                 + 0.835217 w_k1_FDX_s011
                                 + 0.89704 w_k1_IBM_s011
                                 + 1.0331 w_k1_INTC_s011 + 1.03497 w_k1_KO_s011
                                 + 0.697603 w_k1_MAT_s011
                                 + 0.99194 w_k1_MSFT_s011
                                 + 1.02197 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.22409 w_k1_ADM_s012
                                 + 0.862425 w_k1_AXP_s012
                                 + 0.687751 w_k1_DIS_s012
                                 + 0.965624 w_k1_FDX_s012
                                 + 0.756003 w_k1_IBM_s012
                                 + 0.774909 w_k1_INTC_s012
                                 + 0.755562 w_k1_KO_s012
                                 + 0.734349 w_k1_MAT_s012
                                 + 0.856002 w_k1_MSFT_s012
                                 + 0.96385 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.82006 w_k1_ADM_s012
                                 + 0.743125 w_k1_AXP_s012
                                 + 0.702943 w_k1_DIS_s012
                                 + 0.80823 w_k1_FDX_s012
                                 + 1.11744 w_k1_IBM_s012
                                 + 0.662303 w_k1_INTC_s012
                                 + 0.982052 w_k1_KO_s012
                                 + 0.950788 w_k1_MAT_s012
                                 + 0.814453 w_k1_MSFT_s012
                                 + 0.963494 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.07025 w_k1_ADM_s012
                                 + 1.17848 w_k1_AXP_s012
                                 + 1.36549 w_k1_DIS_s012
                                 + 1.13738 w_k1_FDX_s012
                                 + 1.36507 w_k1_IBM_s012
                                 + 0.917191 w_k1_INTC_s012
                                 + 1.1536 w_k1_KO_s012 + 1.22741 w_k1_MAT_s012
                                 + 1.06804 w_k1_MSFT_s012
                                 + 1.22468 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.963972 w_k1_ADM_s012
                                 + 1.13149 w_k1_AXP_s012
                                 + 0.818165 w_k1_DIS_s012
                                 + 0.979819 w_k1_FDX_s012
                                 + 0.998285 w_k1_IBM_s012
                                 + 0.915909 w_k1_INTC_s012
                                 + 1.07044 w_k1_KO_s012 + 1.39375 w_k1_MAT_s012
                                 + 0.920475 w_k1_MSFT_s012
                                 + 1.05059 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.869296 w_k1_ADM_s012
                                 + 0.720159 w_k1_AXP_s012
                                 + 0.711396 w_k1_DIS_s012
                                 + 0.788031 w_k1_FDX_s012
                                 + 0.854673 w_k1_IBM_s012
                                 + 0.633438 w_k1_INTC_s012
                                 + 0.822602 w_k1_KO_s012
                                 + 0.922121 w_k1_MAT_s012
                                 + 0.772925 w_k1_MSFT_s012
                                 + 0.894695 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.77399 w_k1_ADM_s012
                                 + 0.570684 w_k1_AXP_s012
                                 + 0.600593 w_k1_DIS_s012
                                 + 0.659128 w_k1_FDX_s012
                                 + 0.742292 w_k1_IBM_s012
                                 + 0.539133 w_k1_INTC_s012
                                 + 0.761188 w_k1_KO_s012
                                 + 0.856439 w_k1_MAT_s012
                                 + 0.674532 w_k1_MSFT_s012
                                 + 0.78561 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.92396 w_k1_ADM_s012
                                 + 0.947174 w_k1_AXP_s012
                                 + 0.981386 w_k1_DIS_s012
                                 + 0.916145 w_k1_FDX_s012
                                 + 1.4871 w_k1_IBM_s012
                                 + 0.992561 w_k1_INTC_s012
                                 + 1.16903 w_k1_KO_s012 + 1.27866 w_k1_MAT_s012
                                 + 1.06074 w_k1_MSFT_s012
                                 + 0.995588 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.14662 w_k1_ADM_s012
                                 + 1.1647 w_k1_AXP_s012 + 1.04092 w_k1_DIS_s012
                                 + 1.22968 w_k1_FDX_s012
                                 + 0.854802 w_k1_IBM_s012
                                 + 0.993412 w_k1_INTC_s012
                                 + 0.952851 w_k1_KO_s012
                                 + 1.12754 w_k1_MAT_s012
                                 + 1.08292 w_k1_MSFT_s012
                                 + 1.09315 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.27187 w_k1_ADM_s013
                                 + 1.43812 w_k1_AXP_s013
                                 + 1.14297 w_k1_DIS_s013
                                 + 1.11026 w_k1_FDX_s013
                                 + 1.01561 w_k1_IBM_s013
                                 + 1.0314 w_k1_INTC_s013
                                 + 0.818755 w_k1_KO_s013
                                 + 0.970225 w_k1_MAT_s013
                                 + 0.921408 w_k1_MSFT_s013
                                 + 1.05113 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.21476 w_k1_ADM_s013
                                 + 1.03167 w_k1_AXP_s013
                                 + 1.30585 w_k1_DIS_s013
                                 + 1.05091 w_k1_FDX_s013
                                 + 1.05231 w_k1_IBM_s013
                                 + 1.19426 w_k1_INTC_s013
                                 + 1.14167 w_k1_KO_s013 + 0.74274 w_k1_MAT_s013
                                 + 1.02384 w_k1_MSFT_s013
                                 + 0.973922 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.19623 w_k1_ADM_s013
                                 + 0.919154 w_k1_AXP_s013
                                 + 0.979992 w_k1_DIS_s013
                                 + 1.01384 w_k1_FDX_s013
                                 + 0.68003 w_k1_IBM_s013
                                 + 1.1194 w_k1_INTC_s013
                                 + 0.919702 w_k1_KO_s013
                                 + 0.854927 w_k1_MAT_s013
                                 + 1.00068 w_k1_MSFT_s013
                                 + 0.961187 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.2899 w_k1_ADM_s013
                                 + 1.4258 w_k1_AXP_s013 + 1.43467 w_k1_DIS_s013
                                 + 1.1462 w_k1_FDX_s013 + 1.53113 w_k1_IBM_s013
                                 + 1.10115 w_k1_INTC_s013
                                 + 1.45058 w_k1_KO_s013 + 1.34555 w_k1_MAT_s013
                                 + 1.16883 w_k1_MSFT_s013
                                 + 1.16029 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.728397 w_k1_ADM_s013
                                 + 0.717499 w_k1_AXP_s013
                                 + 0.516902 w_k1_DIS_s013
                                 + 0.625956 w_k1_FDX_s013
                                 + 0.825033 w_k1_IBM_s013
                                 + 0.753913 w_k1_INTC_s013
                                 + 0.794369 w_k1_KO_s013
                                 + 0.699406 w_k1_MAT_s013
                                 + 0.71667 w_k1_MSFT_s013
                                 + 0.780171 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.906141 w_k1_ADM_s013
                                 + 0.955783 w_k1_AXP_s013
                                 + 0.71394 w_k1_DIS_s013
                                 + 0.803506 w_k1_FDX_s013
                                 + 0.785929 w_k1_IBM_s013
                                 + 0.843803 w_k1_INTC_s013
                                 + 0.83472 w_k1_KO_s013
                                 + 0.868868 w_k1_MAT_s013
                                 + 0.80755 w_k1_MSFT_s013
                                 + 0.835817 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.97041 w_k1_ADM_s013
                                 + 0.926516 w_k1_AXP_s013
                                 + 0.896534 w_k1_DIS_s013
                                 + 0.746134 w_k1_FDX_s013
                                 + 1.30594 w_k1_IBM_s013
                                 + 1.05314 w_k1_INTC_s013
                                 + 1.11292 w_k1_KO_s013 + 1.23628 w_k1_MAT_s013
                                 + 0.789341 w_k1_MSFT_s013
                                 + 0.946056 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.90603 w_k1_ADM_s013
                                 + 1.36998 w_k1_AXP_s013
                                 + 1.95012 w_k1_DIS_s013
                                 + 1.54584 w_k1_FDX_s013
                                 + 0.924515 w_k1_IBM_s013
                                 + 1.22277 w_k1_INTC_s013
                                 + 1.27138 w_k1_KO_s013 + 1.20509 w_k1_MAT_s013
                                 + 1.27224 w_k1_MSFT_s013
                                 + 1.36972 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.965679 w_k1_ADM_s014
                                 + 1.17249 w_k1_AXP_s014
                                 + 1.34227 w_k1_DIS_s014
                                 + 1.06545 w_k1_FDX_s014
                                 + 1.40546 w_k1_IBM_s014
                                 + 1.16093 w_k1_INTC_s014
                                 + 1.55753 w_k1_KO_s014 + 2.14139 w_k1_MAT_s014
                                 + 1.13922 w_k1_MSFT_s014
                                 + 1.18594 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.792775 w_k1_ADM_s014
                                 + 0.687304 w_k1_AXP_s014
                                 + 0.591141 w_k1_DIS_s014
                                 + 0.775765 w_k1_FDX_s014
                                 + 0.712744 w_k1_IBM_s014
                                 + 0.80835 w_k1_INTC_s014
                                 + 0.621232 w_k1_KO_s014
                                 + 0.674464 w_k1_MAT_s014
                                 + 0.715466 w_k1_MSFT_s014
                                 + 0.721548 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.638224 w_k1_ADM_s014
                                 + 0.730433 w_k1_AXP_s014
                                 + 0.734982 w_k1_DIS_s014
                                 + 0.749435 w_k1_FDX_s014
                                 + 1.1533 w_k1_IBM_s014
                                 + 0.923621 w_k1_INTC_s014
                                 + 1.08597 w_k1_KO_s014 + 1.37036 w_k1_MAT_s014
                                 + 0.795884 w_k1_MSFT_s014
                                 + 0.900114 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.22048 w_k1_ADM_s014
                                 + 1.23697 w_k1_AXP_s014
                                 + 1.70676 w_k1_DIS_s014
                                 + 1.38733 w_k1_FDX_s014
                                 + 1.14796 w_k1_IBM_s014
                                 + 1.22888 w_k1_INTC_s014
                                 + 1.23413 w_k1_KO_s014 + 1.15513 w_k1_MAT_s014
                                 + 1.38099 w_k1_MSFT_s014
                                 + 1.23045 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.792515 w_k1_ADM_s014
                                 + 0.826687 w_k1_AXP_s014
                                 + 0.884508 w_k1_DIS_s014
                                 + 0.981921 w_k1_FDX_s014
                                 + 0.668472 w_k1_IBM_s014
                                 + 0.968713 w_k1_INTC_s014
                                 + 0.847538 w_k1_KO_s014
                                 + 1.12405 w_k1_MAT_s014
                                 + 0.989993 w_k1_MSFT_s014
                                 + 0.910243 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.886981 w_k1_ADM_s014
                                 + 1.07092 w_k1_AXP_s014 + 1.1486 w_k1_DIS_s014
                                 + 1.01899 w_k1_FDX_s014
                                 + 1.03258 w_k1_IBM_s014
                                 + 1.02327 w_k1_INTC_s014
                                 + 1.03415 w_k1_KO_s014 + 1.19037 w_k1_MAT_s014
                                 + 1.04817 w_k1_MSFT_s014
                                 + 0.913108 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.872793 w_k1_ADM_s014
                                 + 0.861962 w_k1_AXP_s014
                                 + 1.01121 w_k1_DIS_s014
                                 + 1.01674 w_k1_FDX_s014
                                 + 0.816955 w_k1_IBM_s014
                                 + 0.904472 w_k1_INTC_s014
                                 + 1.0061 w_k1_KO_s014 + 1.07003 w_k1_MAT_s014
                                 + 1.0533 w_k1_MSFT_s014
                                 + 0.987179 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.07706 w_k1_ADM_s014
                                 + 0.795397 w_k1_AXP_s014
                                 + 1.03448 w_k1_DIS_s014
                                 + 0.894869 w_k1_FDX_s014
                                 + 1.50328 w_k1_IBM_s014
                                 + 0.963519 w_k1_INTC_s014
                                 + 1.36281 w_k1_KO_s014 + 2.01919 w_k1_MAT_s014
                                 + 1.03547 w_k1_MSFT_s014
                                 + 1.11251 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.808248 w_k1_ADM_s015
                                 + 0.822774 w_k1_AXP_s015
                                 + 0.72117 w_k1_DIS_s015
                                 + 0.85291 w_k1_FDX_s015
                                 + 0.943792 w_k1_IBM_s015
                                 + 0.942718 w_k1_INTC_s015
                                 + 0.822533 w_k1_KO_s015
                                 + 1.09595 w_k1_MAT_s015
                                 + 0.846159 w_k1_MSFT_s015
                                 + 0.849131 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.29504 w_k1_ADM_s015
                                 + 1.08834 w_k1_AXP_s015
                                 + 1.27375 w_k1_DIS_s015
                                 + 1.24689 w_k1_FDX_s015 + 1.0096 w_k1_IBM_s015
                                 + 1.1723 w_k1_INTC_s015 + 1.21475 w_k1_KO_s015
                                 + 1.27958 w_k1_MAT_s015 + 1.179 w_k1_MSFT_s015
                                 + 1.19918 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.868503 w_k1_ADM_s015
                                 + 1.23631 w_k1_AXP_s015
                                 + 0.739634 w_k1_DIS_s015
                                 + 0.957483 w_k1_FDX_s015
                                 + 0.704104 w_k1_IBM_s015
                                 + 1.04477 w_k1_INTC_s015
                                 + 0.767716 w_k1_KO_s015
                                 + 0.970691 w_k1_MAT_s015
                                 + 0.809249 w_k1_MSFT_s015
                                 + 0.959301 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.760631 w_k1_ADM_s015
                                 + 0.932968 w_k1_AXP_s015
                                 + 0.84097 w_k1_DIS_s015
                                 + 0.846498 w_k1_FDX_s015
                                 + 1.37349 w_k1_IBM_s015
                                 + 0.927752 w_k1_INTC_s015
                                 + 1.01452 w_k1_KO_s015 + 1.49489 w_k1_MAT_s015
                                 + 0.957628 w_k1_MSFT_s015
                                 + 0.911862 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.679706 w_k1_ADM_s015
                                 + 0.746613 w_k1_AXP_s015
                                 + 0.625299 w_k1_DIS_s015
                                 + 0.71965 w_k1_FDX_s015
                                 + 0.667021 w_k1_IBM_s015
                                 + 0.770088 w_k1_INTC_s015
                                 + 0.76227 w_k1_KO_s015
                                 + 0.948586 w_k1_MAT_s015
                                 + 0.773845 w_k1_MSFT_s015
                                 + 0.773775 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.86565 w_k1_ADM_s015
                                 + 0.86148 w_k1_AXP_s015
                                 + 0.750591 w_k1_DIS_s015
                                 + 0.994429 w_k1_FDX_s015
                                 + 0.705736 w_k1_IBM_s015
                                 + 1.13939 w_k1_INTC_s015
                                 + 0.708986 w_k1_KO_s015
                                 + 0.546407 w_k1_MAT_s015
                                 + 0.952852 w_k1_MSFT_s015
                                 + 0.856047 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.00562 w_k1_ADM_s015
                                 + 1.41825 w_k1_AXP_s015
                                 + 1.28432 w_k1_DIS_s015
                                 + 1.17271 w_k1_FDX_s015
                                 + 1.55638 w_k1_IBM_s015
                                 + 1.32809 w_k1_INTC_s015
                                 + 1.53277 w_k1_KO_s015 + 2.60765 w_k1_MAT_s015
                                 + 1.39941 w_k1_MSFT_s015
                                 + 1.22248 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.896149 w_k1_ADM_s015
                                 + 1.15407 w_k1_AXP_s015
                                 + 1.03583 w_k1_DIS_s015
                                 + 0.963495 w_k1_FDX_s015
                                 + 1.03498 w_k1_IBM_s015
                                 + 1.0178 w_k1_INTC_s015 + 1.10031 w_k1_KO_s015
                                 + 1.47732 w_k1_MAT_s015
                                 + 1.10728 w_k1_MSFT_s015
                                 + 1.05518 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.17719 w_k1_ADM_s016
                                 + 0.701908 w_k1_AXP_s016
                                 + 0.810662 w_k1_DIS_s016
                                 + 0.790023 w_k1_FDX_s016
                                 + 0.959898 w_k1_IBM_s016
                                 + 0.747279 w_k1_INTC_s016
                                 + 0.931652 w_k1_KO_s016
                                 + 0.83667 w_k1_MAT_s016
                                 + 0.914685 w_k1_MSFT_s016
                                 + 0.897477 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.21134 w_k1_ADM_s016
                                 + 0.864659 w_k1_AXP_s016
                                 + 1.13808 w_k1_DIS_s016
                                 + 1.02833 w_k1_FDX_s016
                                 + 0.842587 w_k1_IBM_s016
                                 + 1.04551 w_k1_INTC_s016
                                 + 0.893267 w_k1_KO_s016
                                 + 0.72421 w_k1_MAT_s016
                                 + 1.12457 w_k1_MSFT_s016
                                 + 0.953484 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.712129 w_k1_ADM_s016
                                 + 0.903572 w_k1_AXP_s016
                                 + 0.781327 w_k1_DIS_s016
                                 + 0.862814 w_k1_FDX_s016
                                 + 1.24293 w_k1_IBM_s016
                                 + 0.964671 w_k1_INTC_s016
                                 + 0.890314 w_k1_KO_s016
                                 + 1.01429 w_k1_MAT_s016
                                 + 0.937172 w_k1_MSFT_s016
                                 + 0.797639 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.26252 w_k1_ADM_s016
                                 + 1.00829 w_k1_AXP_s016
                                 + 0.938934 w_k1_DIS_s016
                                 + 1.05055 w_k1_FDX_s016
                                 + 0.936942 w_k1_IBM_s016
                                 + 1.05224 w_k1_INTC_s016
                                 + 0.84184 w_k1_KO_s016
                                 + 0.718868 w_k1_MAT_s016
                                 + 0.977047 w_k1_MSFT_s016
                                 + 0.908832 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.34737 w_k1_ADM_s016
                                 + 1.25874 w_k1_AXP_s016
                                 + 1.36315 w_k1_DIS_s016
                                 + 1.28912 w_k1_FDX_s016
                                 + 1.15429 w_k1_IBM_s016
                                 + 0.975742 w_k1_INTC_s016
                                 + 1.29588 w_k1_KO_s016 + 1.4053 w_k1_MAT_s016
                                 + 1.34522 w_k1_MSFT_s016
                                 + 1.19738 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.34065 w_k1_ADM_s016
                                 + 1.75391 w_k1_AXP_s016
                                 + 2.03125 w_k1_DIS_s016
                                 + 1.46647 w_k1_FDX_s016
                                 + 1.71282 w_k1_IBM_s016
                                 + 1.26753 w_k1_INTC_s016
                                 + 1.51227 w_k1_KO_s016 + 1.80372 w_k1_MAT_s016
                                 + 1.40693 w_k1_MSFT_s016
                                 + 1.48929 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.17963 w_k1_ADM_s016
                                 + 0.967343 w_k1_AXP_s016
                                 + 1.07056 w_k1_DIS_s016
                                 + 1.17792 w_k1_FDX_s016
                                 + 0.703354 w_k1_IBM_s016
                                 + 1.15425 w_k1_INTC_s016
                                 + 0.867041 w_k1_KO_s016
                                 + 0.581399 w_k1_MAT_s016
                                 + 1.15883 w_k1_MSFT_s016
                                 + 0.994808 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.983564 w_k1_ADM_s016
                                 + 0.714609 w_k1_AXP_s016
                                 + 0.588345 w_k1_DIS_s016
                                 + 0.746947 w_k1_FDX_s016
                                 + 0.725635 w_k1_IBM_s016
                                 + 0.661211 w_k1_INTC_s016
                                 + 0.678333 w_k1_KO_s016
                                 + 0.668483 w_k1_MAT_s016
                                 + 0.678145 w_k1_MSFT_s016
                                 + 0.811265 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.19692 w_k1_ADM_s017
                                 + 0.872042 w_k1_AXP_s017
                                 + 0.889862 w_k1_DIS_s017
                                 + 0.994407 w_k1_FDX_s017
                                 + 1.00706 w_k1_IBM_s017
                                 + 0.957064 w_k1_INTC_s017
                                 + 1.06352 w_k1_KO_s017 + 1.14576 w_k1_MAT_s017
                                 + 1.13189 w_k1_MSFT_s017 + 1.0913 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.01278 w_k1_ADM_s017
                                 + 0.871511 w_k1_AXP_s017
                                 + 0.773651 w_k1_DIS_s017
                                 + 0.896848 w_k1_FDX_s017
                                 + 0.743621 w_k1_IBM_s017
                                 + 0.848877 w_k1_INTC_s017
                                 + 0.941281 w_k1_KO_s017
                                 + 1.15986 w_k1_MAT_s017
                                 + 0.939604 w_k1_MSFT_s017
                                 + 0.954075 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.833379 w_k1_ADM_s017
                                 + 0.731613 w_k1_AXP_s017
                                 + 0.716377 w_k1_DIS_s017
                                 + 0.773158 w_k1_FDX_s017
                                 + 1.0323 w_k1_IBM_s017
                                 + 0.914915 w_k1_INTC_s017
                                 + 0.908653 w_k1_KO_s017
                                 + 1.11339 w_k1_MAT_s017
                                 + 0.879325 w_k1_MSFT_s017
                                 + 0.869696 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.32888 w_k1_ADM_s017
                                 + 1.64978 w_k1_AXP_s017 + 1.6198 w_k1_DIS_s017
                                 + 1.36708 w_k1_FDX_s017
                                 + 1.33411 w_k1_IBM_s017
                                 + 1.15246 w_k1_INTC_s017
                                 + 1.33132 w_k1_KO_s017 + 1.54225 w_k1_MAT_s017
                                 + 1.23621 w_k1_MSFT_s017
                                 + 1.20167 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.27128 w_k1_ADM_s017
                                 + 1.3619 w_k1_AXP_s017 + 1.17153 w_k1_DIS_s017
                                 + 1.26348 w_k1_FDX_s017
                                 + 0.993848 w_k1_IBM_s017
                                 + 1.20139 w_k1_INTC_s017 + 1.1519 w_k1_KO_s017
                                 + 1.11092 w_k1_MAT_s017
                                 + 1.14988 w_k1_MSFT_s017
                                 + 1.19829 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.19201 w_k1_ADM_s017
                                 + 1.45367 w_k1_AXP_s017
                                 + 1.17835 w_k1_DIS_s017
                                 + 1.11366 w_k1_FDX_s017
                                 + 1.22757 w_k1_IBM_s017
                                 + 1.3778 w_k1_INTC_s017 + 1.34219 w_k1_KO_s017
                                 + 1.58653 w_k1_MAT_s017
                                 + 1.12385 w_k1_MSFT_s017
                                 + 1.23143 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.11619 w_k1_ADM_s017
                                 + 1.26171 w_k1_AXP_s017
                                 + 1.05512 w_k1_DIS_s017
                                 + 1.05768 w_k1_FDX_s017
                                 + 1.06779 w_k1_IBM_s017
                                 + 1.13118 w_k1_INTC_s017
                                 + 1.20861 w_k1_KO_s017 + 1.24764 w_k1_MAT_s017
                                 + 0.978974 w_k1_MSFT_s017
                                 + 1.09958 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.857679 w_k1_ADM_s017
                                 + 0.954903 w_k1_AXP_s017
                                 + 0.864281 w_k1_DIS_s017
                                 + 0.889954 w_k1_FDX_s017
                                 + 1.34631 w_k1_IBM_s017
                                 + 1.12448 w_k1_INTC_s017
                                 + 1.32326 w_k1_KO_s017 + 1.55782 w_k1_MAT_s017
                                 + 0.862754 w_k1_MSFT_s017
                                 + 1.05146 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.28781 w_k1_ADM_s100
                                 + 1.1337 w_k1_AXP_s100 + 1.15041 w_k1_DIS_s100
                                 + 1.14589 w_k1_FDX_s100
                                 + 0.677807 w_k1_IBM_s100
                                 + 0.874987 w_k1_INTC_s100
                                 + 0.963516 w_k1_KO_s100
                                 + 0.999927 w_k1_MAT_s100
                                 + 1.15749 w_k1_MSFT_s100
                                 + 1.00736 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.24336 w_k1_ADM_s100
                                 + 1.16927 w_k1_AXP_s100 + 1.5894 w_k1_DIS_s100
                                 + 1.35137 w_k1_FDX_s100
                                 + 1.23793 w_k1_IBM_s100
                                 + 0.988237 w_k1_INTC_s100
                                 + 1.33963 w_k1_KO_s100 + 1.24357 w_k1_MAT_s100
                                 + 1.30711 w_k1_MSFT_s100
                                 + 1.12192 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.952708 w_k1_ADM_s100
                                 + 0.735333 w_k1_AXP_s100
                                 + 0.769851 w_k1_DIS_s100
                                 + 0.818335 w_k1_FDX_s100
                                 + 0.840511 w_k1_IBM_s100
                                 + 0.819238 w_k1_INTC_s100
                                 + 0.913783 w_k1_KO_s100
                                 + 0.74743 w_k1_MAT_s100
                                 + 0.907757 w_k1_MSFT_s100
                                 + 0.894732 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.01286 w_k1_ADM_s100
                                 + 0.828765 w_k1_AXP_s100
                                 + 0.916003 w_k1_DIS_s100
                                 + 0.84435 w_k1_FDX_s100
                                 + 1.11546 w_k1_IBM_s100
                                 + 0.885837 w_k1_INTC_s100
                                 + 1.06624 w_k1_KO_s100
                                 + 0.899567 w_k1_MAT_s100
                                 + 0.980024 w_k1_MSFT_s100
                                 + 0.966141 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.960117 w_k1_ADM_s100
                                 + 0.827194 w_k1_AXP_s100
                                 + 0.79596 w_k1_DIS_s100
                                 + 0.89906 w_k1_FDX_s100
                                 + 0.608809 w_k1_IBM_s100
                                 + 1.0643 w_k1_INTC_s100
                                 + 0.824631 w_k1_KO_s100
                                 + 0.455502 w_k1_MAT_s100
                                 + 0.976085 w_k1_MSFT_s100
                                 + 0.868945 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.612911 w_k1_ADM_s100
                                 + 0.59973 w_k1_AXP_s100
                                 + 0.602024 w_k1_DIS_s100
                                 + 0.624722 w_k1_FDX_s100
                                 + 0.730455 w_k1_IBM_s100
                                 + 0.880935 w_k1_INTC_s100
                                 + 0.665631 w_k1_KO_s100
                                 + 0.659205 w_k1_MAT_s100
                                 + 0.829881 w_k1_MSFT_s100
                                 + 0.733918 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.09088 w_k1_ADM_s100
                                 + 1.1309 w_k1_AXP_s100 + 1.04809 w_k1_DIS_s100
                                 + 0.999127 w_k1_FDX_s100
                                 + 0.903387 w_k1_IBM_s100
                                 + 0.896428 w_k1_INTC_s100
                                 + 0.796424 w_k1_KO_s100
                                 + 1.12036 w_k1_MAT_s100
                                 + 1.14418 w_k1_MSFT_s100
                                 + 0.894404 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.98166 w_k1_ADM_s100
                                 + 1.43266 w_k1_AXP_s100
                                 + 1.41651 w_k1_DIS_s100
                                 + 1.18373 w_k1_FDX_s100
                                 + 1.05407 w_k1_IBM_s100
                                 + 1.05187 w_k1_INTC_s100
                                 + 1.04847 w_k1_KO_s100 + 1.51361 w_k1_MAT_s100
                                 + 1.29798 w_k1_MSFT_s100
                                 + 1.13044 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.841316 w_k1_ADM_s101
                                 + 0.948696 w_k1_AXP_s101
                                 + 1.12032 w_k1_DIS_s101
                                 + 0.992682 w_k1_FDX_s101
                                 + 1.4193 w_k1_IBM_s101
                                 + 0.887054 w_k1_INTC_s101
                                 + 1.24958 w_k1_KO_s101 + 1.08209 w_k1_MAT_s101
                                 + 1.01978 w_k1_MSFT_s101
                                 + 1.18729 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.912197 w_k1_ADM_s101
                                 + 1.16261 w_k1_AXP_s101 + 1.0249 w_k1_DIS_s101
                                 + 1.22238 w_k1_FDX_s101
                                 + 0.908497 w_k1_IBM_s101
                                 + 1.31848 w_k1_INTC_s101
                                 + 1.09975 w_k1_KO_s101 + 1.09763 w_k1_MAT_s101
                                 + 1.12886 w_k1_MSFT_s101
                                 + 1.10629 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.846684 w_k1_ADM_s101
                                 + 0.699445 w_k1_AXP_s101
                                 + 0.728363 w_k1_DIS_s101
                                 + 0.791241 w_k1_FDX_s101
                                 + 0.915532 w_k1_IBM_s101
                                 + 0.792568 w_k1_INTC_s101
                                 + 0.853338 w_k1_KO_s101
                                 + 0.816253 w_k1_MAT_s101
                                 + 0.883479 w_k1_MSFT_s101
                                 + 1.02256 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.16826 w_k1_ADM_s101
                                 + 1.03986 w_k1_AXP_s101
                                 + 1.03906 w_k1_DIS_s101
                                 + 1.02952 w_k1_FDX_s101
                                 + 1.14837 w_k1_IBM_s101
                                 + 1.02228 w_k1_INTC_s101
                                 + 1.29502 w_k1_KO_s101 + 1.23654 w_k1_MAT_s101
                                 + 1.1517 w_k1_MSFT_s101 + 1.0875 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.946596 w_k1_ADM_s101
                                 + 0.913187 w_k1_AXP_s101
                                 + 0.866354 w_k1_DIS_s101
                                 + 1.10113 w_k1_FDX_s101
                                 + 0.703342 w_k1_IBM_s101
                                 + 0.957227 w_k1_INTC_s101
                                 + 0.842015 w_k1_KO_s101
                                 + 0.795349 w_k1_MAT_s101
                                 + 1.15618 w_k1_MSFT_s101
                                 + 0.919828 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.09843 w_k1_ADM_s101
                                 + 1.46394 w_k1_AXP_s101
                                 + 1.34572 w_k1_DIS_s101
                                 + 1.21931 w_k1_FDX_s101
                                 + 1.46523 w_k1_IBM_s101
                                 + 1.09194 w_k1_INTC_s101 + 1.293 w_k1_KO_s101
                                 + 1.50179 w_k1_MAT_s101
                                 + 1.10141 w_k1_MSFT_s101
                                 + 1.36905 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.891843 w_k1_ADM_s101
                                 + 0.794341 w_k1_AXP_s101
                                 + 1.05477 w_k1_DIS_s101
                                 + 1.01905 w_k1_FDX_s101
                                 + 0.945789 w_k1_IBM_s101
                                 + 1.11406 w_k1_INTC_s101
                                 + 1.13016 w_k1_KO_s101 + 0.81132 w_k1_MAT_s101
                                 + 1.11487 w_k1_MSFT_s101
                                 + 0.991392 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.693342 w_k1_ADM_s101
                                 + 0.674953 w_k1_AXP_s101
                                 + 0.576868 w_k1_DIS_s101
                                 + 0.737324 w_k1_FDX_s101
                                 + 0.708247 w_k1_IBM_s101
                                 + 0.780409 w_k1_INTC_s101
                                 + 0.723421 w_k1_KO_s101
                                 + 0.790378 w_k1_MAT_s101
                                 + 0.853356 w_k1_MSFT_s101
                                 + 0.903554 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.32332 w_k1_ADM_s102
                                 + 0.615681 w_k1_AXP_s102
                                 + 0.666599 w_k1_DIS_s102
                                 + 0.931059 w_k1_FDX_s102
                                 + 0.690101 w_k1_IBM_s102
                                 + 0.743127 w_k1_INTC_s102
                                 + 0.832705 w_k1_KO_s102
                                 + 0.550712 w_k1_MAT_s102
                                 + 0.836466 w_k1_MSFT_s102
                                 + 0.851073 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.43054 w_k1_ADM_s102
                                 + 1.40519 w_k1_AXP_s102
                                 + 1.23692 w_k1_DIS_s102
                                 + 1.35401 w_k1_FDX_s102
                                 + 1.41967 w_k1_IBM_s102
                                 + 0.963283 w_k1_INTC_s102
                                 + 1.48839 w_k1_KO_s102 + 1.78675 w_k1_MAT_s102
                                 + 1.30886 w_k1_MSFT_s102 + 1.2737 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.3377 w_k1_ADM_s102
                                 + 1.31743 w_k1_AXP_s102
                                 + 1.28252 w_k1_DIS_s102
                                 + 1.28057 w_k1_FDX_s102
                                 + 0.862315 w_k1_IBM_s102
                                 + 1.14026 w_k1_INTC_s102
                                 + 1.13792 w_k1_KO_s102 + 1.16534 w_k1_MAT_s102
                                 + 1.26768 w_k1_MSFT_s102
                                 + 1.15459 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.781154 w_k1_ADM_s102
                                 + 1.0186 w_k1_AXP_s102
                                 + 0.893333 w_k1_DIS_s102
                                 + 0.978551 w_k1_FDX_s102
                                 + 1.34792 w_k1_IBM_s102
                                 + 0.934258 w_k1_INTC_s102
                                 + 0.939616 w_k1_KO_s102
                                 + 1.43648 w_k1_MAT_s102
                                 + 0.927053 w_k1_MSFT_s102
                                 + 1.02834 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.14146 w_k1_ADM_s102
                                 + 1.03938 w_k1_AXP_s102
                                 + 0.875079 w_k1_DIS_s102
                                 + 0.967679 w_k1_FDX_s102
                                 + 1.13527 w_k1_IBM_s102
                                 + 0.845673 w_k1_INTC_s102
                                 + 0.978171 w_k1_KO_s102
                                 + 1.13569 w_k1_MAT_s102
                                 + 0.85472 w_k1_MSFT_s102
                                 + 0.937018 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.36706 w_k1_ADM_s102
                                 + 0.792029 w_k1_AXP_s102
                                 + 0.915599 w_k1_DIS_s102
                                 + 1.04139 w_k1_FDX_s102
                                 + 0.88431 w_k1_IBM_s102
                                 + 0.832211 w_k1_INTC_s102
                                 + 0.971352 w_k1_KO_s102
                                 + 0.681522 w_k1_MAT_s102
                                 + 0.907587 w_k1_MSFT_s102
                                 + 1.00092 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.00535 w_k1_ADM_s102
                                 + 1.15705 w_k1_AXP_s102
                                 + 1.16419 w_k1_DIS_s102
                                 + 1.16163 w_k1_FDX_s102
                                 + 1.02218 w_k1_IBM_s102
                                 + 1.21829 w_k1_INTC_s102
                                 + 1.29671 w_k1_KO_s102 + 1.11806 w_k1_MAT_s102
                                 + 1.20465 w_k1_MSFT_s102
                                 + 1.16399 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.67667 w_k1_ADM_s102
                                 + 1.05537 w_k1_AXP_s102
                                 + 1.24056 w_k1_DIS_s102
                                 + 1.06971 w_k1_FDX_s102
                                 + 1.67492 w_k1_IBM_s102
                                 + 0.995603 w_k1_INTC_s102
                                 + 1.35748 w_k1_KO_s102 + 2.01479 w_k1_MAT_s102
                                 + 1.08009 w_k1_MSFT_s102
                                 + 1.09395 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.818635 w_k1_ADM_s103
                                 + 0.889156 w_k1_AXP_s103
                                 + 0.69746 w_k1_DIS_s103
                                 + 0.786772 w_k1_FDX_s103
                                 + 0.978207 w_k1_IBM_s103
                                 + 0.888022 w_k1_INTC_s103
                                 + 0.930576 w_k1_KO_s103
                                 + 0.767595 w_k1_MAT_s103
                                 + 0.689958 w_k1_MSFT_s103
                                 + 0.926488 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.835101 w_k1_ADM_s103
                                 + 0.869641 w_k1_AXP_s103
                                 + 0.901716 w_k1_DIS_s103
                                 + 0.754254 w_k1_FDX_s103
                                 + 1.41711 w_k1_IBM_s103
                                 + 0.893721 w_k1_INTC_s103
                                 + 1.21926 w_k1_KO_s103 + 1.07486 w_k1_MAT_s103
                                 + 0.784811 w_k1_MSFT_s103
                                 + 0.952097 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.11344 w_k1_ADM_s103
                                 + 1.28479 w_k1_AXP_s103
                                 + 1.14244 w_k1_DIS_s103
                                 + 0.938526 w_k1_FDX_s103
                                 + 1.70693 w_k1_IBM_s103
                                 + 1.33286 w_k1_INTC_s103
                                 + 1.59905 w_k1_KO_s103 + 1.86542 w_k1_MAT_s103
                                 + 1.04192 w_k1_MSFT_s103
                                 + 1.36205 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.20081 w_k1_ADM_s103
                                 + 1.07772 w_k1_AXP_s103
                                 + 1.26719 w_k1_DIS_s103
                                 + 1.09142 w_k1_FDX_s103
                                 + 0.858488 w_k1_IBM_s103
                                 + 1.09218 w_k1_INTC_s103
                                 + 1.07859 w_k1_KO_s103
                                 + 0.666793 w_k1_MAT_s103
                                 + 1.09318 w_k1_MSFT_s103
                                 + 1.02423 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.084 w_k1_ADM_s103
                                 + 1.20653 w_k1_AXP_s103
                                 + 1.11061 w_k1_DIS_s103
                                 + 1.08972 w_k1_FDX_s103
                                 + 1.21422 w_k1_IBM_s103
                                 + 1.11033 w_k1_INTC_s103
                                 + 1.29673 w_k1_KO_s103 + 1.46764 w_k1_MAT_s103
                                 + 0.929999 w_k1_MSFT_s103
                                 + 1.15954 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.071 w_k1_ADM_s103
                                 + 1.59443 w_k1_AXP_s103
                                 + 1.64664 w_k1_DIS_s103
                                 + 1.39274 w_k1_FDX_s103
                                 + 1.00567 w_k1_IBM_s103
                                 + 1.54977 w_k1_INTC_s103
                                 + 1.46409 w_k1_KO_s103 + 1.18184 w_k1_MAT_s103
                                 + 1.20425 w_k1_MSFT_s103
                                 + 1.36524 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.844794 w_k1_ADM_s103
                                 + 0.863821 w_k1_AXP_s103
                                 + 0.833457 w_k1_DIS_s103
                                 + 0.805756 w_k1_FDX_s103
                                 + 1.21389 w_k1_IBM_s103
                                 + 1.15869 w_k1_INTC_s103
                                 + 1.15997 w_k1_KO_s103 + 1.11249 w_k1_MAT_s103
                                 + 0.987665 w_k1_MSFT_s103
                                 + 0.982872 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.769231 w_k1_ADM_s103
                                 + 0.832452 w_k1_AXP_s103
                                 + 0.801477 w_k1_DIS_s103
                                 + 0.825738 w_k1_FDX_s103
                                 + 0.70808 w_k1_IBM_s103
                                 + 1.07634 w_k1_INTC_s103
                                 + 0.911678 w_k1_KO_s103
                                 + 0.709607 w_k1_MAT_s103
                                 + 0.877456 w_k1_MSFT_s103
                                 + 0.972049 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.711952 w_k1_ADM_s104
                                 + 1.17195 w_k1_AXP_s104
                                 + 1.15678 w_k1_DIS_s104
                                 + 1.14652 w_k1_FDX_s104 + 1.2318 w_k1_IBM_s104
                                 + 1.45012 w_k1_INTC_s104
                                 + 1.19688 w_k1_KO_s104 + 1.62447 w_k1_MAT_s104
                                 + 1.15682 w_k1_MSFT_s104
                                 + 1.21617 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.622304 w_k1_ADM_s104
                                 + 0.704356 w_k1_AXP_s104
                                 + 0.812139 w_k1_DIS_s104
                                 + 0.91994 w_k1_FDX_s104
                                 + 0.69476 w_k1_IBM_s104
                                 + 0.969084 w_k1_INTC_s104
                                 + 0.762133 w_k1_KO_s104
                                 + 0.778627 w_k1_MAT_s104
                                 + 0.97482 w_k1_MSFT_s104
                                 + 0.789525 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.11283 w_k1_ADM_s104
                                 + 0.843833 w_k1_AXP_s104
                                 + 1.25366 w_k1_DIS_s104
                                 + 1.20278 w_k1_FDX_s104
                                 + 1.02272 w_k1_IBM_s104
                                 + 1.05854 w_k1_INTC_s104
                                 + 1.09641 w_k1_KO_s104
                                 + 0.697069 w_k1_MAT_s104
                                 + 1.18971 w_k1_MSFT_s104
                                 + 1.10526 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.810715 w_k1_ADM_s104
                                 + 0.603442 w_k1_AXP_s104
                                 + 0.814654 w_k1_DIS_s104
                                 + 0.829921 w_k1_FDX_s104
                                 + 0.906107 w_k1_IBM_s104
                                 + 1.09558 w_k1_INTC_s104
                                 + 1.01041 w_k1_KO_s104 + 0.64189 w_k1_MAT_s104
                                 + 1.07212 w_k1_MSFT_s104
                                 + 0.915854 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.11945 w_k1_ADM_s104
                                 + 1.0766 w_k1_AXP_s104 + 1.00095 w_k1_DIS_s104
                                 + 1.08395 w_k1_FDX_s104
                                 + 0.896547 w_k1_IBM_s104
                                 + 1.11847 w_k1_INTC_s104
                                 + 0.98948 w_k1_KO_s104 + 1.00072 w_k1_MAT_s104
                                 + 1.13871 w_k1_MSFT_s104
                                 + 1.09052 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.817484 w_k1_ADM_s104
                                 + 0.793475 w_k1_AXP_s104
                                 + 0.810524 w_k1_DIS_s104
                                 + 0.859075 w_k1_FDX_s104
                                 + 1.14498 w_k1_IBM_s104
                                 + 0.945839 w_k1_INTC_s104
                                 + 1.02628 w_k1_KO_s104
                                 + 0.951012 w_k1_MAT_s104
                                 + 0.99915 w_k1_MSFT_s104
                                 + 1.01616 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.43959 w_k1_ADM_s104
                                 + 1.53147 w_k1_AXP_s104
                                 + 1.55848 w_k1_DIS_s104 + 1.5861 w_k1_FDX_s104
                                 + 1.03367 w_k1_IBM_s104
                                 + 1.59845 w_k1_INTC_s104
                                 + 1.12571 w_k1_KO_s104 + 1.07305 w_k1_MAT_s104
                                 + 1.49874 w_k1_MSFT_s104
                                 + 1.23024 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.964668 w_k1_ADM_s104
                                 + 1.22954 w_k1_AXP_s104
                                 + 1.40506 w_k1_DIS_s104
                                 + 1.05502 w_k1_FDX_s104
                                 + 1.53407 w_k1_IBM_s104
                                 + 1.1844 w_k1_INTC_s104 + 1.44502 w_k1_KO_s104
                                 + 1.30809 w_k1_MAT_s104
                                 + 1.17155 w_k1_MSFT_s104
                                 + 1.28824 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.00702 w_k1_ADM_s105
                                 + 1.09547 w_k1_AXP_s105
                                 + 0.868702 w_k1_DIS_s105
                                 + 0.966906 w_k1_FDX_s105
                                 + 0.731781 w_k1_IBM_s105
                                 + 1.00763 w_k1_INTC_s105
                                 + 0.737534 w_k1_KO_s105
                                 + 0.608449 w_k1_MAT_s105
                                 + 0.868924 w_k1_MSFT_s105
                                 + 0.83851 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.957592 w_k1_ADM_s105
                                 + 1.10395 w_k1_AXP_s105
                                 + 1.35107 w_k1_DIS_s105 + 1.0148 w_k1_FDX_s105
                                 + 1.5365 w_k1_IBM_s105
                                 + 1.14369 w_k1_INTC_s105
                                 + 1.17452 w_k1_KO_s105 + 1.5156 w_k1_MAT_s105
                                 + 0.972714 w_k1_MSFT_s105
                                 + 1.19727 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.713987 w_k1_ADM_s105
                                 + 0.796069 w_k1_AXP_s105
                                 + 0.728924 w_k1_DIS_s105
                                 + 0.778758 w_k1_FDX_s105
                                 + 0.881701 w_k1_IBM_s105
                                 + 0.819027 w_k1_INTC_s105
                                 + 0.775583 w_k1_KO_s105 + 1.11 w_k1_MAT_s105
                                 + 0.754364 w_k1_MSFT_s105
                                 + 0.878709 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.866162 w_k1_ADM_s105
                                 + 0.939701 w_k1_AXP_s105
                                 + 1.08303 w_k1_DIS_s105
                                 + 1.01411 w_k1_FDX_s105
                                 + 1.02564 w_k1_IBM_s105
                                 + 1.1193 w_k1_INTC_s105 + 1.01541 w_k1_KO_s105
                                 + 1.0406 w_k1_MAT_s105
                                 + 0.990025 w_k1_MSFT_s105
                                 + 0.924847 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.39914 w_k1_ADM_s105
                                 + 1.26871 w_k1_AXP_s105
                                 + 1.60423 w_k1_DIS_s105
                                 + 1.28958 w_k1_FDX_s105
                                 + 1.25741 w_k1_IBM_s105
                                 + 0.921657 w_k1_INTC_s105
                                 + 1.26643 w_k1_KO_s105 + 1.49257 w_k1_MAT_s105
                                 + 1.11143 w_k1_MSFT_s105
                                 + 1.27547 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.14088 w_k1_ADM_s105
                                 + 1.34076 w_k1_AXP_s105
                                 + 1.11104 w_k1_DIS_s105
                                 + 1.18498 w_k1_FDX_s105
                                 + 0.909997 w_k1_IBM_s105
                                 + 1.23015 w_k1_INTC_s105
                                 + 1.03506 w_k1_KO_s105 + 1.17791 w_k1_MAT_s105
                                 + 1.17958 w_k1_MSFT_s105
                                 + 1.06138 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.707781 w_k1_ADM_s105
                                 + 0.549444 w_k1_AXP_s105
                                 + 0.788511 w_k1_DIS_s105
                                 + 0.637661 w_k1_FDX_s105
                                 + 0.891352 w_k1_IBM_s105
                                 + 0.824158 w_k1_INTC_s105
                                 + 0.891396 w_k1_KO_s105
                                 + 0.664714 w_k1_MAT_s105
                                 + 0.768335 w_k1_MSFT_s105
                                 + 0.811537 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.901908 w_k1_ADM_s105
                                 + 0.900321 w_k1_AXP_s105
                                 + 0.933513 w_k1_DIS_s105
                                 + 0.847858 w_k1_FDX_s105
                                 + 0.915874 w_k1_IBM_s105
                                 + 0.942044 w_k1_INTC_s105
                                 + 0.846131 w_k1_KO_s105
                                 + 0.915281 w_k1_MAT_s105
                                 + 0.8956 w_k1_MSFT_s105 + 0.89997 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.22942 w_k1_ADM_s106
                                 + 1.37468 w_k1_AXP_s106 + 1.1493 w_k1_DIS_s106
                                 + 1.27897 w_k1_FDX_s106
                                 + 0.991855 w_k1_IBM_s106
                                 + 1.19057 w_k1_INTC_s106
                                 + 0.956817 w_k1_KO_s106
                                 + 1.06276 w_k1_MAT_s106
                                 + 1.08724 w_k1_MSFT_s106
                                 + 1.08452 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.763388 w_k1_ADM_s106
                                 + 0.819531 w_k1_AXP_s106
                                 + 0.810836 w_k1_DIS_s106
                                 + 0.876406 w_k1_FDX_s106
                                 + 0.900728 w_k1_IBM_s106
                                 + 0.940174 w_k1_INTC_s106
                                 + 0.773325 w_k1_KO_s106
                                 + 0.903347 w_k1_MAT_s106
                                 + 0.990226 w_k1_MSFT_s106
                                 + 0.854512 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.719943 w_k1_ADM_s106
                                 + 0.741228 w_k1_AXP_s106
                                 + 0.621705 w_k1_DIS_s106
                                 + 0.734085 w_k1_FDX_s106
                                 + 0.611255 w_k1_IBM_s106
                                 + 0.898253 w_k1_INTC_s106
                                 + 0.692398 w_k1_KO_s106
                                 + 0.691051 w_k1_MAT_s106
                                 + 0.775245 w_k1_MSFT_s106
                                 + 0.849649 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.792771 w_k1_ADM_s106
                                 + 1.06713 w_k1_AXP_s106
                                 + 1.10953 w_k1_DIS_s106
                                 + 0.902609 w_k1_FDX_s106
                                 + 1.14608 w_k1_IBM_s106
                                 + 0.919799 w_k1_INTC_s106
                                 + 0.998423 w_k1_KO_s106
                                 + 1.37979 w_k1_MAT_s106
                                 + 0.882025 w_k1_MSFT_s106
                                 + 0.891558 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.870184 w_k1_ADM_s106
                                 + 0.829841 w_k1_AXP_s106
                                 + 0.679872 w_k1_DIS_s106
                                 + 0.779548 w_k1_FDX_s106
                                 + 0.770542 w_k1_IBM_s106
                                 + 0.792637 w_k1_INTC_s106
                                 + 0.781027 w_k1_KO_s106
                                 + 0.996749 w_k1_MAT_s106
                                 + 0.711061 w_k1_MSFT_s106
                                 + 0.845221 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.14542 w_k1_ADM_s106
                                 + 1.33752 w_k1_AXP_s106 + 1.3809 w_k1_DIS_s106
                                 + 1.06906 w_k1_FDX_s106
                                 + 1.50931 w_k1_IBM_s106
                                 + 1.10474 w_k1_INTC_s106
                                 + 1.39758 w_k1_KO_s106 + 1.86306 w_k1_MAT_s106
                                 + 1.16007 w_k1_MSFT_s106
                                 + 1.02928 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.941703 w_k1_ADM_s106
                                 + 1.11157 w_k1_AXP_s106
                                 + 1.59491 w_k1_DIS_s106
                                 + 1.16587 w_k1_FDX_s106
                                 + 1.08004 w_k1_IBM_s106
                                 + 1.15876 w_k1_INTC_s106 + 1.1341 w_k1_KO_s106
                                 + 1.23084 w_k1_MAT_s106
                                 + 1.29738 w_k1_MSFT_s106
                                 + 1.11639 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.784236 w_k1_ADM_s106
                                 + 1.03631 w_k1_AXP_s106
                                 + 1.05388 w_k1_DIS_s106
                                 + 1.14502 w_k1_FDX_s106
                                 + 0.706483 w_k1_IBM_s106
                                 + 1.11315 w_k1_INTC_s106
                                 + 0.919988 w_k1_KO_s106
                                 + 0.93296 w_k1_MAT_s106
                                 + 1.15218 w_k1_MSFT_s106
                                 + 1.01979 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.871831 w_k1_ADM_s107
                                 + 0.917702 w_k1_AXP_s107
                                 + 0.736252 w_k1_DIS_s107
                                 + 0.997866 w_k1_FDX_s107
                                 + 0.834277 w_k1_IBM_s107
                                 + 0.77645 w_k1_INTC_s107
                                 + 0.807376 w_k1_KO_s107 + 1.0988 w_k1_MAT_s107
                                 + 0.84301 w_k1_MSFT_s107
                                 + 1.01061 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.22165 w_k1_ADM_s107
                                 + 0.990312 w_k1_AXP_s107
                                 + 0.929163 w_k1_DIS_s107
                                 + 1.08834 w_k1_FDX_s107
                                 + 0.67129 w_k1_IBM_s107
                                 + 0.771827 w_k1_INTC_s107
                                 + 0.837416 w_k1_KO_s107
                                 + 0.804727 w_k1_MAT_s107
                                 + 1.10337 w_k1_MSFT_s107
                                 + 1.09401 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.00282 w_k1_ADM_s107
                                 + 0.926918 w_k1_AXP_s107
                                 + 1.23373 w_k1_DIS_s107
                                 + 1.05787 w_k1_FDX_s107
                                 + 1.07596 w_k1_IBM_s107
                                 + 1.10588 w_k1_INTC_s107
                                 + 1.16269 w_k1_KO_s107 + 1.0999 w_k1_MAT_s107
                                 + 1.1287 w_k1_MSFT_s107 + 1.08763 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.16434 w_k1_ADM_s107
                                 + 1.12574 w_k1_AXP_s107
                                 + 1.03729 w_k1_DIS_s107
                                 + 0.982335 w_k1_FDX_s107
                                 + 1.05528 w_k1_IBM_s107
                                 + 0.918815 w_k1_INTC_s107
                                 + 1.11616 w_k1_KO_s107 + 1.35522 w_k1_MAT_s107
                                 + 0.998208 w_k1_MSFT_s107
                                 + 1.10585 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.14998 w_k1_ADM_s107
                                 + 0.921874 w_k1_AXP_s107
                                 + 1.32001 w_k1_DIS_s107
                                 + 0.965358 w_k1_FDX_s107
                                 + 1.57118 w_k1_IBM_s107
                                 + 1.00926 w_k1_INTC_s107
                                 + 1.45369 w_k1_KO_s107 + 1.57282 w_k1_MAT_s107
                                 + 1.22855 w_k1_MSFT_s107
                                 + 1.22645 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.770196 w_k1_ADM_s107
                                 + 0.640159 w_k1_AXP_s107
                                 + 0.669251 w_k1_DIS_s107
                                 + 0.656497 w_k1_FDX_s107
                                 + 0.907279 w_k1_IBM_s107
                                 + 0.600518 w_k1_INTC_s107
                                 + 0.780955 w_k1_KO_s107
                                 + 1.00623 w_k1_MAT_s107
                                 + 0.746875 w_k1_MSFT_s107
                                 + 0.942689 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.17752 w_k1_ADM_s107
                                 + 1.13081 w_k1_AXP_s107
                                 + 1.75019 w_k1_DIS_s107
                                 + 1.30194 w_k1_FDX_s107
                                 + 1.32163 w_k1_IBM_s107
                                 + 1.3667 w_k1_INTC_s107 + 1.62095 w_k1_KO_s107
                                 + 1.67322 w_k1_MAT_s107
                                 + 1.50508 w_k1_MSFT_s107
                                 + 1.25974 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.58051 w_k1_ADM_s107
                                 + 1.73466 w_k1_AXP_s107
                                 + 1.68991 w_k1_DIS_s107
                                 + 1.44849 w_k1_FDX_s107
                                 + 1.01432 w_k1_IBM_s107
                                 + 1.15357 w_k1_INTC_s107
                                 + 1.27603 w_k1_KO_s107 + 1.75129 w_k1_MAT_s107
                                 + 1.37868 w_k1_MSFT_s107
                                 + 1.22652 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.931266 w_k1_ADM_s110
                                 + 0.860825 w_k1_AXP_s110
                                 + 0.852271 w_k1_DIS_s110
                                 + 0.910294 w_k1_FDX_s110
                                 + 0.818219 w_k1_IBM_s110
                                 + 1.03455 w_k1_INTC_s110
                                 + 0.878604 w_k1_KO_s110
                                 + 0.60882 w_k1_MAT_s110
                                 + 0.892217 w_k1_MSFT_s110
                                 + 0.930431 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.41233 w_k1_ADM_s110
                                 + 1.16053 w_k1_AXP_s110 + 1.0893 w_k1_DIS_s110
                                 + 1.18157 w_k1_FDX_s110
                                 + 1.08672 w_k1_IBM_s110
                                 + 1.07651 w_k1_INTC_s110
                                 + 1.04763 w_k1_KO_s110 + 1.07105 w_k1_MAT_s110
                                 + 1.08022 w_k1_MSFT_s110
                                 + 1.00241 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.11027 w_k1_ADM_s110
                                 + 1.45509 w_k1_AXP_s110
                                 + 1.64694 w_k1_DIS_s110
                                 + 1.26213 w_k1_FDX_s110
                                 + 1.19033 w_k1_IBM_s110
                                 + 1.12806 w_k1_INTC_s110
                                 + 1.37558 w_k1_KO_s110 + 1.27259 w_k1_MAT_s110
                                 + 1.24619 w_k1_MSFT_s110
                                 + 1.18563 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.796215 w_k1_ADM_s110
                                 + 0.653851 w_k1_AXP_s110
                                 + 0.570562 w_k1_DIS_s110
                                 + 0.738331 w_k1_FDX_s110
                                 + 0.731005 w_k1_IBM_s110
                                 + 0.75181 w_k1_INTC_s110
                                 + 0.739672 w_k1_KO_s110
                                 + 0.54778 w_k1_MAT_s110
                                 + 0.757797 w_k1_MSFT_s110
                                 + 0.728283 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.25154 w_k1_ADM_s110
                                 + 1.36522 w_k1_AXP_s110
                                 + 0.94654 w_k1_DIS_s110 + 1.1126 w_k1_FDX_s110
                                 + 1.00406 w_k1_IBM_s110
                                 + 1.09814 w_k1_INTC_s110
                                 + 0.980185 w_k1_KO_s110
                                 + 1.06763 w_k1_MAT_s110
                                 + 0.890275 w_k1_MSFT_s110
                                 + 0.993864 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.801416 w_k1_ADM_s110
                                 + 0.999851 w_k1_AXP_s110
                                 + 1.0028 w_k1_DIS_s110
                                 + 0.826633 w_k1_FDX_s110
                                 + 1.29481 w_k1_IBM_s110
                                 + 1.10295 w_k1_INTC_s110
                                 + 1.23764 w_k1_KO_s110 + 1.17549 w_k1_MAT_s110
                                 + 0.882308 w_k1_MSFT_s110
                                 + 0.994761 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.937683 w_k1_ADM_s110
                                 + 0.639996 w_k1_AXP_s110
                                 + 0.763032 w_k1_DIS_s110
                                 + 0.749435 w_k1_FDX_s110
                                 + 0.917766 w_k1_IBM_s110
                                 + 0.811216 w_k1_INTC_s110
                                 + 0.820765 w_k1_KO_s110
                                 + 0.885853 w_k1_MAT_s110
                                 + 0.851557 w_k1_MSFT_s110
                                 + 0.894111 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.986892 w_k1_ADM_s110
                                 + 0.911259 w_k1_AXP_s110
                                 + 1.11875 w_k1_DIS_s110 + 1.0616 w_k1_FDX_s110
                                 + 0.982385 w_k1_IBM_s110
                                 + 1.12047 w_k1_INTC_s110
                                 + 1.17771 w_k1_KO_s110 + 1.07066 w_k1_MAT_s110
                                 + 1.11018 w_k1_MSFT_s110
                                 + 1.05162 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.922872 w_k1_ADM_s111
                                 + 0.923955 w_k1_AXP_s111
                                 + 1.03457 w_k1_DIS_s111
                                 + 0.982254 w_k1_FDX_s111
                                 + 1.02944 w_k1_IBM_s111
                                 + 1.07372 w_k1_INTC_s111
                                 + 1.01947 w_k1_KO_s111
                                 + 0.882524 w_k1_MAT_s111
                                 + 1.02754 w_k1_MSFT_s111
                                 + 1.01574 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.14203 w_k1_ADM_s111
                                 + 1.15561 w_k1_AXP_s111
                                 + 1.45317 w_k1_DIS_s111
                                 + 1.25902 w_k1_FDX_s111
                                 + 0.935811 w_k1_IBM_s111
                                 + 1.03495 w_k1_INTC_s111
                                 + 1.17817 w_k1_KO_s111
                                 + 0.925242 w_k1_MAT_s111
                                 + 1.21028 w_k1_MSFT_s111
                                 + 1.20868 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.668034 w_k1_ADM_s111
                                 + 0.753257 w_k1_AXP_s111
                                 + 0.725819 w_k1_DIS_s111
                                 + 0.706004 w_k1_FDX_s111
                                 + 1.31782 w_k1_IBM_s111
                                 + 0.85699 w_k1_INTC_s111
                                 + 0.936796 w_k1_KO_s111
                                 + 1.08113 w_k1_MAT_s111
                                 + 0.805251 w_k1_MSFT_s111
                                 + 0.970835 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.730428 w_k1_ADM_s111
                                 + 0.823678 w_k1_AXP_s111
                                 + 0.700166 w_k1_DIS_s111
                                 + 0.738565 w_k1_FDX_s111
                                 + 0.905975 w_k1_IBM_s111
                                 + 0.808529 w_k1_INTC_s111
                                 + 0.848974 w_k1_KO_s111
                                 + 0.849216 w_k1_MAT_s111
                                 + 0.780488 w_k1_MSFT_s111
                                 + 0.885955 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.10842 w_k1_ADM_s111
                                 + 1.56371 w_k1_AXP_s111 + 1.6997 w_k1_DIS_s111
                                 + 1.24026 w_k1_FDX_s111
                                 + 1.62127 w_k1_IBM_s111
                                 + 1.24301 w_k1_INTC_s111
                                 + 1.47744 w_k1_KO_s111 + 1.5438 w_k1_MAT_s111
                                 + 1.28588 w_k1_MSFT_s111
                                 + 1.13245 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.10422 w_k1_ADM_s111
                                 + 1.03629 w_k1_AXP_s111
                                 + 1.28215 w_k1_DIS_s111
                                 + 0.992692 w_k1_FDX_s111
                                 + 1.47759 w_k1_IBM_s111
                                 + 0.915868 w_k1_INTC_s111
                                 + 1.22671 w_k1_KO_s111 + 1.24625 w_k1_MAT_s111
                                 + 1.09564 w_k1_MSFT_s111
                                 + 1.10528 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.859136 w_k1_ADM_s111
                                 + 0.990982 w_k1_AXP_s111
                                 + 1.35553 w_k1_DIS_s111
                                 + 1.24267 w_k1_FDX_s111
                                 + 0.64953 w_k1_IBM_s111
                                 + 1.15629 w_k1_INTC_s111
                                 + 0.932729 w_k1_KO_s111
                                 + 0.721996 w_k1_MAT_s111
                                 + 1.12104 w_k1_MSFT_s111
                                 + 0.976561 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.719389 w_k1_ADM_s111
                                 + 0.655914 w_k1_AXP_s111
                                 + 0.634712 w_k1_DIS_s111
                                 + 0.785681 w_k1_FDX_s111
                                 + 0.616124 w_k1_IBM_s111
                                 + 0.886269 w_k1_INTC_s111
                                 + 0.667676 w_k1_KO_s111
                                 + 0.501541 w_k1_MAT_s111
                                 + 0.849135 w_k1_MSFT_s111
                                 + 0.819681 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.10177 w_k1_ADM_s112
                                 + 0.855095 w_k1_AXP_s112
                                 + 1.30803 w_k1_DIS_s112
                                 + 1.16605 w_k1_FDX_s112
                                 + 0.861319 w_k1_IBM_s112
                                 + 1.26069 w_k1_INTC_s112
                                 + 1.17209 w_k1_KO_s112
                                 + 0.893087 w_k1_MAT_s112
                                 + 1.25939 w_k1_MSFT_s112
                                 + 1.04096 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.04248 w_k1_ADM_s112
                                 + 0.790505 w_k1_AXP_s112
                                 + 0.705619 w_k1_DIS_s112
                                 + 0.85537 w_k1_FDX_s112
                                 + 0.731381 w_k1_IBM_s112
                                 + 0.966418 w_k1_INTC_s112
                                 + 0.84455 w_k1_KO_s112
                                 + 0.844811 w_k1_MAT_s112
                                 + 0.836464 w_k1_MSFT_s112
                                 + 0.857626 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.18975 w_k1_ADM_s112
                                 + 1.31772 w_k1_AXP_s112
                                 + 1.05771 w_k1_DIS_s112
                                 + 1.11269 w_k1_FDX_s112
                                 + 0.939859 w_k1_IBM_s112
                                 + 1.31349 w_k1_INTC_s112
                                 + 1.03897 w_k1_KO_s112 + 1.13341 w_k1_MAT_s112
                                 + 1.10964 w_k1_MSFT_s112
                                 + 0.988324 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.997111 w_k1_ADM_s112
                                 + 0.99669 w_k1_AXP_s112
                                 + 1.01676 w_k1_DIS_s112
                                 + 1.01782 w_k1_FDX_s112
                                 + 0.672058 w_k1_IBM_s112
                                 + 1.08041 w_k1_INTC_s112
                                 + 0.948623 w_k1_KO_s112
                                 + 0.878303 w_k1_MAT_s112
                                 + 1.07153 w_k1_MSFT_s112
                                 + 0.957385 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.999561 w_k1_ADM_s112
                                 + 0.835962 w_k1_AXP_s112
                                 + 0.921127 w_k1_DIS_s112
                                 + 0.893991 w_k1_FDX_s112
                                 + 1.36306 w_k1_IBM_s112
                                 + 0.981657 w_k1_INTC_s112
                                 + 1.29149 w_k1_KO_s112 + 1.1788 w_k1_MAT_s112
                                 + 0.949434 w_k1_MSFT_s112
                                 + 1.00002 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.962064 w_k1_ADM_s112
                                 + 1.05414 w_k1_AXP_s112 + 1.0854 w_k1_DIS_s112
                                 + 0.981097 w_k1_FDX_s112
                                 + 1.12535 w_k1_IBM_s112
                                 + 1.13065 w_k1_INTC_s112
                                 + 1.25452 w_k1_KO_s112 + 1.0294 w_k1_MAT_s112
                                 + 1.08574 w_k1_MSFT_s112
                                 + 1.05646 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.00325 w_k1_ADM_s112
                                 + 0.648823 w_k1_AXP_s112
                                 + 0.737079 w_k1_DIS_s112
                                 + 0.733094 w_k1_FDX_s112
                                 + 0.838444 w_k1_IBM_s112
                                 + 0.904771 w_k1_INTC_s112
                                 + 0.845047 w_k1_KO_s112
                                 + 0.614842 w_k1_MAT_s112
                                 + 0.817661 w_k1_MSFT_s112
                                 + 0.835081 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.68792 w_k1_ADM_s112
                                 + 1.4284 w_k1_AXP_s112 + 1.73731 w_k1_DIS_s112
                                 + 1.35111 w_k1_FDX_s112
                                 + 1.42565 w_k1_IBM_s112
                                 + 1.49985 w_k1_INTC_s112
                                 + 1.52426 w_k1_KO_s112 + 1.55958 w_k1_MAT_s112
                                 + 1.35866 w_k1_MSFT_s112
                                 + 1.22579 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.689811 w_k1_ADM_s113
                                 + 0.818225 w_k1_AXP_s113
                                 + 0.806286 w_k1_DIS_s113
                                 + 0.809932 w_k1_FDX_s113
                                 + 0.790929 w_k1_IBM_s113
                                 + 0.964427 w_k1_INTC_s113
                                 + 0.879808 w_k1_KO_s113
                                 + 0.841431 w_k1_MAT_s113
                                 + 0.851083 w_k1_MSFT_s113
                                 + 0.964092 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.577076 w_k1_ADM_s113
                                 + 0.506949 w_k1_AXP_s113
                                 + 0.537699 w_k1_DIS_s113
                                 + 0.635199 w_k1_FDX_s113
                                 + 0.793768 w_k1_IBM_s113
                                 + 0.773775 w_k1_INTC_s113
                                 + 0.709763 w_k1_KO_s113
                                 + 0.576816 w_k1_MAT_s113
                                 + 0.760572 w_k1_MSFT_s113
                                 + 0.849721 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.89209 w_k1_ADM_s113
                                 + 1.03279 w_k1_AXP_s113
                                 + 1.03497 w_k1_DIS_s113
                                 + 1.06525 w_k1_FDX_s113
                                 + 1.08794 w_k1_IBM_s113
                                 + 1.11624 w_k1_INTC_s113
                                 + 1.14294 w_k1_KO_s113 + 1.20991 w_k1_MAT_s113
                                 + 1.06786 w_k1_MSFT_s113
                                 + 1.15097 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.29146 w_k1_ADM_s113
                                 + 1.37273 w_k1_AXP_s113
                                 + 1.28944 w_k1_DIS_s113
                                 + 1.32543 w_k1_FDX_s113
                                 + 0.776048 w_k1_IBM_s113
                                 + 1.06326 w_k1_INTC_s113
                                 + 1.04688 w_k1_KO_s113
                                 + 0.780394 w_k1_MAT_s113
                                 + 1.13273 w_k1_MSFT_s113
                                 + 1.07236 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.809802 w_k1_ADM_s113
                                 + 1.01241 w_k1_AXP_s113
                                 + 0.991192 w_k1_DIS_s113
                                 + 0.883339 w_k1_FDX_s113
                                 + 1.55692 w_k1_IBM_s113
                                 + 1.14324 w_k1_INTC_s113
                                 + 1.24163 w_k1_KO_s113 + 1.51454 w_k1_MAT_s113
                                 + 0.916663 w_k1_MSFT_s113
                                 + 1.13619 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.677045 w_k1_ADM_s113
                                 + 0.820982 w_k1_AXP_s113
                                 + 0.594925 w_k1_DIS_s113
                                 + 0.742396 w_k1_FDX_s113
                                 + 0.740722 w_k1_IBM_s113
                                 + 0.84672 w_k1_INTC_s113
                                 + 0.688689 w_k1_KO_s113
                                 + 0.774093 w_k1_MAT_s113
                                 + 0.705119 w_k1_MSFT_s113
                                 + 0.82133 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.14274 w_k1_ADM_s113
                                 + 2.02386 w_k1_AXP_s113
                                 + 1.49606 w_k1_DIS_s113
                                 + 1.34745 w_k1_FDX_s113
                                 + 1.63857 w_k1_IBM_s113
                                 + 1.19927 w_k1_INTC_s113 + 1.3719 w_k1_KO_s113
                                 + 2.2914 w_k1_MAT_s113
                                 + 1.21753 w_k1_MSFT_s113
                                 + 1.27916 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.930495 w_k1_ADM_s113
                                 + 0.992137 w_k1_AXP_s113
                                 + 0.802963 w_k1_DIS_s113
                                 + 0.948602 w_k1_FDX_s113
                                 + 0.782538 w_k1_IBM_s113
                                 + 1.07785 w_k1_INTC_s113
                                 + 0.896136 w_k1_KO_s113
                                 + 0.802093 w_k1_MAT_s113
                                 + 0.944833 w_k1_MSFT_s113
                                 + 1.01542 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.908504 w_k1_ADM_s114
                                 + 0.900176 w_k1_AXP_s114
                                 + 1.02452 w_k1_DIS_s114
                                 + 0.928835 w_k1_FDX_s114
                                 + 0.951303 w_k1_IBM_s114
                                 + 0.957636 w_k1_INTC_s114
                                 + 1.02728 w_k1_KO_s114 + 1.05659 w_k1_MAT_s114
                                 + 1.00222 w_k1_MSFT_s114
                                 + 1.01232 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.960891 w_k1_ADM_s114
                                 + 1.10424 w_k1_AXP_s114
                                 + 1.01357 w_k1_DIS_s114
                                 + 0.856098 w_k1_FDX_s114
                                 + 1.5839 w_k1_IBM_s114
                                 + 1.03103 w_k1_INTC_s114
                                 + 1.24887 w_k1_KO_s114 + 1.49842 w_k1_MAT_s114
                                 + 0.918818 w_k1_MSFT_s114
                                 + 1.07774 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.14058 w_k1_ADM_s114
                                 + 0.809096 w_k1_AXP_s114
                                 + 0.823788 w_k1_DIS_s114
                                 + 0.917447 w_k1_FDX_s114
                                 + 0.74409 w_k1_IBM_s114
                                 + 1.0026 w_k1_INTC_s114
                                 + 0.848907 w_k1_KO_s114
                                 + 0.50148 w_k1_MAT_s114
                                 + 0.874278 w_k1_MSFT_s114
                                 + 0.966325 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.989071 w_k1_ADM_s114
                                 + 1.03694 w_k1_AXP_s114
                                 + 0.705204 w_k1_DIS_s114
                                 + 0.912613 w_k1_FDX_s114
                                 + 0.682685 w_k1_IBM_s114
                                 + 0.734985 w_k1_INTC_s114
                                 + 0.670662 w_k1_KO_s114
                                 + 0.670675 w_k1_MAT_s114
                                 + 0.739859 w_k1_MSFT_s114
                                 + 0.791556 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.24966 w_k1_ADM_s114
                                 + 1.05121 w_k1_AXP_s114 + 1.2842 w_k1_DIS_s114
                                 + 1.16503 w_k1_FDX_s114
                                 + 0.891691 w_k1_IBM_s114
                                 + 1.28336 w_k1_INTC_s114
                                 + 1.07187 w_k1_KO_s114
                                 + 0.831334 w_k1_MAT_s114
                                 + 1.18199 w_k1_MSFT_s114
                                 + 1.02153 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.40359 w_k1_ADM_s114
                                 + 1.37365 w_k1_AXP_s114
                                 + 1.76233 w_k1_DIS_s114
                                 + 1.32843 w_k1_FDX_s114
                                 + 1.47004 w_k1_IBM_s114
                                 + 1.25283 w_k1_INTC_s114 + 1.5371 w_k1_KO_s114
                                 + 1.72793 w_k1_MAT_s114
                                 + 1.39747 w_k1_MSFT_s114
                                 + 1.26183 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.908378 w_k1_ADM_s114
                                 + 0.790217 w_k1_AXP_s114
                                 + 0.629222 w_k1_DIS_s114
                                 + 0.683886 w_k1_FDX_s114
                                 + 0.891904 w_k1_IBM_s114
                                 + 0.86526 w_k1_INTC_s114
                                 + 0.890973 w_k1_KO_s114
                                 + 0.793473 w_k1_MAT_s114
                                 + 0.692484 w_k1_MSFT_s114
                                 + 0.848294 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.02805 w_k1_ADM_s114
                                 + 1.24361 w_k1_AXP_s114
                                 + 1.13757 w_k1_DIS_s114
                                 + 1.02647 w_k1_FDX_s114
                                 + 1.33417 w_k1_IBM_s114
                                 + 0.984252 w_k1_INTC_s114
                                 + 1.13713 w_k1_KO_s114 + 1.36181 w_k1_MAT_s114
                                 + 0.968999 w_k1_MSFT_s114
                                 + 1.02509 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.03064 w_k1_ADM_s115
                                 + 1.11767 w_k1_AXP_s115
                                 + 1.18983 w_k1_DIS_s115
                                 + 1.12287 w_k1_FDX_s115
                                 + 0.824328 w_k1_IBM_s115
                                 + 1.20424 w_k1_INTC_s115
                                 + 0.961016 w_k1_KO_s115
                                 + 0.844001 w_k1_MAT_s115
                                 + 1.07485 w_k1_MSFT_s115
                                 + 1.08699 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.599765 w_k1_ADM_s115
                                 + 1.08273 w_k1_AXP_s115
                                 + 1.03655 w_k1_DIS_s115
                                 + 0.937943 w_k1_FDX_s115
                                 + 1.63033 w_k1_IBM_s115
                                 + 1.21658 w_k1_INTC_s115
                                 + 1.32642 w_k1_KO_s115 + 1.63328 w_k1_MAT_s115
                                 + 0.967594 w_k1_MSFT_s115
                                 + 1.06675 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.880924 w_k1_ADM_s115
                                 + 0.923471 w_k1_AXP_s115
                                 + 0.900423 w_k1_DIS_s115
                                 + 1.05136 w_k1_FDX_s115
                                 + 0.662438 w_k1_IBM_s115
                                 + 1.01789 w_k1_INTC_s115
                                 + 0.75406 w_k1_KO_s115
                                 + 0.624507 w_k1_MAT_s115
                                 + 0.904364 w_k1_MSFT_s115
                                 + 0.962938 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.17781 w_k1_ADM_s115
                                 + 1.21803 w_k1_AXP_s115 + 1.5344 w_k1_DIS_s115
                                 + 1.39811 w_k1_FDX_s115
                                 + 1.01185 w_k1_IBM_s115
                                 + 1.37384 w_k1_INTC_s115
                                 + 1.28897 w_k1_KO_s115 + 1.18559 w_k1_MAT_s115
                                 + 1.27495 w_k1_MSFT_s115
                                 + 1.30475 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.951714 w_k1_ADM_s115
                                 + 0.984944 w_k1_AXP_s115
                                 + 1.06716 w_k1_DIS_s115
                                 + 1.02217 w_k1_FDX_s115
                                 + 1.04467 w_k1_IBM_s115
                                 + 1.0481 w_k1_INTC_s115 + 1.06579 w_k1_KO_s115
                                 + 1.04275 w_k1_MAT_s115
                                 + 1.01905 w_k1_MSFT_s115
                                 + 1.12313 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.0055 w_k1_ADM_s115
                                 + 1.67012 w_k1_AXP_s115
                                 + 1.42912 w_k1_DIS_s115
                                 + 1.31365 w_k1_FDX_s115
                                 + 1.36635 w_k1_IBM_s115
                                 + 1.33478 w_k1_INTC_s115
                                 + 1.31012 w_k1_KO_s115 + 1.7177 w_k1_MAT_s115
                                 + 1.1729 w_k1_MSFT_s115 + 1.27884 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.703248 w_k1_ADM_s115
                                 + 0.659174 w_k1_AXP_s115
                                 + 0.744184 w_k1_DIS_s115
                                 + 0.738069 w_k1_FDX_s115
                                 + 0.749675 w_k1_IBM_s115
                                 + 0.847406 w_k1_INTC_s115
                                 + 0.807922 w_k1_KO_s115
                                 + 0.714529 w_k1_MAT_s115
                                 + 0.89109 w_k1_MSFT_s115
                                 + 0.853613 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.733198 w_k1_ADM_s115
                                 + 0.869926 w_k1_AXP_s115
                                 + 0.848667 w_k1_DIS_s115
                                 + 0.789128 w_k1_FDX_s115
                                 + 1.10781 w_k1_IBM_s115
                                 + 1.10939 w_k1_INTC_s115
                                 + 1.02138 w_k1_KO_s115 + 1.04126 w_k1_MAT_s115
                                 + 0.869038 w_k1_MSFT_s115
                                 + 1.03787 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.28501 w_k1_ADM_s116
                                 + 1.3115 w_k1_AXP_s116 + 1.34986 w_k1_DIS_s116
                                 + 1.14816 w_k1_FDX_s116
                                 + 1.22965 w_k1_IBM_s116
                                 + 0.813986 w_k1_INTC_s116
                                 + 1.28425 w_k1_KO_s116 + 1.52956 w_k1_MAT_s116
                                 + 1.11252 w_k1_MSFT_s116
                                 + 1.03314 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.994506 w_k1_ADM_s116
                                 + 1.11845 w_k1_AXP_s116
                                 + 0.986319 w_k1_DIS_s116
                                 + 0.973064 w_k1_FDX_s116
                                 + 1.12824 w_k1_IBM_s116
                                 + 0.918857 w_k1_INTC_s116
                                 + 1.07641 w_k1_KO_s116 + 1.39109 w_k1_MAT_s116
                                 + 0.936674 w_k1_MSFT_s116
                                 + 1.03517 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.968658 w_k1_ADM_s116
                                 + 0.802191 w_k1_AXP_s116
                                 + 0.993807 w_k1_DIS_s116
                                 + 0.977642 w_k1_FDX_s116
                                 + 1.19242 w_k1_IBM_s116
                                 + 0.874898 w_k1_INTC_s116
                                 + 1.23715 w_k1_KO_s116 + 1.33377 w_k1_MAT_s116
                                 + 1.11434 w_k1_MSFT_s116
                                 + 0.962862 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.786735 w_k1_ADM_s116
                                 + 0.730082 w_k1_AXP_s116
                                 + 0.673288 w_k1_DIS_s116
                                 + 0.74943 w_k1_FDX_s116
                                 + 1.16412 w_k1_IBM_s116
                                 + 0.793796 w_k1_INTC_s116
                                 + 1.0174 w_k1_KO_s116 + 1.50248 w_k1_MAT_s116
                                 + 0.713259 w_k1_MSFT_s116
                                 + 0.829028 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.14768 w_k1_ADM_s116
                                 + 1.00254 w_k1_AXP_s116
                                 + 0.883252 w_k1_DIS_s116
                                 + 1.05551 w_k1_FDX_s116
                                 + 0.620317 w_k1_IBM_s116
                                 + 0.784763 w_k1_INTC_s116
                                 + 0.745942 w_k1_KO_s116
                                 + 0.547456 w_k1_MAT_s116
                                 + 0.930645 w_k1_MSFT_s116
                                 + 0.879765 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.688798 w_k1_ADM_s116
                                 + 0.639836 w_k1_AXP_s116
                                 + 0.641728 w_k1_DIS_s116
                                 + 0.725921 w_k1_FDX_s116
                                 + 0.756667 w_k1_IBM_s116
                                 + 0.736772 w_k1_INTC_s116
                                 + 0.80983 w_k1_KO_s116
                                 + 0.956308 w_k1_MAT_s116
                                 + 0.792926 w_k1_MSFT_s116
                                 + 0.821135 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.19325 w_k1_ADM_s116
                                 + 1.31784 w_k1_AXP_s116
                                 + 1.68465 w_k1_DIS_s116
                                 + 1.40641 w_k1_FDX_s116
                                 + 0.892469 w_k1_IBM_s116
                                 + 1.00219 w_k1_INTC_s116
                                 + 1.23604 w_k1_KO_s116 + 1.35143 w_k1_MAT_s116
                                 + 1.31024 w_k1_MSFT_s116 + 1.1836 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.02865 w_k1_ADM_s116
                                 + 0.81522 w_k1_AXP_s116
                                 + 0.821639 w_k1_DIS_s116
                                 + 0.855048 w_k1_FDX_s116
                                 + 0.882927 w_k1_IBM_s116
                                 + 0.742022 w_k1_INTC_s116
                                 + 0.848822 w_k1_KO_s116
                                 + 0.961935 w_k1_MAT_s116
                                 + 0.971307 w_k1_MSFT_s116
                                 + 0.861387 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.32706 w_k1_ADM_s117
                                 + 1.70757 w_k1_AXP_s117
                                 + 1.38882 w_k1_DIS_s117
                                 + 1.39322 w_k1_FDX_s117
                                 + 0.938723 w_k1_IBM_s117
                                 + 1.08648 w_k1_INTC_s117 + 1.0013 w_k1_KO_s117
                                 + 1.23214 w_k1_MAT_s117
                                 + 1.2044 w_k1_MSFT_s117 + 1.1086 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.20131 w_k1_ADM_s117
                                 + 1.03105 w_k1_AXP_s117
                                 + 1.18665 w_k1_DIS_s117
                                 + 1.06421 w_k1_FDX_s117
                                 + 1.15691 w_k1_IBM_s117
                                 + 0.936037 w_k1_INTC_s117
                                 + 1.19196 w_k1_KO_s117
                                 + 0.958556 w_k1_MAT_s117
                                 + 1.14384 w_k1_MSFT_s117
                                 + 0.92534 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.24551 w_k1_ADM_s117
                                 + 1.48826 w_k1_AXP_s117
                                 + 1.53343 w_k1_DIS_s117
                                 + 1.28411 w_k1_FDX_s117
                                 + 1.24334 w_k1_IBM_s117
                                 + 1.12651 w_k1_INTC_s117
                                 + 1.38557 w_k1_KO_s117 + 1.84615 w_k1_MAT_s117
                                 + 1.23079 w_k1_MSFT_s117 + 1.1525 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.2975 w_k1_ADM_s117
                                 + 1.06938 w_k1_AXP_s117
                                 + 1.17535 w_k1_DIS_s117 + 1.0969 w_k1_FDX_s117
                                 + 0.823739 w_k1_IBM_s117
                                 + 1.00964 w_k1_INTC_s117 + 0.9565 w_k1_KO_s117
                                 + 0.803692 w_k1_MAT_s117
                                 + 1.16227 w_k1_MSFT_s117
                                 + 1.04305 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.677734 w_k1_ADM_s117
                                 + 0.785431 w_k1_AXP_s117
                                 + 0.679936 w_k1_DIS_s117
                                 + 0.765724 w_k1_FDX_s117
                                 + 0.719959 w_k1_IBM_s117
                                 + 0.682845 w_k1_INTC_s117
                                 + 0.747883 w_k1_KO_s117
                                 + 0.75182 w_k1_MAT_s117
                                 + 0.753294 w_k1_MSFT_s117
                                 + 0.746313 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.949583 w_k1_ADM_s117
                                 + 0.717221 w_k1_AXP_s117
                                 + 0.915161 w_k1_DIS_s117
                                 + 0.799042 w_k1_FDX_s117
                                 + 1.00258 w_k1_IBM_s117
                                 + 0.74332 w_k1_INTC_s117
                                 + 0.941059 w_k1_KO_s117
                                 + 0.947166 w_k1_MAT_s117
                                 + 0.842285 w_k1_MSFT_s117
                                 + 0.863978 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.875421 w_k1_ADM_s117
                                 + 1.08947 w_k1_AXP_s117
                                 + 1.27869 w_k1_DIS_s117
                                 + 0.913532 w_k1_FDX_s117
                                 + 1.49102 w_k1_IBM_s117
                                 + 0.815181 w_k1_INTC_s117
                                 + 1.35503 w_k1_KO_s117 + 1.49779 w_k1_MAT_s117
                                 + 1.06825 w_k1_MSFT_s117
                                 + 1.05335 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.16654 w_k1_ADM_s117
                                 + 1.04151 w_k1_AXP_s117
                                 + 0.92952 w_k1_DIS_s117
                                 + 1.02716 w_k1_FDX_s117
                                 + 0.593589 w_k1_IBM_s117
                                 + 0.91597 w_k1_INTC_s117
                                 + 0.657527 w_k1_KO_s117
                                 + 0.444324 w_k1_MAT_s117
                                 + 0.952105 w_k1_MSFT_s117
                                 + 0.871019 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0141493309045019 w_k1_ADM_s000
                                 + 0.0124088181888222 w_k1_ADM_s001
                                 + 0.00954013592093546 w_k1_ADM_s002
                                 + 0.0222649673371816 w_k1_ADM_s003
                                 + 0.0144760622929483 w_k1_ADM_s004
                                 + 0.0119995600141012 w_k1_ADM_s005
                                 + 0.0255379006923838 w_k1_ADM_s006
                                 + 0.040093760035165 w_k1_ADM_s007
                                 + 0.0130781763240623 w_k1_ADM_s010
                                 + 0.0165030356725779 w_k1_ADM_s011
                                 + 0.0144227846019378 w_k1_ADM_s012
                                 + 0.0437357915651667 w_k1_ADM_s013
                                 + 0.0141067570742588 w_k1_ADM_s014
                                 + 0.0302690384517565 w_k1_ADM_s015
                                 + 0.0365768414336166 w_k1_ADM_s016
                                 + 0.0412059134845038 w_k1_ADM_s017
                                 + 0.0111484033289576 w_k1_ADM_s100
                                 + 0.0229181848066894 w_k1_ADM_s101
                                 + 0.00878071169953943 w_k1_ADM_s102
                                 + 0.0296705621120195 w_k1_ADM_s103
                                 + 0.00743400338724224 w_k1_ADM_s104
                                 + 0.00997429152856843 w_k1_ADM_s105
                                 + 0.0158608580793151 w_k1_ADM_s106
                                 + 0.0118733413423827 w_k1_ADM_s107
                                 + 0.0368929755909035 w_k1_ADM_s110
                                 + 0.131334249644302 w_k1_ADM_s111
                                 + 0.0760593878345197 w_k1_ADM_s112
                                 + 0.0283292192633175 w_k1_ADM_s113
                                 + 0.0676428443184967 w_k1_ADM_s114
                                 + 0.0686731857445466 w_k1_ADM_s115
                                 + 0.0393305940553928 w_k1_ADM_s116
                                 + 0.0805614500265066 w_k1_ADM_s117
                                 + 0.0163119226082601 w_k1_AXP_s000
                                 + 0.0112514559528689 w_k1_AXP_s001
                                 + 0.0086505468384616 w_k1_AXP_s002
                                 + 0.0225873103327379 w_k1_AXP_s003
                                 + 0.0134832862979035 w_k1_AXP_s004
                                 + 0.0120940912099849 w_k1_AXP_s005
                                 + 0.0259649187978711 w_k1_AXP_s006
                                 + 0.0437605787746331 w_k1_AXP_s007
                                 + 0.0147762605604052 w_k1_AXP_s010
                                 + 0.0152640703523058 w_k1_AXP_s011
                                 + 0.0137583804122742 w_k1_AXP_s012
                                 + 0.041799812609994 w_k1_AXP_s013
                                 + 0.0150559291337854 w_k1_AXP_s014
                                 + 0.0332864847393613 w_k1_AXP_s015
                                 + 0.0329077385647117 w_k1_AXP_s016
                                 + 0.0412515265165249 w_k1_AXP_s017
                                 + 0.0110091140468808 w_k1_AXP_s100
                                 + 0.0247130288571643 w_k1_AXP_s101
                                 + 0.00675735057595669 w_k1_AXP_s102
                                 + 0.0320572681700662 w_k1_AXP_s103
                                 + 0.00770173134106678 w_k1_AXP_s104
                                 + 0.0105812721027202 w_k1_AXP_s105
                                 + 0.0182200776310668 w_k1_AXP_s106
                                 + 0.010873005464097 w_k1_AXP_s107
                                 + 0.0384195642704588 w_k1_AXP_s110
                                 + 0.140939215539654 w_k1_AXP_s111
                                 + 0.070326648218016 w_k1_AXP_s112
                                 + 0.0333722169067537 w_k1_AXP_s113
                                 + 0.0653337293580723 w_k1_AXP_s114
                                 + 0.0810170933224056 w_k1_AXP_s115
                                 + 0.0374102878409525 w_k1_AXP_s116
                                 + 0.0785099772275653 w_k1_AXP_s117
                                 + 0.0164152887031617 w_k1_DIS_s000
                                 + 0.0114499645911767 w_k1_DIS_s001
                                 + 0.00795473471471693 w_k1_DIS_s002
                                 + 0.0239524579867563 w_k1_DIS_s003
                                 + 0.013730589648553 w_k1_DIS_s004
                                 + 0.0122596514638787 w_k1_DIS_s005
                                 + 0.0266293226748287 w_k1_DIS_s006
                                 + 0.0431812583408029 w_k1_DIS_s007
                                 + 0.0134547348808308 w_k1_DIS_s010
                                 + 0.0153732391540779 w_k1_DIS_s011
                                 + 0.0133301204616011 w_k1_DIS_s012
                                 + 0.040382156529284 w_k1_DIS_s013
                                 + 0.0169925448858057 w_k1_DIS_s014
                                 + 0.0310381291628817 w_k1_DIS_s015
                                 + 0.0336866342985375 w_k1_DIS_s016
                                 + 0.0373557375394566 w_k1_DIS_s017
                                 + 0.0118967629523508 w_k1_DIS_s100
                                 + 0.0239478021270188 w_k1_DIS_s101
                                 + 0.00681573617234598 w_k1_DIS_s102
                                 + 0.0315651770953994 w_k1_DIS_s103
                                 + 0.00850729135905696 w_k1_DIS_s104
                                 + 0.0110751059407166 w_k1_DIS_s105
                                 + 0.0179230155111326 w_k1_DIS_s106
                                 + 0.0120578459310842 w_k1_DIS_s107
                                 + 0.0403784966555711 w_k1_DIS_s110
                                 + 0.15803973945448 w_k1_DIS_s111
                                 + 0.0725502391858693 w_k1_DIS_s112
                                 + 0.0299880919004562 w_k1_DIS_s113
                                 + 0.0659452261775472 w_k1_DIS_s114
                                 + 0.0833110154811518 w_k1_DIS_s115
                                 + 0.0383982195441404 w_k1_DIS_s116
                                 + 0.0830946089774976 w_k1_DIS_s117
                                 + 0.0153286831745521 w_k1_FDX_s000
                                 + 0.0113068055750823 w_k1_FDX_s001
                                 + 0.00880923800079057 w_k1_FDX_s002
                                 + 0.0235223228856706 w_k1_FDX_s003
                                 + 0.0135451727692553 w_k1_FDX_s004
                                 + 0.0119484837869852 w_k1_FDX_s005
                                 + 0.0272993289950268 w_k1_FDX_s006
                                 + 0.0405681574114177 w_k1_FDX_s007
                                 + 0.0135906981646044 w_k1_FDX_s010
                                 + 0.0157402144769229 w_k1_FDX_s011
                                 + 0.0140699335626888 w_k1_FDX_s012
                                 + 0.0372853674801074 w_k1_FDX_s013
                                 + 0.0159383985934047 w_k1_FDX_s014
                                 + 0.0320195677093791 w_k1_FDX_s015
                                 + 0.0338526675860182 w_k1_FDX_s016
                                 + 0.0386897889516379 w_k1_FDX_s017
                                 + 0.0109351637787386 w_k1_FDX_s100
                                 + 0.0254592192230744 w_k1_FDX_s101
                                 + 0.00744754389441496 w_k1_FDX_s102
                                 + 0.0295441235812095 w_k1_FDX_s103
                                 + 0.00842881355860055 w_k1_FDX_s104
                                 + 0.0101491433158434 w_k1_FDX_s105
                                 + 0.017133080748875 w_k1_FDX_s106
                                 + 0.0113553011288137 w_k1_FDX_s107
                                 + 0.0369893838472457 w_k1_FDX_s110
                                 + 0.142393103103138 w_k1_FDX_s111
                                 + 0.0688195856521961 w_k1_FDX_s112
                                 + 0.031136232938107 w_k1_FDX_s113
                                 + 0.0621103011821172 w_k1_FDX_s114
                                 + 0.079386990678954 w_k1_FDX_s115
                                 + 0.0367879518111852 w_k1_FDX_s116
                                 + 0.0752485597873267 w_k1_FDX_s117
                                 + 0.0155572797534449 w_k1_IBM_s000
                                 + 0.0112584701228245 w_k1_IBM_s001
                                 + 0.00839988377351198 w_k1_IBM_s002
                                 + 0.024284888987751 w_k1_IBM_s003
                                 + 0.0137000309356553 w_k1_IBM_s004
                                 + 0.0115521303722972 w_k1_IBM_s005
                                 + 0.0250322731625564 w_k1_IBM_s006
                                 + 0.04240776568524 w_k1_IBM_s007
                                 + 0.0141763984496463 w_k1_IBM_s010
                                 + 0.0161943932092672 w_k1_IBM_s011
                                 + 0.0144146722302429 w_k1_IBM_s012
                                 + 0.039705720511712 w_k1_IBM_s013
                                 + 0.0158172586252769 w_k1_IBM_s014
                                 + 0.0320953932552372 w_k1_IBM_s015
                                 + 0.0336691266578338 w_k1_IBM_s016
                                 + 0.0389687122664096 w_k1_IBM_s017
                                 + 0.0103392200520909 w_k1_IBM_s100
                                 + 0.0248465499989332 w_k1_IBM_s101
                                 + 0.00729177053261091 w_k1_IBM_s102
                                 + 0.0312063725314727 w_k1_IBM_s103
                                 + 0.00809031410253808 w_k1_IBM_s104
                                 + 0.0105938291608492 w_k1_IBM_s105
                                 + 0.0169280677683526 w_k1_IBM_s106
                                 + 0.0115544093220503 w_k1_IBM_s107
                                 + 0.0366034671551225 w_k1_IBM_s110
                                 + 0.150196533883227 w_k1_IBM_s111
                                 + 0.0700620103127669 w_k1_IBM_s112
                                 + 0.0297943778242532 w_k1_IBM_s113
                                 + 0.067168649908698 w_k1_IBM_s114
                                 + 0.0810856802945615 w_k1_IBM_s115
                                 + 0.0382644954171948 w_k1_IBM_s116
                                 + 0.075700145861194 w_k1_IBM_s117
                                 + 0.0163589770020395 w_k1_INTC_s000
                                 + 0.012257683175047 w_k1_INTC_s001
                                 + 0.00871913875011014 w_k1_INTC_s002
                                 + 0.0252447402200841 w_k1_INTC_s003
                                 + 0.0127478645584472 w_k1_INTC_s004
                                 + 0.0128787294886128 w_k1_INTC_s005
                                 + 0.0248248281107221 w_k1_INTC_s006
                                 + 0.0416294586526741 w_k1_INTC_s007
                                 + 0.013379443271192 w_k1_INTC_s010
                                 + 0.0157967975016026 w_k1_INTC_s011
                                 + 0.0117930871797122 w_k1_INTC_s012
                                 + 0.039849486523733 w_k1_INTC_s013
                                 + 0.0159979330580301 w_k1_INTC_s014
                                 + 0.0337018508674434 w_k1_INTC_s015
                                 + 0.0309809342656321 w_k1_INTC_s016
                                 + 0.0394087643008302 w_k1_INTC_s017
                                 + 0.0100343696304696 w_k1_INTC_s100
                                 + 0.0252803660431558 w_k1_INTC_s101
                                 + 0.00650543907850349 w_k1_INTC_s102
                                 + 0.0340266266576254 w_k1_INTC_s103
                                 + 0.00898079037441627 w_k1_INTC_s104
                                 + 0.0102102888736762 w_k1_INTC_s105
                                 + 0.0173069370813326 w_k1_INTC_s106
                                 + 0.0101470928351526 w_k1_INTC_s107
                                 + 0.0374980933959986 w_k1_INTC_s110
                                 + 0.143958779029241 w_k1_INTC_s111
                                 + 0.0787673064449153 w_k1_INTC_s112
                                 + 0.0318833615831802 w_k1_INTC_s113
                                 + 0.0648712110104671 w_k1_INTC_s114
                                 + 0.0875162803175268 w_k1_INTC_s115
                                 + 0.0303903905430362 w_k1_INTC_s116
                                 + 0.0665921836097889 w_k1_INTC_s117
                                 + 0.0148067465217487 w_k1_KO_s000
                                 + 0.0119084146295718 w_k1_KO_s001
                                 + 0.00834913978088644 w_k1_KO_s002
                                 + 0.022018744816786 w_k1_KO_s003
                                 + 0.0131945726188656 w_k1_KO_s004
                                 + 0.0118224996909844 w_k1_KO_s005
                                 + 0.0256719329452105 w_k1_KO_s006
                                 + 0.0424719675074936 w_k1_KO_s007
                                 + 0.0127628129749767 w_k1_KO_s010
                                 + 0.0160038540113204 w_k1_KO_s011
                                 + 0.0135525242284627 w_k1_KO_s012
                                 + 0.0401691740922031 w_k1_KO_s013
                                 + 0.0169090467119128 w_k1_KO_s014
                                 + 0.0331472349603896 w_k1_KO_s015
                                 + 0.0321787993416772 w_k1_KO_s016
                                 + 0.0422566972849921 w_k1_KO_s017
                                 + 0.0107323463197608 w_k1_KO_s100
                                 + 0.0262099668136556 w_k1_KO_s101
                                 + 0.00752093355890667 w_k1_KO_s102
                                 + 0.0347010658755188 w_k1_KO_s103
                                 + 0.00829560017528459 w_k1_KO_s104
                                 + 0.00999659540241944 w_k1_KO_s105
                                 + 0.0163867349078674 w_k1_KO_s106
                                 + 0.0120308063972471 w_k1_KO_s107
                                 + 0.0392018714122876 w_k1_KO_s110
                                 + 0.147883868067485 w_k1_KO_s111
                                 + 0.0773023540845181 w_k1_KO_s112
                                 + 0.030907526247281 w_k1_KO_s113
                                 + 0.0663149957985981 w_k1_KO_s114
                                 + 0.0824192283841782 w_k1_KO_s115
                                 + 0.0393404690955406 w_k1_KO_s116
                                 + 0.0771652686990106 w_k1_KO_s117
                                 + 0.014336342570693 w_k1_MAT_s000
                                 + 0.0121757484522552 w_k1_MAT_s001
                                 + 0.00835140304450322 w_k1_MAT_s002
                                 + 0.0214408888098199 w_k1_MAT_s003
                                 + 0.0142944293611328 w_k1_MAT_s004
                                 + 0.0102891180062972 w_k1_MAT_s005
                                 + 0.0236244463662852 w_k1_MAT_s006
                                 + 0.0358953355118218 w_k1_MAT_s007
                                 + 0.0143029998670762 w_k1_MAT_s010
                                 + 0.0183695666858882 w_k1_MAT_s011
                                 + 0.0149344112469752 w_k1_MAT_s012
                                 + 0.0388491790765333 w_k1_MAT_s013
                                 + 0.0207389756872389 w_k1_MAT_s014
                                 + 0.0425366996076582 w_k1_MAT_s015
                                 + 0.0324185439274208 w_k1_MAT_s016
                                 + 0.0487760634319853 w_k1_MAT_s017
                                 + 0.0109893638614285 w_k1_MAT_s100
                                 + 0.0257672818828962 w_k1_MAT_s101
                                 + 0.00737871088012448 w_k1_MAT_s102
                                 + 0.0303441158274376 w_k1_MAT_s103
                                 + 0.007689048672425 w_k1_MAT_s104
                                 + 0.011666805892184 w_k1_MAT_s105
                                 + 0.0197144475407274 w_k1_MAT_s106
                                 + 0.0138226781125331 w_k1_MAT_s107
                                 + 0.035602850803681 w_k1_MAT_s110
                                 + 0.13543132192332 w_k1_MAT_s111
                                 + 0.0701206983244147 w_k1_MAT_s112
                                 + 0.032108010248395 w_k1_MAT_s113
                                 + 0.0653795836726507 w_k1_MAT_s114
                                 + 0.0844501687303076 w_k1_MAT_s115
                                 + 0.0460570642778813 w_k1_MAT_s116
                                 + 0.0784547404657969 w_k1_MAT_s117
                                 + 0.0160167815547608 w_k1_MSFT_s000
                                 + 0.0118981978420767 w_k1_MSFT_s001
                                 + 0.00819851975447079 w_k1_MSFT_s002
                                 + 0.0241962457170021 w_k1_MSFT_s003
                                 + 0.0142579551650736 w_k1_MSFT_s004
                                 + 0.0113596708701492 w_k1_MSFT_s005
                                 + 0.0260964426815342 w_k1_MSFT_s006
                                 + 0.0439840028606349 w_k1_MSFT_s007
                                 + 0.0127241517899338 w_k1_MSFT_s010
                                 + 0.0150613383229073 w_k1_MSFT_s011
                                 + 0.0132740444739748 w_k1_MSFT_s012
                                 + 0.0365261564864129 w_k1_MSFT_s013
                                 + 0.0164051844166071 w_k1_MSFT_s014
                                 + 0.0332926242754275 w_k1_MSFT_s015
                                 + 0.0343809204702656 w_k1_MSFT_s016
                                 + 0.0388784644169677 w_k1_MSFT_s017
                                 + 0.0118441610208522 w_k1_MSFT_s100
                                 + 0.0257610781916382 w_k1_MSFT_s101
                                 + 0.00698245952631427 w_k1_MSFT_s102
                                 + 0.0289088932992526 w_k1_MSFT_s103
                                 + 0.00891454737600653 w_k1_MSFT_s104
                                 + 0.00971584158731022 w_k1_MSFT_s105
                                 + 0.0170632240435139 w_k1_MSFT_s106
                                 + 0.0119644123123231 w_k1_MSFT_s107
                                 + 0.036707594235425 w_k1_MSFT_s110
                                 + 0.146846804558568 w_k1_MSFT_s111
                                 + 0.0724171709156766 w_k1_MSFT_s112
                                 + 0.0300047409716818 w_k1_MSFT_s113
                                 + 0.0617570682436058 w_k1_MSFT_s114
                                 + 0.0791464098854864 w_k1_MSFT_s115
                                 + 0.0372257827577008 w_k1_MSFT_s116
                                 + 0.0766105136230997 w_k1_MSFT_s117
                                 + 0.0161064149823818 w_k1_PG_s000
                                 + 0.0116390317678101 w_k1_PG_s001
                                 + 0.00839215366355464 w_k1_PG_s002
                                 + 0.023298320869554 w_k1_PG_s003
                                 + 0.0124897020526049 w_k1_PG_s004
                                 + 0.0120062236655784 w_k1_PG_s005
                                 + 0.0243216141044577 w_k1_PG_s006
                                 + 0.0389119393211561 w_k1_PG_s007
                                 + 0.0141106986228764 w_k1_PG_s010
                                 + 0.0176518998562837 w_k1_PG_s011
                                 + 0.0145413685891254 w_k1_PG_s012
                                 + 0.0381973662184335 w_k1_PG_s013
                                 + 0.0155686948389786 w_k1_PG_s014
                                 + 0.032381292669458 w_k1_PG_s015
                                 + 0.0323249215782855 w_k1_PG_s016
                                 + 0.0402887416210452 w_k1_PG_s017
                                 + 0.0104756755715838 w_k1_PG_s100
                                 + 0.0262957510424271 w_k1_PG_s101
                                 + 0.0072072430833572 w_k1_PG_s102
                                 + 0.0325346013902179 w_k1_PG_s103
                                 + 0.00834046186360897 w_k1_PG_s104
                                 + 0.0102885495667426 w_k1_PG_s105
                                 + 0.0163283355723886 w_k1_PG_s106
                                 + 0.0123601914729567 w_k1_PG_s107
                                 + 0.0365914969938027 w_k1_PG_s110
                                 + 0.144886630125714 w_k1_PG_s111
                                 + 0.068737743936833 w_k1_PG_s112
                                 + 0.0320879666068407 w_k1_PG_s113
                                 + 0.0637412439998254 w_k1_PG_s114
                                 + 0.0845214201484416 w_k1_PG_s115
                                 + 0.0353962076317814 w_k1_PG_s116
                                 + 0.0717695837041331 w_k1_PG_s117 >= 1.13
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
