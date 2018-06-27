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
 w_k1_ADM_s0_return_constraint:  1.09922 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.897541 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.30801 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.01437 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.01791 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.914717 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.711028 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.986758 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.00838 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.32705 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.09077 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.861949 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.718196 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.862913 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.29063 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.859675 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.25527 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.753348 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.892135 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.08248 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.925337 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.07362 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.955443 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.47815 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.25652 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.92622 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.09713 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.17243 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.17868 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.11774 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.884341 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.82246 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.13426 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.01196 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.31538 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.915026 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.946301 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.03928 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.17047 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.865476 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.45938 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.92893 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.16919 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.821934 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.704287 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.13486 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.906553 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.61865 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.932752 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.960466 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.763014 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.918198 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.30747 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.79689 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.58502 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.633224 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.916455 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.38228 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.02892 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.36229 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.803564 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.39677 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.983553 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.70632 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.870135 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.44489 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.12061 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.978503 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.13545 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.647987 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.11505 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.970079 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.53341 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.00159 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.888068 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.12009 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25613 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.815512 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.52658 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.889376 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.20694 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.785156 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.556332 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.21701 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.894868 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.67044 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.04655 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.923068 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.764981 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.786937 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.41559 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.782752 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.74636 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.556828 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.02613 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.66667 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.07218 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.1129 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.844512 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.64654 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.956878 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.644063 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.944812 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.65681 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.03558 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.12577 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.11724 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.683518 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.20015 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.827001 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.78261 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.07921 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.953028 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.28063 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15592 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.88221 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.29791 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.953115 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.13131 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.896159 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.795632 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.13565 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.02235 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.44811 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.13173 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.901303 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.897504 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.834742 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.34758 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.813015 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.18449 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.733231 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.873608 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.24161 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.05705 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.02918 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.732009 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.45104 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.10694 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.766777 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.947177 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.20603 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.00211 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.07711 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.0055 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.76659 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.31217 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.857183 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.37826 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.899761 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.998556 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.086 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.16062 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.916525 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.08557 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.999668 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.21881 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.801454 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.616691 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.35555 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.929225 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.40574 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.675257 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.57001 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.642934 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.01556 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.782765 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.929061 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 2.24998 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.551602 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.51703 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.35428 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.832648 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.40715 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.22154 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.917777 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.579801 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.69693 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.749578 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.61958 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.03986 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.811247 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.13998 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.666588 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.765094 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.957521 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.07517 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.54862 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.815653 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.928078 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09407 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.911389 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.26583 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.992847 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.08127 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.930033 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.777645 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.04332 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.973671 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.33898 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.942999 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.02625 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.855491 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.90941 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.21082 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.86414 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.09285 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.757468 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.885885 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.13563 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.04276 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.979589 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.770644 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.13401 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.952445 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.715359 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.903332 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.31072 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.992473 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.04872 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.900786 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.843521 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.25222 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.814016 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.19481 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.995136 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.982246 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.01164 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.22501 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.903923 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.19031 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.856525 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.201 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.770265 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.670836 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.19423 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.960148 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.39202 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.856484 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.07312 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.70957 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.972951 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.08577 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.888002 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.61644 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.642331 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.20094 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.39403 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.964176 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.15395 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.973369 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.05195 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.708601 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.690238 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.813444 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.53799 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.986754 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.02489 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.05988 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.82808 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.01756 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.970827 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.30748 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.38055 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.981458 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.1851 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.16216 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.814056 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.24098 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.03676 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.58548 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.819958 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.731351 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.17718 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.0134 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.43491 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.686741 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.51089 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.761217 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.859816 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.672047 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.844245 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 2.45008 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.56096 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.35736 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.34178 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.783897 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.78013 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.04445 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.674531 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.58794 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.559521 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.703942 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.52125 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.1143 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.783993 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.19129 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.599216 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.731479 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.06313 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.0764 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.8089 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.610463 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.21646 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18351 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.908459 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.2264 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.929989 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.10276 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.891259 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.703684 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.05221 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.04263 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.40269 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.09324 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.929155 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.858607 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.767123 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.16504 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.828865 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.08206 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.671701 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.926991 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.17479 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.03958 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.987647 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.853707 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.29435 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.9443 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.743454 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.907519 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.24915 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.979658 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.12991 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.0751 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.798365 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.20661 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.920759 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.34086 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.00464 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.964448 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.09744 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13342 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.934322 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.15928 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.953011 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.062 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.894947 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.796678 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.0308 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.94028 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.35618 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.933215 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.02657 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.795482 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.932201 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.10418 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.888183 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.18228 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.750799 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.04294 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11751 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.983209 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.12886 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.964738 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.20072 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.908064 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.817444 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.874148 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.22946 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.10794 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.08356 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.00643 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.783045 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.07652 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.921577 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.18546 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.10302 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.970088 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.98061 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.179999992229035 z_k1_s0000
                                 - 0.0970001144624092 z_k1_s0001
                                 - 0.0840000980736319 z_k1_s0002
                                 - 0.208000058820273 z_k1_s0003
                                 - 0.138999604148774 z_k1_s0004
                                 - 0.0960000975555676 z_k1_s0005
                                 - 0.0630000481302034 z_k1_s0006
                                 - 0.150000502024606 z_k1_s0007
                                 - 0.135000223381596 z_k1_s0010
                                 - 0.272000036851211 z_k1_s0011
                                 - 0.230000250190679 z_k1_s0012
                                 - 0.649000217354731 z_k1_s0013
                                 - 0.159000178844869 z_k1_s0014
                                 - 0.673001256019568 z_k1_s0015
                                 - 0.171000156576505 z_k1_s0016
                                 - 0.419001321166008 z_k1_s0017
                                 - 0.183999939458339 z_k1_s0020
                                 - 0.108000106377797 z_k1_s0021
                                 - 0.0870003350266938 z_k1_s0022
                                 - 0.157999717404319 z_k1_s0023
                                 - 0.0530000446483594 z_k1_s0024
                                 - 0.0700000266374384 z_k1_s0025
                                 - 0.10400007221564 z_k1_s0026
                                 - 0.0520000361078202 z_k1_s0027
                                 - 0.168999926152363 z_k1_s0030
                                 - 0.10599975824696 z_k1_s0031
                                 - 0.0799998132618675 z_k1_s0032
                                 - 0.0879997264882311 z_k1_s0033
                                 - 0.0679998299226168 z_k1_s0034
                                 - 0.185999912007943 z_k1_s0035
                                 - 0.264999111868136 z_k1_s0036
                                 - 0.172999655766134 z_k1_s0037
                                 - 0.383000013797692 z_k1_s0040
                                 - 0.191000486651306 z_k1_s0041
                                 - 0.0810000625400114 z_k1_s0042
                                 - 0.260000288844525 z_k1_s0043
                                 - 0.10400011553781 z_k1_s0044
                                 - 0.124000014248866 z_k1_s0045
                                 - 0.384999878788736 z_k1_s0046
                                 - 0.256000558862437 z_k1_s0047
                                 - 0.0520000309694 z_k1_s0050
                                 - 0.126999748682768 z_k1_s0051
                                 - 0.0600000474108529 z_k1_s0052
                                 - 0.0760002320938469 z_k1_s0053
                                 - 0.170999763210715 z_k1_s0054
                                 - 0.172000125791106 z_k1_s0055
                                 - 0.0660000217919206 z_k1_s0056
                                 - 0.0350000150063235 z_k1_s0057
                                 - 0.0829998068829464 z_k1_s0060
                                 - 0.0869997112627529 z_k1_s0061
                                 - 0.106000252067923 z_k1_s0062
                                 - 0.106000252067923 z_k1_s0063
                                 - 0.0810002526934789 z_k1_s0064
                                 - 0.174999995621111 z_k1_s0065
                                 - 0.096999870212705 z_k1_s0066
                                 - 0.061000093993749 z_k1_s0067
                                 - 0.0969996299743734 z_k1_s0070
                                 - 0.0969996299743734 z_k1_s0071
                                 - 0.100999648562894 z_k1_s0072
                                 - 0.254999209024805 z_k1_s0073
                                 - 0.0839996607613761 z_k1_s0074
                                 - 0.169999878015178 z_k1_s0075
                                 - 0.301998667442469 z_k1_s0076
                                 - 0.109999599187371 z_k1_s0077
                                 - 0.201999836552377 z_k1_s0100
                                 - 0.314999851069255 z_k1_s0101
                                 - 0.960999324828837 z_k1_s0102
                                 - 0.35799823408257 z_k1_s0103
                                 - 0.232999906824235 z_k1_s0104
                                 - 0.394998709174619 z_k1_s0105
                                 - 0.511001066300673 z_k1_s0106
                                 - 0.592000467475865 z_k1_s0107
                                 - 0.848999705035948 z_k1_s0110
                                 - 0.742999551617503 z_k1_s0111
                                 - 1.91699884972443 z_k1_s0112
                                 - 1.47700301504259 z_k1_s0113
                                 - 0.486000204724348 z_k1_s0114
                                 - 2.16899699807841 z_k1_s0115
                                 - 1.0150015152285 z_k1_s0116
                                 - 1.13299808440298 z_k1_s0117
                                 - 0.364000054464982 z_k1_s0120
                                 - 0.403999823746911 z_k1_s0121
                                 - 0.125000032905828 z_k1_s0122
                                 - 0.349999058216595 z_k1_s0123
                                 - 0.148999894474986 z_k1_s0124
                                 - 0.180000038768394 z_k1_s0125
                                 - 0.265999542724543 z_k1_s0126
                                 - 0.316000331326666 z_k1_s0127
                                 - 0.531001138986448 z_k1_s0130
                                 - 0.381001074018379 z_k1_s0131
                                 - 0.442000999438784 z_k1_s0132
                                 - 0.187999879426761 z_k1_s0133
                                 - 0.47100111299922 z_k1_s0134
                                 - 0.540000233885047 z_k1_s0135
                                 - 0.156999966999969 z_k1_s0136
                                 - 0.320001148597974 z_k1_s0137
                                 - 0.311001183011502 z_k1_s0140
                                 - 0.404000034657743 z_k1_s0141
                                 - 0.152000058441536 z_k1_s0142
                                 - 0.238000136759782 z_k1_s0143
                                 - 0.737000637843566 z_k1_s0144
                                 - 0.493000815545363 z_k1_s0145
                                 - 0.151000110851799 z_k1_s0146
                                 - 0.380000146103839 z_k1_s0147
                                 - 0.529000162523714 z_k1_s0150
                                 - 0.546003298817403 z_k1_s0151
                                 - 0.502001467258247 z_k1_s0152
                                 - 0.506000512850164 z_k1_s0153
                                 - 1.02800542810004 z_k1_s0154
                                 - 0.243000998438881 z_k1_s0155
                                 - 0.250001177731946 z_k1_s0156
                                 - 0.506000512850164 z_k1_s0157
                                 - 1.73800294305315 z_k1_s0160
                                 - 0.842001815342523 z_k1_s0161
                                 - 2.41300421667125 z_k1_s0162
                                 - 0.571000587590924 z_k1_s0163
                                 - 0.532001193869703 z_k1_s0164
                                 - 1.00400468695524 z_k1_s0165
                                 - 2.08100062222441 z_k1_s0166
                                 - 0.68800074255795 z_k1_s0167
                                 - 0.304999025312289 z_k1_s0170
                                 - 0.206999586424806 z_k1_s0171
                                 - 0.172999874504562 z_k1_s0172
                                 - 0.378000109992231 z_k1_s0173
                                 - 0.333999265390819 z_k1_s0174
                                 - 0.110999896474049 z_k1_s0175
                                 - 0.314998764994895 z_k1_s0176
                                 - 0.166999831695204 z_k1_s0177
                                 - 0.649003554653033 z_k1_s1000
                                 - 0.277000653974331 z_k1_s1001
                                 - 0.801004614973017 z_k1_s1002
                                 - 0.542004058773344 z_k1_s1003
                                 - 0.458001138484793 z_k1_s1004
                                 - 0.909002033612206 z_k1_s1005
                                 - 1.18900120787818 z_k1_s1006
                                 - 0.455001034188773 z_k1_s1007
                                 - 0.27600089417363 z_k1_s1010
                                 - 1.01400281091878 z_k1_s1011
                                 - 0.414000349756835 z_k1_s1012
                                 - 0.590000172734399 z_k1_s1013
                                 - 0.48499994033178 z_k1_s1014
                                 - 0.538001757360245 z_k1_s1015
                                 - 0.295000879570088 z_k1_s1016
                                 - 0.354001293444973 z_k1_s1017
                                 - 0.270998887122623 z_k1_s1020
                                 - 0.356000967468281 z_k1_s1021
                                 - 0.331999038334879 z_k1_s1022
                                 - 0.33000060557442 z_k1_s1023
                                 - 0.254000026254083 z_k1_s1024
                                 - 0.256000014215134 z_k1_s1025
                                 - 0.153000115820804 z_k1_s1026
                                 - 0.640001331538328 z_k1_s1027
                                 - 0.393001017251629 z_k1_s1030
                                 - 0.677001765603982 z_k1_s1031
                                 - 0.24400046307695 z_k1_s1032
                                 - 0.405001055534138 z_k1_s1033
                                 - 0.9630018896686 z_k1_s1034
                                 - 0.511002024364459 z_k1_s1035
                                 - 0.868004345771903 z_k1_s1036
                                 - 0.669003316752692 z_k1_s1037
                                 - 0.216999981327306 z_k1_s1040
                                 - 0.213000485957895 z_k1_s1041
                                 - 0.185000488367275 z_k1_s1042
                                 - 0.316999720579183 z_k1_s1043
                                 - 0.107000091650005 z_k1_s1044
                                 - 0.37900072381691 z_k1_s1045
                                 - 0.222000233040256 z_k1_s1046
                                 - 0.125000115315438 z_k1_s1047
                                 - 0.782998873113986 z_k1_s1050
                                 - 0.54600090450205 z_k1_s1051
                                 - 0.597999677644241 z_k1_s1052
                                 - 0.731000099971796 z_k1_s1053
                                 - 0.194000210279776 z_k1_s1054
                                 - 0.349000355952264 z_k1_s1055
                                 - 0.491000611389074 z_k1_s1056
                                 - 0.247000277879413 z_k1_s1057
                                 - 0.986000611330188 z_k1_s1060
                                 - 0.357000221343689 z_k1_s1061
                                 - 0.850000527008783 z_k1_s1062
                                 - 0.768998482226419 z_k1_s1063
                                 - 0.986000611330188 z_k1_s1064
                                 - 0.692003692875103 z_k1_s1065
                                 - 0.791001397047049 z_k1_s1066
                                 - 0.733999729792484 z_k1_s1067
                                 - 0.149000026668333 z_k1_s1070
                                 - 0.439000762933885 z_k1_s1071
                                 - 0.170000116593443 z_k1_s1072
                                 - 0.585000212412636 z_k1_s1073
                                 - 0.266000316050983 z_k1_s1074
                                 - 0.223000190704359 z_k1_s1075
                                 - 0.395000692090856 z_k1_s1076
                                 - 0.242000142736537 z_k1_s1077
                                 - 0.195999786059327 z_k1_s1100
                                 - 0.479999809736975 z_k1_s1101
                                 - 0.530998706334525 z_k1_s1102
                                 - 0.446998984880442 z_k1_s1103
                                 - 0.310999747287974 z_k1_s1104
                                 - 0.468999534784797 z_k1_s1105
                                 - 0.357999435720687 z_k1_s1106
                                 - 0.477998570655306 z_k1_s1107
                                 - 0.608001339802423 z_k1_s1110
                                 - 0.378999721710351 z_k1_s1111
                                 - 0.421999855892749 z_k1_s1112
                                 - 0.348999935666738 z_k1_s1113
                                 - 0.292000094203937 z_k1_s1114
                                 - 1.08099826950998 z_k1_s1115
                                 - 0.451001109394531 z_k1_s1116
                                 - 0.550999018540264 z_k1_s1117
                                 - 0.195000600948896 z_k1_s1120
                                 - 0.214999861405895 z_k1_s1121
                                 - 0.229999766248242 z_k1_s1122
                                 - 0.140999840717415 z_k1_s1123
                                 - 0.271999867406491 z_k1_s1124
                                 - 0.484999986566525 z_k1_s1125
                                 - 0.123999826029529 z_k1_s1126
                                 - 0.175999925015955 z_k1_s1127
                                 - 0.146999938745065 z_k1_s1130
                                 - 0.169999947751422 z_k1_s1131
                                 - 0.345999558374313 z_k1_s1132
                                 - 0.409999736200588 z_k1_s1133
                                 - 0.156999832330517 z_k1_s1134
                                 - 0.32100031241033 z_k1_s1135
                                 - 0.26300057825496 z_k1_s1136
                                 - 0.137999858838285 z_k1_s1137
                                 - 0.154999243938969 z_k1_s1140
                                 - 0.285999331599771 z_k1_s1141
                                 - 0.360998860957878 z_k1_s1142
                                 - 0.165999218231268 z_k1_s1143
                                 - 0.169999149810591 z_k1_s1144
                                 - 0.575997078586682 z_k1_s1145
                                 - 0.257999593945632 z_k1_s1146
                                 - 0.193998955885408 z_k1_s1147
                                 - 0.534997548988747 z_k1_s1150
                                 - 0.274999197703319 z_k1_s1151
                                 - 0.609997956088042 z_k1_s1152
                                 - 0.350999074731982 z_k1_s1153
                                 - 0.445997677896009 z_k1_s1154
                                 - 0.376999062860132 z_k1_s1155
                                 - 0.666997959484294 z_k1_s1156
                                 - 0.563998624398113 z_k1_s1157
                                 - 0.464001107257681 z_k1_s1160
                                 - 0.354999075648733 z_k1_s1161
                                 - 0.14300001482492 z_k1_s1162
                                 - 0.298999638997338 z_k1_s1163
                                 - 0.157000008737845 z_k1_s1164
                                 - 0.677000552646921 z_k1_s1165
                                 - 0.17400007834644 z_k1_s1166
                                 - 0.426001585780003 z_k1_s1167
                                 - 0.449999954179732 z_k1_s1170
                                 - 0.287999520155368 z_k1_s1171
                                 - 0.12099992689504 z_k1_s1172
                                 - 0.131999847112826 z_k1_s1173
                                 - 0.178000317977508 z_k1_s1174
                                 - 0.170999564248613 z_k1_s1175
                                 - 0.116999955906755 z_k1_s1176
                                 - 0.151999862954134 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.25306 w_k1_ADM_s000
                                 + 1.01435 w_k1_AXP_s000
                                 + 1.32182 w_k1_DIS_s000
                                 + 1.29788 w_k1_FDX_s000
                                 + 0.873557 w_k1_IBM_s000
                                 + 1.20621 w_k1_INTC_s000
                                 + 0.991113 w_k1_KO_s000
                                 + 0.654948 w_k1_MAT_s000
                                 + 1.21609 w_k1_MSFT_s000
                                 + 1.06813 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.85395 w_k1_ADM_s000
                                 + 1.29021 w_k1_AXP_s000 + 1.498 w_k1_DIS_s000
                                 + 1.27311 w_k1_FDX_s000
                                 + 1.48134 w_k1_IBM_s000
                                 + 1.20122 w_k1_INTC_s000
                                 + 1.52724 w_k1_KO_s000 + 1.57858 w_k1_MAT_s000
                                 + 1.4239 w_k1_MSFT_s000 + 1.25145 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.01472 w_k1_ADM_s000
                                 + 0.652264 w_k1_AXP_s000
                                 + 0.775364 w_k1_DIS_s000
                                 + 0.731622 w_k1_FDX_s000
                                 + 1.19274 w_k1_IBM_s000
                                 + 0.875396 w_k1_INTC_s000
                                 + 0.935103 w_k1_KO_s000
                                 + 1.07527 w_k1_MAT_s000
                                 + 0.951124 w_k1_MSFT_s000
                                 + 0.889211 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.3616 w_k1_ADM_s000
                                 + 0.96078 w_k1_AXP_s000
                                 + 1.10887 w_k1_DIS_s000
                                 + 0.996786 w_k1_FDX_s000
                                 + 0.838061 w_k1_IBM_s000
                                 + 0.896387 w_k1_INTC_s000
                                 + 0.930169 w_k1_KO_s000
                                 + 0.503625 w_k1_MAT_s000
                                 + 1.01838 w_k1_MSFT_s000
                                 + 0.95147 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.994302 w_k1_ADM_s000
                                 + 0.68798 w_k1_AXP_s000
                                 + 0.759962 w_k1_DIS_s000
                                 + 0.812073 w_k1_FDX_s000
                                 + 0.599984 w_k1_IBM_s000
                                 + 0.808027 w_k1_INTC_s000
                                 + 0.75432 w_k1_KO_s000
                                 + 0.554591 w_k1_MAT_s000
                                 + 0.921858 w_k1_MSFT_s000
                                 + 0.777729 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.851022 w_k1_ADM_s000
                                 + 0.791361 w_k1_AXP_s000
                                 + 0.781521 w_k1_DIS_s000
                                 + 0.839254 w_k1_FDX_s000
                                 + 0.963899 w_k1_IBM_s000
                                 + 0.864123 w_k1_INTC_s000
                                 + 0.8262 w_k1_KO_s000 + 0.890175 w_k1_MAT_s000
                                 + 0.913984 w_k1_MSFT_s000
                                 + 0.898019 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.24305 w_k1_ADM_s000
                                 + 0.836336 w_k1_AXP_s000
                                 + 1.23827 w_k1_DIS_s000
                                 + 0.935727 w_k1_FDX_s000
                                 + 1.69089 w_k1_IBM_s000
                                 + 1.07304 w_k1_INTC_s000
                                 + 1.36594 w_k1_KO_s000 + 1.17007 w_k1_MAT_s000
                                 + 1.1465 w_k1_MSFT_s000 + 1.11048 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.11511 w_k1_ADM_s000
                                 + 0.986519 w_k1_AXP_s000
                                 + 1.02227 w_k1_DIS_s000
                                 + 0.972895 w_k1_FDX_s000
                                 + 1.06949 w_k1_IBM_s000
                                 + 0.851977 w_k1_INTC_s000
                                 + 1.03685 w_k1_KO_s000 + 1.06707 w_k1_MAT_s000
                                 + 1.14348 w_k1_MSFT_s000
                                 + 1.07189 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.973671 w_k1_ADM_s001
                                 + 1.1882 w_k1_AXP_s001
                                 + 0.949844 w_k1_DIS_s001
                                 + 0.844165 w_k1_FDX_s001
                                 + 1.60654 w_k1_IBM_s001
                                 + 1.1215 w_k1_INTC_s001 + 1.15896 w_k1_KO_s001
                                 + 2.08314 w_k1_MAT_s001
                                 + 0.931424 w_k1_MSFT_s001
                                 + 1.02773 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.00247 w_k1_ADM_s001
                                 + 1.42621 w_k1_AXP_s001
                                 + 1.51708 w_k1_DIS_s001
                                 + 1.16043 w_k1_FDX_s001
                                 + 1.76615 w_k1_IBM_s001
                                 + 1.5212 w_k1_INTC_s001 + 1.48307 w_k1_KO_s001
                                 + 2.18718 w_k1_MAT_s001
                                 + 1.31979 w_k1_MSFT_s001 + 1.2244 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.04174 w_k1_ADM_s001
                                 + 0.647966 w_k1_AXP_s001
                                 + 0.754548 w_k1_DIS_s001
                                 + 0.915234 w_k1_FDX_s001
                                 + 0.64714 w_k1_IBM_s001
                                 + 1.14721 w_k1_INTC_s001
                                 + 0.702486 w_k1_KO_s001
                                 + 0.489691 w_k1_MAT_s001
                                 + 0.993404 w_k1_MSFT_s001
                                 + 0.966503 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.0481 w_k1_ADM_s001
                                 + 1.02141 w_k1_AXP_s001
                                 + 0.794703 w_k1_DIS_s001
                                 + 0.845073 w_k1_FDX_s001
                                 + 0.951686 w_k1_IBM_s001
                                 + 0.821875 w_k1_INTC_s001
                                 + 0.95322 w_k1_KO_s001 + 1.54844 w_k1_MAT_s001
                                 + 0.813866 w_k1_MSFT_s001
                                 + 0.990148 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.940419 w_k1_ADM_s001
                                 + 0.78263 w_k1_AXP_s001
                                 + 1.00421 w_k1_DIS_s001
                                 + 1.05748 w_k1_FDX_s001
                                 + 0.609762 w_k1_IBM_s001
                                 + 1.18404 w_k1_INTC_s001
                                 + 0.855105 w_k1_KO_s001
                                 + 0.610598 w_k1_MAT_s001
                                 + 1.12305 w_k1_MSFT_s001 + 1.0293 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.18383 w_k1_ADM_s001
                                 + 1.2903 w_k1_AXP_s001 + 1.32624 w_k1_DIS_s001
                                 + 1.27498 w_k1_FDX_s001
                                 + 0.878718 w_k1_IBM_s001
                                 + 1.08051 w_k1_INTC_s001
                                 + 0.973119 w_k1_KO_s001
                                 + 1.32164 w_k1_MAT_s001
                                 + 1.12379 w_k1_MSFT_s001
                                 + 1.25413 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05549 w_k1_ADM_s001
                                 + 0.886283 w_k1_AXP_s001
                                 + 0.7234 w_k1_DIS_s001
                                 + 0.795986 w_k1_FDX_s001
                                 + 0.738772 w_k1_IBM_s001
                                 + 0.943678 w_k1_INTC_s001
                                 + 0.790824 w_k1_KO_s001 + 0.6608 w_k1_MAT_s001
                                 + 0.844209 w_k1_MSFT_s001
                                 + 0.903371 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.914316 w_k1_ADM_s001
                                 + 0.935371 w_k1_AXP_s001
                                 + 1.05528 w_k1_DIS_s001
                                 + 0.983499 w_k1_FDX_s001
                                 + 1.03239 w_k1_IBM_s001
                                 + 0.999742 w_k1_INTC_s001
                                 + 1.07057 w_k1_KO_s001 + 1.08359 w_k1_MAT_s001
                                 + 0.970703 w_k1_MSFT_s001
                                 + 1.05666 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.846156 w_k1_ADM_s002
                                 + 1.21993 w_k1_AXP_s002 + 1.3382 w_k1_DIS_s002
                                 + 1.13775 w_k1_FDX_s002
                                 + 1.40797 w_k1_IBM_s002
                                 + 1.08628 w_k1_INTC_s002
                                 + 1.25399 w_k1_KO_s002 + 1.38911 w_k1_MAT_s002
                                 + 1.05157 w_k1_MSFT_s002
                                 + 1.06195 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.04348 w_k1_ADM_s002
                                 + 1.12747 w_k1_AXP_s002
                                 + 1.35392 w_k1_DIS_s002
                                 + 1.22163 w_k1_FDX_s002
                                 + 0.858832 w_k1_IBM_s002
                                 + 1.18068 w_k1_INTC_s002
                                 + 1.13082 w_k1_KO_s002
                                 + 0.954722 w_k1_MAT_s002
                                 + 1.2107 w_k1_MSFT_s002 + 1.16969 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.19255 w_k1_ADM_s002
                                 + 1.03875 w_k1_AXP_s002
                                 + 0.979131 w_k1_DIS_s002
                                 + 1.03278 w_k1_FDX_s002
                                 + 0.797372 w_k1_IBM_s002
                                 + 0.879981 w_k1_INTC_s002
                                 + 0.853254 w_k1_KO_s002
                                 + 0.900536 w_k1_MAT_s002
                                 + 0.926281 w_k1_MSFT_s002
                                 + 1.02846 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.979788 w_k1_ADM_s002
                                 + 1.03983 w_k1_AXP_s002
                                 + 1.12469 w_k1_DIS_s002
                                 + 1.01163 w_k1_FDX_s002
                                 + 1.11131 w_k1_IBM_s002
                                 + 1.02029 w_k1_INTC_s002
                                 + 1.00369 w_k1_KO_s002
                                 + 0.745043 w_k1_MAT_s002
                                 + 0.895264 w_k1_MSFT_s002
                                 + 0.97354 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.879502 w_k1_ADM_s002
                                 + 0.965828 w_k1_AXP_s002
                                 + 0.797202 w_k1_DIS_s002
                                 + 0.93686 w_k1_FDX_s002
                                 + 0.732718 w_k1_IBM_s002
                                 + 0.891928 w_k1_INTC_s002
                                 + 0.761666 w_k1_KO_s002
                                 + 0.838655 w_k1_MAT_s002
                                 + 0.80825 w_k1_MSFT_s002
                                 + 0.854807 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.814795 w_k1_ADM_s002
                                 + 0.815987 w_k1_AXP_s002
                                 + 0.839463 w_k1_DIS_s002
                                 + 0.844511 w_k1_FDX_s002
                                 + 1.2564 w_k1_IBM_s002
                                 + 0.940863 w_k1_INTC_s002
                                 + 0.961237 w_k1_KO_s002
                                 + 1.14899 w_k1_MAT_s002
                                 + 0.82679 w_k1_MSFT_s002
                                 + 0.981096 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.651062 w_k1_ADM_s002
                                 + 0.667763 w_k1_AXP_s002
                                 + 1.00312 w_k1_DIS_s002
                                 + 0.807353 w_k1_FDX_s002
                                 + 0.780903 w_k1_IBM_s002
                                 + 0.916868 w_k1_INTC_s002
                                 + 0.925004 w_k1_KO_s002
                                 + 0.599748 w_k1_MAT_s002
                                 + 0.888145 w_k1_MSFT_s002
                                 + 0.997048 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.708103 w_k1_ADM_s002
                                 + 0.592481 w_k1_AXP_s002
                                 + 0.634243 w_k1_DIS_s002
                                 + 0.685245 w_k1_FDX_s002
                                 + 0.79628 w_k1_IBM_s002
                                 + 0.751146 w_k1_INTC_s002
                                 + 0.70464 w_k1_KO_s002
                                 + 0.746424 w_k1_MAT_s002
                                 + 0.833025 w_k1_MSFT_s002
                                 + 0.85992 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.02517 w_k1_ADM_s003
                                 + 0.776439 w_k1_AXP_s003
                                 + 0.827099 w_k1_DIS_s003 + 0.858 w_k1_FDX_s003
                                 + 1.10682 w_k1_IBM_s003
                                 + 0.78831 w_k1_INTC_s003
                                 + 0.926702 w_k1_KO_s003
                                 + 0.892121 w_k1_MAT_s003
                                 + 0.774987 w_k1_MSFT_s003
                                 + 0.870072 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.31921 w_k1_ADM_s003
                                 + 0.812996 w_k1_AXP_s003
                                 + 1.30607 w_k1_DIS_s003
                                 + 1.22349 w_k1_FDX_s003
                                 + 0.707445 w_k1_IBM_s003
                                 + 1.25442 w_k1_INTC_s003
                                 + 1.04988 w_k1_KO_s003
                                 + 0.791008 w_k1_MAT_s003
                                 + 1.16382 w_k1_MSFT_s003
                                 + 0.92528 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.812402 w_k1_ADM_s003
                                 + 0.547995 w_k1_AXP_s003
                                 + 0.762438 w_k1_DIS_s003
                                 + 0.729833 w_k1_FDX_s003
                                 + 0.620929 w_k1_IBM_s003
                                 + 0.806757 w_k1_INTC_s003
                                 + 0.755656 w_k1_KO_s003
                                 + 0.407862 w_k1_MAT_s003
                                 + 0.759587 w_k1_MSFT_s003
                                 + 0.751479 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.890743 w_k1_ADM_s003
                                 + 1.41433 w_k1_AXP_s003
                                 + 1.30689 w_k1_DIS_s003
                                 + 1.13457 w_k1_FDX_s003
                                 + 1.16554 w_k1_IBM_s003
                                 + 0.974808 w_k1_INTC_s003
                                 + 1.22654 w_k1_KO_s003 + 2.05655 w_k1_MAT_s003
                                 + 0.998066 w_k1_MSFT_s003
                                 + 1.0993 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.841571 w_k1_ADM_s003
                                 + 1.10037 w_k1_AXP_s003
                                 + 0.817162 w_k1_DIS_s003
                                 + 0.83154 w_k1_FDX_s003
                                 + 1.49761 w_k1_IBM_s003
                                 + 1.0617 w_k1_INTC_s003 + 1.04984 w_k1_KO_s003
                                 + 1.47794 w_k1_MAT_s003
                                 + 0.831376 w_k1_MSFT_s003
                                 + 0.937556 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.53914 w_k1_ADM_s003
                                 + 1.30285 w_k1_AXP_s003
                                 + 1.45663 w_k1_DIS_s003
                                 + 1.13464 w_k1_FDX_s003
                                 + 1.68313 w_k1_IBM_s003
                                 + 0.963086 w_k1_INTC_s003
                                 + 1.53142 w_k1_KO_s003 + 2.32324 w_k1_MAT_s003
                                 + 1.02045 w_k1_MSFT_s003
                                 + 1.03985 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.20581 w_k1_ADM_s003
                                 + 1.05151 w_k1_AXP_s003
                                 + 1.17267 w_k1_DIS_s003
                                 + 1.13133 w_k1_FDX_s003
                                 + 0.967386 w_k1_IBM_s003
                                 + 0.934126 w_k1_INTC_s003
                                 + 1.01061 w_k1_KO_s003 + 1.02678 w_k1_MAT_s003
                                 + 0.910794 w_k1_MSFT_s003
                                 + 0.951845 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.887159 w_k1_ADM_s003
                                 + 0.678765 w_k1_AXP_s003
                                 + 0.782261 w_k1_DIS_s003
                                 + 0.962046 w_k1_FDX_s003
                                 + 0.59166 w_k1_IBM_s003
                                 + 0.745588 w_k1_INTC_s003
                                 + 0.734346 w_k1_KO_s003
                                 + 0.577965 w_k1_MAT_s003
                                 + 0.841264 w_k1_MSFT_s003
                                 + 0.807789 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.939877 w_k1_ADM_s004
                                 + 0.867984 w_k1_AXP_s004
                                 + 0.766738 w_k1_DIS_s004
                                 + 0.859117 w_k1_FDX_s004
                                 + 1.03514 w_k1_IBM_s004
                                 + 1.01369 w_k1_INTC_s004
                                 + 0.903046 w_k1_KO_s004
                                 + 0.891562 w_k1_MAT_s004
                                 + 0.809525 w_k1_MSFT_s004
                                 + 0.92919 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.26839 w_k1_ADM_s004
                                 + 1.71804 w_k1_AXP_s004 + 1.5965 w_k1_DIS_s004
                                 + 1.38925 w_k1_FDX_s004
                                 + 1.35397 w_k1_IBM_s004
                                 + 1.39876 w_k1_INTC_s004
                                 + 1.22823 w_k1_KO_s004 + 2.01359 w_k1_MAT_s004
                                 + 1.23446 w_k1_MSFT_s004
                                 + 1.30295 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.1621 w_k1_ADM_s004
                                 + 1.33518 w_k1_AXP_s004
                                 + 0.982844 w_k1_DIS_s004
                                 + 1.25313 w_k1_FDX_s004
                                 + 0.801906 w_k1_IBM_s004
                                 + 1.19948 w_k1_INTC_s004
                                 + 0.86422 w_k1_KO_s004
                                 + 0.970961 w_k1_MAT_s004
                                 + 1.08398 w_k1_MSFT_s004
                                 + 1.11498 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.01287 w_k1_ADM_s004
                                 + 0.917953 w_k1_AXP_s004
                                 + 1.13924 w_k1_DIS_s004
                                 + 0.962373 w_k1_FDX_s004
                                 + 1.26591 w_k1_IBM_s004
                                 + 0.938573 w_k1_INTC_s004
                                 + 1.10075 w_k1_KO_s004
                                 + 0.919321 w_k1_MAT_s004
                                 + 0.868875 w_k1_MSFT_s004
                                 + 1.01536 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.04942 w_k1_ADM_s004
                                 + 0.936838 w_k1_AXP_s004
                                 + 1.27128 w_k1_DIS_s004
                                 + 1.10413 w_k1_FDX_s004
                                 + 0.925304 w_k1_IBM_s004
                                 + 0.977349 w_k1_INTC_s004
                                 + 1.09848 w_k1_KO_s004
                                 + 0.820191 w_k1_MAT_s004
                                 + 1.10372 w_k1_MSFT_s004
                                 + 1.08033 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.762682 w_k1_ADM_s004
                                 + 0.685969 w_k1_AXP_s004
                                 + 0.550291 w_k1_DIS_s004
                                 + 0.679494 w_k1_FDX_s004
                                 + 0.752565 w_k1_IBM_s004
                                 + 0.683302 w_k1_INTC_s004
                                 + 0.633808 w_k1_KO_s004
                                 + 0.650169 w_k1_MAT_s004
                                 + 0.707779 w_k1_MSFT_s004
                                 + 0.714186 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.78336 w_k1_ADM_s004
                                 + 0.86679 w_k1_AXP_s004
                                 + 0.79652 w_k1_DIS_s004
                                 + 0.972521 w_k1_FDX_s004
                                 + 0.696087 w_k1_IBM_s004
                                 + 1.06183 w_k1_INTC_s004
                                 + 0.811804 w_k1_KO_s004
                                 + 1.00215 w_k1_MAT_s004
                                 + 0.898531 w_k1_MSFT_s004
                                 + 0.990414 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.13297 w_k1_ADM_s004
                                 + 1.52484 w_k1_AXP_s004
                                 + 1.01699 w_k1_DIS_s004
                                 + 1.15304 w_k1_FDX_s004
                                 + 1.31333 w_k1_IBM_s004
                                 + 1.10576 w_k1_INTC_s004
                                 + 1.08268 w_k1_KO_s004 + 1.51271 w_k1_MAT_s004
                                 + 0.948435 w_k1_MSFT_s004
                                 + 1.05257 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.964603 w_k1_ADM_s005
                                 + 0.698764 w_k1_AXP_s005
                                 + 0.703474 w_k1_DIS_s005
                                 + 0.715451 w_k1_FDX_s005
                                 + 1.01736 w_k1_IBM_s005
                                 + 0.563824 w_k1_INTC_s005
                                 + 0.814217 w_k1_KO_s005
                                 + 0.766776 w_k1_MAT_s005
                                 + 0.673591 w_k1_MSFT_s005
                                 + 0.756923 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.25033 w_k1_ADM_s005
                                 + 1.33511 w_k1_AXP_s005
                                 + 1.48465 w_k1_DIS_s005
                                 + 1.13347 w_k1_FDX_s005
                                 + 1.63194 w_k1_IBM_s005
                                 + 1.2116 w_k1_INTC_s005 + 1.52456 w_k1_KO_s005
                                 + 1.18719 w_k1_MAT_s005
                                 + 1.2539 w_k1_MSFT_s005 + 1.26891 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.04472 w_k1_ADM_s005
                                 + 0.975596 w_k1_AXP_s005
                                 + 0.861176 w_k1_DIS_s005
                                 + 1.02186 w_k1_FDX_s005
                                 + 0.72151 w_k1_IBM_s005
                                 + 1.13053 w_k1_INTC_s005
                                 + 0.854025 w_k1_KO_s005
                                 + 0.581745 w_k1_MAT_s005
                                 + 1.05469 w_k1_MSFT_s005
                                 + 0.896719 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.817071 w_k1_ADM_s005
                                 + 0.59428 w_k1_AXP_s005
                                 + 0.754388 w_k1_DIS_s005
                                 + 0.853718 w_k1_FDX_s005
                                 + 0.630238 w_k1_IBM_s005
                                 + 0.790892 w_k1_INTC_s005
                                 + 0.740768 w_k1_KO_s005
                                 + 0.398303 w_k1_MAT_s005
                                 + 0.803592 w_k1_MSFT_s005
                                 + 0.722569 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.993188 w_k1_ADM_s005
                                 + 1.13651 w_k1_AXP_s005
                                 + 1.18068 w_k1_DIS_s005
                                 + 1.08577 w_k1_FDX_s005
                                 + 1.00594 w_k1_IBM_s005
                                 + 0.985487 w_k1_INTC_s005
                                 + 1.24627 w_k1_KO_s005 + 1.06775 w_k1_MAT_s005
                                 + 1.26467 w_k1_MSFT_s005
                                 + 1.11424 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.91153 w_k1_ADM_s005
                                 + 0.886248 w_k1_AXP_s005
                                 + 0.973102 w_k1_DIS_s005
                                 + 0.909376 w_k1_FDX_s005
                                 + 1.00532 w_k1_IBM_s005
                                 + 0.826205 w_k1_INTC_s005 + 1.099 w_k1_KO_s005
                                 + 0.930185 w_k1_MAT_s005
                                 + 0.932685 w_k1_MSFT_s005
                                 + 1.03839 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.08659 w_k1_ADM_s005
                                 + 0.776137 w_k1_AXP_s005
                                 + 0.741307 w_k1_DIS_s005
                                 + 0.866077 w_k1_FDX_s005
                                 + 0.839656 w_k1_IBM_s005
                                 + 0.8103 w_k1_INTC_s005
                                 + 0.891399 w_k1_KO_s005
                                 + 0.837531 w_k1_MAT_s005
                                 + 0.879599 w_k1_MSFT_s005
                                 + 0.86857 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.813293 w_k1_ADM_s005
                                 + 0.601323 w_k1_AXP_s005
                                 + 0.55924 w_k1_DIS_s005
                                 + 0.683235 w_k1_FDX_s005
                                 + 0.768457 w_k1_IBM_s005
                                 + 0.531742 w_k1_INTC_s005
                                 + 0.741278 w_k1_KO_s005
                                 + 0.741108 w_k1_MAT_s005
                                 + 0.590204 w_k1_MSFT_s005
                                 + 0.750786 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.38748 w_k1_ADM_s006
                                 + 1.47987 w_k1_AXP_s006
                                 + 1.06671 w_k1_DIS_s006
                                 + 1.13415 w_k1_FDX_s006
                                 + 1.42757 w_k1_IBM_s006
                                 + 1.04269 w_k1_INTC_s006
                                 + 0.995889 w_k1_KO_s006
                                 + 1.20632 w_k1_MAT_s006
                                 + 1.04517 w_k1_MSFT_s006
                                 + 1.19856 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.08177 w_k1_ADM_s006
                                 + 0.890407 w_k1_AXP_s006
                                 + 0.708225 w_k1_DIS_s006
                                 + 0.774827 w_k1_FDX_s006
                                 + 1.31769 w_k1_IBM_s006
                                 + 0.74459 w_k1_INTC_s006
                                 + 1.12381 w_k1_KO_s006 + 1.21203 w_k1_MAT_s006
                                 + 0.79649 w_k1_MSFT_s006
                                 + 1.00053 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.1385 w_k1_ADM_s006
                                 + 0.680482 w_k1_AXP_s006
                                 + 0.964527 w_k1_DIS_s006
                                 + 0.895984 w_k1_FDX_s006
                                 + 0.870684 w_k1_IBM_s006
                                 + 0.894458 w_k1_INTC_s006
                                 + 0.878258 w_k1_KO_s006
                                 + 0.46703 w_k1_MAT_s006
                                 + 0.983756 w_k1_MSFT_s006
                                 + 0.902646 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.05128 w_k1_ADM_s006
                                 + 0.749729 w_k1_AXP_s006
                                 + 0.658103 w_k1_DIS_s006
                                 + 0.768817 w_k1_FDX_s006
                                 + 0.783742 w_k1_IBM_s006
                                 + 0.858873 w_k1_INTC_s006
                                 + 0.750516 w_k1_KO_s006
                                 + 0.414263 w_k1_MAT_s006
                                 + 0.806941 w_k1_MSFT_s006
                                 + 0.893869 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.22599 w_k1_ADM_s006
                                 + 0.919942 w_k1_AXP_s006
                                 + 1.36329 w_k1_DIS_s006
                                 + 1.23414 w_k1_FDX_s006 + 1.0351 w_k1_IBM_s006
                                 + 1.47021 w_k1_INTC_s006
                                 + 1.18367 w_k1_KO_s006
                                 + 0.954424 w_k1_MAT_s006
                                 + 1.27755 w_k1_MSFT_s006
                                 + 1.08097 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.53303 w_k1_ADM_s006
                                 + 1.24278 w_k1_AXP_s006
                                 + 0.992977 w_k1_DIS_s006
                                 + 1.12614 w_k1_FDX_s006
                                 + 1.15315 w_k1_IBM_s006
                                 + 1.19733 w_k1_INTC_s006 + 1.3221 w_k1_KO_s006
                                 + 0.982565 w_k1_MAT_s006
                                 + 1.07025 w_k1_MSFT_s006
                                 + 1.19994 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.18192 w_k1_ADM_s006
                                 + 0.971758 w_k1_AXP_s006
                                 + 0.856227 w_k1_DIS_s006
                                 + 1.02938 w_k1_FDX_s006
                                 + 0.783464 w_k1_IBM_s006
                                 + 0.90255 w_k1_INTC_s006
                                 + 0.951934 w_k1_KO_s006
                                 + 0.784828 w_k1_MAT_s006
                                 + 1.0354 w_k1_MSFT_s006 + 1.07341 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.42828 w_k1_ADM_s006
                                 + 1.37918 w_k1_AXP_s006
                                 + 1.51866 w_k1_DIS_s006
                                 + 1.33215 w_k1_FDX_s006
                                 + 1.43761 w_k1_IBM_s006
                                 + 1.37342 w_k1_INTC_s006
                                 + 1.46149 w_k1_KO_s006 + 1.26515 w_k1_MAT_s006
                                 + 1.3369 w_k1_MSFT_s006 + 1.30402 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.40788 w_k1_ADM_s007
                                 + 1.36741 w_k1_AXP_s007
                                 + 1.00896 w_k1_DIS_s007
                                 + 1.15034 w_k1_FDX_s007
                                 + 1.05155 w_k1_IBM_s007
                                 + 1.10329 w_k1_INTC_s007
                                 + 0.959016 w_k1_KO_s007
                                 + 1.31583 w_k1_MAT_s007
                                 + 1.18746 w_k1_MSFT_s007
                                 + 1.04492 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.00299 w_k1_ADM_s007
                                 + 0.941717 w_k1_AXP_s007
                                 + 1.20204 w_k1_DIS_s007
                                 + 1.06962 w_k1_FDX_s007
                                 + 1.11878 w_k1_IBM_s007
                                 + 1.12987 w_k1_INTC_s007
                                 + 0.989724 w_k1_KO_s007
                                 + 1.24374 w_k1_MAT_s007
                                 + 1.2977 w_k1_MSFT_s007 + 1.03134 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.39835 w_k1_ADM_s007
                                 + 1.41273 w_k1_AXP_s007
                                 + 1.47146 w_k1_DIS_s007
                                 + 1.40402 w_k1_FDX_s007
                                 + 1.24043 w_k1_IBM_s007
                                 + 1.23979 w_k1_INTC_s007
                                 + 1.34743 w_k1_KO_s007 + 1.68752 w_k1_MAT_s007
                                 + 1.36572 w_k1_MSFT_s007
                                 + 1.25755 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.793686 w_k1_ADM_s007
                                 + 0.612434 w_k1_AXP_s007
                                 + 0.72265 w_k1_DIS_s007
                                 + 0.787885 w_k1_FDX_s007
                                 + 0.838853 w_k1_IBM_s007
                                 + 0.857777 w_k1_INTC_s007
                                 + 0.824832 w_k1_KO_s007
                                 + 0.899112 w_k1_MAT_s007
                                 + 0.889354 w_k1_MSFT_s007
                                 + 0.79921 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.27556 w_k1_ADM_s007
                                 + 0.842779 w_k1_AXP_s007
                                 + 0.922071 w_k1_DIS_s007
                                 + 1.11204 w_k1_FDX_s007
                                 + 0.722851 w_k1_IBM_s007
                                 + 0.907622 w_k1_INTC_s007
                                 + 0.803092 w_k1_KO_s007
                                 + 0.711223 w_k1_MAT_s007
                                 + 1.09541 w_k1_MSFT_s007 + 1.0235 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.12559 w_k1_ADM_s007
                                 + 1.0471 w_k1_AXP_s007 + 1.07475 w_k1_DIS_s007
                                 + 1.08656 w_k1_FDX_s007
                                 + 1.41402 w_k1_IBM_s007
                                 + 0.945762 w_k1_INTC_s007
                                 + 1.04085 w_k1_KO_s007 + 2.07005 w_k1_MAT_s007
                                 + 1.08428 w_k1_MSFT_s007
                                 + 1.09165 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.06785 w_k1_ADM_s007
                                 + 1.00368 w_k1_AXP_s007
                                 + 0.855499 w_k1_DIS_s007
                                 + 0.96758 w_k1_FDX_s007
                                 + 1.02484 w_k1_IBM_s007
                                 + 0.885534 w_k1_INTC_s007
                                 + 0.8348 w_k1_KO_s007 + 1.16721 w_k1_MAT_s007
                                 + 0.966245 w_k1_MSFT_s007
                                 + 0.839372 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.0382 w_k1_ADM_s007
                                 + 0.740702 w_k1_AXP_s007
                                 + 1.28006 w_k1_DIS_s007
                                 + 1.12181 w_k1_FDX_s007
                                 + 0.975188 w_k1_IBM_s007
                                 + 0.982269 w_k1_INTC_s007
                                 + 0.964779 w_k1_KO_s007
                                 + 0.991162 w_k1_MAT_s007
                                 + 1.19987 w_k1_MSFT_s007
                                 + 0.969332 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.1422 w_k1_ADM_s010
                                 + 1.3075 w_k1_AXP_s010 + 1.90993 w_k1_DIS_s010
                                 + 1.29612 w_k1_FDX_s010
                                 + 1.19448 w_k1_IBM_s010
                                 + 1.50623 w_k1_INTC_s010
                                 + 1.54681 w_k1_KO_s010 + 1.36076 w_k1_MAT_s010
                                 + 1.56853 w_k1_MSFT_s010
                                 + 1.02816 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.2155 w_k1_ADM_s010
                                 + 1.48623 w_k1_AXP_s010
                                 + 1.80792 w_k1_DIS_s010
                                 + 1.23135 w_k1_FDX_s010
                                 + 1.70971 w_k1_IBM_s010
                                 + 1.17268 w_k1_INTC_s010
                                 + 1.46735 w_k1_KO_s010 + 2.28528 w_k1_MAT_s010
                                 + 1.24589 w_k1_MSFT_s010
                                 + 1.19068 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.738664 w_k1_ADM_s010
                                 + 0.732338 w_k1_AXP_s010
                                 + 0.881523 w_k1_DIS_s010
                                 + 0.883188 w_k1_FDX_s010
                                 + 0.698513 w_k1_IBM_s010
                                 + 0.894466 w_k1_INTC_s010
                                 + 0.821553 w_k1_KO_s010
                                 + 1.03971 w_k1_MAT_s010
                                 + 0.917788 w_k1_MSFT_s010
                                 + 0.922037 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.935112 w_k1_ADM_s010
                                 + 1.03796 w_k1_AXP_s010
                                 + 1.23761 w_k1_DIS_s010
                                 + 1.12762 w_k1_FDX_s010
                                 + 0.992312 w_k1_IBM_s010
                                 + 1.14842 w_k1_INTC_s010
                                 + 1.15043 w_k1_KO_s010 + 1.24043 w_k1_MAT_s010
                                 + 1.19538 w_k1_MSFT_s010
                                 + 1.03763 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.871196 w_k1_ADM_s010
                                 + 1.1362 w_k1_AXP_s010 + 1.15726 w_k1_DIS_s010
                                 + 1.05903 w_k1_FDX_s010
                                 + 1.40608 w_k1_IBM_s010
                                 + 1.03196 w_k1_INTC_s010
                                 + 1.26009 w_k1_KO_s010 + 2.03594 w_k1_MAT_s010
                                 + 1.12351 w_k1_MSFT_s010
                                 + 1.12249 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.35405 w_k1_ADM_s010
                                 + 1.49272 w_k1_AXP_s010
                                 + 1.09634 w_k1_DIS_s010
                                 + 1.22753 w_k1_FDX_s010
                                 + 1.01322 w_k1_IBM_s010
                                 + 1.15273 w_k1_INTC_s010
                                 + 1.28812 w_k1_KO_s010 + 1.61501 w_k1_MAT_s010
                                 + 1.34396 w_k1_MSFT_s010 + 1.1449 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.781181 w_k1_ADM_s010
                                 + 0.682928 w_k1_AXP_s010
                                 + 0.596252 w_k1_DIS_s010
                                 + 0.735711 w_k1_FDX_s010
                                 + 0.641754 w_k1_IBM_s010
                                 + 0.730656 w_k1_INTC_s010
                                 + 0.696736 w_k1_KO_s010
                                 + 0.73828 w_k1_MAT_s010
                                 + 0.721289 w_k1_MSFT_s010
                                 + 0.788479 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.811475 w_k1_ADM_s010
                                 + 0.966902 w_k1_AXP_s010
                                 + 0.992267 w_k1_DIS_s010
                                 + 0.948719 w_k1_FDX_s010
                                 + 1.06537 w_k1_IBM_s010
                                 + 1.01354 w_k1_INTC_s010
                                 + 0.954722 w_k1_KO_s010
                                 + 1.24008 w_k1_MAT_s010
                                 + 1.11224 w_k1_MSFT_s010
                                 + 0.99622 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.917136 w_k1_ADM_s011
                                 + 0.836208 w_k1_AXP_s011
                                 + 0.716564 w_k1_DIS_s011
                                 + 0.739388 w_k1_FDX_s011
                                 + 1.06374 w_k1_IBM_s011
                                 + 0.758661 w_k1_INTC_s011
                                 + 0.831989 w_k1_KO_s011
                                 + 1.09504 w_k1_MAT_s011
                                 + 0.719127 w_k1_MSFT_s011
                                 + 0.877233 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.934394 w_k1_ADM_s011
                                 + 0.884215 w_k1_AXP_s011
                                 + 1.02536 w_k1_DIS_s011
                                 + 0.888576 w_k1_FDX_s011
                                 + 1.14727 w_k1_IBM_s011
                                 + 1.14941 w_k1_INTC_s011
                                 + 1.14049 w_k1_KO_s011 + 1.02987 w_k1_MAT_s011
                                 + 1.05565 w_k1_MSFT_s011
                                 + 0.922256 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.21223 w_k1_ADM_s011
                                 + 1.1407 w_k1_AXP_s011 + 1.32296 w_k1_DIS_s011
                                 + 1.11536 w_k1_FDX_s011
                                 + 0.986023 w_k1_IBM_s011
                                 + 1.01286 w_k1_INTC_s011
                                 + 1.11336 w_k1_KO_s011 + 1.06187 w_k1_MAT_s011
                                 + 1.28729 w_k1_MSFT_s011
                                 + 1.06768 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.945678 w_k1_ADM_s011
                                 + 0.996596 w_k1_AXP_s011
                                 + 1.00744 w_k1_DIS_s011
                                 + 1.09429 w_k1_FDX_s011
                                 + 0.657564 w_k1_IBM_s011
                                 + 0.872857 w_k1_INTC_s011
                                 + 0.885658 w_k1_KO_s011
                                 + 0.862868 w_k1_MAT_s011
                                 + 0.993395 w_k1_MSFT_s011
                                 + 0.927718 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.897024 w_k1_ADM_s011
                                 + 1.10717 w_k1_AXP_s011
                                 + 0.806329 w_k1_DIS_s011
                                 + 0.798986 w_k1_FDX_s011
                                 + 1.46491 w_k1_IBM_s011
                                 + 0.820513 w_k1_INTC_s011
                                 + 1.07906 w_k1_KO_s011 + 1.5435 w_k1_MAT_s011
                                 + 0.82631 w_k1_MSFT_s011
                                 + 0.857546 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.746361 w_k1_ADM_s011
                                 + 0.797861 w_k1_AXP_s011
                                 + 0.720909 w_k1_DIS_s011
                                 + 0.814221 w_k1_FDX_s011
                                 + 0.747531 w_k1_IBM_s011
                                 + 0.852488 w_k1_INTC_s011
                                 + 0.800223 w_k1_KO_s011
                                 + 0.852148 w_k1_MAT_s011
                                 + 0.760584 w_k1_MSFT_s011
                                 + 0.834177 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.09089 w_k1_ADM_s011
                                 + 1.39542 w_k1_AXP_s011
                                 + 1.43322 w_k1_DIS_s011
                                 + 1.21055 w_k1_FDX_s011
                                 + 1.72723 w_k1_IBM_s011
                                 + 1.36703 w_k1_INTC_s011
                                 + 1.76374 w_k1_KO_s011 + 1.76967 w_k1_MAT_s011
                                 + 1.22085 w_k1_MSFT_s011
                                 + 1.22847 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.962864 w_k1_ADM_s011
                                 + 1.31909 w_k1_AXP_s011 + 1.0979 w_k1_DIS_s011
                                 + 1.06319 w_k1_FDX_s011
                                 + 1.21309 w_k1_IBM_s011
                                 + 0.857233 w_k1_INTC_s011
                                 + 1.22823 w_k1_KO_s011 + 1.80333 w_k1_MAT_s011
                                 + 1.04952 w_k1_MSFT_s011 + 1.0823 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.960203 w_k1_ADM_s012
                                 + 1.40585 w_k1_AXP_s012
                                 + 1.25086 w_k1_DIS_s012
                                 + 1.00902 w_k1_FDX_s012 + 1.4769 w_k1_IBM_s012
                                 + 0.95449 w_k1_INTC_s012 + 1.3419 w_k1_KO_s012
                                 + 1.74966 w_k1_MAT_s012
                                 + 1.10727 w_k1_MSFT_s012
                                 + 1.17227 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.937552 w_k1_ADM_s012
                                 + 1.02125 w_k1_AXP_s012
                                 + 0.799642 w_k1_DIS_s012
                                 + 0.906437 w_k1_FDX_s012
                                 + 1.0038 w_k1_IBM_s012
                                 + 0.980494 w_k1_INTC_s012
                                 + 1.05936 w_k1_KO_s012
                                 + 0.876034 w_k1_MAT_s012
                                 + 1.10529 w_k1_MSFT_s012 + 0.9806 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.718524 w_k1_ADM_s012
                                 + 0.58134 w_k1_AXP_s012
                                 + 0.543558 w_k1_DIS_s012
                                 + 0.665708 w_k1_FDX_s012
                                 + 0.804636 w_k1_IBM_s012
                                 + 0.901841 w_k1_INTC_s012
                                 + 0.853829 w_k1_KO_s012
                                 + 0.563813 w_k1_MAT_s012
                                 + 0.799306 w_k1_MSFT_s012
                                 + 0.861723 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.933336 w_k1_ADM_s012
                                 + 1.21393 w_k1_AXP_s012
                                 + 0.967963 w_k1_DIS_s012
                                 + 1.03749 w_k1_FDX_s012
                                 + 0.895987 w_k1_IBM_s012
                                 + 0.960491 w_k1_INTC_s012
                                 + 1.13556 w_k1_KO_s012 + 1.01072 w_k1_MAT_s012
                                 + 1.00598 w_k1_MSFT_s012
                                 + 1.07061 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.813809 w_k1_ADM_s012
                                 + 0.911622 w_k1_AXP_s012
                                 + 0.635706 w_k1_DIS_s012
                                 + 0.79563 w_k1_FDX_s012
                                 + 0.756471 w_k1_IBM_s012
                                 + 1.01437 w_k1_INTC_s012
                                 + 0.877242 w_k1_KO_s012
                                 + 0.912323 w_k1_MAT_s012
                                 + 0.893877 w_k1_MSFT_s012
                                 + 0.893876 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.733464 w_k1_ADM_s012
                                 + 0.77889 w_k1_AXP_s012
                                 + 0.699313 w_k1_DIS_s012
                                 + 0.898596 w_k1_FDX_s012
                                 + 0.612013 w_k1_IBM_s012
                                 + 1.08584 w_k1_INTC_s012
                                 + 1.01728 w_k1_KO_s012
                                 + 0.622535 w_k1_MAT_s012
                                 + 1.07018 w_k1_MSFT_s012
                                 + 0.97688 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.843051 w_k1_ADM_s012
                                 + 0.834718 w_k1_AXP_s012
                                 + 0.65163 w_k1_DIS_s012
                                 + 0.74535 w_k1_FDX_s012
                                 + 1.17052 w_k1_IBM_s012
                                 + 0.795105 w_k1_INTC_s012
                                 + 0.948259 w_k1_KO_s012
                                 + 1.17357 w_k1_MAT_s012
                                 + 0.761313 w_k1_MSFT_s012
                                 + 0.87394 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.361 w_k1_ADM_s012
                                 + 1.29213 w_k1_AXP_s012
                                 + 1.61113 w_k1_DIS_s012
                                 + 1.44457 w_k1_FDX_s012
                                 + 0.932376 w_k1_IBM_s012
                                 + 1.37317 w_k1_INTC_s012
                                 + 1.36662 w_k1_KO_s012
                                 + 0.941969 w_k1_MAT_s012
                                 + 1.42822 w_k1_MSFT_s012
                                 + 1.26655 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.05173 w_k1_ADM_s013
                                 + 0.836757 w_k1_AXP_s013
                                 + 1.13803 w_k1_DIS_s013
                                 + 1.09716 w_k1_FDX_s013
                                 + 0.746599 w_k1_IBM_s013
                                 + 1.12619 w_k1_INTC_s013
                                 + 0.939009 w_k1_KO_s013
                                 + 0.482238 w_k1_MAT_s013
                                 + 1.20056 w_k1_MSFT_s013
                                 + 0.918947 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.851629 w_k1_ADM_s013
                                 + 0.813407 w_k1_AXP_s013
                                 + 0.707588 w_k1_DIS_s013
                                 + 0.689083 w_k1_FDX_s013
                                 + 1.03208 w_k1_IBM_s013
                                 + 0.703216 w_k1_INTC_s013
                                 + 0.904567 w_k1_KO_s013
                                 + 0.971198 w_k1_MAT_s013
                                 + 0.795336 w_k1_MSFT_s013
                                 + 0.843309 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.00202 w_k1_ADM_s013
                                 + 1.15663 w_k1_AXP_s013
                                 + 1.27232 w_k1_DIS_s013
                                 + 1.13793 w_k1_FDX_s013
                                 + 1.09468 w_k1_IBM_s013
                                 + 1.17204 w_k1_INTC_s013
                                 + 1.11912 w_k1_KO_s013 + 1.27256 w_k1_MAT_s013
                                 + 1.2858 w_k1_MSFT_s013 + 1.09033 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.964104 w_k1_ADM_s013
                                 + 1.25513 w_k1_AXP_s013
                                 + 1.43498 w_k1_DIS_s013
                                 + 1.02562 w_k1_FDX_s013 + 1.7464 w_k1_IBM_s013
                                 + 1.29032 w_k1_INTC_s013
                                 + 1.24303 w_k1_KO_s013 + 1.00126 w_k1_MAT_s013
                                 + 1.14807 w_k1_MSFT_s013
                                 + 1.22438 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.844781 w_k1_ADM_s013
                                 + 1.01919 w_k1_AXP_s013
                                 + 1.03851 w_k1_DIS_s013
                                 + 0.867857 w_k1_FDX_s013
                                 + 1.27462 w_k1_IBM_s013
                                 + 1.07003 w_k1_INTC_s013
                                 + 1.09762 w_k1_KO_s013 + 1.17603 w_k1_MAT_s013
                                 + 1.11747 w_k1_MSFT_s013
                                 + 1.08261 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.915115 w_k1_ADM_s013
                                 + 0.781293 w_k1_AXP_s013
                                 + 0.780209 w_k1_DIS_s013
                                 + 0.789244 w_k1_FDX_s013
                                 + 0.689833 w_k1_IBM_s013
                                 + 0.962873 w_k1_INTC_s013
                                 + 0.769287 w_k1_KO_s013
                                 + 0.494292 w_k1_MAT_s013
                                 + 0.88859 w_k1_MSFT_s013
                                 + 0.833136 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.40327 w_k1_ADM_s013
                                 + 1.67163 w_k1_AXP_s013
                                 + 1.40254 w_k1_DIS_s013
                                 + 1.34015 w_k1_FDX_s013
                                 + 1.08772 w_k1_IBM_s013
                                 + 1.16095 w_k1_INTC_s013
                                 + 1.07207 w_k1_KO_s013 + 1.15299 w_k1_MAT_s013
                                 + 1.40584 w_k1_MSFT_s013
                                 + 1.20423 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.14418 w_k1_ADM_s013
                                 + 1.30438 w_k1_AXP_s013
                                 + 1.64686 w_k1_DIS_s013
                                 + 1.45911 w_k1_FDX_s013
                                 + 0.818253 w_k1_IBM_s013
                                 + 1.22987 w_k1_INTC_s013
                                 + 1.18933 w_k1_KO_s013 + 1.0726 w_k1_MAT_s013
                                 + 1.58002 w_k1_MSFT_s013
                                 + 1.15629 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.0518 w_k1_ADM_s014
                                 + 0.952149 w_k1_AXP_s014
                                 + 1.19394 w_k1_DIS_s014
                                 + 1.01009 w_k1_FDX_s014
                                 + 0.996047 w_k1_IBM_s014
                                 + 1.11559 w_k1_INTC_s014
                                 + 1.03748 w_k1_KO_s014
                                 + 0.907488 w_k1_MAT_s014
                                 + 1.07638 w_k1_MSFT_s014
                                 + 0.992971 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.672232 w_k1_ADM_s014
                                 + 0.769032 w_k1_AXP_s014
                                 + 0.574977 w_k1_DIS_s014
                                 + 0.776953 w_k1_FDX_s014
                                 + 0.790778 w_k1_IBM_s014
                                 + 0.908456 w_k1_INTC_s014
                                 + 0.796801 w_k1_KO_s014
                                 + 0.853225 w_k1_MAT_s014
                                 + 0.826409 w_k1_MSFT_s014
                                 + 0.830472 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.862302 w_k1_ADM_s014
                                 + 0.836773 w_k1_AXP_s014
                                 + 0.889324 w_k1_DIS_s014
                                 + 0.812455 w_k1_FDX_s014
                                 + 1.09777 w_k1_IBM_s014
                                 + 1.0017 w_k1_INTC_s014
                                 + 0.930195 w_k1_KO_s014
                                 + 0.701146 w_k1_MAT_s014
                                 + 0.853223 w_k1_MSFT_s014
                                 + 0.807327 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.719414 w_k1_ADM_s014
                                 + 0.872908 w_k1_AXP_s014
                                 + 0.779748 w_k1_DIS_s014
                                 + 0.829642 w_k1_FDX_s014
                                 + 1.07468 w_k1_IBM_s014
                                 + 0.936996 w_k1_INTC_s014
                                 + 0.931985 w_k1_KO_s014
                                 + 1.17644 w_k1_MAT_s014
                                 + 0.785757 w_k1_MSFT_s014
                                 + 0.945733 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.914233 w_k1_ADM_s014
                                 + 1.14343 w_k1_AXP_s014
                                 + 0.954605 w_k1_DIS_s014
                                 + 1.0864 w_k1_FDX_s014
                                 + 0.756557 w_k1_IBM_s014
                                 + 1.04345 w_k1_INTC_s014
                                 + 0.886684 w_k1_KO_s014
                                 + 0.90154 w_k1_MAT_s014
                                 + 0.946851 w_k1_MSFT_s014
                                 + 0.903514 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.05311 w_k1_ADM_s014
                                 + 1.44506 w_k1_AXP_s014
                                 + 1.55143 w_k1_DIS_s014
                                 + 1.32527 w_k1_FDX_s014 + 1.1299 w_k1_IBM_s014
                                 + 1.56602 w_k1_INTC_s014
                                 + 1.43949 w_k1_KO_s014 + 1.45582 w_k1_MAT_s014
                                 + 1.4659 w_k1_MSFT_s014 + 1.22896 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.10336 w_k1_ADM_s014
                                 + 1.45789 w_k1_AXP_s014
                                 + 1.00315 w_k1_DIS_s014
                                 + 1.03972 w_k1_FDX_s014
                                 + 1.20878 w_k1_IBM_s014
                                 + 1.23216 w_k1_INTC_s014
                                 + 1.15745 w_k1_KO_s014 + 1.29884 w_k1_MAT_s014
                                 + 1.03442 w_k1_MSFT_s014
                                 + 0.937956 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.966276 w_k1_ADM_s014
                                 + 1.16594 w_k1_AXP_s014
                                 + 1.15969 w_k1_DIS_s014
                                 + 0.884767 w_k1_FDX_s014
                                 + 1.48332 w_k1_IBM_s014
                                 + 1.09582 w_k1_INTC_s014
                                 + 1.23561 w_k1_KO_s014 + 1.37734 w_k1_MAT_s014
                                 + 1.01654 w_k1_MSFT_s014
                                 + 1.01282 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.13575 w_k1_ADM_s015
                                 + 1.26676 w_k1_AXP_s015
                                 + 0.913376 w_k1_DIS_s015
                                 + 1.10899 w_k1_FDX_s015
                                 + 1.02879 w_k1_IBM_s015
                                 + 1.19752 w_k1_INTC_s015
                                 + 0.896194 w_k1_KO_s015
                                 + 1.34364 w_k1_MAT_s015
                                 + 1.09957 w_k1_MSFT_s015
                                 + 1.01187 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.04382 w_k1_ADM_s015
                                 + 0.915562 w_k1_AXP_s015
                                 + 1.07834 w_k1_DIS_s015
                                 + 0.902979 w_k1_FDX_s015
                                 + 1.06756 w_k1_IBM_s015
                                 + 0.929691 w_k1_INTC_s015
                                 + 1.0886 w_k1_KO_s015 + 1.00778 w_k1_MAT_s015
                                 + 1.07936 w_k1_MSFT_s015
                                 + 1.02121 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.30228 w_k1_ADM_s015
                                 + 1.20647 w_k1_AXP_s015
                                 + 1.07138 w_k1_DIS_s015
                                 + 1.03621 w_k1_FDX_s015
                                 + 0.945951 w_k1_IBM_s015
                                 + 1.03316 w_k1_INTC_s015
                                 + 1.01237 w_k1_KO_s015 + 1.11962 w_k1_MAT_s015
                                 + 1.20138 w_k1_MSFT_s015
                                 + 1.07209 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.01754 w_k1_ADM_s015
                                 + 0.940866 w_k1_AXP_s015
                                 + 0.849065 w_k1_DIS_s015
                                 + 0.900313 w_k1_FDX_s015
                                 + 1.13265 w_k1_IBM_s015
                                 + 0.931713 w_k1_INTC_s015
                                 + 1.04659 w_k1_KO_s015 + 1.07502 w_k1_MAT_s015
                                 + 0.931026 w_k1_MSFT_s015
                                 + 1.04437 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.77913 w_k1_ADM_s015
                                 + 0.662501 w_k1_AXP_s015
                                 + 0.648271 w_k1_DIS_s015
                                 + 0.749991 w_k1_FDX_s015
                                 + 0.818002 w_k1_IBM_s015
                                 + 0.843198 w_k1_INTC_s015
                                 + 0.850978 w_k1_KO_s015
                                 + 0.670502 w_k1_MAT_s015
                                 + 0.935037 w_k1_MSFT_s015
                                 + 0.831399 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.27545 w_k1_ADM_s015
                                 + 0.885255 w_k1_AXP_s015
                                 + 1.22998 w_k1_DIS_s015 + 1.2187 w_k1_FDX_s015
                                 + 0.867814 w_k1_IBM_s015
                                 + 1.20084 w_k1_INTC_s015
                                 + 1.09157 w_k1_KO_s015
                                 + 0.691697 w_k1_MAT_s015
                                 + 1.21384 w_k1_MSFT_s015
                                 + 1.12216 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.219 w_k1_ADM_s015
                                 + 1.55416 w_k1_AXP_s015
                                 + 1.57521 w_k1_DIS_s015
                                 + 1.35116 w_k1_FDX_s015
                                 + 1.08135 w_k1_IBM_s015
                                 + 1.2794 w_k1_INTC_s015 + 1.24948 w_k1_KO_s015
                                 + 1.36531 w_k1_MAT_s015
                                 + 1.38306 w_k1_MSFT_s015
                                 + 1.27293 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.09193 w_k1_ADM_s015
                                 + 1.23105 w_k1_AXP_s015
                                 + 1.44545 w_k1_DIS_s015
                                 + 1.16113 w_k1_FDX_s015
                                 + 1.46231 w_k1_IBM_s015
                                 + 1.00082 w_k1_INTC_s015
                                 + 1.57024 w_k1_KO_s015 + 1.83839 w_k1_MAT_s015
                                 + 1.30068 w_k1_MSFT_s015
                                 + 1.27886 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.899108 w_k1_ADM_s016
                                 + 0.984177 w_k1_AXP_s016
                                 + 0.802206 w_k1_DIS_s016
                                 + 0.947388 w_k1_FDX_s016
                                 + 0.777382 w_k1_IBM_s016
                                 + 0.824577 w_k1_INTC_s016
                                 + 0.853193 w_k1_KO_s016
                                 + 0.955444 w_k1_MAT_s016
                                 + 0.909927 w_k1_MSFT_s016
                                 + 0.956761 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.958853 w_k1_ADM_s016
                                 + 0.890428 w_k1_AXP_s016
                                 + 0.687182 w_k1_DIS_s016
                                 + 0.7716 w_k1_FDX_s016
                                 + 0.818126 w_k1_IBM_s016
                                 + 0.835157 w_k1_INTC_s016
                                 + 0.78069 w_k1_KO_s016
                                 + 0.663447 w_k1_MAT_s016
                                 + 0.733637 w_k1_MSFT_s016
                                 + 0.814935 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.212 w_k1_ADM_s016
                                 + 1.13433 w_k1_AXP_s016
                                 + 0.993973 w_k1_DIS_s016
                                 + 1.0447 w_k1_FDX_s016 + 1.03609 w_k1_IBM_s016
                                 + 0.949372 w_k1_INTC_s016
                                 + 1.06896 w_k1_KO_s016
                                 + 0.882859 w_k1_MAT_s016
                                 + 1.05932 w_k1_MSFT_s016
                                 + 1.07869 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.48347 w_k1_ADM_s016
                                 + 1.16776 w_k1_AXP_s016
                                 + 1.65125 w_k1_DIS_s016
                                 + 1.22158 w_k1_FDX_s016
                                 + 1.40304 w_k1_IBM_s016
                                 + 1.43546 w_k1_INTC_s016
                                 + 1.56576 w_k1_KO_s016 + 1.48715 w_k1_MAT_s016
                                 + 1.33905 w_k1_MSFT_s016
                                 + 1.26984 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.37571 w_k1_ADM_s016
                                 + 1.91442 w_k1_AXP_s016
                                 + 1.63616 w_k1_DIS_s016
                                 + 1.31662 w_k1_FDX_s016 + 1.7703 w_k1_IBM_s016
                                 + 1.21516 w_k1_INTC_s016
                                 + 1.74817 w_k1_KO_s016 + 2.65304 w_k1_MAT_s016
                                 + 1.30366 w_k1_MSFT_s016
                                 + 1.26195 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.33664 w_k1_ADM_s016
                                 + 1.51167 w_k1_AXP_s016
                                 + 1.28506 w_k1_DIS_s016
                                 + 1.28273 w_k1_FDX_s016
                                 + 1.14556 w_k1_IBM_s016
                                 + 1.03739 w_k1_INTC_s016
                                 + 1.18056 w_k1_KO_s016 + 1.41886 w_k1_MAT_s016
                                 + 1.11927 w_k1_MSFT_s016
                                 + 1.15012 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.845872 w_k1_ADM_s016
                                 + 1.06615 w_k1_AXP_s016
                                 + 1.03295 w_k1_DIS_s016
                                 + 0.879496 w_k1_FDX_s016
                                 + 1.28441 w_k1_IBM_s016
                                 + 0.981231 w_k1_INTC_s016
                                 + 1.20919 w_k1_KO_s016 + 1.46294 w_k1_MAT_s016
                                 + 1.06664 w_k1_MSFT_s016
                                 + 0.977697 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.865145 w_k1_ADM_s016
                                 + 0.691999 w_k1_AXP_s016
                                 + 0.635749 w_k1_DIS_s016
                                 + 0.751131 w_k1_FDX_s016
                                 + 0.616909 w_k1_IBM_s016
                                 + 0.856325 w_k1_INTC_s016
                                 + 0.754516 w_k1_KO_s016
                                 + 0.53319 w_k1_MAT_s016
                                 + 0.820404 w_k1_MSFT_s016
                                 + 0.786261 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.817184 w_k1_ADM_s017
                                 + 0.853407 w_k1_AXP_s017
                                 + 0.832909 w_k1_DIS_s017
                                 + 0.863801 w_k1_FDX_s017
                                 + 0.786906 w_k1_IBM_s017
                                 + 0.936694 w_k1_INTC_s017
                                 + 0.896273 w_k1_KO_s017
                                 + 0.704851 w_k1_MAT_s017
                                 + 0.913053 w_k1_MSFT_s017
                                 + 0.953766 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.874411 w_k1_ADM_s017
                                 + 0.796104 w_k1_AXP_s017
                                 + 0.633983 w_k1_DIS_s017
                                 + 0.769182 w_k1_FDX_s017
                                 + 0.804379 w_k1_IBM_s017
                                 + 0.877746 w_k1_INTC_s017
                                 + 0.792126 w_k1_KO_s017
                                 + 0.604452 w_k1_MAT_s017
                                 + 0.739855 w_k1_MSFT_s017
                                 + 0.841123 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.22817 w_k1_ADM_s017
                                 + 1.69395 w_k1_AXP_s017
                                 + 1.34743 w_k1_DIS_s017 + 1.2276 w_k1_FDX_s017
                                 + 1.43874 w_k1_IBM_s017
                                 + 1.3842 w_k1_INTC_s017 + 1.51481 w_k1_KO_s017
                                 + 1.63822 w_k1_MAT_s017
                                 + 1.27375 w_k1_MSFT_s017
                                 + 0.989915 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.10034 w_k1_ADM_s017
                                 + 0.979488 w_k1_AXP_s017
                                 + 1.12637 w_k1_DIS_s017
                                 + 1.01086 w_k1_FDX_s017 + 1.0522 w_k1_IBM_s017
                                 + 1.04532 w_k1_INTC_s017 + 1.2767 w_k1_KO_s017
                                 + 0.824359 w_k1_MAT_s017
                                 + 1.09657 w_k1_MSFT_s017
                                 + 1.05309 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.967783 w_k1_ADM_s017
                                 + 1.13735 w_k1_AXP_s017
                                 + 1.03249 w_k1_DIS_s017 + 1.1233 w_k1_FDX_s017
                                 + 0.819757 w_k1_IBM_s017
                                 + 1.01131 w_k1_INTC_s017
                                 + 1.07141 w_k1_KO_s017
                                 + 0.840554 w_k1_MAT_s017
                                 + 0.943723 w_k1_MSFT_s017
                                 + 0.910964 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.10624 w_k1_ADM_s017
                                 + 1.27759 w_k1_AXP_s017
                                 + 1.55142 w_k1_DIS_s017
                                 + 1.23313 w_k1_FDX_s017
                                 + 1.44803 w_k1_IBM_s017
                                 + 0.968096 w_k1_INTC_s017
                                 + 1.52211 w_k1_KO_s017 + 1.56834 w_k1_MAT_s017
                                 + 1.12557 w_k1_MSFT_s017
                                 + 1.08065 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.728893 w_k1_ADM_s017
                                 + 1.03648 w_k1_AXP_s017
                                 + 0.809961 w_k1_DIS_s017
                                 + 0.874953 w_k1_FDX_s017
                                 + 1.25729 w_k1_IBM_s017
                                 + 1.12893 w_k1_INTC_s017
                                 + 1.04349 w_k1_KO_s017 + 1.04026 w_k1_MAT_s017
                                 + 0.80302 w_k1_MSFT_s017
                                 + 1.02986 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.08146 w_k1_ADM_s017
                                 + 0.985179 w_k1_AXP_s017
                                 + 1.38536 w_k1_DIS_s017
                                 + 1.10159 w_k1_FDX_s017
                                 + 1.32944 w_k1_IBM_s017
                                 + 1.22229 w_k1_INTC_s017
                                 + 1.35978 w_k1_KO_s017
                                 + 0.948371 w_k1_MAT_s017
                                 + 1.19665 w_k1_MSFT_s017
                                 + 1.02769 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.837333 w_k1_ADM_s100
                                 + 0.70985 w_k1_AXP_s100
                                 + 0.571864 w_k1_DIS_s100
                                 + 0.756043 w_k1_FDX_s100
                                 + 0.728138 w_k1_IBM_s100
                                 + 0.906773 w_k1_INTC_s100
                                 + 0.668486 w_k1_KO_s100
                                 + 0.566715 w_k1_MAT_s100
                                 + 0.694708 w_k1_MSFT_s100
                                 + 0.772137 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.08112 w_k1_ADM_s100
                                 + 1.59847 w_k1_AXP_s100
                                 + 1.12536 w_k1_DIS_s100
                                 + 1.39556 w_k1_FDX_s100
                                 + 0.901253 w_k1_IBM_s100
                                 + 1.20839 w_k1_INTC_s100
                                 + 0.958594 w_k1_KO_s100
                                 + 1.13689 w_k1_MAT_s100
                                 + 1.04993 w_k1_MSFT_s100
                                 + 1.04985 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.839334 w_k1_ADM_s100
                                 + 1.14443 w_k1_AXP_s100
                                 + 0.987416 w_k1_DIS_s100
                                 + 0.886034 w_k1_FDX_s100
                                 + 1.55072 w_k1_IBM_s100
                                 + 1.01138 w_k1_INTC_s100
                                 + 1.07974 w_k1_KO_s100 + 1.31101 w_k1_MAT_s100
                                 + 0.856493 w_k1_MSFT_s100
                                 + 0.916628 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.886751 w_k1_ADM_s100
                                 + 0.972413 w_k1_AXP_s100
                                 + 1.12417 w_k1_DIS_s100 + 1.042 w_k1_FDX_s100
                                 + 1.18803 w_k1_IBM_s100
                                 + 1.15454 w_k1_INTC_s100 + 1.1881 w_k1_KO_s100
                                 + 0.789713 w_k1_MAT_s100
                                 + 1.04716 w_k1_MSFT_s100
                                 + 1.01898 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.811841 w_k1_ADM_s100
                                 + 1.01944 w_k1_AXP_s100
                                 + 0.751154 w_k1_DIS_s100
                                 + 1.01496 w_k1_FDX_s100
                                 + 0.613998 w_k1_IBM_s100
                                 + 1.11769 w_k1_INTC_s100
                                 + 0.70069 w_k1_KO_s100
                                 + 0.688306 w_k1_MAT_s100
                                 + 0.943461 w_k1_MSFT_s100
                                 + 0.903321 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.28611 w_k1_ADM_s100
                                 + 1.58212 w_k1_AXP_s100
                                 + 1.35239 w_k1_DIS_s100
                                 + 1.40651 w_k1_FDX_s100 + 1.3622 w_k1_IBM_s100
                                 + 1.29962 w_k1_INTC_s100
                                 + 1.37424 w_k1_KO_s100 + 1.53287 w_k1_MAT_s100
                                 + 1.19147 w_k1_MSFT_s100
                                 + 1.24525 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.08592 w_k1_ADM_s100
                                 + 1.12136 w_k1_AXP_s100
                                 + 1.00238 w_k1_DIS_s100
                                 + 1.11567 w_k1_FDX_s100
                                 + 0.892999 w_k1_IBM_s100
                                 + 1.06282 w_k1_INTC_s100
                                 + 0.992048 w_k1_KO_s100
                                 + 1.09856 w_k1_MAT_s100
                                 + 1.10005 w_k1_MSFT_s100
                                 + 0.984324 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.759248 w_k1_ADM_s100
                                 + 0.724599 w_k1_AXP_s100
                                 + 0.682992 w_k1_DIS_s100
                                 + 0.751691 w_k1_FDX_s100
                                 + 0.976709 w_k1_IBM_s100
                                 + 0.761496 w_k1_INTC_s100
                                 + 0.748445 w_k1_KO_s100
                                 + 0.823048 w_k1_MAT_s100
                                 + 0.72509 w_k1_MSFT_s100
                                 + 0.787093 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.968822 w_k1_ADM_s101
                                 + 1.18486 w_k1_AXP_s101
                                 + 1.55162 w_k1_DIS_s101
                                 + 1.07808 w_k1_FDX_s101
                                 + 1.69025 w_k1_IBM_s101
                                 + 1.42034 w_k1_INTC_s101
                                 + 1.40215 w_k1_KO_s101 + 1.76203 w_k1_MAT_s101
                                 + 1.26236 w_k1_MSFT_s101
                                 + 1.15998 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.867788 w_k1_ADM_s101
                                 + 0.943957 w_k1_AXP_s101
                                 + 0.865229 w_k1_DIS_s101
                                 + 0.978854 w_k1_FDX_s101
                                 + 0.835924 w_k1_IBM_s101
                                 + 0.904447 w_k1_INTC_s101
                                 + 0.892727 w_k1_KO_s101
                                 + 1.32051 w_k1_MAT_s101
                                 + 1.10553 w_k1_MSFT_s101
                                 + 0.915296 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.28854 w_k1_ADM_s101
                                 + 1.56136 w_k1_AXP_s101 + 1.7103 w_k1_DIS_s101
                                 + 1.37666 w_k1_FDX_s101
                                 + 1.05146 w_k1_IBM_s101
                                 + 1.55203 w_k1_INTC_s101
                                 + 1.30689 w_k1_KO_s101 + 1.54213 w_k1_MAT_s101
                                 + 1.34255 w_k1_MSFT_s101
                                 + 1.26568 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.14955 w_k1_ADM_s101
                                 + 1.20542 w_k1_AXP_s101
                                 + 0.888609 w_k1_DIS_s101
                                 + 1.03203 w_k1_FDX_s101 + 1.3109 w_k1_IBM_s101
                                 + 1.12463 w_k1_INTC_s101
                                 + 1.14551 w_k1_KO_s101 + 1.894 w_k1_MAT_s101
                                 + 1.0137 w_k1_MSFT_s101 + 1.04872 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.755599 w_k1_ADM_s101
                                 + 0.666279 w_k1_AXP_s101
                                 + 0.594316 w_k1_DIS_s101
                                 + 0.706959 w_k1_FDX_s101
                                 + 0.71591 w_k1_IBM_s101
                                 + 0.740015 w_k1_INTC_s101
                                 + 0.867608 w_k1_KO_s101
                                 + 1.06318 w_k1_MAT_s101
                                 + 0.794417 w_k1_MSFT_s101
                                 + 0.862818 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.19816 w_k1_ADM_s101
                                 + 0.899242 w_k1_AXP_s101
                                 + 1.26804 w_k1_DIS_s101
                                 + 1.01905 w_k1_FDX_s101
                                 + 0.906183 w_k1_IBM_s101
                                 + 1.19616 w_k1_INTC_s101
                                 + 1.15447 w_k1_KO_s101 + 1.133 w_k1_MAT_s101
                                 + 1.12081 w_k1_MSFT_s101
                                 + 1.11757 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.11767 w_k1_ADM_s101
                                 + 0.955066 w_k1_AXP_s101
                                 + 0.625112 w_k1_DIS_s101
                                 + 0.957057 w_k1_FDX_s101
                                 + 0.590158 w_k1_IBM_s101
                                 + 0.962721 w_k1_INTC_s101
                                 + 0.593081 w_k1_KO_s101
                                 + 0.763569 w_k1_MAT_s101
                                 + 0.899305 w_k1_MSFT_s101
                                 + 0.805809 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.965769 w_k1_ADM_s101
                                 + 0.674528 w_k1_AXP_s101
                                 + 0.847556 w_k1_DIS_s101
                                 + 0.784603 w_k1_FDX_s101
                                 + 1.2053 w_k1_IBM_s101
                                 + 0.956156 w_k1_INTC_s101
                                 + 1.22367 w_k1_KO_s101 + 1.09763 w_k1_MAT_s101
                                 + 0.930008 w_k1_MSFT_s101
                                 + 0.885067 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.07635 w_k1_ADM_s102
                                 + 0.948026 w_k1_AXP_s102
                                 + 1.15053 w_k1_DIS_s102 + 1.0949 w_k1_FDX_s102
                                 + 1.04011 w_k1_IBM_s102
                                 + 1.32546 w_k1_INTC_s102
                                 + 1.17405 w_k1_KO_s102
                                 + 0.804587 w_k1_MAT_s102
                                 + 1.18868 w_k1_MSFT_s102
                                 + 1.09424 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.918257 w_k1_ADM_s102
                                 + 0.810461 w_k1_AXP_s102
                                 + 1.09372 w_k1_DIS_s102
                                 + 0.928397 w_k1_FDX_s102
                                 + 1.08302 w_k1_IBM_s102
                                 + 1.0899 w_k1_INTC_s102 + 1.13675 w_k1_KO_s102
                                 + 0.752379 w_k1_MAT_s102
                                 + 1.16855 w_k1_MSFT_s102
                                 + 1.02955 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.4826 w_k1_ADM_s102
                                 + 1.35879 w_k1_AXP_s102
                                 + 1.58235 w_k1_DIS_s102
                                 + 1.31839 w_k1_FDX_s102
                                 + 1.26713 w_k1_IBM_s102
                                 + 1.22678 w_k1_INTC_s102
                                 + 1.46231 w_k1_KO_s102 + 1.45655 w_k1_MAT_s102
                                 + 1.3553 w_k1_MSFT_s102 + 1.40079 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.813062 w_k1_ADM_s102
                                 + 0.827059 w_k1_AXP_s102
                                 + 0.851605 w_k1_DIS_s102
                                 + 0.967535 w_k1_FDX_s102
                                 + 0.606716 w_k1_IBM_s102
                                 + 0.925235 w_k1_INTC_s102
                                 + 0.809456 w_k1_KO_s102
                                 + 0.665798 w_k1_MAT_s102
                                 + 1.09338 w_k1_MSFT_s102
                                 + 0.929801 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.984768 w_k1_ADM_s102
                                 + 1.32195 w_k1_AXP_s102
                                 + 0.94942 w_k1_DIS_s102
                                 + 1.02288 w_k1_FDX_s102
                                 + 1.29557 w_k1_IBM_s102
                                 + 1.3037 w_k1_INTC_s102 + 1.18165 w_k1_KO_s102
                                 + 0.871563 w_k1_MAT_s102
                                 + 1.10868 w_k1_MSFT_s102
                                 + 1.09305 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04142 w_k1_ADM_s102
                                 + 0.991768 w_k1_AXP_s102
                                 + 1.32388 w_k1_DIS_s102
                                 + 1.11061 w_k1_FDX_s102
                                 + 0.785501 w_k1_IBM_s102
                                 + 1.20923 w_k1_INTC_s102
                                 + 1.05967 w_k1_KO_s102
                                 + 0.784596 w_k1_MAT_s102
                                 + 1.23077 w_k1_MSFT_s102
                                 + 1.04555 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.13467 w_k1_ADM_s102
                                 + 1.172 w_k1_AXP_s102 + 1.10551 w_k1_DIS_s102
                                 + 1.0442 w_k1_FDX_s102
                                 + 0.816028 w_k1_IBM_s102
                                 + 1.08268 w_k1_INTC_s102
                                 + 0.972737 w_k1_KO_s102
                                 + 0.654182 w_k1_MAT_s102
                                 + 1.11094 w_k1_MSFT_s102
                                 + 1.09862 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.800603 w_k1_ADM_s102
                                 + 0.743162 w_k1_AXP_s102
                                 + 0.745278 w_k1_DIS_s102
                                 + 0.765825 w_k1_FDX_s102
                                 + 0.960975 w_k1_IBM_s102
                                 + 1.0942 w_k1_INTC_s102
                                 + 0.929455 w_k1_KO_s102
                                 + 0.718082 w_k1_MAT_s102
                                 + 0.963145 w_k1_MSFT_s102
                                 + 0.955083 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.751976 w_k1_ADM_s103
                                 + 0.877878 w_k1_AXP_s103
                                 + 0.831314 w_k1_DIS_s103
                                 + 0.823127 w_k1_FDX_s103
                                 + 1.27407 w_k1_IBM_s103
                                 + 1.10131 w_k1_INTC_s103
                                 + 1.03916 w_k1_KO_s103 + 1.2435 w_k1_MAT_s103
                                 + 0.968329 w_k1_MSFT_s103
                                 + 0.99291 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.12976 w_k1_ADM_s103
                                 + 1.43903 w_k1_AXP_s103 + 1.2775 w_k1_DIS_s103
                                 + 1.03509 w_k1_FDX_s103
                                 + 1.63134 w_k1_IBM_s103
                                 + 1.14754 w_k1_INTC_s103
                                 + 1.23123 w_k1_KO_s103 + 1.61388 w_k1_MAT_s103
                                 + 1.04879 w_k1_MSFT_s103 + 1.2897 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.819178 w_k1_ADM_s103
                                 + 0.829065 w_k1_AXP_s103
                                 + 0.633077 w_k1_DIS_s103
                                 + 0.766626 w_k1_FDX_s103
                                 + 1.13688 w_k1_IBM_s103
                                 + 0.896624 w_k1_INTC_s103
                                 + 0.81822 w_k1_KO_s103 + 1.1275 w_k1_MAT_s103
                                 + 0.807359 w_k1_MSFT_s103
                                 + 0.84605 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.778371 w_k1_ADM_s103
                                 + 0.605192 w_k1_AXP_s103
                                 + 0.613604 w_k1_DIS_s103
                                 + 0.733738 w_k1_FDX_s103
                                 + 0.69993 w_k1_IBM_s103
                                 + 0.666547 w_k1_INTC_s103
                                 + 0.682992 w_k1_KO_s103
                                 + 0.680257 w_k1_MAT_s103
                                 + 0.776033 w_k1_MSFT_s103
                                 + 0.788642 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.20376 w_k1_ADM_s103
                                 + 1.62936 w_k1_AXP_s103
                                 + 1.42965 w_k1_DIS_s103
                                 + 1.31144 w_k1_FDX_s103
                                 + 0.971324 w_k1_IBM_s103
                                 + 1.35588 w_k1_INTC_s103
                                 + 1.25526 w_k1_KO_s103 + 1.31583 w_k1_MAT_s103
                                 + 1.33267 w_k1_MSFT_s103
                                 + 1.18433 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.01886 w_k1_ADM_s103
                                 + 0.934052 w_k1_AXP_s103
                                 + 1.00903 w_k1_DIS_s103
                                 + 1.01692 w_k1_FDX_s103
                                 + 0.824871 w_k1_IBM_s103
                                 + 1.14806 w_k1_INTC_s103
                                 + 0.939734 w_k1_KO_s103
                                 + 0.798643 w_k1_MAT_s103
                                 + 1.0097 w_k1_MSFT_s103 + 1.02214 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.836177 w_k1_ADM_s103
                                 + 0.787661 w_k1_AXP_s103
                                 + 0.848964 w_k1_DIS_s103
                                 + 0.878063 w_k1_FDX_s103
                                 + 0.712252 w_k1_IBM_s103
                                 + 1.18993 w_k1_INTC_s103
                                 + 0.809731 w_k1_KO_s103
                                 + 0.803257 w_k1_MAT_s103
                                 + 0.955831 w_k1_MSFT_s103
                                 + 0.954944 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.957968 w_k1_ADM_s103
                                 + 1.10566 w_k1_AXP_s103
                                 + 1.07116 w_k1_DIS_s103
                                 + 1.03824 w_k1_FDX_s103
                                 + 1.10955 w_k1_IBM_s103
                                 + 1.36128 w_k1_INTC_s103 + 1.0331 w_k1_KO_s103
                                 + 1.26239 w_k1_MAT_s103
                                 + 1.01117 w_k1_MSFT_s103
                                 + 1.09266 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.788068 w_k1_ADM_s104
                                 + 0.797288 w_k1_AXP_s104
                                 + 1.19774 w_k1_DIS_s104
                                 + 0.99318 w_k1_FDX_s104
                                 + 1.03057 w_k1_IBM_s104
                                 + 1.16091 w_k1_INTC_s104
                                 + 1.31097 w_k1_KO_s104 + 1.06837 w_k1_MAT_s104
                                 + 1.13929 w_k1_MSFT_s104
                                 + 1.14548 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.993206 w_k1_ADM_s104
                                 + 1.09789 w_k1_AXP_s104
                                 + 1.07206 w_k1_DIS_s104
                                 + 1.00244 w_k1_FDX_s104
                                 + 1.36521 w_k1_IBM_s104
                                 + 1.18337 w_k1_INTC_s104
                                 + 1.22779 w_k1_KO_s104
                                 + 0.972939 w_k1_MAT_s104
                                 + 1.23228 w_k1_MSFT_s104
                                 + 1.19402 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.24526 w_k1_ADM_s104
                                 + 1.53264 w_k1_AXP_s104
                                 + 1.37655 w_k1_DIS_s104
                                 + 1.31366 w_k1_FDX_s104
                                 + 1.22779 w_k1_IBM_s104
                                 + 1.30061 w_k1_INTC_s104 + 1.2677 w_k1_KO_s104
                                 + 1.2534 w_k1_MAT_s104
                                 + 1.34903 w_k1_MSFT_s104
                                 + 1.47787 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.908894 w_k1_ADM_s104
                                 + 0.90708 w_k1_AXP_s104
                                 + 0.847621 w_k1_DIS_s104
                                 + 0.904873 w_k1_FDX_s104
                                 + 0.977302 w_k1_IBM_s104
                                 + 1.01386 w_k1_INTC_s104
                                 + 0.870987 w_k1_KO_s104
                                 + 0.736842 w_k1_MAT_s104
                                 + 1.05637 w_k1_MSFT_s104
                                 + 1.02023 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.892943 w_k1_ADM_s104
                                 + 1.13119 w_k1_AXP_s104
                                 + 1.07834 w_k1_DIS_s104
                                 + 1.14297 w_k1_FDX_s104
                                 + 0.751905 w_k1_IBM_s104
                                 + 1.12219 w_k1_INTC_s104
                                 + 0.929438 w_k1_KO_s104
                                 + 0.734527 w_k1_MAT_s104
                                 + 1.01396 w_k1_MSFT_s104
                                 + 1.07507 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.862738 w_k1_ADM_s104
                                 + 0.833946 w_k1_AXP_s104
                                 + 0.740974 w_k1_DIS_s104
                                 + 0.842703 w_k1_FDX_s104
                                 + 0.815106 w_k1_IBM_s104
                                 + 1.0095 w_k1_INTC_s104
                                 + 0.773968 w_k1_KO_s104
                                 + 0.658585 w_k1_MAT_s104
                                 + 0.881185 w_k1_MSFT_s104
                                 + 0.939445 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.15217 w_k1_ADM_s104
                                 + 1.04362 w_k1_AXP_s104
                                 + 1.29044 w_k1_DIS_s104
                                 + 1.15457 w_k1_FDX_s104
                                 + 1.08615 w_k1_IBM_s104
                                 + 1.09815 w_k1_INTC_s104
                                 + 1.12919 w_k1_KO_s104 + 1.20183 w_k1_MAT_s104
                                 + 1.18601 w_k1_MSFT_s104
                                 + 1.15219 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.990794 w_k1_ADM_s104
                                 + 0.928223 w_k1_AXP_s104
                                 + 1.31609 w_k1_DIS_s104
                                 + 1.10354 w_k1_FDX_s104
                                 + 0.797462 w_k1_IBM_s104
                                 + 1.37104 w_k1_INTC_s104
                                 + 1.09682 w_k1_KO_s104
                                 + 0.731885 w_k1_MAT_s104
                                 + 1.31583 w_k1_MSFT_s104
                                 + 1.10533 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.38387 w_k1_ADM_s105
                                 + 1.0899 w_k1_AXP_s105 + 1.07005 w_k1_DIS_s105
                                 + 1.04395 w_k1_FDX_s105
                                 + 1.07325 w_k1_IBM_s105
                                 + 0.984243 w_k1_INTC_s105
                                 + 0.97638 w_k1_KO_s105
                                 + 0.949262 w_k1_MAT_s105
                                 + 1.00133 w_k1_MSFT_s105
                                 + 1.02981 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.11066 w_k1_ADM_s105
                                 + 1.14098 w_k1_AXP_s105
                                 + 0.946427 w_k1_DIS_s105
                                 + 0.981963 w_k1_FDX_s105
                                 + 1.42848 w_k1_IBM_s105 + 1.097 w_k1_INTC_s105
                                 + 0.99738 w_k1_KO_s105 + 1.29381 w_k1_MAT_s105
                                 + 0.98625 w_k1_MSFT_s105
                                 + 1.10984 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.55676 w_k1_ADM_s105
                                 + 1.13641 w_k1_AXP_s105
                                 + 1.14232 w_k1_DIS_s105
                                 + 1.21572 w_k1_FDX_s105
                                 + 0.785515 w_k1_IBM_s105
                                 + 1.07718 w_k1_INTC_s105
                                 + 0.984532 w_k1_KO_s105
                                 + 0.992232 w_k1_MAT_s105
                                 + 1.19535 w_k1_MSFT_s105
                                 + 1.10228 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.18156 w_k1_ADM_s105
                                 + 0.916628 w_k1_AXP_s105
                                 + 0.642992 w_k1_DIS_s105
                                 + 0.878123 w_k1_FDX_s105
                                 + 0.738547 w_k1_IBM_s105
                                 + 0.809177 w_k1_INTC_s105
                                 + 0.67205 w_k1_KO_s105 + 0.67484 w_k1_MAT_s105
                                 + 0.840628 w_k1_MSFT_s105
                                 + 0.846256 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.06334 w_k1_ADM_s105
                                 + 0.761323 w_k1_AXP_s105
                                 + 0.951496 w_k1_DIS_s105
                                 + 0.80789 w_k1_FDX_s105
                                 + 1.00383 w_k1_IBM_s105
                                 + 0.774797 w_k1_INTC_s105
                                 + 0.979383 w_k1_KO_s105
                                 + 0.921273 w_k1_MAT_s105
                                 + 0.91429 w_k1_MSFT_s105
                                 + 0.976944 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.84082 w_k1_ADM_s105
                                 + 1.54875 w_k1_AXP_s105
                                 + 1.72106 w_k1_DIS_s105
                                 + 1.55132 w_k1_FDX_s105
                                 + 1.16392 w_k1_IBM_s105
                                 + 1.31729 w_k1_INTC_s105
                                 + 1.24288 w_k1_KO_s105 + 1.28953 w_k1_MAT_s105
                                 + 1.48889 w_k1_MSFT_s105
                                 + 1.45979 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.919425 w_k1_ADM_s105
                                 + 0.72793 w_k1_AXP_s105
                                 + 0.724184 w_k1_DIS_s105
                                 + 0.699758 w_k1_FDX_s105
                                 + 0.914878 w_k1_IBM_s105
                                 + 0.744828 w_k1_INTC_s105
                                 + 0.856447 w_k1_KO_s105
                                 + 0.874691 w_k1_MAT_s105
                                 + 0.762221 w_k1_MSFT_s105
                                 + 0.891814 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.16547 w_k1_ADM_s105
                                 + 0.99842 w_k1_AXP_s105
                                 + 0.849201 w_k1_DIS_s105
                                 + 0.805329 w_k1_FDX_s105
                                 + 1.75117 w_k1_IBM_s105
                                 + 0.738036 w_k1_INTC_s105
                                 + 1.04096 w_k1_KO_s105 + 1.9644 w_k1_MAT_s105
                                 + 0.855551 w_k1_MSFT_s105
                                 + 1.04083 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.22487 w_k1_ADM_s106
                                 + 1.24684 w_k1_AXP_s106
                                 + 1.51831 w_k1_DIS_s106
                                 + 1.33895 w_k1_FDX_s106 + 1.2588 w_k1_IBM_s106
                                 + 1.17653 w_k1_INTC_s106
                                 + 1.25061 w_k1_KO_s106 + 1.07689 w_k1_MAT_s106
                                 + 1.22448 w_k1_MSFT_s106
                                 + 1.06182 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.749101 w_k1_ADM_s106
                                 + 0.571263 w_k1_AXP_s106
                                 + 0.518124 w_k1_DIS_s106
                                 + 0.720114 w_k1_FDX_s106
                                 + 0.667074 w_k1_IBM_s106
                                 + 0.690237 w_k1_INTC_s106
                                 + 0.628314 w_k1_KO_s106
                                 + 0.627787 w_k1_MAT_s106
                                 + 0.787848 w_k1_MSFT_s106
                                 + 0.727546 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.894464 w_k1_ADM_s106
                                 + 1.28752 w_k1_AXP_s106
                                 + 0.992556 w_k1_DIS_s106
                                 + 0.913452 w_k1_FDX_s106
                                 + 1.4784 w_k1_IBM_s106
                                 + 1.03114 w_k1_INTC_s106 + 1.1536 w_k1_KO_s106
                                 + 1.20336 w_k1_MAT_s106
                                 + 0.963721 w_k1_MSFT_s106
                                 + 1.03043 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.817444 w_k1_ADM_s106
                                 + 0.971035 w_k1_AXP_s106
                                 + 0.621642 w_k1_DIS_s106
                                 + 0.90213 w_k1_FDX_s106
                                 + 0.854827 w_k1_IBM_s106
                                 + 0.963426 w_k1_INTC_s106
                                 + 0.83302 w_k1_KO_s106 + 1.02775 w_k1_MAT_s106
                                 + 0.851381 w_k1_MSFT_s106
                                 + 0.811341 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.7965 w_k1_ADM_s106
                                 + 0.823027 w_k1_AXP_s106
                                 + 1.05302 w_k1_DIS_s106
                                 + 0.900681 w_k1_FDX_s106
                                 + 1.03167 w_k1_IBM_s106
                                 + 1.12189 w_k1_INTC_s106
                                 + 1.00375 w_k1_KO_s106
                                 + 0.615661 w_k1_MAT_s106
                                 + 0.956245 w_k1_MSFT_s106
                                 + 0.937897 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.12316 w_k1_ADM_s106
                                 + 1.18135 w_k1_AXP_s106
                                 + 1.10702 w_k1_DIS_s106 + 1.186 w_k1_FDX_s106
                                 + 0.832178 w_k1_IBM_s106
                                 + 1.04853 w_k1_INTC_s106
                                 + 0.953148 w_k1_KO_s106
                                 + 0.981387 w_k1_MAT_s106
                                 + 1.10128 w_k1_MSFT_s106
                                 + 0.994425 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.943557 w_k1_ADM_s106
                                 + 0.931393 w_k1_AXP_s106
                                 + 1.12749 w_k1_DIS_s106 + 1.039 w_k1_FDX_s106
                                 + 1.17576 w_k1_IBM_s106
                                 + 0.885059 w_k1_INTC_s106
                                 + 1.12389 w_k1_KO_s106 + 1.0489 w_k1_MAT_s106
                                 + 1.07082 w_k1_MSFT_s106
                                 + 0.942298 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.845578 w_k1_ADM_s106
                                 + 0.77542 w_k1_AXP_s106
                                 + 0.808891 w_k1_DIS_s106
                                 + 0.835974 w_k1_FDX_s106
                                 + 0.81121 w_k1_IBM_s106
                                 + 1.02962 w_k1_INTC_s106
                                 + 0.855084 w_k1_KO_s106
                                 + 0.540344 w_k1_MAT_s106
                                 + 0.929192 w_k1_MSFT_s106
                                 + 0.859242 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.680023 w_k1_ADM_s107
                                 + 0.520348 w_k1_AXP_s107
                                 + 0.591638 w_k1_DIS_s107
                                 + 0.688477 w_k1_FDX_s107
                                 + 0.795752 w_k1_IBM_s107
                                 + 0.797996 w_k1_INTC_s107
                                 + 0.713686 w_k1_KO_s107
                                 + 0.595869 w_k1_MAT_s107
                                 + 0.681645 w_k1_MSFT_s107
                                 + 0.76052 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.817199 w_k1_ADM_s107
                                 + 1.42172 w_k1_AXP_s107
                                 + 1.36836 w_k1_DIS_s107
                                 + 1.14444 w_k1_FDX_s107
                                 + 1.40173 w_k1_IBM_s107
                                 + 1.07317 w_k1_INTC_s107
                                 + 1.18305 w_k1_KO_s107 + 1.29412 w_k1_MAT_s107
                                 + 0.948666 w_k1_MSFT_s107
                                 + 1.12387 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.02176 w_k1_ADM_s107
                                 + 0.890219 w_k1_AXP_s107
                                 + 1.25661 w_k1_DIS_s107
                                 + 1.05538 w_k1_FDX_s107
                                 + 0.808521 w_k1_IBM_s107
                                 + 1.12938 w_k1_INTC_s107
                                 + 0.92732 w_k1_KO_s107
                                 + 0.525906 w_k1_MAT_s107
                                 + 1.16463 w_k1_MSFT_s107
                                 + 1.04886 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.01283 w_k1_ADM_s107
                                 + 0.852321 w_k1_AXP_s107
                                 + 1.02231 w_k1_DIS_s107
                                 + 0.970776 w_k1_FDX_s107
                                 + 1.08664 w_k1_IBM_s107
                                 + 1.09762 w_k1_INTC_s107
                                 + 0.951413 w_k1_KO_s107
                                 + 0.685895 w_k1_MAT_s107
                                 + 0.96089 w_k1_MSFT_s107
                                 + 0.890498 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.2448 w_k1_ADM_s107
                                 + 1.22669 w_k1_AXP_s107
                                 + 1.49045 w_k1_DIS_s107
                                 + 1.28238 w_k1_FDX_s107 + 1.0421 w_k1_IBM_s107
                                 + 1.16117 w_k1_INTC_s107
                                 + 1.09281 w_k1_KO_s107
                                 + 0.747099 w_k1_MAT_s107
                                 + 1.19118 w_k1_MSFT_s107
                                 + 1.09356 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.19343 w_k1_ADM_s107
                                 + 1.30039 w_k1_AXP_s107
                                 + 1.07862 w_k1_DIS_s107
                                 + 1.23665 w_k1_FDX_s107
                                 + 0.912186 w_k1_IBM_s107
                                 + 1.12908 w_k1_INTC_s107
                                 + 0.83063 w_k1_KO_s107 + 1.19034 w_k1_MAT_s107
                                 + 0.931046 w_k1_MSFT_s107
                                 + 1.04944 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.936849 w_k1_ADM_s107
                                 + 1.1142 w_k1_AXP_s107
                                 + 0.811585 w_k1_DIS_s107
                                 + 0.984688 w_k1_FDX_s107
                                 + 0.91408 w_k1_IBM_s107
                                 + 1.12146 w_k1_INTC_s107
                                 + 0.754124 w_k1_KO_s107
                                 + 0.782253 w_k1_MAT_s107
                                 + 0.958324 w_k1_MSFT_s107
                                 + 0.930256 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.865439 w_k1_ADM_s107
                                 + 0.774499 w_k1_AXP_s107
                                 + 0.722693 w_k1_DIS_s107
                                 + 0.838659 w_k1_FDX_s107
                                 + 0.870813 w_k1_IBM_s107
                                 + 1.00553 w_k1_INTC_s107
                                 + 0.720899 w_k1_KO_s107
                                 + 0.61826 w_k1_MAT_s107
                                 + 0.896961 w_k1_MSFT_s107
                                 + 0.85846 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.50511 w_k1_ADM_s110
                                 + 1.57518 w_k1_AXP_s110
                                 + 1.82329 w_k1_DIS_s110
                                 + 1.54271 w_k1_FDX_s110
                                 + 1.21024 w_k1_IBM_s110 + 1.812 w_k1_INTC_s110
                                 + 1.53674 w_k1_KO_s110 + 1.45472 w_k1_MAT_s110
                                 + 1.43557 w_k1_MSFT_s110
                                 + 1.33952 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.09794 w_k1_ADM_s110
                                 + 0.936141 w_k1_AXP_s110
                                 + 0.768944 w_k1_DIS_s110
                                 + 0.902951 w_k1_FDX_s110
                                 + 0.886642 w_k1_IBM_s110
                                 + 0.879617 w_k1_INTC_s110
                                 + 0.868914 w_k1_KO_s110
                                 + 0.885462 w_k1_MAT_s110
                                 + 0.750125 w_k1_MSFT_s110
                                 + 0.929835 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.889268 w_k1_ADM_s110
                                 + 0.740061 w_k1_AXP_s110
                                 + 0.913327 w_k1_DIS_s110
                                 + 0.96999 w_k1_FDX_s110
                                 + 0.798363 w_k1_IBM_s110
                                 + 1.16839 w_k1_INTC_s110
                                 + 0.939134 w_k1_KO_s110
                                 + 0.67213 w_k1_MAT_s110
                                 + 0.9664 w_k1_MSFT_s110
                                 + 0.907474 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.24764 w_k1_ADM_s110
                                 + 1.27025 w_k1_AXP_s110
                                 + 1.22616 w_k1_DIS_s110
                                 + 1.16598 w_k1_FDX_s110
                                 + 1.09889 w_k1_IBM_s110
                                 + 1.11196 w_k1_INTC_s110
                                 + 1.24333 w_k1_KO_s110 + 1.40011 w_k1_MAT_s110
                                 + 1.14564 w_k1_MSFT_s110
                                 + 1.22103 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.72036 w_k1_ADM_s110
                                 + 0.534263 w_k1_AXP_s110
                                 + 0.56623 w_k1_DIS_s110
                                 + 0.683986 w_k1_FDX_s110
                                 + 0.759865 w_k1_IBM_s110
                                 + 0.771747 w_k1_INTC_s110
                                 + 0.724633 w_k1_KO_s110
                                 + 0.702074 w_k1_MAT_s110
                                 + 0.685294 w_k1_MSFT_s110
                                 + 0.79877 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.995282 w_k1_ADM_s110
                                 + 1.11806 w_k1_AXP_s110
                                 + 1.11698 w_k1_DIS_s110
                                 + 1.05826 w_k1_FDX_s110
                                 + 1.47288 w_k1_IBM_s110
                                 + 1.24902 w_k1_INTC_s110
                                 + 1.28069 w_k1_KO_s110 + 1.24434 w_k1_MAT_s110
                                 + 1.01568 w_k1_MSFT_s110
                                 + 1.19006 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.27001 w_k1_ADM_s110
                                 + 0.930374 w_k1_AXP_s110
                                 + 1.26356 w_k1_DIS_s110
                                 + 1.12854 w_k1_FDX_s110
                                 + 1.09733 w_k1_IBM_s110
                                 + 1.42415 w_k1_INTC_s110
                                 + 1.44903 w_k1_KO_s110 + 1.0459 w_k1_MAT_s110
                                 + 1.1921 w_k1_MSFT_s110 + 1.05623 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.981829 w_k1_ADM_s110
                                 + 0.847228 w_k1_AXP_s110
                                 + 0.780649 w_k1_DIS_s110
                                 + 0.83379 w_k1_FDX_s110
                                 + 1.14478 w_k1_IBM_s110
                                 + 0.91206 w_k1_INTC_s110
                                 + 1.03563 w_k1_KO_s110 + 1.51312 w_k1_MAT_s110
                                 + 0.816153 w_k1_MSFT_s110
                                 + 1.02833 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.20647 w_k1_ADM_s111
                                 + 1.26923 w_k1_AXP_s111
                                 + 1.12751 w_k1_DIS_s111
                                 + 0.962914 w_k1_FDX_s111
                                 + 1.52442 w_k1_IBM_s111
                                 + 0.941119 w_k1_INTC_s111
                                 + 1.19231 w_k1_KO_s111 + 1.47963 w_k1_MAT_s111
                                 + 1.0017 w_k1_MSFT_s111 + 1.15671 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.76893 w_k1_ADM_s111
                                 + 0.635824 w_k1_AXP_s111
                                 + 0.6725 w_k1_DIS_s111
                                 + 0.695177 w_k1_FDX_s111
                                 + 1.18284 w_k1_IBM_s111
                                 + 0.860817 w_k1_INTC_s111
                                 + 0.928275 w_k1_KO_s111 + 1.2032 w_k1_MAT_s111
                                 + 0.77683 w_k1_MSFT_s111
                                 + 0.878786 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.915548 w_k1_ADM_s111
                                 + 0.790595 w_k1_AXP_s111
                                 + 0.773363 w_k1_DIS_s111
                                 + 0.896652 w_k1_FDX_s111
                                 + 0.721086 w_k1_IBM_s111
                                 + 0.856544 w_k1_INTC_s111
                                 + 0.756271 w_k1_KO_s111
                                 + 0.801553 w_k1_MAT_s111
                                 + 0.906445 w_k1_MSFT_s111
                                 + 0.927148 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.722682 w_k1_ADM_s111
                                 + 0.652881 w_k1_AXP_s111
                                 + 0.633551 w_k1_DIS_s111
                                 + 0.680474 w_k1_FDX_s111
                                 + 0.839246 w_k1_IBM_s111
                                 + 0.705509 w_k1_INTC_s111
                                 + 0.750195 w_k1_KO_s111
                                 + 0.734809 w_k1_MAT_s111
                                 + 0.778802 w_k1_MSFT_s111
                                 + 0.808112 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.05059 w_k1_ADM_s111
                                 + 1.04734 w_k1_AXP_s111
                                 + 1.16078 w_k1_DIS_s111
                                 + 1.15737 w_k1_FDX_s111
                                 + 0.673112 w_k1_IBM_s111
                                 + 0.967858 w_k1_INTC_s111
                                 + 0.830356 w_k1_KO_s111
                                 + 0.677633 w_k1_MAT_s111
                                 + 1.10837 w_k1_MSFT_s111
                                 + 0.940536 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.944257 w_k1_ADM_s111
                                 + 1.19746 w_k1_AXP_s111
                                 + 1.21996 w_k1_DIS_s111 + 1.1335 w_k1_FDX_s111
                                 + 1.08093 w_k1_IBM_s111
                                 + 0.970802 w_k1_INTC_s111
                                 + 0.985052 w_k1_KO_s111
                                 + 1.12492 w_k1_MAT_s111
                                 + 1.07966 w_k1_MSFT_s111
                                 + 1.05567 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.766348 w_k1_ADM_s111
                                 + 1.07373 w_k1_AXP_s111
                                 + 0.83266 w_k1_DIS_s111
                                 + 0.868231 w_k1_FDX_s111
                                 + 0.978132 w_k1_IBM_s111
                                 + 0.924096 w_k1_INTC_s111
                                 + 0.866857 w_k1_KO_s111
                                 + 1.04057 w_k1_MAT_s111
                                 + 0.874925 w_k1_MSFT_s111
                                 + 0.924575 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.432 w_k1_ADM_s111
                                 + 1.61647 w_k1_AXP_s111
                                 + 1.69189 w_k1_DIS_s111
                                 + 1.51463 w_k1_FDX_s111
                                 + 0.934115 w_k1_IBM_s111
                                 + 1.21878 w_k1_INTC_s111
                                 + 1.22401 w_k1_KO_s111 + 1.19737 w_k1_MAT_s111
                                 + 1.42466 w_k1_MSFT_s111 + 1.2855 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.25189 w_k1_ADM_s112
                                 + 1.41314 w_k1_AXP_s112
                                 + 1.32042 w_k1_DIS_s112
                                 + 1.20821 w_k1_FDX_s112
                                 + 0.925602 w_k1_IBM_s112
                                 + 0.967583 w_k1_INTC_s112
                                 + 1.00415 w_k1_KO_s112 + 1.6071 w_k1_MAT_s112
                                 + 1.02656 w_k1_MSFT_s112
                                 + 1.15156 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.848213 w_k1_ADM_s112
                                 + 0.876998 w_k1_AXP_s112
                                 + 1.22249 w_k1_DIS_s112 + 1.0008 w_k1_FDX_s112
                                 + 0.853708 w_k1_IBM_s112
                                 + 0.921746 w_k1_INTC_s112
                                 + 0.936565 w_k1_KO_s112
                                 + 0.970496 w_k1_MAT_s112
                                 + 0.999964 w_k1_MSFT_s112
                                 + 0.976441 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.951199 w_k1_ADM_s112
                                 + 1.3545 w_k1_AXP_s112 + 1.21937 w_k1_DIS_s112
                                 + 1.04204 w_k1_FDX_s112
                                 + 1.42359 w_k1_IBM_s112
                                 + 1.08717 w_k1_INTC_s112
                                 + 1.33943 w_k1_KO_s112 + 1.70844 w_k1_MAT_s112
                                 + 1.01612 w_k1_MSFT_s112
                                 + 1.12894 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.35738 w_k1_ADM_s112
                                 + 1.40119 w_k1_AXP_s112
                                 + 1.69836 w_k1_DIS_s112
                                 + 1.29871 w_k1_FDX_s112
                                 + 1.16038 w_k1_IBM_s112
                                 + 1.16266 w_k1_INTC_s112
                                 + 1.41964 w_k1_KO_s112 + 1.10972 w_k1_MAT_s112
                                 + 1.31619 w_k1_MSFT_s112
                                 + 1.13503 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.880262 w_k1_ADM_s112
                                 + 0.725537 w_k1_AXP_s112
                                 + 0.714755 w_k1_DIS_s112
                                 + 0.791241 w_k1_FDX_s112
                                 + 0.746601 w_k1_IBM_s112
                                 + 0.866435 w_k1_INTC_s112
                                 + 0.775841 w_k1_KO_s112
                                 + 0.473801 w_k1_MAT_s112
                                 + 0.850714 w_k1_MSFT_s112
                                 + 0.782872 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.01818 w_k1_ADM_s112
                                 + 1.0101 w_k1_AXP_s112
                                 + 0.935931 w_k1_DIS_s112
                                 + 0.849681 w_k1_FDX_s112
                                 + 0.975442 w_k1_IBM_s112
                                 + 0.899757 w_k1_INTC_s112
                                 + 0.895194 w_k1_KO_s112
                                 + 1.01308 w_k1_MAT_s112
                                 + 0.858201 w_k1_MSFT_s112
                                 + 0.868743 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.24923 w_k1_ADM_s112
                                 + 1.58798 w_k1_AXP_s112
                                 + 0.958919 w_k1_DIS_s112
                                 + 1.11603 w_k1_FDX_s112
                                 + 0.827057 w_k1_IBM_s112
                                 + 0.870905 w_k1_INTC_s112
                                 + 0.905992 w_k1_KO_s112
                                 + 1.30466 w_k1_MAT_s112
                                 + 0.989589 w_k1_MSFT_s112
                                 + 0.977889 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.16138 w_k1_ADM_s112
                                 + 1.17659 w_k1_AXP_s112
                                 + 1.17352 w_k1_DIS_s112
                                 + 1.05467 w_k1_FDX_s112
                                 + 0.898488 w_k1_IBM_s112
                                 + 0.966501 w_k1_INTC_s112
                                 + 0.921233 w_k1_KO_s112
                                 + 0.963054 w_k1_MAT_s112
                                 + 1.03691 w_k1_MSFT_s112
                                 + 0.957951 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.13788 w_k1_ADM_s113
                                 + 0.960379 w_k1_AXP_s113
                                 + 1.01291 w_k1_DIS_s113
                                 + 0.938063 w_k1_FDX_s113
                                 + 1.02679 w_k1_IBM_s113
                                 + 0.956207 w_k1_INTC_s113
                                 + 1.03173 w_k1_KO_s113
                                 + 0.714473 w_k1_MAT_s113
                                 + 0.858753 w_k1_MSFT_s113
                                 + 1.04204 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.875291 w_k1_ADM_s113
                                 + 0.908201 w_k1_AXP_s113
                                 + 1.25466 w_k1_DIS_s113
                                 + 1.04091 w_k1_FDX_s113
                                 + 0.955741 w_k1_IBM_s113
                                 + 1.29884 w_k1_INTC_s113
                                 + 1.03961 w_k1_KO_s113
                                 + 0.860403 w_k1_MAT_s113
                                 + 1.15128 w_k1_MSFT_s113
                                 + 1.07963 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.742254 w_k1_ADM_s113
                                 + 0.851417 w_k1_AXP_s113
                                 + 0.778689 w_k1_DIS_s113
                                 + 0.760179 w_k1_FDX_s113
                                 + 1.20247 w_k1_IBM_s113
                                 + 0.935292 w_k1_INTC_s113
                                 + 0.934517 w_k1_KO_s113
                                 + 0.98608 w_k1_MAT_s113
                                 + 0.760697 w_k1_MSFT_s113
                                 + 0.847865 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.20747 w_k1_ADM_s113
                                 + 1.28049 w_k1_AXP_s113
                                 + 1.36919 w_k1_DIS_s113
                                 + 1.09898 w_k1_FDX_s113 + 1.5181 w_k1_IBM_s113
                                 + 1.29726 w_k1_INTC_s113
                                 + 1.22832 w_k1_KO_s113 + 1.14693 w_k1_MAT_s113
                                 + 1.12085 w_k1_MSFT_s113
                                 + 1.05624 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.11014 w_k1_ADM_s113
                                 + 1.40912 w_k1_AXP_s113
                                 + 1.12216 w_k1_DIS_s113
                                 + 1.19779 w_k1_FDX_s113
                                 + 1.03954 w_k1_IBM_s113
                                 + 1.31095 w_k1_INTC_s113 + 1.211 w_k1_KO_s113
                                 + 0.949918 w_k1_MAT_s113
                                 + 1.10769 w_k1_MSFT_s113
                                 + 1.14631 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.826051 w_k1_ADM_s113
                                 + 0.731405 w_k1_AXP_s113
                                 + 0.696224 w_k1_DIS_s113
                                 + 0.800376 w_k1_FDX_s113
                                 + 0.768205 w_k1_IBM_s113
                                 + 0.779771 w_k1_INTC_s113
                                 + 0.76244 w_k1_KO_s113
                                 + 0.512225 w_k1_MAT_s113
                                 + 0.714349 w_k1_MSFT_s113
                                 + 0.818664 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.883748 w_k1_ADM_s113
                                 + 0.840812 w_k1_AXP_s113
                                 + 0.945075 w_k1_DIS_s113
                                 + 1.01154 w_k1_FDX_s113
                                 + 0.869091 w_k1_IBM_s113
                                 + 1.09407 w_k1_INTC_s113
                                 + 0.958487 w_k1_KO_s113
                                 + 0.761523 w_k1_MAT_s113
                                 + 0.927442 w_k1_MSFT_s113
                                 + 1.0143 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.15342 w_k1_ADM_s113
                                 + 1.44573 w_k1_AXP_s113
                                 + 1.46881 w_k1_DIS_s113
                                 + 1.30713 w_k1_FDX_s113
                                 + 0.911818 w_k1_IBM_s113
                                 + 1.20745 w_k1_INTC_s113 + 1.0145 w_k1_KO_s113
                                 + 0.921181 w_k1_MAT_s113
                                 + 1.15546 w_k1_MSFT_s113
                                 + 1.25322 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.24579 w_k1_ADM_s114
                                 + 1.40469 w_k1_AXP_s114
                                 + 1.74671 w_k1_DIS_s114
                                 + 1.45867 w_k1_FDX_s114
                                 + 1.17259 w_k1_IBM_s114
                                 + 1.22181 w_k1_INTC_s114
                                 + 1.49086 w_k1_KO_s114 + 1.66 w_k1_MAT_s114
                                 + 1.47688 w_k1_MSFT_s114
                                 + 1.42751 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.844922 w_k1_ADM_s114
                                 + 0.714571 w_k1_AXP_s114
                                 + 0.596693 w_k1_DIS_s114
                                 + 0.746755 w_k1_FDX_s114
                                 + 0.828187 w_k1_IBM_s114
                                 + 0.847312 w_k1_INTC_s114
                                 + 0.885861 w_k1_KO_s114
                                 + 0.897007 w_k1_MAT_s114
                                 + 0.814269 w_k1_MSFT_s114
                                 + 0.842416 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.0298 w_k1_ADM_s114
                                 + 0.76329 w_k1_AXP_s114
                                 + 0.989457 w_k1_DIS_s114
                                 + 0.992159 w_k1_FDX_s114
                                 + 1.15348 w_k1_IBM_s114 + 1.099 w_k1_INTC_s114
                                 + 1.21085 w_k1_KO_s114 + 1.38924 w_k1_MAT_s114
                                 + 0.947532 w_k1_MSFT_s114
                                 + 1.12087 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.19095 w_k1_ADM_s114
                                 + 1.22263 w_k1_AXP_s114 + 1.1779 w_k1_DIS_s114
                                 + 1.34386 w_k1_FDX_s114
                                 + 0.610541 w_k1_IBM_s114
                                 + 1.03615 w_k1_INTC_s114
                                 + 0.857751 w_k1_KO_s114
                                 + 0.620512 w_k1_MAT_s114
                                 + 1.03098 w_k1_MSFT_s114
                                 + 1.06303 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.19726 w_k1_ADM_s114
                                 + 1.051 w_k1_AXP_s114 + 1.14855 w_k1_DIS_s114
                                 + 1.02154 w_k1_FDX_s114
                                 + 1.49992 w_k1_IBM_s114 + 1.09 w_k1_INTC_s114
                                 + 1.52594 w_k1_KO_s114 + 1.75255 w_k1_MAT_s114
                                 + 1.05893 w_k1_MSFT_s114
                                 + 1.14723 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.09612 w_k1_ADM_s114
                                 + 0.806564 w_k1_AXP_s114
                                 + 0.822255 w_k1_DIS_s114
                                 + 1.03685 w_k1_FDX_s114
                                 + 0.691221 w_k1_IBM_s114
                                 + 1.14291 w_k1_INTC_s114
                                 + 0.820558 w_k1_KO_s114
                                 + 0.869296 w_k1_MAT_s114
                                 + 1.03985 w_k1_MSFT_s114
                                 + 0.967441 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.11621 w_k1_ADM_s114
                                 + 1.32368 w_k1_AXP_s114
                                 + 0.949454 w_k1_DIS_s114
                                 + 1.10925 w_k1_FDX_s114
                                 + 1.11416 w_k1_IBM_s114
                                 + 1.07446 w_k1_INTC_s114
                                 + 1.07294 w_k1_KO_s114 + 1.30398 w_k1_MAT_s114
                                 + 0.864382 w_k1_MSFT_s114
                                 + 0.985894 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.38276 w_k1_ADM_s114
                                 + 1.3163 w_k1_AXP_s114 + 1.23473 w_k1_DIS_s114
                                 + 1.28743 w_k1_FDX_s114
                                 + 1.33333 w_k1_IBM_s114
                                 + 1.29098 w_k1_INTC_s114
                                 + 1.52484 w_k1_KO_s114 + 1.92038 w_k1_MAT_s114
                                 + 1.25294 w_k1_MSFT_s114
                                 + 1.14764 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.14438 w_k1_ADM_s115
                                 + 1.23877 w_k1_AXP_s115
                                 + 1.36535 w_k1_DIS_s115
                                 + 1.26241 w_k1_FDX_s115
                                 + 0.992602 w_k1_IBM_s115
                                 + 1.08908 w_k1_INTC_s115
                                 + 1.00017 w_k1_KO_s115
                                 + 0.913458 w_k1_MAT_s115
                                 + 1.12577 w_k1_MSFT_s115
                                 + 1.05764 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.07577 w_k1_ADM_s115
                                 + 0.912724 w_k1_AXP_s115
                                 + 0.830005 w_k1_DIS_s115
                                 + 0.940515 w_k1_FDX_s115
                                 + 0.851761 w_k1_IBM_s115
                                 + 0.935385 w_k1_INTC_s115
                                 + 0.829961 w_k1_KO_s115
                                 + 0.834169 w_k1_MAT_s115
                                 + 0.946459 w_k1_MSFT_s115
                                 + 0.835158 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.679456 w_k1_ADM_s115
                                 + 0.706619 w_k1_AXP_s115
                                 + 0.502267 w_k1_DIS_s115
                                 + 0.723677 w_k1_FDX_s115
                                 + 0.705603 w_k1_IBM_s115
                                 + 0.802623 w_k1_INTC_s115
                                 + 0.675821 w_k1_KO_s115
                                 + 0.523203 w_k1_MAT_s115
                                 + 0.745702 w_k1_MSFT_s115
                                 + 0.708702 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.29084 w_k1_ADM_s115
                                 + 1.67446 w_k1_AXP_s115
                                 + 1.74011 w_k1_DIS_s115
                                 + 1.56319 w_k1_FDX_s115
                                 + 1.03892 w_k1_IBM_s115
                                 + 1.27243 w_k1_INTC_s115
                                 + 1.06024 w_k1_KO_s115
                                 + 0.720351 w_k1_MAT_s115
                                 + 1.21781 w_k1_MSFT_s115
                                 + 1.14582 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04264 w_k1_ADM_s115
                                 + 1.14573 w_k1_AXP_s115
                                 + 0.983545 w_k1_DIS_s115
                                 + 1.06165 w_k1_FDX_s115
                                 + 0.812812 w_k1_IBM_s115
                                 + 1.17294 w_k1_INTC_s115
                                 + 0.94206 w_k1_KO_s115
                                 + 0.665643 w_k1_MAT_s115
                                 + 1.03724 w_k1_MSFT_s115
                                 + 0.926161 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.10465 w_k1_ADM_s115
                                 + 1.53416 w_k1_AXP_s115
                                 + 1.29718 w_k1_DIS_s115
                                 + 1.17514 w_k1_FDX_s115
                                 + 1.68304 w_k1_IBM_s115
                                 + 1.0312 w_k1_INTC_s115 + 1.29232 w_k1_KO_s115
                                 + 1.86799 w_k1_MAT_s115
                                 + 0.96239 w_k1_MSFT_s115
                                 + 1.05993 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.984917 w_k1_ADM_s115
                                 + 0.858311 w_k1_AXP_s115
                                 + 0.743739 w_k1_DIS_s115
                                 + 0.776339 w_k1_FDX_s115
                                 + 1.01764 w_k1_IBM_s115
                                 + 0.751298 w_k1_INTC_s115
                                 + 0.786727 w_k1_KO_s115
                                 + 0.784667 w_k1_MAT_s115
                                 + 0.750122 w_k1_MSFT_s115
                                 + 0.905234 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.20876 w_k1_ADM_s115
                                 + 1.0703 w_k1_AXP_s115 + 1.25687 w_k1_DIS_s115
                                 + 1.11362 w_k1_FDX_s115
                                 + 1.27213 w_k1_IBM_s115
                                 + 0.881672 w_k1_INTC_s115
                                 + 1.14444 w_k1_KO_s115
                                 + 0.993984 w_k1_MAT_s115
                                 + 0.944785 w_k1_MSFT_s115
                                 + 1.03872 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.175 w_k1_ADM_s116
                                 + 1.27335 w_k1_AXP_s116
                                 + 1.31678 w_k1_DIS_s116
                                 + 1.16384 w_k1_FDX_s116
                                 + 1.12361 w_k1_IBM_s116
                                 + 0.969095 w_k1_INTC_s116
                                 + 1.03157 w_k1_KO_s116 + 1.29417 w_k1_MAT_s116
                                 + 1.23022 w_k1_MSFT_s116
                                 + 1.13346 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.10529 w_k1_ADM_s116
                                 + 1.20001 w_k1_AXP_s116
                                 + 1.21292 w_k1_DIS_s116
                                 + 1.08106 w_k1_FDX_s116
                                 + 1.55003 w_k1_IBM_s116
                                 + 1.11085 w_k1_INTC_s116
                                 + 1.19478 w_k1_KO_s116 + 1.78216 w_k1_MAT_s116
                                 + 0.971823 w_k1_MSFT_s116
                                 + 1.08764 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.35099 w_k1_ADM_s116
                                 + 1.46724 w_k1_AXP_s116
                                 + 0.997798 w_k1_DIS_s116
                                 + 1.05113 w_k1_FDX_s116
                                 + 1.02048 w_k1_IBM_s116
                                 + 0.937286 w_k1_INTC_s116
                                 + 1.00893 w_k1_KO_s116 + 1.62791 w_k1_MAT_s116
                                 + 0.944102 w_k1_MSFT_s116
                                 + 1.09688 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.03231 w_k1_ADM_s116
                                 + 0.823 w_k1_AXP_s116 + 1.03657 w_k1_DIS_s116
                                 + 0.897237 w_k1_FDX_s116
                                 + 0.826507 w_k1_IBM_s116
                                 + 0.988388 w_k1_INTC_s116
                                 + 1.00346 w_k1_KO_s116
                                 + 0.819138 w_k1_MAT_s116
                                 + 1.11874 w_k1_MSFT_s116
                                 + 0.860843 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.927362 w_k1_ADM_s116
                                 + 0.976507 w_k1_AXP_s116
                                 + 1.08867 w_k1_DIS_s116
                                 + 1.11989 w_k1_FDX_s116
                                 + 0.877298 w_k1_IBM_s116
                                 + 1.16901 w_k1_INTC_s116
                                 + 0.938156 w_k1_KO_s116
                                 + 1.05081 w_k1_MAT_s116
                                 + 1.24403 w_k1_MSFT_s116
                                 + 0.956723 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.772978 w_k1_ADM_s116
                                 + 0.693799 w_k1_AXP_s116
                                 + 0.766844 w_k1_DIS_s116
                                 + 0.757694 w_k1_FDX_s116
                                 + 1.0853 w_k1_IBM_s116
                                 + 0.752668 w_k1_INTC_s116
                                 + 0.939153 w_k1_KO_s116
                                 + 1.24418 w_k1_MAT_s116
                                 + 0.820364 w_k1_MSFT_s116
                                 + 0.840302 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.25789 w_k1_ADM_s116
                                 + 1.31167 w_k1_AXP_s116
                                 + 1.10821 w_k1_DIS_s116
                                 + 1.16406 w_k1_FDX_s116
                                 + 0.75696 w_k1_IBM_s116
                                 + 0.908335 w_k1_INTC_s116
                                 + 0.915981 w_k1_KO_s116
                                 + 1.10359 w_k1_MAT_s116
                                 + 1.08276 w_k1_MSFT_s116
                                 + 1.02612 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.920784 w_k1_ADM_s116
                                 + 0.848859 w_k1_AXP_s116
                                 + 0.69472 w_k1_DIS_s116
                                 + 0.890865 w_k1_FDX_s116
                                 + 0.806452 w_k1_IBM_s116
                                 + 0.902466 w_k1_INTC_s116
                                 + 0.814142 w_k1_KO_s116
                                 + 0.884531 w_k1_MAT_s116
                                 + 0.904269 w_k1_MSFT_s116
                                 + 0.791497 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.15041 w_k1_ADM_s117
                                 + 0.959419 w_k1_AXP_s117
                                 + 1.18546 w_k1_DIS_s117
                                 + 1.14895 w_k1_FDX_s117
                                 + 1.10054 w_k1_IBM_s117
                                 + 1.41354 w_k1_INTC_s117
                                 + 0.935819 w_k1_KO_s117
                                 + 0.759117 w_k1_MAT_s117
                                 + 1.14645 w_k1_MSFT_s117
                                 + 1.08882 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.92443 w_k1_ADM_s117
                                 + 0.852469 w_k1_AXP_s117
                                 + 0.988167 w_k1_DIS_s117
                                 + 0.883749 w_k1_FDX_s117
                                 + 1.23216 w_k1_IBM_s117
                                 + 1.05479 w_k1_INTC_s117
                                 + 0.960913 w_k1_KO_s117
                                 + 0.928937 w_k1_MAT_s117
                                 + 0.968246 w_k1_MSFT_s117
                                 + 0.959705 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.993321 w_k1_ADM_s117
                                 + 0.681956 w_k1_AXP_s117
                                 + 0.753624 w_k1_DIS_s117
                                 + 0.868541 w_k1_FDX_s117
                                 + 0.65771 w_k1_IBM_s117
                                 + 0.802955 w_k1_INTC_s117
                                 + 0.682888 w_k1_KO_s117
                                 + 0.347752 w_k1_MAT_s117
                                 + 0.778834 w_k1_MSFT_s117
                                 + 0.846731 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.864024 w_k1_ADM_s117
                                 + 0.976827 w_k1_AXP_s117
                                 + 0.812953 w_k1_DIS_s117
                                 + 1.0291 w_k1_FDX_s117
                                 + 0.818814 w_k1_IBM_s117
                                 + 1.01935 w_k1_INTC_s117
                                 + 0.752568 w_k1_KO_s117
                                 + 0.927808 w_k1_MAT_s117
                                 + 1.00835 w_k1_MSFT_s117
                                 + 1.03366 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.0232 w_k1_ADM_s117
                                 + 1.40206 w_k1_AXP_s117
                                 + 1.43972 w_k1_DIS_s117
                                 + 1.41051 w_k1_FDX_s117
                                 + 1.26427 w_k1_IBM_s117
                                 + 1.3703 w_k1_INTC_s117 + 1.11795 w_k1_KO_s117
                                 + 1.22847 w_k1_MAT_s117
                                 + 1.2939 w_k1_MSFT_s117 + 1.10182 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.818884 w_k1_ADM_s117
                                 + 0.934407 w_k1_AXP_s117
                                 + 0.614893 w_k1_DIS_s117
                                 + 0.858657 w_k1_FDX_s117
                                 + 0.820527 w_k1_IBM_s117
                                 + 0.718088 w_k1_INTC_s117
                                 + 0.674742 w_k1_KO_s117
                                 + 1.00818 w_k1_MAT_s117
                                 + 0.74361 w_k1_MSFT_s117
                                 + 0.895762 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.12595 w_k1_ADM_s117
                                 + 0.965212 w_k1_AXP_s117
                                 + 0.979221 w_k1_DIS_s117
                                 + 1.14729 w_k1_FDX_s117
                                 + 0.861041 w_k1_IBM_s117
                                 + 1.29102 w_k1_INTC_s117
                                 + 0.933998 w_k1_KO_s117
                                 + 0.613494 w_k1_MAT_s117
                                 + 1.07815 w_k1_MSFT_s117
                                 + 1.06485 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.872848 w_k1_ADM_s117
                                 + 0.7688 w_k1_AXP_s117
                                 + 0.999139 w_k1_DIS_s117
                                 + 0.848158 w_k1_FDX_s117
                                 + 0.893233 w_k1_IBM_s117
                                 + 0.95015 w_k1_INTC_s117
                                 + 0.995943 w_k1_KO_s117
                                 + 0.724939 w_k1_MAT_s117
                                 + 0.973509 w_k1_MSFT_s117
                                 + 1.01762 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0123931866114424 w_k1_ADM_s000
                                 + 0.0283376970121646 w_k1_ADM_s001
                                 + 0.00735128273984929 w_k1_ADM_s002
                                 + 0.0127299011416272 w_k1_ADM_s003
                                 + 0.0175505951480885 w_k1_ADM_s004
                                 + 0.00760530235181431 w_k1_ADM_s005
                                 + 0.0101074852561786 w_k1_ADM_s006
                                 + 0.0131266328957896 w_k1_ADM_s007
                                 + 0.0327564507453194 w_k1_ADM_s010
                                 + 0.0944650197598649 w_k1_ADM_s011
                                 + 0.0205237566668135 w_k1_ADM_s012
                                 + 0.0298559403195111 w_k1_ADM_s013
                                 + 0.0262775027685339 w_k1_ADM_s014
                                 + 0.043075173988907 w_k1_ADM_s015
                                 + 0.105709844169812 w_k1_ADM_s016
                                 + 0.0191488030509691 w_k1_ADM_s017
                                 + 0.0517335561366056 w_k1_ADM_s100
                                 + 0.0404170068257084 w_k1_ADM_s101
                                 + 0.0258184901925388 w_k1_ADM_s102
                                 + 0.0462204945202608 w_k1_ADM_s103
                                 + 0.0170321163617969 w_k1_ADM_s104
                                 + 0.0507269714125209 w_k1_ADM_s105
                                 + 0.0579364914163036 w_k1_ADM_s106
                                 + 0.0240302418979926 w_k1_ADM_s107
                                 + 0.034667004951029 w_k1_ADM_s110
                                 + 0.0412570368269091 w_k1_ADM_s111
                                 + 0.0192920642040434 w_k1_ADM_s112
                                 + 0.018990021073814 w_k1_ADM_s113
                                 + 0.0239532785420845 w_k1_ADM_s114
                                 + 0.0399577043906263 w_k1_ADM_s115
                                 + 0.0271946091471234 w_k1_ADM_s116
                                 + 0.01604730257248 w_k1_ADM_s117
                                 + 0.00934633280655089 w_k1_AXP_s000
                                 + 0.028965521392658 w_k1_AXP_s001
                                 + 0.00809462842713267 w_k1_AXP_s002
                                 + 0.0109892594962816 w_k1_AXP_s003
                                 + 0.0191396743183604 w_k1_AXP_s004
                                 + 0.00728644689592612 w_k1_AXP_s005
                                 + 0.00822289341162323 w_k1_AXP_s006
                                 + 0.0115622685168044 w_k1_AXP_s007
                                 + 0.0358338152504451 w_k1_AXP_s010
                                 + 0.10205130396626 w_k1_AXP_s011
                                 + 0.0232823609885993 w_k1_AXP_s012
                                 + 0.0308321074544955 w_k1_AXP_s013
                                 + 0.0316007350813663 w_k1_AXP_s014
                                 + 0.0415937074159953 w_k1_AXP_s015
                                 + 0.110951313641105 w_k1_AXP_s016
                                 + 0.0210108228045898 w_k1_AXP_s017
                                 + 0.0591526457982025 w_k1_AXP_s100
                                 + 0.0396926721121966 w_k1_AXP_s101
                                 + 0.0251409459729882 w_k1_AXP_s102
                                 + 0.0523639104050956 w_k1_AXP_s103
                                 + 0.0176276127621668 w_k1_AXP_s104
                                 + 0.0411822977299504 w_k1_AXP_s105
                                 + 0.0620933806179426 w_k1_AXP_s106
                                 + 0.0259543911575619 w_k1_AXP_s107
                                 + 0.0314742675692009 w_k1_AXP_s110
                                 + 0.045493594338027 w_k1_AXP_s111
                                 + 0.0206445329761849 w_k1_AXP_s112
                                 + 0.0199181766027375 w_k1_AXP_s113
                                 + 0.0214071250340588 w_k1_AXP_s114
                                 + 0.0419801655296357 w_k1_AXP_s115
                                 + 0.0268559017216908 w_k1_AXP_s116
                                 + 0.0152784438152201 w_k1_AXP_s117
                                 + 0.0109102203067239 w_k1_DIS_s000
                                 + 0.0284827959944595 w_k1_DIS_s001
                                 + 0.00893657035509653 w_k1_DIS_s002
                                 + 0.0122681095542929 w_k1_DIS_s003
                                 + 0.0174186556658058 w_k1_DIS_s004
                                 + 0.00771904463967785 w_k1_DIS_s005
                                 + 0.00782041399285664 w_k1_DIS_s006
                                 + 0.0120668496817834 w_k1_DIS_s007
                                 + 0.0384030757350833 w_k1_DIS_s010
                                 + 0.100484739450766 w_k1_DIS_s011
                                 + 0.0208814663132772 w_k1_DIS_s012
                                 + 0.0336367802638527 w_k1_DIS_s013
                                 + 0.0298492379077013 w_k1_DIS_s014
                                 + 0.0412993017854039 w_k1_DIS_s015
                                 + 0.100617813541102 w_k1_DIS_s016
                                 + 0.0204769381921226 w_k1_DIS_s017
                                 + 0.0515904444489389 w_k1_DIS_s100
                                 + 0.0399283289022677 w_k1_DIS_s101
                                 + 0.0273371464591256 w_k1_DIS_s102
                                 + 0.0494044441044384 w_k1_DIS_s103
                                 + 0.0185881844467935 w_k1_DIS_s104
                                 + 0.0385830079669765 w_k1_DIS_s105
                                 + 0.0629365208086381 w_k1_DIS_s106
                                 + 0.0263300165801124 w_k1_DIS_s107
                                 + 0.0328498851483742 w_k1_DIS_s110
                                 + 0.044533560677332 w_k1_DIS_s111
                                 + 0.0201402618736244 w_k1_DIS_s112
                                 + 0.0204390132882963 w_k1_DIS_s113
                                 + 0.02147479936311 w_k1_DIS_s114
                                 + 0.0400850823284716 w_k1_DIS_s115
                                 + 0.0267304750124237 w_k1_DIS_s116
                                 + 0.0164440154588474 w_k1_DIS_s117
                                 + 0.0102422994788928 w_k1_FDX_s000
                                 + 0.0276295983752066 w_k1_FDX_s001
                                 + 0.00819338510570723 w_k1_FDX_s002
                                 + 0.0116674252504591 w_k1_FDX_s003
                                 + 0.0181479638004212 w_k1_FDX_s004
                                 + 0.00730501829305532 w_k1_FDX_s005
                                 + 0.00816164430439025 w_k1_FDX_s006
                                 + 0.0125178344815331 w_k1_FDX_s007
                                 + 0.0357133798896907 w_k1_FDX_s010
                                 + 0.0963001781180535 w_k1_FDX_s011
                                 + 0.021148612266813 w_k1_FDX_s012
                                 + 0.0305319114558956 w_k1_FDX_s013
                                 + 0.0289622244911205 w_k1_FDX_s014
                                 + 0.04047887285727 w_k1_FDX_s015
                                 + 0.0984996088885031 w_k1_FDX_s016
                                 + 0.0198879179225598 w_k1_FDX_s017
                                 + 0.0556365762519206 w_k1_FDX_s100
                                 + 0.0392015624852762 w_k1_FDX_s101
                                 + 0.0257823772307765 w_k1_FDX_s102
                                 + 0.0474778514962917 w_k1_FDX_s103
                                 + 0.0179485285633886 w_k1_FDX_s104
                                 + 0.0396311268335563 w_k1_FDX_s105
                                 + 0.0619170119611018 w_k1_FDX_s106
                                 + 0.0256110672408092 w_k1_FDX_s107
                                 + 0.0328365248194651 w_k1_FDX_s110
                                 + 0.0425419376384352 w_k1_FDX_s111
                                 + 0.0182488248435964 w_k1_FDX_s112
                                 + 0.0191984609315231 w_k1_FDX_s113
                                 + 0.0232774178411648 w_k1_FDX_s114
                                 + 0.039865636511831 w_k1_FDX_s115
                                 + 0.0261322192450811 w_k1_FDX_s116
                                 + 0.0167353515598693 w_k1_FDX_s117
                                 + 0.0101831954186317 w_k1_IBM_s000
                                 + 0.0271099639309081 w_k1_IBM_s001
                                 + 0.00846181316536592 w_k1_IBM_s002
                                 + 0.0118789556684955 w_k1_IBM_s003
                                 + 0.0184291538322421 w_k1_IBM_s004
                                 + 0.00778591578948085 w_k1_IBM_s005
                                 + 0.00857831114829616 w_k1_IBM_s006
                                 + 0.0126758412224335 w_k1_IBM_s007
                                 + 0.0349283324261462 w_k1_IBM_s010
                                 + 0.100778737774343 w_k1_IBM_s011
                                 + 0.0218616620873912 w_k1_IBM_s012
                                 + 0.0300731247527721 w_k1_IBM_s013
                                 + 0.0291269356963391 w_k1_IBM_s014
                                 + 0.0423717017786472 w_k1_IBM_s015
                                 + 0.105304231662979 w_k1_IBM_s016
                                 + 0.0210573570789931 w_k1_IBM_s017
                                 + 0.0563389762709915 w_k1_IBM_s100
                                 + 0.0395839507313176 w_k1_IBM_s101
                                 + 0.0255836814839764 w_k1_IBM_s102
                                 + 0.048834305613208 w_k1_IBM_s103
                                 + 0.0178155833770095 w_k1_IBM_s104
                                 + 0.0411261741959687 w_k1_IBM_s105
                                 + 0.0651187780310845 w_k1_IBM_s106
                                 + 0.0255947658797083 w_k1_IBM_s107
                                 + 0.0344507178934685 w_k1_IBM_s110
                                 + 0.0429320771355358 w_k1_IBM_s111
                                 + 0.0179193228497183 w_k1_IBM_s112
                                 + 0.0211609251719084 w_k1_IBM_s113
                                 + 0.0213560964360867 w_k1_IBM_s114
                                 + 0.0395360971615375 w_k1_IBM_s115
                                 + 0.0281241487472489 w_k1_IBM_s116
                                 + 0.0163963389714324 w_k1_IBM_s117
                                 + 0.00984288041500697 w_k1_INTC_s000
                                 + 0.0285813431072432 w_k1_INTC_s001
                                 + 0.00812699809540277 w_k1_INTC_s002
                                 + 0.0104437415365278 w_k1_INTC_s003
                                 + 0.0187484779673114 w_k1_INTC_s004
                                 + 0.00693847793339856 w_k1_INTC_s005
                                 + 0.00837121819829294 w_k1_INTC_s006
                                 + 0.0117306515348516 w_k1_INTC_s007
                                 + 0.0361352138800731 w_k1_INTC_s010
                                 + 0.0933557222260656 w_k1_INTC_s011
                                 + 0.0218446692171323 w_k1_INTC_s012
                                 + 0.0322632297024519 w_k1_INTC_s013
                                 + 0.0323277020933397 w_k1_INTC_s014
                                 + 0.0411608209923734 w_k1_INTC_s015
                                 + 0.0956591365588287 w_k1_INTC_s016
                                 + 0.0210695045017594 w_k1_INTC_s017
                                 + 0.0566254538071089 w_k1_INTC_s100
                                 + 0.0424012718289745 w_k1_INTC_s101
                                 + 0.0296646388369927 w_k1_INTC_s102
                                 + 0.055343726986288 w_k1_INTC_s103
                                 + 0.0198382803282958 w_k1_INTC_s104
                                 + 0.0376333747304999 w_k1_INTC_s105
                                 + 0.0631141130137878 w_k1_INTC_s106
                                 + 0.0267109906085312 w_k1_INTC_s107
                                 + 0.0366643736834342 w_k1_INTC_s110
                                 + 0.0392649987075157 w_k1_INTC_s111
                                 + 0.0175098675944556 w_k1_INTC_s112
                                 + 0.0212734661630756 w_k1_INTC_s113
                                 + 0.0237171904021739 w_k1_INTC_s114
                                 + 0.0368637137543453 w_k1_INTC_s115
                                 + 0.025091649837584 w_k1_INTC_s116
                                 + 0.0183376201322771 w_k1_INTC_s117
                                 + 0.0102431450580051 w_k1_KO_s000
                                 + 0.0271474032589839 w_k1_KO_s001
                                 + 0.0082617562045471 w_k1_KO_s002
                                 + 0.0118737240352211 w_k1_KO_s003
                                 + 0.0171920172479004 w_k1_KO_s004
                                 + 0.00830415234175559 w_k1_KO_s005
                                 + 0.00861813759891072 w_k1_KO_s006
                                 + 0.0113808614243535 w_k1_KO_s007
                                 + 0.0369966948448249 w_k1_KO_s010
                                 + 0.104380570252217 w_k1_KO_s011
                                 + 0.0241851474806489 w_k1_KO_s012
                                 + 0.0305289481791451 w_k1_KO_s013
                                 + 0.0301523086715405 w_k1_KO_s014
                                 + 0.0435322858480749 w_k1_KO_s015
                                 + 0.107643982206698 w_k1_KO_s016
                                 + 0.0226457212391692 w_k1_KO_s017
                                 + 0.0529840599203393 w_k1_KO_s100
                                 + 0.0415916325256852 w_k1_KO_s101
                                 + 0.0279055280448384 w_k1_KO_s102
                                 + 0.0480121544260494 w_k1_KO_s103
                                 + 0.0183719553992733 w_k1_KO_s104
                                 + 0.0369049273791386 w_k1_KO_s105
                                 + 0.0624447221978685 w_k1_KO_s106
                                 + 0.0228817601744271 w_k1_KO_s107
                                 + 0.0361251072900295 w_k1_KO_s110
                                 + 0.0403039161885391 w_k1_KO_s111
                                 + 0.0182508631583172 w_k1_KO_s112
                                 + 0.0198230500493419 w_k1_KO_s113
                                 + 0.0236862225863033 w_k1_KO_s114
                                 + 0.0362528768610383 w_k1_KO_s115
                                 + 0.0263641074955441 w_k1_KO_s116
                                 + 0.014548662168414 w_k1_KO_s117
                                 + 0.00862410208994591 w_k1_MAT_s000
                                 + 0.0354727747128831 w_k1_MAT_s001
                                 + 0.00780835743933511 w_k1_MAT_s002
                                 + 0.0135292539957437 w_k1_MAT_s003
                                 + 0.0198273971343462 w_k1_MAT_s004
                                 + 0.00679613844959947 w_k1_MAT_s005
                                 + 0.00701547806156194 w_k1_MAT_s006
                                 + 0.0152116314422736 w_k1_MAT_s007
                                 + 0.0466165901729916 w_k1_MAT_s010
                                 + 0.114427787282789 w_k1_MAT_s011
                                 + 0.0227284556288139 w_k1_MAT_s012
                                 + 0.0272188781793317 w_k1_MAT_s013
                                 + 0.0311517027254747 w_k1_MAT_s014
                                 + 0.0449596668490836 w_k1_MAT_s015
                                 + 0.114458707795869 w_k1_MAT_s016
                                 + 0.0187600823790362 w_k1_MAT_s017
                                 + 0.0555017577151663 w_k1_MAT_s100
                                 + 0.0532023958792415 w_k1_MAT_s101
                                 + 0.0217107374592987 w_k1_MAT_s102
                                 + 0.0534894032961338 w_k1_MAT_s103
                                 + 0.0159102196757537 w_k1_MAT_s104
                                 + 0.0407980977653604 w_k1_MAT_s105
                                 + 0.0561158838645097 w_k1_MAT_s106
                                 + 0.0207034106799163 w_k1_MAT_s107
                                 + 0.0359253483425692 w_k1_MAT_s110
                                 + 0.0449328642419658 w_k1_MAT_s111
                                 + 0.020229457555519 w_k1_MAT_s112
                                 + 0.0170368538331354 w_k1_MAT_s113
                                 + 0.0262598270372333 w_k1_MAT_s114
                                 + 0.0337517150024276 w_k1_MAT_s115
                                 + 0.0328699866796844 w_k1_MAT_s116
                                 + 0.0134672029710845 w_k1_MAT_s117
                                 + 0.0110836484219639 w_k1_MSFT_s000
                                 + 0.0272737069913395 w_k1_MSFT_s001
                                 + 0.00782679843786207 w_k1_MSFT_s002
                                 + 0.0103616875789888 w_k1_MSFT_s003
                                 + 0.0165082657542211 w_k1_MSFT_s004
                                 + 0.0077401727613466 w_k1_MSFT_s005
                                 + 0.0081861795098132 w_k1_MSFT_s006
                                 + 0.0130591286463429 w_k1_MSFT_s007
                                 + 0.0383890178402453 w_k1_MSFT_s010
                                 + 0.0980942693829998 w_k1_MSFT_s011
                                 + 0.0228123555762362 w_k1_MSFT_s012
                                 + 0.0345717797426508 w_k1_MSFT_s013
                                 + 0.0294832750697349 w_k1_MSFT_s014
                                 + 0.0450529980487663 w_k1_MSFT_s015
                                 + 0.101213395206393 w_k1_MSFT_s016
                                 + 0.0195942372390814 w_k1_MSFT_s017
                                 + 0.0514835141766061 w_k1_MSFT_s100
                                 + 0.0420613210917241 w_k1_MSFT_s101
                                 + 0.0293198016419559 w_k1_MSFT_s102
                                 + 0.049073411357105 w_k1_MSFT_s103
                                 + 0.0196437931508255 w_k1_MSFT_s104
                                 + 0.0393441962437558 w_k1_MSFT_s105
                                 + 0.0619646876801481 w_k1_MSFT_s106
                                 + 0.023982193361457 w_k1_MSFT_s107
                                 + 0.0317305914005911 w_k1_MSFT_s110
                                 + 0.0422810794663071 w_k1_MSFT_s111
                                 + 0.0178728839591626 w_k1_MSFT_s112
                                 + 0.0185128770425691 w_k1_MSFT_s113
                                 + 0.0222004253854248 w_k1_MSFT_s114
                                 + 0.0360350057867734 w_k1_MSFT_s115
                                 + 0.0270964912003671 w_k1_MSFT_s116
                                 + 0.0165368606820879 w_k1_MSFT_s117
                                 + 0.0101131207842202 w_k1_PG_s000
                                 + 0.0294159129148678 w_k1_PG_s001
                                 + 0.00827411279932094 w_k1_PG_s002
                                 + 0.010511287946689 w_k1_PG_s003
                                 + 0.0181073250710284 w_k1_PG_s004
                                 + 0.00761971011735014 w_k1_PG_s005
                                 + 0.00858171423645567 w_k1_PG_s006
                                 + 0.0116387585421812 w_k1_PG_s007
                                 + 0.0354674921972879 w_k1_PG_s010
                                 + 0.0954622764629612 w_k1_PG_s011
                                 + 0.0224702019364313 w_k1_PG_s012
                                 + 0.0304025612866304 w_k1_PG_s013
                                 + 0.0279039609152229 w_k1_PG_s014
                                 + 0.0425221243223542 w_k1_PG_s015
                                 + 0.100786140821683 w_k1_PG_s016
                                 + 0.0195457515018495 w_k1_PG_s017
                                 + 0.0515259073099548 w_k1_PG_s100
                                 + 0.0396175570691773 w_k1_PG_s101
                                 + 0.0275959336680718 w_k1_PG_s102
                                 + 0.0501189759459048 w_k1_PG_s103
                                 + 0.0196475102500871 w_k1_PG_s104
                                 + 0.0408405110816718 w_k1_PG_s105
                                 + 0.0579543051068713 w_k1_PG_s106
                                 + 0.0240605696127231 w_k1_PG_s107
                                 + 0.034127578157102 w_k1_PG_s110
                                 + 0.0425073587043634 w_k1_PG_s111
                                 + 0.0177832335058869 w_k1_PG_s112
                                 + 0.0194560344386153 w_k1_PG_s113
                                 + 0.0227256085363046 w_k1_PG_s114
                                 + 0.0363227527299868 w_k1_PG_s115
                                 + 0.025610978674484 w_k1_PG_s116
                                 + 0.0163382560627823 w_k1_PG_s117 >= 1.13
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
