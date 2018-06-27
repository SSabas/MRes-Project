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
 w_k1_ADM_s0_return_constraint:  1.11512 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.975554 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.01581 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.99042 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.04797 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.924671 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.0237 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.45957 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.966695 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.833474 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.4527 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.73622 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.25872 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.10599 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.863817 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.804507 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.20316 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.37756 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.41978 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.993024 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.24384 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.18567 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.01008 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.786593 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.19503 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.03989 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.806938 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.42618 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.277 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.23646 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.21103 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.01379 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.01258 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.71491 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.718807 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.911197 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.0607 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.964617 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.31683 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.889105 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.861104 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.06155 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.28941 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.833455 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.935533 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.79352 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.912091 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.81395 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.25407 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.723621 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.17879 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.21283 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.717029 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.825906 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.44689 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.46892 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.20057 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.770739 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.03761 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.06337 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.785332 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.1147 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.28951 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.836985 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.625366 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.32529 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.6333 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.963397 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.55761 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.25714 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.02229 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.669333 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.751328 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.955599 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.28374 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.983431 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.2868 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.884345 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.84632 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.12973 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.34921 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.823258 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.12455 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 2.01048 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.797353 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.799865 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.26401 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.61168 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.62191 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.14916 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.76914 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.775157 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.6698 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.40018 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.47611 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.766667 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.811351 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.884861 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.771376 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.992323 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.08016 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.766073 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.535573 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.71698 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.65713 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.06237 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.61004 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.3165 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.15117 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.519979 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.6672 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.00726 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.05404 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.873462 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19295 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.924936 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.966018 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.01852 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.18509 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.858058 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.10053 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.60178 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.903717 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.799449 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.26451 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.699998 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.34174 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.0093 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.749272 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.739561 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.29938 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.33433 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.29736 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.857958 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.06382 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.999861 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.950465 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.907092 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.13629 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.836722 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.713045 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.42309 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.36982 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.17856 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.53905 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.11478 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.02027 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.743766 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.741278 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.02611 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.14431 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.01161 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.22888 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.931546 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.713011 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.20757 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.16836 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.846588 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.776613 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.27195 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.748563 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.24227 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.02395 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.783193 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.00965 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.31885 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.942632 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.2962 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.53849 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.12203 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.78963 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.730453 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.610965 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.920364 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.583268 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.64657 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.01828 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.03729 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.633633 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.976961 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.56308 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.72089 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.988071 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.92736 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.19243 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.66064 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.24247 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.880196 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.16097 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.730089 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18705 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.953924 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.805615 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.97279 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.16666 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.962501 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.18204 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.77169 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.999322 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.960569 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.3406 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.933459 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.2711 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.02614 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.960497 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.813363 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.26193 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.19751 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.19836 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.849384 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.997356 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.982865 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.960421 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.993629 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.08732 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.822043 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.761066 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.25754 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.29031 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.1128 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.28186 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.10578 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.995153 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.65424 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.84343 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.899255 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.984772 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.863924 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.16543 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.876818 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.762416 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.13339 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.34372 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.900141 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.989525 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.62677 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.939918 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.04724 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.31019 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.845258 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.28935 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.31932 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.887526 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.970557 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.48397 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.19671 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.12771 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.797818 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.831477 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.92927 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.684012 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.06397 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.874264 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.863463 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.620424 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.16271 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.47336 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.856733 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.35181 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.36274 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.08809 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.65022 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.02929 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.959595 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.05797 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.891133 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.27546 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.886542 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.747602 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.18454 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.35795 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.845101 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.750303 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.13676 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.816924 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.06196 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.01229 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.773692 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.02695 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.30715 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.820915 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.3807 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.92719 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.35099 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.946162 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.825411 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.850856 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.01584 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.513316 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.49431 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.982059 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.844841 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.60278 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.1817 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.56289 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.581605 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.10877 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.90137 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.18999 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.602058 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.02292 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.762527 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.28117 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.856479 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11778 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.953669 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.862897 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.07555 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.11952 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.886398 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.14929 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.66597 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.0411 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.984014 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.29645 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.836612 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.4056 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.11344 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.818164 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.790607 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.27784 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.14566 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.32334 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.889046 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.933876 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.956004 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.958583 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.992629 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.00258 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.890358 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.750979 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.39787 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.33778 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.12951 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.44985 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.08581 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.06426 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.713872 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.783239 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.02444 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.08321 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.891876 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13515 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.965675 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.913723 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.00515 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.13706 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.887179 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.00057 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.36532 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.911133 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.997671 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.12049 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.83165 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.13918 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.12362 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.933563 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.917004 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.40805 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.18366 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.12836 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.948384 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.03627 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.05066 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.850861 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.02291 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.974539 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.946456 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.770124 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.151 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.33143 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.01002 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.2304 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.21264 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.997305 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.760034 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.891829 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.02008 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.15442 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.908122 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.164999753887445 z_k1_s0000
                                 - 0.130000564759786 z_k1_s0001
                                 - 0.109000027083216 z_k1_s0002
                                 - 0.104999957182917 z_k1_s0003
                                 - 0.0980000055573877 z_k1_s0004
                                 - 0.190000475264312 z_k1_s0005
                                 - 0.183999801413876 z_k1_s0006
                                 - 0.156999614086845 z_k1_s0007
                                 - 0.11500000692267 z_k1_s0010
                                 - 0.101999847322758 z_k1_s0011
                                 - 0.0499999699229087 z_k1_s0012
                                 - 0.0770002428612037 z_k1_s0013
                                 - 0.074999954884363 z_k1_s0014
                                 - 0.074999954884363 z_k1_s0015
                                 - 0.0609999937459406 z_k1_s0016
                                 - 0.205999602122455 z_k1_s0017
                                 - 0.313999654806393 z_k1_s0020
                                 - 0.101999716052152 z_k1_s0021
                                 - 0.100999693345817 z_k1_s0022
                                 - 0.242998692655104 z_k1_s0023
                                 - 0.240999947239452 z_k1_s0024
                                 - 0.392998848612836 z_k1_s0025
                                 - 0.103999761464823 z_k1_s0026
                                 - 0.126999633712023 z_k1_s0027
                                 - 0.142000321182582 z_k1_s0030
                                 - 0.253999681188096 z_k1_s0031
                                 - 0.415000105155093 z_k1_s0032
                                 - 0.274001406175084 z_k1_s0033
                                 - 0.237001196505415 z_k1_s0034
                                 - 0.124000214297327 z_k1_s0035
                                 - 0.118000238902369 z_k1_s0036
                                 - 0.243000449098856 z_k1_s0037
                                 - 0.105999794524364 z_k1_s0040
                                 - 0.0819998410471492 z_k1_s0041
                                 - 0.0859998332933516 z_k1_s0042
                                 - 0.108999788709015 z_k1_s0043
                                 - 0.0679999374853757 z_k1_s0044
                                 - 0.0859998332933516 z_k1_s0045
                                 - 0.0609999510545216 z_k1_s0046
                                 - 0.094999815847307 z_k1_s0047
                                 - 0.210000178539892 z_k1_s0050
                                 - 0.246000107146592 z_k1_s0051
                                 - 0.11000005952085 z_k1_s0052
                                 - 0.210000178539892 z_k1_s0053
                                 - 0.338000916365056 z_k1_s0054
                                 - 0.268999863200606 z_k1_s0055
                                 - 0.128000202324441 z_k1_s0056
                                 - 0.273999654951268 z_k1_s0057
                                 - 0.0709996418418173 z_k1_s0060
                                 - 0.134999517761643 z_k1_s0061
                                 - 0.134999517761643 z_k1_s0062
                                 - 0.234999076918822 z_k1_s0063
                                 - 0.0769995702314679 z_k1_s0064
                                 - 0.146999600339845 z_k1_s0065
                                 - 0.158999682918047 z_k1_s0066
                                 - 0.169999476366106 z_k1_s0067
                                 - 0.226999323264066 z_k1_s0070
                                 - 0.284998645888094 z_k1_s0071
                                 - 0.107999681283335 z_k1_s0072
                                 - 0.281999126751051 z_k1_s0073
                                 - 0.265998613029251 z_k1_s0074
                                 - 0.257999279665623 z_k1_s0075
                                 - 0.164999595160408 z_k1_s0076
                                 - 0.25599898457608 z_k1_s0077
                                 - 0.140000148067311 z_k1_s0100
                                 - 0.279000419228489 z_k1_s0101
                                 - 0.152000043141592 z_k1_s0102
                                 - 0.14900018372308 z_k1_s0103
                                 - 0.275000454203729 z_k1_s0104
                                 - 0.583000048111442 z_k1_s0105
                                 - 0.129000129899162 z_k1_s0106
                                 - 0.579999502592582 z_k1_s0107
                                 - 0.587998569309871 z_k1_s0110
                                 - 0.381999429822911 z_k1_s0111
                                 - 0.266999264121523 z_k1_s0112
                                 - 0.608997763715577 z_k1_s0113
                                 - 0.496998539226795 z_k1_s0114
                                 - 0.472999459905988 z_k1_s0115
                                 - 0.516997771994134 z_k1_s0116
                                 - 0.187999477897186 z_k1_s0117
                                 - 0.714000309353676 z_k1_s0120
                                 - 0.186999941402218 z_k1_s0121
                                 - 0.275999880587246 z_k1_s0122
                                 - 0.213000025761842 z_k1_s0123
                                 - 0.172999941085477 z_k1_s0124
                                 - 0.424999067187229 z_k1_s0125
                                 - 0.311000027979032 z_k1_s0126
                                 - 0.633000914863677 z_k1_s0127
                                 - 0.141000220569608 z_k1_s0130
                                 - 0.433000647428614 z_k1_s0131
                                 - 0.410001289713881 z_k1_s0132
                                 - 0.291999489257795 z_k1_s0133
                                 - 0.243000449951967 z_k1_s0134
                                 - 0.230000374756117 z_k1_s0135
                                 - 0.222000292420162 z_k1_s0136
                                 - 0.37300120191047 z_k1_s0137
                                 - 0.309001248567054 z_k1_s0140
                                 - 0.566002287019264 z_k1_s0141
                                 - 0.508002052660399 z_k1_s0142
                                 - 0.209000844500046 z_k1_s0143
                                 - 0.228000921272778 z_k1_s0144
                                 - 0.604002440564727 z_k1_s0145
                                 - 0.308001244526384 z_k1_s0146
                                 - 0.39300158798334 z_k1_s0147
                                 - 0.605999682104552 z_k1_s0150
                                 - 0.267999778295766 z_k1_s0151
                                 - 0.348999812652293 z_k1_s0152
                                 - 0.68399942772203 z_k1_s0153
                                 - 0.84600122123459 z_k1_s0154
                                 - 0.933000108269708 z_k1_s0155
                                 - 0.244000055582638 z_k1_s0156
                                 - 0.381999970382688 z_k1_s0157
                                 - 0.114999993965102 z_k1_s0160
                                 - 0.0990000166682285 z_k1_s0161
                                 - 0.356999706042842 z_k1_s0162
                                 - 0.0760000770352354 z_k1_s0163
                                 - 0.354000292656197 z_k1_s0164
                                 - 0.163000665356065 z_k1_s0165
                                 - 0.124000008925856 z_k1_s0166
                                 - 0.190999627300132 z_k1_s0167
                                 - 0.501997724211475 z_k1_s0170
                                 - 0.470999793803398 z_k1_s0171
                                 - 0.447998236411199 z_k1_s0172
                                 - 0.314999249362963 z_k1_s0173
                                 - 0.461999879170018 z_k1_s0174
                                 - 0.463000224795106 z_k1_s0175
                                 - 0.118999581484106 z_k1_s0176
                                 - 0.415999960378034 z_k1_s0177
                                 - 0.238000068190302 z_k1_s1000
                                 - 0.25499990940393 z_k1_s1001
                                 - 0.837000959422045 z_k1_s1002
                                 - 0.350000774162067 z_k1_s1003
                                 - 0.182000001604359 z_k1_s1004
                                 - 0.332000537959488 z_k1_s1005
                                 - 0.272999859206569 z_k1_s1006
                                 - 0.396999004247083 z_k1_s1007
                                 - 0.708995501359545 z_k1_s1010
                                 - 0.583996070830076 z_k1_s1011
                                 - 0.66899717606494 z_k1_s1012
                                 - 0.202999429642628 z_k1_s1013
                                 - 0.818995559903537 z_k1_s1014
                                 - 0.708995501359545 z_k1_s1015
                                 - 0.470997070685755 z_k1_s1016
                                 - 0.499997722117874 z_k1_s1017
                                 - 0.37600043494936 z_k1_s1020
                                 - 0.211000380335002 z_k1_s1021
                                 - 0.453001598917082 z_k1_s1022
                                 - 0.893000118153907 z_k1_s1023
                                 - 0.241000168482868 z_k1_s1024
                                 - 0.358000236780348 z_k1_s1025
                                 - 0.646998765178632 z_k1_s1026
                                 - 0.887002762766672 z_k1_s1027
                                 - 2.84000524405154 z_k1_s1030
                                 - 1.52800380450157 z_k1_s1031
                                 - 0.583000767642785 z_k1_s1032
                                 - 1.49600100880353 z_k1_s1033
                                 - 0.688000685871281 z_k1_s1034
                                 - 0.701000819472993 z_k1_s1035
                                 - 0.617000318969847 z_k1_s1036
                                 - 0.979004474587014 z_k1_s1037
                                 - 0.374999555993228 z_k1_s1040
                                 - 0.204000487460182 z_k1_s1041
                                 - 0.330000014273966 z_k1_s1042
                                 - 0.344000861697555 z_k1_s1043
                                 - 0.255000103075321 z_k1_s1044
                                 - 0.146000043133324 z_k1_s1045
                                 - 0.106000022994059 z_k1_s1046
                                 - 0.264999551235239 z_k1_s1047
                                 - 0.395000458067867 z_k1_s1050
                                 - 0.449998349295224 z_k1_s1051
                                 - 0.129999719440597 z_k1_s1052
                                 - 0.218999634149795 z_k1_s1053
                                 - 0.129999719440597 z_k1_s1054
                                 - 0.45100037604223 z_k1_s1055
                                 - 0.367998739832426 z_k1_s1056
                                 - 0.380999847575066 z_k1_s1057
                                 - 0.51000080791982 z_k1_s1060
                                 - 0.153000015375987 z_k1_s1061
                                 - 0.188000004056771 z_k1_s1062
                                 - 0.170000042306648 z_k1_s1063
                                 - 0.219999961279208 z_k1_s1064
                                 - 0.408000646335856 z_k1_s1065
                                 - 0.194000066973465 z_k1_s1066
                                 - 0.427000542905442 z_k1_s1067
                                 - 0.235000152595757 z_k1_s1070
                                 - 0.272000357534424 z_k1_s1071
                                 - 0.536999177070801 z_k1_s1072
                                 - 0.819000169986542 z_k1_s1073
                                 - 0.452002481688556 z_k1_s1074
                                 - 0.223000118958926 z_k1_s1075
                                 - 0.89199966603487 z_k1_s1076
                                 - 0.618999609372689 z_k1_s1077
                                 - 0.158999958160326 z_k1_s1100
                                 - 0.151999983529377 z_k1_s1101
                                 - 0.219000074711442 z_k1_s1102
                                 - 0.1640000068396 z_k1_s1103
                                 - 0.484999578287676 z_k1_s1104
                                 - 0.20000007677027 z_k1_s1105
                                 - 0.403001211468491 z_k1_s1106
                                 - 0.555999988973266 z_k1_s1107
                                 - 0.450001558067083 z_k1_s1110
                                 - 0.308999201871244 z_k1_s1111
                                 - 0.147000041968209 z_k1_s1112
                                 - 0.171000031665046 z_k1_s1113
                                 - 0.31999986262449 z_k1_s1114
                                 - 0.467001305593814 z_k1_s1115
                                 - 0.598000513830129 z_k1_s1116
                                 - 0.340000554539078 z_k1_s1117
                                 - 0.38699951603711 z_k1_s1120
                                 - 0.340001382514512 z_k1_s1121
                                 - 0.342000183940208 z_k1_s1122
                                 - 0.185000117955973 z_k1_s1123
                                 - 0.298999653269001 z_k1_s1124
                                 - 0.368999783198349 z_k1_s1125
                                 - 0.308998920401229 z_k1_s1126
                                 - 0.399000214596909 z_k1_s1127
                                 - 0.503001754341587 z_k1_s1130
                                 - 0.685001864770578 z_k1_s1131
                                 - 0.562002197712021 z_k1_s1132
                                 - 0.430999536937925 z_k1_s1133
                                 - 0.453998871606528 z_k1_s1134
                                 - 0.685001864770578 z_k1_s1135
                                 - 0.406000690142246 z_k1_s1136
                                 - 0.39500048873911 z_k1_s1137
                                 - 0.538000408550508 z_k1_s1140
                                 - 0.304999471799004 z_k1_s1141
                                 - 0.541998097123957 z_k1_s1142
                                 - 0.288999296722712 z_k1_s1143
                                 - 0.519998521992818 z_k1_s1144
                                 - 1.03499993326883 z_k1_s1145
                                 - 0.419998963763137 z_k1_s1146
                                 - 0.446999745603477 z_k1_s1147
                                 - 0.50199774852513 z_k1_s1150
                                 - 0.318999536413718 z_k1_s1151
                                 - 0.59500088382896 z_k1_s1152
                                 - 0.435999578175214 z_k1_s1153
                                 - 0.537997821805512 z_k1_s1154
                                 - 0.383999603258937 z_k1_s1155
                                 - 0.381999468254641 z_k1_s1156
                                 - 0.772000262125445 z_k1_s1157
                                 - 0.446000362717542 z_k1_s1160
                                 - 1.09399755627436 z_k1_s1161
                                 - 0.834001712540692 z_k1_s1162
                                 - 1.17600062096261 z_k1_s1163
                                 - 1.13300433523952 z_k1_s1164
                                 - 1.18400301049462 z_k1_s1165
                                 - 1.56800082795015 z_k1_s1166
                                 - 0.357000077192145 z_k1_s1167
                                 - 0.747998116732817 z_k1_s1170
                                 - 0.665000599460053 z_k1_s1171
                                 - 0.822998811663958 z_k1_s1172
                                 - 0.512999362212819 z_k1_s1173
                                 - 1.24799847094345 z_k1_s1174
                                 - 0.767996590582332 z_k1_s1175
                                 - 0.783996653261036 z_k1_s1176
                                 - 0.868996585116935 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.00976 w_k1_ADM_s000
                                 + 1.00556 w_k1_AXP_s000 + 1.1184 w_k1_DIS_s000
                                 + 1.11851 w_k1_FDX_s000
                                 + 0.77563 w_k1_IBM_s000
                                 + 1.03099 w_k1_INTC_s000
                                 + 1.03761 w_k1_KO_s000
                                 + 0.722304 w_k1_MAT_s000
                                 + 1.06314 w_k1_MSFT_s000
                                 + 0.94768 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.866936 w_k1_ADM_s000
                                 + 1.03639 w_k1_AXP_s000
                                 + 0.971273 w_k1_DIS_s000
                                 + 0.854614 w_k1_FDX_s000
                                 + 1.16233 w_k1_IBM_s000
                                 + 0.894037 w_k1_INTC_s000
                                 + 1.08905 w_k1_KO_s000 + 1.18316 w_k1_MAT_s000
                                 + 0.906274 w_k1_MSFT_s000
                                 + 0.987269 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.815243 w_k1_ADM_s000
                                 + 0.636851 w_k1_AXP_s000
                                 + 0.798381 w_k1_DIS_s000
                                 + 0.80621 w_k1_FDX_s000
                                 + 0.690198 w_k1_IBM_s000
                                 + 0.926817 w_k1_INTC_s000
                                 + 0.903573 w_k1_KO_s000
                                 + 0.595266 w_k1_MAT_s000
                                 + 0.91421 w_k1_MSFT_s000
                                 + 0.783948 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.01759 w_k1_ADM_s000
                                 + 1.46655 w_k1_AXP_s000
                                 + 1.51953 w_k1_DIS_s000
                                 + 1.43865 w_k1_FDX_s000 + 1.0247 w_k1_IBM_s000
                                 + 1.52459 w_k1_INTC_s000
                                 + 1.21395 w_k1_KO_s000 + 1.05209 w_k1_MAT_s000
                                 + 1.26048 w_k1_MSFT_s000
                                 + 1.07378 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.31361 w_k1_ADM_s000
                                 + 1.16346 w_k1_AXP_s000
                                 + 1.35951 w_k1_DIS_s000
                                 + 1.18457 w_k1_FDX_s000
                                 + 1.18568 w_k1_IBM_s000
                                 + 1.10106 w_k1_INTC_s000
                                 + 1.24581 w_k1_KO_s000 + 1.16941 w_k1_MAT_s000
                                 + 1.25811 w_k1_MSFT_s000
                                 + 1.10861 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.19965 w_k1_ADM_s000
                                 + 1.246 w_k1_AXP_s000 + 0.9472 w_k1_DIS_s000
                                 + 1.07107 w_k1_FDX_s000 + 1.3145 w_k1_IBM_s000
                                 + 1.1283 w_k1_INTC_s000 + 1.14147 w_k1_KO_s000
                                 + 1.29263 w_k1_MAT_s000
                                 + 1.05272 w_k1_MSFT_s000
                                 + 1.09929 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.896677 w_k1_ADM_s000
                                 + 1.03251 w_k1_AXP_s000
                                 + 0.722011 w_k1_DIS_s000
                                 + 0.904446 w_k1_FDX_s000
                                 + 0.710028 w_k1_IBM_s000
                                 + 0.893644 w_k1_INTC_s000
                                 + 0.866971 w_k1_KO_s000
                                 + 0.75806 w_k1_MAT_s000
                                 + 1.0097 w_k1_MSFT_s000
                                 + 0.921729 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.858945 w_k1_ADM_s000
                                 + 0.752947 w_k1_AXP_s000
                                 + 0.824664 w_k1_DIS_s000
                                 + 0.879782 w_k1_FDX_s000
                                 + 1.09752 w_k1_IBM_s000
                                 + 0.91159 w_k1_INTC_s000
                                 + 0.945785 w_k1_KO_s000
                                 + 0.946492 w_k1_MAT_s000
                                 + 0.970017 w_k1_MSFT_s000
                                 + 1.00312 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.960775 w_k1_ADM_s001
                                 + 0.861062 w_k1_AXP_s001
                                 + 1.03578 w_k1_DIS_s001
                                 + 1.11869 w_k1_FDX_s001
                                 + 0.793176 w_k1_IBM_s001
                                 + 0.859348 w_k1_INTC_s001
                                 + 0.94384 w_k1_KO_s001 + 1.37565 w_k1_MAT_s001
                                 + 0.98672 w_k1_MSFT_s001
                                 + 0.932123 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.855152 w_k1_ADM_s001
                                 + 0.957812 w_k1_AXP_s001
                                 + 1.18646 w_k1_DIS_s001
                                 + 0.998603 w_k1_FDX_s001
                                 + 1.27296 w_k1_IBM_s001
                                 + 0.976732 w_k1_INTC_s001
                                 + 1.19203 w_k1_KO_s001 + 1.72427 w_k1_MAT_s001
                                 + 0.996719 w_k1_MSFT_s001
                                 + 1.11217 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.833849 w_k1_ADM_s001
                                 + 0.667099 w_k1_AXP_s001
                                 + 0.565521 w_k1_DIS_s001
                                 + 0.680335 w_k1_FDX_s001
                                 + 0.940254 w_k1_IBM_s001
                                 + 0.693472 w_k1_INTC_s001
                                 + 0.790344 w_k1_KO_s001
                                 + 0.971945 w_k1_MAT_s001
                                 + 0.6723 w_k1_MSFT_s001
                                 + 0.829214 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.13766 w_k1_ADM_s001
                                 + 1.12088 w_k1_AXP_s001
                                 + 1.59141 w_k1_DIS_s001
                                 + 1.40919 w_k1_FDX_s001
                                 + 0.868869 w_k1_IBM_s001
                                 + 0.831111 w_k1_INTC_s001
                                 + 1.0846 w_k1_KO_s001 + 1.46896 w_k1_MAT_s001
                                 + 1.28306 w_k1_MSFT_s001
                                 + 1.14437 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.28388 w_k1_ADM_s001
                                 + 1.13641 w_k1_AXP_s001
                                 + 1.56197 w_k1_DIS_s001
                                 + 1.17924 w_k1_FDX_s001
                                 + 1.52252 w_k1_IBM_s001
                                 + 0.854942 w_k1_INTC_s001
                                 + 1.57152 w_k1_KO_s001 + 2.34343 w_k1_MAT_s001
                                 + 1.20459 w_k1_MSFT_s001
                                 + 1.08728 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.17511 w_k1_ADM_s001
                                 + 1.48679 w_k1_AXP_s001 + 1.0257 w_k1_DIS_s001
                                 + 1.03943 w_k1_FDX_s001
                                 + 1.52074 w_k1_IBM_s001
                                 + 0.743042 w_k1_INTC_s001
                                 + 1.02316 w_k1_KO_s001 + 1.91226 w_k1_MAT_s001
                                 + 0.757311 w_k1_MSFT_s001
                                 + 0.868758 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.911635 w_k1_ADM_s001
                                 + 0.771505 w_k1_AXP_s001
                                 + 0.877941 w_k1_DIS_s001
                                 + 0.766523 w_k1_FDX_s001
                                 + 1.25436 w_k1_IBM_s001
                                 + 0.735639 w_k1_INTC_s001
                                 + 0.950591 w_k1_KO_s001
                                 + 1.39432 w_k1_MAT_s001
                                 + 0.884891 w_k1_MSFT_s001
                                 + 0.934241 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.922337 w_k1_ADM_s001
                                 + 0.838428 w_k1_AXP_s001
                                 + 0.722613 w_k1_DIS_s001
                                 + 0.878313 w_k1_FDX_s001
                                 + 0.776734 w_k1_IBM_s001
                                 + 0.641937 w_k1_INTC_s001
                                 + 0.789624 w_k1_KO_s001
                                 + 0.89764 w_k1_MAT_s001
                                 + 0.80737 w_k1_MSFT_s001
                                 + 0.823151 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.846634 w_k1_ADM_s002
                                 + 1.01627 w_k1_AXP_s002
                                 + 1.23476 w_k1_DIS_s002
                                 + 1.16576 w_k1_FDX_s002
                                 + 1.06004 w_k1_IBM_s002
                                 + 1.19874 w_k1_INTC_s002
                                 + 1.13696 w_k1_KO_s002 + 1.02548 w_k1_MAT_s002
                                 + 1.12356 w_k1_MSFT_s002
                                 + 1.05833 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.906868 w_k1_ADM_s002
                                 + 0.607642 w_k1_AXP_s002
                                 + 0.64323 w_k1_DIS_s002
                                 + 0.74353 w_k1_FDX_s002
                                 + 0.684438 w_k1_IBM_s002
                                 + 0.808643 w_k1_INTC_s002
                                 + 0.615647 w_k1_KO_s002
                                 + 0.406036 w_k1_MAT_s002
                                 + 0.806168 w_k1_MSFT_s002
                                 + 0.773809 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.864656 w_k1_ADM_s002
                                 + 0.955771 w_k1_AXP_s002
                                 + 0.917585 w_k1_DIS_s002
                                 + 0.909331 w_k1_FDX_s002
                                 + 1.68029 w_k1_IBM_s002
                                 + 0.902863 w_k1_INTC_s002
                                 + 0.996549 w_k1_KO_s002 + 1.4486 w_k1_MAT_s002
                                 + 0.733618 w_k1_MSFT_s002
                                 + 1.10519 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.867292 w_k1_ADM_s002
                                 + 0.972601 w_k1_AXP_s002
                                 + 0.750521 w_k1_DIS_s002
                                 + 0.860523 w_k1_FDX_s002
                                 + 0.95542 w_k1_IBM_s002
                                 + 0.912999 w_k1_INTC_s002
                                 + 0.769298 w_k1_KO_s002
                                 + 0.721528 w_k1_MAT_s002
                                 + 0.750636 w_k1_MSFT_s002
                                 + 0.941221 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.54702 w_k1_ADM_s002
                                 + 1.38631 w_k1_AXP_s002
                                 + 1.65494 w_k1_DIS_s002 + 1.3424 w_k1_FDX_s002
                                 + 1.33908 w_k1_IBM_s002
                                 + 1.12005 w_k1_INTC_s002
                                 + 1.29731 w_k1_KO_s002 + 1.13422 w_k1_MAT_s002
                                 + 1.23313 w_k1_MSFT_s002
                                 + 1.21591 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.96863 w_k1_ADM_s002
                                 + 0.89236 w_k1_AXP_s002
                                 + 1.01828 w_k1_DIS_s002 + 0.9933 w_k1_FDX_s002
                                 + 1.23492 w_k1_IBM_s002
                                 + 1.23392 w_k1_INTC_s002
                                 + 1.09328 w_k1_KO_s002 + 0.83806 w_k1_MAT_s002
                                 + 1.03089 w_k1_MSFT_s002
                                 + 0.944055 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.02486 w_k1_ADM_s002
                                 + 1.53956 w_k1_AXP_s002
                                 + 1.03824 w_k1_DIS_s002
                                 + 1.11817 w_k1_FDX_s002
                                 + 1.30357 w_k1_IBM_s002
                                 + 1.09938 w_k1_INTC_s002 + 1.043 w_k1_KO_s002
                                 + 1.47427 w_k1_MAT_s002
                                 + 0.991162 w_k1_MSFT_s002
                                 + 0.928673 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.907458 w_k1_ADM_s002
                                 + 0.877803 w_k1_AXP_s002
                                 + 0.772942 w_k1_DIS_s002
                                 + 1.00647 w_k1_FDX_s002
                                 + 0.664476 w_k1_IBM_s002
                                 + 0.899199 w_k1_INTC_s002
                                 + 0.663028 w_k1_KO_s002
                                 + 0.541031 w_k1_MAT_s002
                                 + 0.834221 w_k1_MSFT_s002
                                 + 0.826188 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.03427 w_k1_ADM_s003
                                 + 1.01448 w_k1_AXP_s003
                                 + 1.08734 w_k1_DIS_s003
                                 + 1.08342 w_k1_FDX_s003
                                 + 0.65704 w_k1_IBM_s003
                                 + 1.00006 w_k1_INTC_s003
                                 + 0.880275 w_k1_KO_s003
                                 + 0.735958 w_k1_MAT_s003
                                 + 1.07151 w_k1_MSFT_s003
                                 + 0.783094 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.857037 w_k1_ADM_s003
                                 + 0.826512 w_k1_AXP_s003
                                 + 0.957911 w_k1_DIS_s003
                                 + 0.903301 w_k1_FDX_s003
                                 + 0.801735 w_k1_IBM_s003
                                 + 0.939092 w_k1_INTC_s003
                                 + 0.899193 w_k1_KO_s003
                                 + 0.874512 w_k1_MAT_s003
                                 + 0.890409 w_k1_MSFT_s003
                                 + 0.777792 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.09297 w_k1_ADM_s003
                                 + 1.09066 w_k1_AXP_s003 + 1.0556 w_k1_DIS_s003
                                 + 0.876594 w_k1_FDX_s003
                                 + 1.3814 w_k1_IBM_s003
                                 + 0.864747 w_k1_INTC_s003
                                 + 1.33472 w_k1_KO_s003 + 1.3892 w_k1_MAT_s003
                                 + 1.03305 w_k1_MSFT_s003
                                 + 0.962368 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.89778 w_k1_ADM_s003
                                 + 0.9006 w_k1_AXP_s003
                                 + 0.994011 w_k1_DIS_s003
                                 + 0.948341 w_k1_FDX_s003
                                 + 0.879359 w_k1_IBM_s003
                                 + 1.02117 w_k1_INTC_s003
                                 + 0.911295 w_k1_KO_s003
                                 + 0.816684 w_k1_MAT_s003
                                 + 0.997491 w_k1_MSFT_s003
                                 + 0.825039 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.06922 w_k1_ADM_s003
                                 + 1.41326 w_k1_AXP_s003
                                 + 1.10395 w_k1_DIS_s003
                                 + 1.01615 w_k1_FDX_s003
                                 + 1.13556 w_k1_IBM_s003
                                 + 0.958221 w_k1_INTC_s003
                                 + 0.998138 w_k1_KO_s003
                                 + 1.40836 w_k1_MAT_s003
                                 + 0.851271 w_k1_MSFT_s003
                                 + 0.968646 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.05906 w_k1_ADM_s003
                                 + 1.30552 w_k1_AXP_s003 + 1.9394 w_k1_DIS_s003
                                 + 1.34352 w_k1_FDX_s003
                                 + 1.15518 w_k1_IBM_s003
                                 + 1.40777 w_k1_INTC_s003
                                 + 1.29217 w_k1_KO_s003 + 1.45426 w_k1_MAT_s003
                                 + 1.31039 w_k1_MSFT_s003
                                 + 1.14985 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.937743 w_k1_ADM_s003
                                 + 0.701393 w_k1_AXP_s003
                                 + 0.624114 w_k1_DIS_s003
                                 + 0.733456 w_k1_FDX_s003
                                 + 0.690625 w_k1_IBM_s003
                                 + 0.714944 w_k1_INTC_s003
                                 + 0.811288 w_k1_KO_s003
                                 + 0.702414 w_k1_MAT_s003
                                 + 0.744442 w_k1_MSFT_s003
                                 + 0.728182 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.26903 w_k1_ADM_s003
                                 + 1.26784 w_k1_AXP_s003
                                 + 1.32426 w_k1_DIS_s003
                                 + 1.22382 w_k1_FDX_s003
                                 + 0.903613 w_k1_IBM_s003
                                 + 1.30213 w_k1_INTC_s003
                                 + 1.29295 w_k1_KO_s003 + 1.13283 w_k1_MAT_s003
                                 + 1.08097 w_k1_MSFT_s003
                                 + 1.05545 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.867858 w_k1_ADM_s004
                                 + 0.926435 w_k1_AXP_s004
                                 + 1.09101 w_k1_DIS_s004
                                 + 1.03895 w_k1_FDX_s004
                                 + 1.00252 w_k1_IBM_s004
                                 + 1.15723 w_k1_INTC_s004
                                 + 0.940363 w_k1_KO_s004
                                 + 0.70246 w_k1_MAT_s004
                                 + 1.10823 w_k1_MSFT_s004
                                 + 0.956113 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.07493 w_k1_ADM_s004
                                 + 1.46722 w_k1_AXP_s004
                                 + 1.41676 w_k1_DIS_s004
                                 + 1.28471 w_k1_FDX_s004
                                 + 1.19623 w_k1_IBM_s004
                                 + 1.06382 w_k1_INTC_s004 + 1.2118 w_k1_KO_s004
                                 + 1.76939 w_k1_MAT_s004
                                 + 1.20542 w_k1_MSFT_s004
                                 + 1.29997 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.802912 w_k1_ADM_s004
                                 + 1.07598 w_k1_AXP_s004
                                 + 0.777787 w_k1_DIS_s004
                                 + 0.880026 w_k1_FDX_s004
                                 + 0.757335 w_k1_IBM_s004
                                 + 0.67513 w_k1_INTC_s004
                                 + 0.682244 w_k1_KO_s004
                                 + 0.970348 w_k1_MAT_s004
                                 + 0.723167 w_k1_MSFT_s004
                                 + 0.872637 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.02855 w_k1_ADM_s004
                                 + 1.34542 w_k1_AXP_s004
                                 + 0.867276 w_k1_DIS_s004
                                 + 1.03448 w_k1_FDX_s004
                                 + 0.964766 w_k1_IBM_s004
                                 + 0.880359 w_k1_INTC_s004
                                 + 0.887512 w_k1_KO_s004
                                 + 1.48325 w_k1_MAT_s004
                                 + 0.892585 w_k1_MSFT_s004
                                 + 0.955473 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.798376 w_k1_ADM_s004
                                 + 0.81261 w_k1_AXP_s004
                                 + 0.66229 w_k1_DIS_s004
                                 + 0.726753 w_k1_FDX_s004
                                 + 1.03323 w_k1_IBM_s004
                                 + 0.874525 w_k1_INTC_s004
                                 + 0.839579 w_k1_KO_s004
                                 + 0.502256 w_k1_MAT_s004
                                 + 0.800189 w_k1_MSFT_s004
                                 + 0.76931 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.848242 w_k1_ADM_s004
                                 + 1.10393 w_k1_AXP_s004
                                 + 1.17359 w_k1_DIS_s004
                                 + 0.935872 w_k1_FDX_s004
                                 + 1.2703 w_k1_IBM_s004
                                 + 1.02769 w_k1_INTC_s004 + 1.1165 w_k1_KO_s004
                                 + 1.32563 w_k1_MAT_s004
                                 + 1.02777 w_k1_MSFT_s004
                                 + 1.03214 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.914128 w_k1_ADM_s004
                                 + 1.11101 w_k1_AXP_s004
                                 + 0.964549 w_k1_DIS_s004
                                 + 1.17358 w_k1_FDX_s004
                                 + 0.68065 w_k1_IBM_s004
                                 + 0.961208 w_k1_INTC_s004
                                 + 0.784357 w_k1_KO_s004
                                 + 0.962075 w_k1_MAT_s004
                                 + 1.11017 w_k1_MSFT_s004
                                 + 0.907675 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.982975 w_k1_ADM_s004
                                 + 0.851134 w_k1_AXP_s004
                                 + 1.00686 w_k1_DIS_s004
                                 + 0.871879 w_k1_FDX_s004
                                 + 0.947926 w_k1_IBM_s004
                                 + 0.760028 w_k1_INTC_s004
                                 + 0.838123 w_k1_KO_s004
                                 + 0.764859 w_k1_MAT_s004
                                 + 0.924101 w_k1_MSFT_s004
                                 + 0.904707 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.975251 w_k1_ADM_s005
                                 + 0.745314 w_k1_AXP_s005
                                 + 0.983944 w_k1_DIS_s005
                                 + 0.992578 w_k1_FDX_s005
                                 + 1.07356 w_k1_IBM_s005
                                 + 0.989581 w_k1_INTC_s005
                                 + 0.972912 w_k1_KO_s005
                                 + 0.90091 w_k1_MAT_s005
                                 + 1.10976 w_k1_MSFT_s005
                                 + 1.04418 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.855978 w_k1_ADM_s005
                                 + 0.637953 w_k1_AXP_s005
                                 + 0.672044 w_k1_DIS_s005
                                 + 0.740125 w_k1_FDX_s005
                                 + 0.973205 w_k1_IBM_s005
                                 + 0.731058 w_k1_INTC_s005
                                 + 0.697179 w_k1_KO_s005
                                 + 0.607242 w_k1_MAT_s005
                                 + 0.728486 w_k1_MSFT_s005
                                 + 0.779729 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.39076 w_k1_ADM_s005
                                 + 1.24553 w_k1_AXP_s005
                                 + 1.26997 w_k1_DIS_s005 + 1.2627 w_k1_FDX_s005
                                 + 1.09024 w_k1_IBM_s005
                                 + 1.10075 w_k1_INTC_s005
                                 + 1.21283 w_k1_KO_s005 + 1.19191 w_k1_MAT_s005
                                 + 1.27333 w_k1_MSFT_s005
                                 + 1.16887 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.855656 w_k1_ADM_s005
                                 + 0.632304 w_k1_AXP_s005
                                 + 0.629944 w_k1_DIS_s005
                                 + 0.799163 w_k1_FDX_s005
                                 + 0.667506 w_k1_IBM_s005
                                 + 0.714459 w_k1_INTC_s005
                                 + 0.680257 w_k1_KO_s005
                                 + 0.725301 w_k1_MAT_s005
                                 + 0.774475 w_k1_MSFT_s005
                                 + 0.795702 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.35218 w_k1_ADM_s005
                                 + 1.08289 w_k1_AXP_s005 + 1.43 w_k1_DIS_s005
                                 + 1.25911 w_k1_FDX_s005
                                 + 1.46099 w_k1_IBM_s005
                                 + 1.03328 w_k1_INTC_s005
                                 + 1.46942 w_k1_KO_s005 + 1.31658 w_k1_MAT_s005
                                 + 1.32198 w_k1_MSFT_s005
                                 + 1.23688 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.25628 w_k1_ADM_s005
                                 + 0.873925 w_k1_AXP_s005
                                 + 1.19143 w_k1_DIS_s005
                                 + 1.13587 w_k1_FDX_s005
                                 + 0.806819 w_k1_IBM_s005
                                 + 1.01226 w_k1_INTC_s005
                                 + 1.05364 w_k1_KO_s005
                                 + 0.746259 w_k1_MAT_s005
                                 + 1.09264 w_k1_MSFT_s005
                                 + 1.04841 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.10288 w_k1_ADM_s005
                                 + 1.13439 w_k1_AXP_s005
                                 + 0.796175 w_k1_DIS_s005
                                 + 0.984275 w_k1_FDX_s005
                                 + 0.968929 w_k1_IBM_s005
                                 + 0.909195 w_k1_INTC_s005
                                 + 0.973703 w_k1_KO_s005
                                 + 1.32411 w_k1_MAT_s005
                                 + 0.994228 w_k1_MSFT_s005
                                 + 1.03138 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.855567 w_k1_ADM_s005
                                 + 0.842915 w_k1_AXP_s005
                                 + 0.937187 w_k1_DIS_s005
                                 + 0.897263 w_k1_FDX_s005
                                 + 1.40104 w_k1_IBM_s005
                                 + 0.925034 w_k1_INTC_s005
                                 + 1.10481 w_k1_KO_s005 + 1.17449 w_k1_MAT_s005
                                 + 0.991706 w_k1_MSFT_s005
                                 + 0.882346 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.809023 w_k1_ADM_s006
                                 + 0.878878 w_k1_AXP_s006
                                 + 0.773528 w_k1_DIS_s006
                                 + 0.824735 w_k1_FDX_s006
                                 + 0.890481 w_k1_IBM_s006
                                 + 1.16646 w_k1_INTC_s006
                                 + 0.917127 w_k1_KO_s006
                                 + 0.962677 w_k1_MAT_s006
                                 + 0.84361 w_k1_MSFT_s006
                                 + 1.07532 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.07603 w_k1_ADM_s006
                                 + 1.87154 w_k1_AXP_s006
                                 + 1.29616 w_k1_DIS_s006
                                 + 1.18624 w_k1_FDX_s006
                                 + 1.20152 w_k1_IBM_s006
                                 + 1.26243 w_k1_INTC_s006
                                 + 1.25201 w_k1_KO_s006 + 1.79414 w_k1_MAT_s006
                                 + 1.165 w_k1_MSFT_s006 + 1.17223 w_k1_PG_s006
                                 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.3389 w_k1_ADM_s006
                                 + 1.19085 w_k1_AXP_s006
                                 + 1.36575 w_k1_DIS_s006
                                 + 1.24055 w_k1_FDX_s006
                                 + 0.911087 w_k1_IBM_s006
                                 + 1.28018 w_k1_INTC_s006
                                 + 1.14668 w_k1_KO_s006
                                 + 0.910399 w_k1_MAT_s006
                                 + 1.26114 w_k1_MSFT_s006
                                 + 1.13053 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.986883 w_k1_ADM_s006
                                 + 1.05112 w_k1_AXP_s006
                                 + 1.03536 w_k1_DIS_s006
                                 + 1.01188 w_k1_FDX_s006
                                 + 1.27137 w_k1_IBM_s006
                                 + 0.983489 w_k1_INTC_s006
                                 + 1.28579 w_k1_KO_s006 + 1.23116 w_k1_MAT_s006
                                 + 1.08933 w_k1_MSFT_s006
                                 + 1.01933 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.68855 w_k1_ADM_s006
                                 + 0.689918 w_k1_AXP_s006
                                 + 0.579897 w_k1_DIS_s006
                                 + 0.661917 w_k1_FDX_s006
                                 + 1.03555 w_k1_IBM_s006
                                 + 0.77068 w_k1_INTC_s006
                                 + 0.795349 w_k1_KO_s006
                                 + 1.00932 w_k1_MAT_s006
                                 + 0.770112 w_k1_MSFT_s006
                                 + 0.769264 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.936524 w_k1_ADM_s006
                                 + 0.842833 w_k1_AXP_s006
                                 + 0.740035 w_k1_DIS_s006 + 0.895 w_k1_FDX_s006
                                 + 0.727231 w_k1_IBM_s006
                                 + 0.924635 w_k1_INTC_s006
                                 + 0.755545 w_k1_KO_s006
                                 + 0.513755 w_k1_MAT_s006
                                 + 0.961761 w_k1_MSFT_s006
                                 + 0.793802 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.738634 w_k1_ADM_s006
                                 + 0.737315 w_k1_AXP_s006
                                 + 0.985586 w_k1_DIS_s006
                                 + 0.808567 w_k1_FDX_s006
                                 + 1.25663 w_k1_IBM_s006
                                 + 0.896438 w_k1_INTC_s006
                                 + 1.08085 w_k1_KO_s006
                                 + 0.898539 w_k1_MAT_s006
                                 + 0.987595 w_k1_MSFT_s006
                                 + 0.955768 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.858954 w_k1_ADM_s006
                                 + 1.14706 w_k1_AXP_s006
                                 + 0.941867 w_k1_DIS_s006
                                 + 0.957582 w_k1_FDX_s006
                                 + 0.89879 w_k1_IBM_s006
                                 + 0.871038 w_k1_INTC_s006
                                 + 0.874597 w_k1_KO_s006
                                 + 0.897767 w_k1_MAT_s006
                                 + 1.00892 w_k1_MSFT_s006
                                 + 0.970788 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.86736 w_k1_ADM_s007
                                 + 1.22871 w_k1_AXP_s007
                                 + 0.821769 w_k1_DIS_s007
                                 + 1.01015 w_k1_FDX_s007
                                 + 0.976727 w_k1_IBM_s007
                                 + 1.15386 w_k1_INTC_s007
                                 + 0.874362 w_k1_KO_s007 + 1.0992 w_k1_MAT_s007
                                 + 0.931864 w_k1_MSFT_s007
                                 + 0.918598 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.03198 w_k1_ADM_s007
                                 + 1.068 w_k1_AXP_s007 + 1.39995 w_k1_DIS_s007
                                 + 1.19061 w_k1_FDX_s007
                                 + 1.03278 w_k1_IBM_s007
                                 + 1.29307 w_k1_INTC_s007
                                 + 1.16164 w_k1_KO_s007 + 1.12737 w_k1_MAT_s007
                                 + 1.38367 w_k1_MSFT_s007
                                 + 1.02379 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.00176 w_k1_ADM_s007
                                 + 1.75941 w_k1_AXP_s007
                                 + 1.86062 w_k1_DIS_s007
                                 + 1.37409 w_k1_FDX_s007
                                 + 1.60501 w_k1_IBM_s007
                                 + 1.57663 w_k1_INTC_s007
                                 + 1.66547 w_k1_KO_s007 + 2.43696 w_k1_MAT_s007
                                 + 1.63731 w_k1_MSFT_s007
                                 + 1.28244 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.06017 w_k1_ADM_s007
                                 + 0.918106 w_k1_AXP_s007
                                 + 1.19196 w_k1_DIS_s007
                                 + 1.15377 w_k1_FDX_s007
                                 + 1.28431 w_k1_IBM_s007
                                 + 1.32413 w_k1_INTC_s007
                                 + 1.27064 w_k1_KO_s007 + 1.11392 w_k1_MAT_s007
                                 + 1.24464 w_k1_MSFT_s007
                                 + 1.04597 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.904855 w_k1_ADM_s007
                                 + 0.84429 w_k1_AXP_s007
                                 + 1.05028 w_k1_DIS_s007
                                 + 0.903583 w_k1_FDX_s007
                                 + 1.42669 w_k1_IBM_s007
                                 + 0.996315 w_k1_INTC_s007
                                 + 1.09559 w_k1_KO_s007 + 1.33425 w_k1_MAT_s007
                                 + 0.966794 w_k1_MSFT_s007
                                 + 0.90546 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.692172 w_k1_ADM_s007
                                 + 0.61295 w_k1_AXP_s007
                                 + 0.717697 w_k1_DIS_s007
                                 + 0.79932 w_k1_FDX_s007
                                 + 0.768805 w_k1_IBM_s007
                                 + 0.83847 w_k1_INTC_s007
                                 + 0.74157 w_k1_KO_s007
                                 + 0.731256 w_k1_MAT_s007
                                 + 0.872022 w_k1_MSFT_s007
                                 + 0.897156 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.932429 w_k1_ADM_s007
                                 + 0.91668 w_k1_AXP_s007
                                 + 0.728288 w_k1_DIS_s007
                                 + 0.869803 w_k1_FDX_s007
                                 + 0.94648 w_k1_IBM_s007
                                 + 0.931763 w_k1_INTC_s007
                                 + 0.7197 w_k1_KO_s007 + 0.740699 w_k1_MAT_s007
                                 + 0.78198 w_k1_MSFT_s007
                                 + 0.856953 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.972354 w_k1_ADM_s007
                                 + 0.979162 w_k1_AXP_s007
                                 + 1.08145 w_k1_DIS_s007
                                 + 1.13294 w_k1_FDX_s007
                                 + 0.737952 w_k1_IBM_s007
                                 + 1.3461 w_k1_INTC_s007 + 1.00788 w_k1_KO_s007
                                 + 0.813543 w_k1_MAT_s007
                                 + 1.26514 w_k1_MSFT_s007
                                 + 0.983201 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.14342 w_k1_ADM_s010
                                 + 1.05821 w_k1_AXP_s010
                                 + 1.60068 w_k1_DIS_s010
                                 + 1.17798 w_k1_FDX_s010
                                 + 1.15349 w_k1_IBM_s010
                                 + 1.1003 w_k1_INTC_s010 + 1.36193 w_k1_KO_s010
                                 + 1.10142 w_k1_MAT_s010
                                 + 1.31891 w_k1_MSFT_s010
                                 + 1.06047 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.884915 w_k1_ADM_s010
                                 + 0.90208 w_k1_AXP_s010
                                 + 0.88216 w_k1_DIS_s010
                                 + 0.868234 w_k1_FDX_s010
                                 + 1.25532 w_k1_IBM_s010
                                 + 0.88373 w_k1_INTC_s010
                                 + 1.16401 w_k1_KO_s010 + 1.64538 w_k1_MAT_s010
                                 + 0.980133 w_k1_MSFT_s010
                                 + 0.933098 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.14835 w_k1_ADM_s010
                                 + 1.14586 w_k1_AXP_s010
                                 + 1.20724 w_k1_DIS_s010 + 1.1528 w_k1_FDX_s010
                                 + 0.917187 w_k1_IBM_s010
                                 + 0.834293 w_k1_INTC_s010
                                 + 1.07013 w_k1_KO_s010
                                 + 0.959014 w_k1_MAT_s010
                                 + 1.06275 w_k1_MSFT_s010
                                 + 0.990818 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.63361 w_k1_ADM_s010
                                 + 1.74793 w_k1_AXP_s010
                                 + 2.02257 w_k1_DIS_s010
                                 + 1.58201 w_k1_FDX_s010
                                 + 1.43588 w_k1_IBM_s010
                                 + 1.19083 w_k1_INTC_s010
                                 + 1.58919 w_k1_KO_s010 + 1.81967 w_k1_MAT_s010
                                 + 1.75692 w_k1_MSFT_s010
                                 + 1.46635 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.44114 w_k1_ADM_s010
                                 + 1.32476 w_k1_AXP_s010
                                 + 1.39818 w_k1_DIS_s010
                                 + 1.37511 w_k1_FDX_s010
                                 + 1.05417 w_k1_IBM_s010
                                 + 0.945617 w_k1_INTC_s010
                                 + 1.0385 w_k1_KO_s010 + 1.20745 w_k1_MAT_s010
                                 + 1.33368 w_k1_MSFT_s010
                                 + 1.09247 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.741952 w_k1_ADM_s010
                                 + 0.721383 w_k1_AXP_s010
                                 + 0.74076 w_k1_DIS_s010
                                 + 0.818148 w_k1_FDX_s010
                                 + 0.721286 w_k1_IBM_s010
                                 + 0.636152 w_k1_INTC_s010
                                 + 0.737673 w_k1_KO_s010
                                 + 0.908192 w_k1_MAT_s010
                                 + 0.763116 w_k1_MSFT_s010
                                 + 0.832935 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.31528 w_k1_ADM_s010
                                 + 1.20518 w_k1_AXP_s010
                                 + 1.50432 w_k1_DIS_s010
                                 + 1.18021 w_k1_FDX_s010
                                 + 1.59079 w_k1_IBM_s010
                                 + 1.11566 w_k1_INTC_s010
                                 + 1.45883 w_k1_KO_s010 + 1.32362 w_k1_MAT_s010
                                 + 1.16592 w_k1_MSFT_s010 + 1.1877 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.993102 w_k1_ADM_s010
                                 + 0.987498 w_k1_AXP_s010
                                 + 0.882036 w_k1_DIS_s010
                                 + 0.972112 w_k1_FDX_s010
                                 + 0.950005 w_k1_IBM_s010
                                 + 0.882857 w_k1_INTC_s010
                                 + 0.949769 w_k1_KO_s010
                                 + 0.941102 w_k1_MAT_s010
                                 + 0.898755 w_k1_MSFT_s010
                                 + 0.946841 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.13208 w_k1_ADM_s011
                                 + 1.02733 w_k1_AXP_s011
                                 + 1.13815 w_k1_DIS_s011
                                 + 1.05013 w_k1_FDX_s011
                                 + 1.21183 w_k1_IBM_s011
                                 + 1.12959 w_k1_INTC_s011 + 1.203 w_k1_KO_s011
                                 + 0.966939 w_k1_MAT_s011
                                 + 1.14562 w_k1_MSFT_s011 + 1.035 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.271 w_k1_ADM_s011
                                 + 1.12385 w_k1_AXP_s011
                                 + 1.22518 w_k1_DIS_s011
                                 + 1.22637 w_k1_FDX_s011
                                 + 0.946405 w_k1_IBM_s011
                                 + 1.38121 w_k1_INTC_s011
                                 + 1.10202 w_k1_KO_s011
                                 + 0.893537 w_k1_MAT_s011
                                 + 1.19761 w_k1_MSFT_s011
                                 + 1.25177 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.78857 w_k1_ADM_s011
                                 + 1.74644 w_k1_AXP_s011
                                 + 1.68596 w_k1_DIS_s011 + 1.4855 w_k1_FDX_s011
                                 + 1.2409 w_k1_IBM_s011
                                 + 1.25607 w_k1_INTC_s011
                                 + 1.23575 w_k1_KO_s011 + 1.16036 w_k1_MAT_s011
                                 + 1.31265 w_k1_MSFT_s011
                                 + 1.36281 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.965104 w_k1_ADM_s011
                                 + 0.78455 w_k1_AXP_s011
                                 + 0.681622 w_k1_DIS_s011
                                 + 0.827723 w_k1_FDX_s011
                                 + 0.651611 w_k1_IBM_s011
                                 + 0.83152 w_k1_INTC_s011
                                 + 0.759159 w_k1_KO_s011
                                 + 0.650831 w_k1_MAT_s011
                                 + 0.866597 w_k1_MSFT_s011
                                 + 0.816207 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.04714 w_k1_ADM_s011
                                 + 1.08301 w_k1_AXP_s011
                                 + 1.04757 w_k1_DIS_s011
                                 + 1.20111 w_k1_FDX_s011
                                 + 0.776885 w_k1_IBM_s011
                                 + 1.26175 w_k1_INTC_s011
                                 + 1.05021 w_k1_KO_s011
                                 + 0.900315 w_k1_MAT_s011
                                 + 1.1083 w_k1_MSFT_s011 + 1.0769 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.816695 w_k1_ADM_s011
                                 + 1.42881 w_k1_AXP_s011
                                 + 1.20523 w_k1_DIS_s011 + 1.0148 w_k1_FDX_s011
                                 + 1.5707 w_k1_IBM_s011
                                 + 1.31739 w_k1_INTC_s011
                                 + 1.39448 w_k1_KO_s011 + 1.77845 w_k1_MAT_s011
                                 + 1.01128 w_k1_MSFT_s011
                                 + 1.21381 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.06427 w_k1_ADM_s011
                                 + 0.763316 w_k1_AXP_s011
                                 + 0.675994 w_k1_DIS_s011
                                 + 0.819566 w_k1_FDX_s011
                                 + 0.921293 w_k1_IBM_s011
                                 + 0.798769 w_k1_INTC_s011
                                 + 0.848643 w_k1_KO_s011
                                 + 0.723531 w_k1_MAT_s011
                                 + 0.957774 w_k1_MSFT_s011
                                 + 0.911886 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.988298 w_k1_ADM_s011
                                 + 0.953419 w_k1_AXP_s011
                                 + 1.10613 w_k1_DIS_s011
                                 + 0.939862 w_k1_FDX_s011
                                 + 0.961706 w_k1_IBM_s011
                                 + 1.14589 w_k1_INTC_s011
                                 + 1.18942 w_k1_KO_s011
                                 + 0.892883 w_k1_MAT_s011
                                 + 1.10318 w_k1_MSFT_s011
                                 + 1.06356 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + w_k1_ADM_s012
                                 + 0.571631 w_k1_AXP_s012
                                 + 0.730909 w_k1_DIS_s012
                                 + 0.769576 w_k1_FDX_s012
                                 + 0.756124 w_k1_IBM_s012
                                 + 0.873562 w_k1_INTC_s012
                                 + 0.851822 w_k1_KO_s012
                                 + 0.485845 w_k1_MAT_s012
                                 + 0.762592 w_k1_MSFT_s012
                                 + 0.793579 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.17209 w_k1_ADM_s012
                                 + 1.01565 w_k1_AXP_s012
                                 + 1.11291 w_k1_DIS_s012
                                 + 1.02247 w_k1_FDX_s012
                                 + 0.891285 w_k1_IBM_s012
                                 + 0.950193 w_k1_INTC_s012
                                 + 1.01006 w_k1_KO_s012
                                 + 0.735547 w_k1_MAT_s012
                                 + 0.995972 w_k1_MSFT_s012
                                 + 0.884102 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.19389 w_k1_ADM_s012
                                 + 0.709649 w_k1_AXP_s012
                                 + 0.766342 w_k1_DIS_s012
                                 + 0.864445 w_k1_FDX_s012
                                 + 1.01093 w_k1_IBM_s012
                                 + 0.841176 w_k1_INTC_s012
                                 + 1.00373 w_k1_KO_s012 + 0.81707 w_k1_MAT_s012
                                 + 0.806462 w_k1_MSFT_s012
                                 + 0.871903 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.00672 w_k1_ADM_s012
                                 + 0.778288 w_k1_AXP_s012
                                 + 1.05958 w_k1_DIS_s012
                                 + 0.959019 w_k1_FDX_s012
                                 + 1.12143 w_k1_IBM_s012
                                 + 0.874767 w_k1_INTC_s012
                                 + 1.16313 w_k1_KO_s012 + 1.00059 w_k1_MAT_s012
                                 + 1.1193 w_k1_MSFT_s012 + 1.0024 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.27461 w_k1_ADM_s012
                                 + 1.16285 w_k1_AXP_s012
                                 + 1.15677 w_k1_DIS_s012
                                 + 1.20969 w_k1_FDX_s012
                                 + 0.636147 w_k1_IBM_s012
                                 + 1.25904 w_k1_INTC_s012
                                 + 0.930828 w_k1_KO_s012
                                 + 0.74861 w_k1_MAT_s012
                                 + 1.30644 w_k1_MSFT_s012
                                 + 0.884589 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.11411 w_k1_ADM_s012
                                 + 1.15099 w_k1_AXP_s012
                                 + 1.23869 w_k1_DIS_s012
                                 + 1.03772 w_k1_FDX_s012
                                 + 1.75521 w_k1_IBM_s012
                                 + 1.10455 w_k1_INTC_s012
                                 + 1.19842 w_k1_KO_s012 + 1.52465 w_k1_MAT_s012
                                 + 1.03464 w_k1_MSFT_s012
                                 + 0.907987 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.834612 w_k1_ADM_s012
                                 + 1.19336 w_k1_AXP_s012
                                 + 0.799387 w_k1_DIS_s012
                                 + 0.855009 w_k1_FDX_s012
                                 + 1.1611 w_k1_IBM_s012
                                 + 0.90221 w_k1_INTC_s012
                                 + 0.95396 w_k1_KO_s012 + 1.10122 w_k1_MAT_s012
                                 + 0.902003 w_k1_MSFT_s012
                                 + 0.831262 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.08516 w_k1_ADM_s012
                                 + 1.29347 w_k1_AXP_s012 + 1.5823 w_k1_DIS_s012
                                 + 1.3878 w_k1_FDX_s012 + 1.06468 w_k1_IBM_s012
                                 + 1.07181 w_k1_INTC_s012
                                 + 1.07228 w_k1_KO_s012 + 1.07546 w_k1_MAT_s012
                                 + 1.16026 w_k1_MSFT_s012
                                 + 1.07417 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.699946 w_k1_ADM_s013
                                 + 0.602631 w_k1_AXP_s013
                                 + 0.502857 w_k1_DIS_s013
                                 + 0.627664 w_k1_FDX_s013
                                 + 0.904926 w_k1_IBM_s013
                                 + 0.698016 w_k1_INTC_s013
                                 + 0.681894 w_k1_KO_s013
                                 + 0.799853 w_k1_MAT_s013
                                 + 0.746683 w_k1_MSFT_s013
                                 + 0.804098 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.34536 w_k1_ADM_s013
                                 + 1.40539 w_k1_AXP_s013
                                 + 1.30851 w_k1_DIS_s013
                                 + 1.28438 w_k1_FDX_s013
                                 + 1.19637 w_k1_IBM_s013
                                 + 1.18835 w_k1_INTC_s013
                                 + 1.23254 w_k1_KO_s013 + 1.36248 w_k1_MAT_s013
                                 + 1.28354 w_k1_MSFT_s013
                                 + 1.24525 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.908088 w_k1_ADM_s013
                                 + 0.773647 w_k1_AXP_s013
                                 + 0.782425 w_k1_DIS_s013
                                 + 0.88799 w_k1_FDX_s013
                                 + 0.777707 w_k1_IBM_s013
                                 + 0.950947 w_k1_INTC_s013
                                 + 0.78673 w_k1_KO_s013
                                 + 0.579233 w_k1_MAT_s013
                                 + 0.91613 w_k1_MSFT_s013
                                 + 0.945933 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.01144 w_k1_ADM_s013
                                 + 0.89136 w_k1_AXP_s013
                                 + 0.822971 w_k1_DIS_s013
                                 + 0.846097 w_k1_FDX_s013
                                 + 1.09555 w_k1_IBM_s013
                                 + 0.807156 w_k1_INTC_s013
                                 + 0.974015 w_k1_KO_s013
                                 + 1.10234 w_k1_MAT_s013
                                 + 0.963425 w_k1_MSFT_s013
                                 + 0.967681 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.926726 w_k1_ADM_s013
                                 + 0.995941 w_k1_AXP_s013
                                 + 1.22077 w_k1_DIS_s013
                                 + 0.963239 w_k1_FDX_s013
                                 + 1.18452 w_k1_IBM_s013
                                 + 1.27077 w_k1_INTC_s013
                                 + 1.23475 w_k1_KO_s013 + 1.19178 w_k1_MAT_s013
                                 + 1.15695 w_k1_MSFT_s013
                                 + 1.15387 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.09301 w_k1_ADM_s013
                                 + 1.3844 w_k1_AXP_s013 + 1.57856 w_k1_DIS_s013
                                 + 1.19842 w_k1_FDX_s013
                                 + 1.50564 w_k1_IBM_s013
                                 + 1.40504 w_k1_INTC_s013 + 1.5354 w_k1_KO_s013
                                 + 1.53405 w_k1_MAT_s013
                                 + 1.26332 w_k1_MSFT_s013
                                 + 1.41303 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.0369 w_k1_ADM_s013
                                 + 0.891601 w_k1_AXP_s013
                                 + 0.759475 w_k1_DIS_s013
                                 + 0.840259 w_k1_FDX_s013
                                 + 1.33216 w_k1_IBM_s013
                                 + 0.954182 w_k1_INTC_s013
                                 + 1.05305 w_k1_KO_s013 + 1.19932 w_k1_MAT_s013
                                 + 0.906014 w_k1_MSFT_s013
                                 + 0.956776 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.922288 w_k1_ADM_s013
                                 + 1.3081 w_k1_AXP_s013 + 1.02089 w_k1_DIS_s013
                                 + 1.13586 w_k1_FDX_s013
                                 + 0.898686 w_k1_IBM_s013
                                 + 1.2133 w_k1_INTC_s013
                                 + 0.876741 w_k1_KO_s013
                                 + 0.989137 w_k1_MAT_s013
                                 + 1.08418 w_k1_MSFT_s013
                                 + 0.948843 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.938529 w_k1_ADM_s014
                                 + 0.911101 w_k1_AXP_s014
                                 + 0.844449 w_k1_DIS_s014
                                 + 1.0853 w_k1_FDX_s014
                                 + 0.774461 w_k1_IBM_s014
                                 + 1.0744 w_k1_INTC_s014
                                 + 0.884415 w_k1_KO_s014
                                 + 0.796746 w_k1_MAT_s014
                                 + 0.937779 w_k1_MSFT_s014
                                 + 0.955577 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.889567 w_k1_ADM_s014
                                 + 0.669324 w_k1_AXP_s014
                                 + 0.827993 w_k1_DIS_s014
                                 + 0.856888 w_k1_FDX_s014
                                 + 0.769688 w_k1_IBM_s014
                                 + 0.931037 w_k1_INTC_s014
                                 + 0.84537 w_k1_KO_s014
                                 + 0.884679 w_k1_MAT_s014
                                 + 0.801491 w_k1_MSFT_s014
                                 + 0.910189 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.24004 w_k1_ADM_s014
                                 + 1.40849 w_k1_AXP_s014
                                 + 1.42835 w_k1_DIS_s014
                                 + 1.25379 w_k1_FDX_s014
                                 + 1.15657 w_k1_IBM_s014
                                 + 1.16327 w_k1_INTC_s014
                                 + 1.17095 w_k1_KO_s014 + 1.34025 w_k1_MAT_s014
                                 + 1.11792 w_k1_MSFT_s014
                                 + 1.05733 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.815939 w_k1_ADM_s014
                                 + 1.18128 w_k1_AXP_s014
                                 + 0.843278 w_k1_DIS_s014
                                 + 1.07861 w_k1_FDX_s014
                                 + 1.02655 w_k1_IBM_s014
                                 + 1.19369 w_k1_INTC_s014
                                 + 0.92049 w_k1_KO_s014 + 1.12809 w_k1_MAT_s014
                                 + 0.907717 w_k1_MSFT_s014
                                 + 0.983255 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.848522 w_k1_ADM_s014
                                 + 1.08815 w_k1_AXP_s014
                                 + 1.16831 w_k1_DIS_s014
                                 + 1.04364 w_k1_FDX_s014
                                 + 1.60642 w_k1_IBM_s014
                                 + 0.991674 w_k1_INTC_s014
                                 + 1.16179 w_k1_KO_s014 + 1.1738 w_k1_MAT_s014
                                 + 1.0077 w_k1_MSFT_s014 + 1.04643 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.13423 w_k1_ADM_s014
                                 + 1.15145 w_k1_AXP_s014
                                 + 1.24896 w_k1_DIS_s014
                                 + 1.37618 w_k1_FDX_s014
                                 + 0.891998 w_k1_IBM_s014
                                 + 1.25533 w_k1_INTC_s014
                                 + 1.02262 w_k1_KO_s014
                                 + 0.979868 w_k1_MAT_s014
                                 + 1.21849 w_k1_MSFT_s014
                                 + 1.10739 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.08315 w_k1_ADM_s014
                                 + 1.13753 w_k1_AXP_s014
                                 + 0.917936 w_k1_DIS_s014
                                 + 0.96852 w_k1_FDX_s014
                                 + 1.38661 w_k1_IBM_s014
                                 + 1.15533 w_k1_INTC_s014
                                 + 1.15404 w_k1_KO_s014 + 1.49441 w_k1_MAT_s014
                                 + 0.971464 w_k1_MSFT_s014
                                 + 1.13807 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.836204 w_k1_ADM_s014
                                 + 0.671281 w_k1_AXP_s014
                                 + 0.598924 w_k1_DIS_s014
                                 + 0.764062 w_k1_FDX_s014
                                 + 0.994777 w_k1_IBM_s014
                                 + 0.732377 w_k1_INTC_s014
                                 + 0.796321 w_k1_KO_s014
                                 + 0.686332 w_k1_MAT_s014
                                 + 0.686715 w_k1_MSFT_s014
                                 + 0.844474 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.16871 w_k1_ADM_s015
                                 + 1.21567 w_k1_AXP_s015 + 1.4225 w_k1_DIS_s015
                                 + 1.24634 w_k1_FDX_s015
                                 + 1.22247 w_k1_IBM_s015
                                 + 1.11412 w_k1_INTC_s015
                                 + 1.39495 w_k1_KO_s015 + 1.29378 w_k1_MAT_s015
                                 + 1.27034 w_k1_MSFT_s015
                                 + 1.15218 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.40116 w_k1_ADM_s015
                                 + 1.43755 w_k1_AXP_s015
                                 + 1.97533 w_k1_DIS_s015
                                 + 1.42912 w_k1_FDX_s015
                                 + 1.72738 w_k1_IBM_s015
                                 + 1.18404 w_k1_INTC_s015
                                 + 1.67804 w_k1_KO_s015 + 1.74951 w_k1_MAT_s015
                                 + 1.56726 w_k1_MSFT_s015
                                 + 1.35503 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.96043 w_k1_ADM_s015
                                 + 0.928501 w_k1_AXP_s015
                                 + 1.12196 w_k1_DIS_s015
                                 + 1.09418 w_k1_FDX_s015
                                 + 1.14149 w_k1_IBM_s015
                                 + 0.99761 w_k1_INTC_s015
                                 + 1.03527 w_k1_KO_s015 + 0.88628 w_k1_MAT_s015
                                 + 1.04476 w_k1_MSFT_s015
                                 + 1.04245 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.14396 w_k1_ADM_s015
                                 + 0.997228 w_k1_AXP_s015
                                 + 0.895798 w_k1_DIS_s015
                                 + 1.06569 w_k1_FDX_s015
                                 + 0.704833 w_k1_IBM_s015
                                 + 1.14702 w_k1_INTC_s015
                                 + 0.894341 w_k1_KO_s015
                                 + 0.620911 w_k1_MAT_s015
                                 + 0.933488 w_k1_MSFT_s015
                                 + 0.942566 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.910775 w_k1_ADM_s015
                                 + 0.962768 w_k1_AXP_s015
                                 + 1.06721 w_k1_DIS_s015
                                 + 1.01535 w_k1_FDX_s015
                                 + 0.884573 w_k1_IBM_s015
                                 + 1.17481 w_k1_INTC_s015
                                 + 0.96369 w_k1_KO_s015
                                 + 0.677819 w_k1_MAT_s015
                                 + 0.927128 w_k1_MSFT_s015
                                 + 0.963292 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.758116 w_k1_ADM_s015
                                 + 0.767741 w_k1_AXP_s015
                                 + 0.637864 w_k1_DIS_s015
                                 + 0.744724 w_k1_FDX_s015
                                 + 0.975126 w_k1_IBM_s015
                                 + 0.793337 w_k1_INTC_s015
                                 + 0.925841 w_k1_KO_s015 + 1.0769 w_k1_MAT_s015
                                 + 0.819411 w_k1_MSFT_s015
                                 + 0.875673 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.62944 w_k1_ADM_s015
                                 + 0.624232 w_k1_AXP_s015
                                 + 0.718898 w_k1_DIS_s015
                                 + 0.762505 w_k1_FDX_s015
                                 + 0.811687 w_k1_IBM_s015
                                 + 0.951455 w_k1_INTC_s015
                                 + 0.911419 w_k1_KO_s015
                                 + 0.576788 w_k1_MAT_s015
                                 + 0.836206 w_k1_MSFT_s015
                                 + 0.895093 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.05693 w_k1_ADM_s015
                                 + 1.19472 w_k1_AXP_s015
                                 + 1.07799 w_k1_DIS_s015
                                 + 0.923377 w_k1_FDX_s015
                                 + 1.44388 w_k1_IBM_s015
                                 + 1.20537 w_k1_INTC_s015
                                 + 1.24652 w_k1_KO_s015 + 1.09049 w_k1_MAT_s015
                                 + 0.938427 w_k1_MSFT_s015
                                 + 1.05542 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.48377 w_k1_ADM_s016
                                 + 1.82582 w_k1_AXP_s016
                                 + 1.53749 w_k1_DIS_s016
                                 + 1.42162 w_k1_FDX_s016
                                 + 1.42649 w_k1_IBM_s016
                                 + 0.941932 w_k1_INTC_s016
                                 + 1.14451 w_k1_KO_s016 + 1.67851 w_k1_MAT_s016
                                 + 1.20917 w_k1_MSFT_s016
                                 + 1.01997 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.02676 w_k1_ADM_s016
                                 + 0.946483 w_k1_AXP_s016
                                 + 1.24109 w_k1_DIS_s016
                                 + 1.20581 w_k1_FDX_s016
                                 + 0.673493 w_k1_IBM_s016
                                 + 1.11625 w_k1_INTC_s016
                                 + 0.892721 w_k1_KO_s016
                                 + 0.425676 w_k1_MAT_s016
                                 + 1.28586 w_k1_MSFT_s016
                                 + 0.974361 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.800165 w_k1_ADM_s016
                                 + 0.70971 w_k1_AXP_s016
                                 + 0.912571 w_k1_DIS_s016
                                 + 0.869493 w_k1_FDX_s016
                                 + 0.83803 w_k1_IBM_s016
                                 + 0.964708 w_k1_INTC_s016
                                 + 0.91399 w_k1_KO_s016
                                 + 0.640462 w_k1_MAT_s016
                                 + 1.00661 w_k1_MSFT_s016
                                 + 0.908892 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.08463 w_k1_ADM_s016
                                 + 1.36776 w_k1_AXP_s016
                                 + 1.59722 w_k1_DIS_s016
                                 + 1.49748 w_k1_FDX_s016
                                 + 0.981374 w_k1_IBM_s016
                                 + 1.1743 w_k1_INTC_s016 + 1.07743 w_k1_KO_s016
                                 + 0.946279 w_k1_MAT_s016
                                 + 1.36458 w_k1_MSFT_s016
                                 + 1.18434 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.715757 w_k1_ADM_s016
                                 + 0.760023 w_k1_AXP_s016
                                 + 0.764004 w_k1_DIS_s016
                                 + 0.828407 w_k1_FDX_s016
                                 + 1.05548 w_k1_IBM_s016
                                 + 0.908555 w_k1_INTC_s016
                                 + 0.750883 w_k1_KO_s016
                                 + 0.862398 w_k1_MAT_s016
                                 + 0.907636 w_k1_MSFT_s016
                                 + 0.789498 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.25505 w_k1_ADM_s016
                                 + 1.39641 w_k1_AXP_s016
                                 + 1.44199 w_k1_DIS_s016 + 1.1567 w_k1_FDX_s016
                                 + 1.06403 w_k1_IBM_s016
                                 + 1.16409 w_k1_INTC_s016
                                 + 1.13492 w_k1_KO_s016 + 1.10513 w_k1_MAT_s016
                                 + 1.25442 w_k1_MSFT_s016
                                 + 1.05092 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.16845 w_k1_ADM_s016
                                 + 1.35958 w_k1_AXP_s016
                                 + 1.09959 w_k1_DIS_s016
                                 + 0.996705 w_k1_FDX_s016
                                 + 1.37485 w_k1_IBM_s016
                                 + 0.791168 w_k1_INTC_s016
                                 + 1.03443 w_k1_KO_s016 + 1.32514 w_k1_MAT_s016
                                 + 0.919492 w_k1_MSFT_s016
                                 + 1.00938 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.959249 w_k1_ADM_s016
                                 + 1.19078 w_k1_AXP_s016
                                 + 1.02258 w_k1_DIS_s016
                                 + 1.09699 w_k1_FDX_s016
                                 + 1.07322 w_k1_IBM_s016
                                 + 0.891095 w_k1_INTC_s016
                                 + 1.00399 w_k1_KO_s016 + 1.03011 w_k1_MAT_s016
                                 + 0.943613 w_k1_MSFT_s016
                                 + 0.980488 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.892908 w_k1_ADM_s017
                                 + 1.10383 w_k1_AXP_s017
                                 + 1.45305 w_k1_DIS_s017
                                 + 1.23863 w_k1_FDX_s017
                                 + 1.17597 w_k1_IBM_s017
                                 + 1.11251 w_k1_INTC_s017
                                 + 1.30938 w_k1_KO_s017 + 1.22216 w_k1_MAT_s017
                                 + 1.29395 w_k1_MSFT_s017
                                 + 1.09548 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.816015 w_k1_ADM_s017
                                 + 0.839525 w_k1_AXP_s017
                                 + 0.725299 w_k1_DIS_s017
                                 + 0.787366 w_k1_FDX_s017
                                 + 0.832409 w_k1_IBM_s017
                                 + 0.927055 w_k1_INTC_s017
                                 + 0.779066 w_k1_KO_s017
                                 + 0.800808 w_k1_MAT_s017
                                 + 0.832452 w_k1_MSFT_s017
                                 + 0.802123 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.713854 w_k1_ADM_s017
                                 + 0.928616 w_k1_AXP_s017
                                 + 0.954465 w_k1_DIS_s017
                                 + 0.813186 w_k1_FDX_s017
                                 + 1.21956 w_k1_IBM_s017
                                 + 0.846032 w_k1_INTC_s017
                                 + 1.00548 w_k1_KO_s017 + 1.14154 w_k1_MAT_s017
                                 + 0.86667 w_k1_MSFT_s017
                                 + 0.830244 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.775043 w_k1_ADM_s017
                                 + 0.877074 w_k1_AXP_s017
                                 + 0.816058 w_k1_DIS_s017
                                 + 0.885563 w_k1_FDX_s017
                                 + 0.647837 w_k1_IBM_s017
                                 + 0.913752 w_k1_INTC_s017
                                 + 0.690562 w_k1_KO_s017
                                 + 0.534802 w_k1_MAT_s017
                                 + 0.831939 w_k1_MSFT_s017
                                 + 0.85217 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0188 w_k1_ADM_s017
                                 + 1.23647 w_k1_AXP_s017
                                 + 1.00961 w_k1_DIS_s017
                                 + 1.06026 w_k1_FDX_s017
                                 + 1.11511 w_k1_IBM_s017
                                 + 1.09043 w_k1_INTC_s017
                                 + 1.09001 w_k1_KO_s017 + 1.26073 w_k1_MAT_s017
                                 + 1.08799 w_k1_MSFT_s017
                                 + 0.926663 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.20983 w_k1_ADM_s017
                                 + 1.26617 w_k1_AXP_s017
                                 + 1.21996 w_k1_DIS_s017
                                 + 1.27667 w_k1_FDX_s017
                                 + 0.924865 w_k1_IBM_s017
                                 + 1.19557 w_k1_INTC_s017
                                 + 1.09958 w_k1_KO_s017 + 1.27641 w_k1_MAT_s017
                                 + 1.18873 w_k1_MSFT_s017
                                 + 1.14567 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.852936 w_k1_ADM_s017
                                 + 0.756572 w_k1_AXP_s017
                                 + 0.516252 w_k1_DIS_s017
                                 + 0.634306 w_k1_FDX_s017
                                 + 0.780919 w_k1_IBM_s017
                                 + 0.788558 w_k1_INTC_s017
                                 + 0.647554 w_k1_KO_s017
                                 + 0.708988 w_k1_MAT_s017
                                 + 0.625099 w_k1_MSFT_s017
                                 + 0.745706 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.878523 w_k1_ADM_s017
                                 + 0.970113 w_k1_AXP_s017
                                 + 1.1012 w_k1_DIS_s017
                                 + 0.971071 w_k1_FDX_s017
                                 + 1.25627 w_k1_IBM_s017
                                 + 1.0314 w_k1_INTC_s017 + 1.17016 w_k1_KO_s017
                                 + 1.30372 w_k1_MAT_s017
                                 + 1.04551 w_k1_MSFT_s017
                                 + 1.01059 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.944205 w_k1_ADM_s100
                                 + 0.936996 w_k1_AXP_s100
                                 + 0.919098 w_k1_DIS_s100
                                 + 0.885367 w_k1_FDX_s100
                                 + 1.55226 w_k1_IBM_s100
                                 + 0.956751 w_k1_INTC_s100
                                 + 1.06723 w_k1_KO_s100 + 1.10707 w_k1_MAT_s100
                                 + 0.891753 w_k1_MSFT_s100
                                 + 1.05295 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.916971 w_k1_ADM_s100
                                 + 1.21355 w_k1_AXP_s100
                                 + 0.992316 w_k1_DIS_s100
                                 + 0.989566 w_k1_FDX_s100
                                 + 1.19962 w_k1_IBM_s100
                                 + 0.851925 w_k1_INTC_s100
                                 + 1.1429 w_k1_KO_s100 + 1.25995 w_k1_MAT_s100
                                 + 1.00724 w_k1_MSFT_s100
                                 + 1.02205 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.928151 w_k1_ADM_s100
                                 + 0.780381 w_k1_AXP_s100
                                 + 0.740352 w_k1_DIS_s100
                                 + 0.851639 w_k1_FDX_s100
                                 + 0.780551 w_k1_IBM_s100
                                 + 0.839747 w_k1_INTC_s100
                                 + 0.778849 w_k1_KO_s100
                                 + 0.616384 w_k1_MAT_s100
                                 + 0.929559 w_k1_MSFT_s100
                                 + 0.860492 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.1749 w_k1_ADM_s100
                                 + 1.24604 w_k1_AXP_s100
                                 + 1.51865 w_k1_DIS_s100
                                 + 1.31565 w_k1_FDX_s100
                                 + 1.28946 w_k1_IBM_s100
                                 + 1.12744 w_k1_INTC_s100
                                 + 1.27521 w_k1_KO_s100
                                 + 0.969771 w_k1_MAT_s100
                                 + 1.31843 w_k1_MSFT_s100
                                 + 1.17854 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.02236 w_k1_ADM_s100
                                 + 0.997253 w_k1_AXP_s100
                                 + 1.44584 w_k1_DIS_s100
                                 + 1.03097 w_k1_FDX_s100
                                 + 1.34929 w_k1_IBM_s100
                                 + 1.03698 w_k1_INTC_s100 + 1.4664 w_k1_KO_s100
                                 + 1.33781 w_k1_MAT_s100
                                 + 1.07899 w_k1_MSFT_s100
                                 + 1.05788 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.19783 w_k1_ADM_s100
                                 + 1.26081 w_k1_AXP_s100
                                 + 1.20546 w_k1_DIS_s100
                                 + 1.14554 w_k1_FDX_s100
                                 + 1.53539 w_k1_IBM_s100
                                 + 0.914668 w_k1_INTC_s100
                                 + 1.30557 w_k1_KO_s100 + 1.75503 w_k1_MAT_s100
                                 + 1.05397 w_k1_MSFT_s100
                                 + 1.21039 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.861826 w_k1_ADM_s100
                                 + 0.780451 w_k1_AXP_s100
                                 + 0.930844 w_k1_DIS_s100
                                 + 0.809039 w_k1_FDX_s100
                                 + 1.19505 w_k1_IBM_s100
                                 + 0.861166 w_k1_INTC_s100
                                 + 1.1944 w_k1_KO_s100 + 1.37013 w_k1_MAT_s100
                                 + 0.885895 w_k1_MSFT_s100
                                 + 1.00801 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.18832 w_k1_ADM_s100
                                 + 1.05144 w_k1_AXP_s100
                                 + 1.13812 w_k1_DIS_s100
                                 + 1.11402 w_k1_FDX_s100
                                 + 0.821654 w_k1_IBM_s100
                                 + 1.11243 w_k1_INTC_s100
                                 + 0.880005 w_k1_KO_s100
                                 + 0.760797 w_k1_MAT_s100
                                 + 1.19511 w_k1_MSFT_s100
                                 + 0.991067 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.03977 w_k1_ADM_s101
                                 + 0.717606 w_k1_AXP_s101
                                 + 0.630774 w_k1_DIS_s101
                                 + 0.714237 w_k1_FDX_s101
                                 + 1.05533 w_k1_IBM_s101
                                 + 0.701436 w_k1_INTC_s101
                                 + 0.995137 w_k1_KO_s101
                                 + 0.944846 w_k1_MAT_s101
                                 + 0.683939 w_k1_MSFT_s101
                                 + 0.836741 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.18635 w_k1_ADM_s101
                                 + 0.991909 w_k1_AXP_s101
                                 + 0.873793 w_k1_DIS_s101
                                 + 0.866659 w_k1_FDX_s101
                                 + 1.0098 w_k1_IBM_s101
                                 + 0.925628 w_k1_INTC_s101
                                 + 0.980868 w_k1_KO_s101
                                 + 1.10067 w_k1_MAT_s101
                                 + 0.916747 w_k1_MSFT_s101
                                 + 0.891733 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.11155 w_k1_ADM_s101
                                 + 1.26295 w_k1_AXP_s101
                                 + 1.24143 w_k1_DIS_s101
                                 + 1.11186 w_k1_FDX_s101
                                 + 1.54076 w_k1_IBM_s101
                                 + 0.852381 w_k1_INTC_s101
                                 + 1.34512 w_k1_KO_s101 + 1.53398 w_k1_MAT_s101
                                 + 0.884336 w_k1_MSFT_s101
                                 + 1.15493 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.64159 w_k1_ADM_s101
                                 + 1.56 w_k1_AXP_s101 + 1.59039 w_k1_DIS_s101
                                 + 1.48567 w_k1_FDX_s101
                                 + 0.894983 w_k1_IBM_s101
                                 + 1.32635 w_k1_INTC_s101
                                 + 1.37538 w_k1_KO_s101 + 1.04286 w_k1_MAT_s101
                                 + 1.46162 w_k1_MSFT_s101
                                 + 1.29229 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.64094 w_k1_ADM_s101
                                 + 1.00047 w_k1_AXP_s101
                                 + 1.48842 w_k1_DIS_s101
                                 + 1.22302 w_k1_FDX_s101
                                 + 1.16025 w_k1_IBM_s101
                                 + 1.03683 w_k1_INTC_s101
                                 + 1.30569 w_k1_KO_s101 + 1.09128 w_k1_MAT_s101
                                 + 1.16452 w_k1_MSFT_s101
                                 + 1.21267 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.17139 w_k1_ADM_s101
                                 + 0.750421 w_k1_AXP_s101
                                 + 0.765238 w_k1_DIS_s101
                                 + 0.895621 w_k1_FDX_s101
                                 + 0.697127 w_k1_IBM_s101
                                 + 0.953252 w_k1_INTC_s101
                                 + 0.931259 w_k1_KO_s101
                                 + 0.709691 w_k1_MAT_s101
                                 + 0.858325 w_k1_MSFT_s101
                                 + 0.930221 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.50258 w_k1_ADM_s101
                                 + 1.05096 w_k1_AXP_s101
                                 + 1.14526 w_k1_DIS_s101
                                 + 1.18186 w_k1_FDX_s101
                                 + 0.784186 w_k1_IBM_s101
                                 + 1.10444 w_k1_INTC_s101
                                 + 1.06625 w_k1_KO_s101
                                 + 0.906186 w_k1_MAT_s101
                                 + 1.15249 w_k1_MSFT_s101
                                 + 1.13563 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.35832 w_k1_ADM_s101
                                 + 0.780611 w_k1_AXP_s101
                                 + 1.04727 w_k1_DIS_s101
                                 + 0.946912 w_k1_FDX_s101
                                 + 0.932275 w_k1_IBM_s101
                                 + 0.849147 w_k1_INTC_s101
                                 + 1.15238 w_k1_KO_s101
                                 + 0.849457 w_k1_MAT_s101
                                 + 0.959998 w_k1_MSFT_s101
                                 + 0.965301 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.13415 w_k1_ADM_s102
                                 + 1.01107 w_k1_AXP_s102
                                 + 1.03836 w_k1_DIS_s102
                                 + 1.12808 w_k1_FDX_s102
                                 + 0.860372 w_k1_IBM_s102
                                 + 1.22441 w_k1_INTC_s102
                                 + 1.06974 w_k1_KO_s102
                                 + 0.706261 w_k1_MAT_s102
                                 + 1.19303 w_k1_MSFT_s102
                                 + 0.987027 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.24459 w_k1_ADM_s102
                                 + 1.3234 w_k1_AXP_s102
                                 + 0.871475 w_k1_DIS_s102
                                 + 0.99159 w_k1_FDX_s102
                                 + 1.33204 w_k1_IBM_s102
                                 + 0.979964 w_k1_INTC_s102
                                 + 1.15458 w_k1_KO_s102 + 1.48637 w_k1_MAT_s102
                                 + 0.960804 w_k1_MSFT_s102
                                 + 1.09577 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.30528 w_k1_ADM_s102
                                 + 1.66195 w_k1_AXP_s102
                                 + 1.69574 w_k1_DIS_s102
                                 + 1.36573 w_k1_FDX_s102
                                 + 1.33523 w_k1_IBM_s102
                                 + 1.19929 w_k1_INTC_s102 + 1.4899 w_k1_KO_s102
                                 + 1.66278 w_k1_MAT_s102
                                 + 1.28485 w_k1_MSFT_s102
                                 + 1.23639 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.08943 w_k1_ADM_s102
                                 + 0.802538 w_k1_AXP_s102
                                 + 0.804479 w_k1_DIS_s102
                                 + 0.949819 w_k1_FDX_s102
                                 + 0.760806 w_k1_IBM_s102
                                 + 0.874304 w_k1_INTC_s102
                                 + 0.941171 w_k1_KO_s102
                                 + 0.938431 w_k1_MAT_s102
                                 + 0.922294 w_k1_MSFT_s102
                                 + 0.979391 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.0573 w_k1_ADM_s102
                                 + 1.03338 w_k1_AXP_s102
                                 + 1.46382 w_k1_DIS_s102
                                 + 1.15034 w_k1_FDX_s102
                                 + 1.30298 w_k1_IBM_s102
                                 + 0.841256 w_k1_INTC_s102
                                 + 1.42513 w_k1_KO_s102 + 1.48453 w_k1_MAT_s102
                                 + 1.15899 w_k1_MSFT_s102
                                 + 1.18573 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.910633 w_k1_ADM_s102
                                 + 0.984519 w_k1_AXP_s102
                                 + 1.0594 w_k1_DIS_s102
                                 + 0.997517 w_k1_FDX_s102
                                 + 1.44387 w_k1_IBM_s102
                                 + 0.982285 w_k1_INTC_s102
                                 + 1.35879 w_k1_KO_s102 + 1.44226 w_k1_MAT_s102
                                 + 1.15236 w_k1_MSFT_s102
                                 + 1.12212 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.31473 w_k1_ADM_s102
                                 + 1.07801 w_k1_AXP_s102
                                 + 1.16886 w_k1_DIS_s102
                                 + 1.19995 w_k1_FDX_s102
                                 + 1.00966 w_k1_IBM_s102
                                 + 1.14856 w_k1_INTC_s102
                                 + 1.24372 w_k1_KO_s102 + 1.22038 w_k1_MAT_s102
                                 + 1.26945 w_k1_MSFT_s102
                                 + 1.11384 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.789839 w_k1_ADM_s102
                                 + 0.782378 w_k1_AXP_s102
                                 + 0.697702 w_k1_DIS_s102
                                 + 0.804875 w_k1_FDX_s102
                                 + 1.04026 w_k1_IBM_s102
                                 + 0.860396 w_k1_INTC_s102
                                 + 0.90375 w_k1_KO_s102
                                 + 0.927667 w_k1_MAT_s102
                                 + 0.857069 w_k1_MSFT_s102
                                 + 0.924864 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.791835 w_k1_ADM_s103
                                 + 0.752192 w_k1_AXP_s103
                                 + 0.750146 w_k1_DIS_s103
                                 + 0.857394 w_k1_FDX_s103
                                 + 0.743146 w_k1_IBM_s103
                                 + 0.887093 w_k1_INTC_s103
                                 + 0.868269 w_k1_KO_s103
                                 + 0.698479 w_k1_MAT_s103
                                 + 0.841235 w_k1_MSFT_s103
                                 + 0.823659 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.813285 w_k1_ADM_s103
                                 + 1.08438 w_k1_AXP_s103
                                 + 1.09938 w_k1_DIS_s103
                                 + 1.06095 w_k1_FDX_s103
                                 + 0.912891 w_k1_IBM_s103
                                 + 1.21396 w_k1_INTC_s103
                                 + 1.05478 w_k1_KO_s103
                                 + 0.872494 w_k1_MAT_s103
                                 + 1.12669 w_k1_MSFT_s103
                                 + 0.92398 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.901716 w_k1_ADM_s103
                                 + 0.926174 w_k1_AXP_s103
                                 + 1.69938 w_k1_DIS_s103
                                 + 1.11714 w_k1_FDX_s103 + 1.0422 w_k1_IBM_s103
                                 + 1.09744 w_k1_INTC_s103
                                 + 1.20153 w_k1_KO_s103 + 1.05852 w_k1_MAT_s103
                                 + 1.33757 w_k1_MSFT_s103
                                 + 1.05755 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.03035 w_k1_ADM_s103
                                 + 1.25119 w_k1_AXP_s103
                                 + 1.08608 w_k1_DIS_s103
                                 + 1.04314 w_k1_FDX_s103
                                 + 1.22721 w_k1_IBM_s103
                                 + 1.26444 w_k1_INTC_s103
                                 + 1.07852 w_k1_KO_s103 + 1.47867 w_k1_MAT_s103
                                 + 1.04919 w_k1_MSFT_s103
                                 + 0.982903 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.788004 w_k1_ADM_s103
                                 + 0.782004 w_k1_AXP_s103
                                 + 0.856161 w_k1_DIS_s103
                                 + 0.771859 w_k1_FDX_s103
                                 + 1.22261 w_k1_IBM_s103
                                 + 0.959216 w_k1_INTC_s103
                                 + 0.985567 w_k1_KO_s103
                                 + 1.01689 w_k1_MAT_s103
                                 + 0.800048 w_k1_MSFT_s103
                                 + 0.847336 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.88799 w_k1_ADM_s103
                                 + 1.01327 w_k1_AXP_s103
                                 + 1.09426 w_k1_DIS_s103
                                 + 0.944929 w_k1_FDX_s103
                                 + 1.72152 w_k1_IBM_s103 + 1.124 w_k1_INTC_s103
                                 + 1.26393 w_k1_KO_s103 + 1.86784 w_k1_MAT_s103
                                 + 1.01109 w_k1_MSFT_s103
                                 + 1.02291 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.983905 w_k1_ADM_s103
                                 + 1.1438 w_k1_AXP_s103 + 1.34647 w_k1_DIS_s103
                                 + 1.32582 w_k1_FDX_s103
                                 + 0.72521 w_k1_IBM_s103
                                 + 1.25822 w_k1_INTC_s103
                                 + 0.972223 w_k1_KO_s103
                                 + 0.892212 w_k1_MAT_s103
                                 + 1.07411 w_k1_MSFT_s103
                                 + 0.976198 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.11584 w_k1_ADM_s103
                                 + 1.54308 w_k1_AXP_s103 + 1.6955 w_k1_DIS_s103
                                 + 1.44303 w_k1_FDX_s103
                                 + 1.16427 w_k1_IBM_s103
                                 + 1.20795 w_k1_INTC_s103
                                 + 1.14279 w_k1_KO_s103 + 1.4823 w_k1_MAT_s103
                                 + 1.29241 w_k1_MSFT_s103
                                 + 1.19066 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.24679 w_k1_ADM_s104
                                 + 1.30879 w_k1_AXP_s104
                                 + 1.29351 w_k1_DIS_s104
                                 + 1.20022 w_k1_FDX_s104
                                 + 0.970279 w_k1_IBM_s104
                                 + 0.996644 w_k1_INTC_s104
                                 + 1.09501 w_k1_KO_s104 + 0.94606 w_k1_MAT_s104
                                 + 1.07799 w_k1_MSFT_s104
                                 + 1.03747 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.829195 w_k1_ADM_s104
                                 + 1.11571 w_k1_AXP_s104
                                 + 1.46852 w_k1_DIS_s104
                                 + 1.17594 w_k1_FDX_s104
                                 + 1.25868 w_k1_IBM_s104
                                 + 1.34215 w_k1_INTC_s104
                                 + 1.24382 w_k1_KO_s104 + 1.15077 w_k1_MAT_s104
                                 + 1.34496 w_k1_MSFT_s104
                                 + 1.08624 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.07446 w_k1_ADM_s104
                                 + 0.998595 w_k1_AXP_s104
                                 + 1.12316 w_k1_DIS_s104
                                 + 1.01463 w_k1_FDX_s104
                                 + 0.869824 w_k1_IBM_s104
                                 + 1.0187 w_k1_INTC_s104
                                 + 0.955284 w_k1_KO_s104
                                 + 0.790503 w_k1_MAT_s104
                                 + 1.08902 w_k1_MSFT_s104
                                 + 0.963503 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.845698 w_k1_ADM_s104
                                 + 0.705368 w_k1_AXP_s104
                                 + 0.695569 w_k1_DIS_s104
                                 + 0.817541 w_k1_FDX_s104
                                 + 0.87977 w_k1_IBM_s104
                                 + 0.735337 w_k1_INTC_s104
                                 + 0.836323 w_k1_KO_s104
                                 + 0.767676 w_k1_MAT_s104
                                 + 0.7494 w_k1_MSFT_s104
                                 + 0.862214 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.975291 w_k1_ADM_s104
                                 + 0.872051 w_k1_AXP_s104
                                 + 0.767829 w_k1_DIS_s104
                                 + 0.813029 w_k1_FDX_s104
                                 + 1.26865 w_k1_IBM_s104
                                 + 0.705682 w_k1_INTC_s104
                                 + 0.966292 w_k1_KO_s104
                                 + 1.13793 w_k1_MAT_s104
                                 + 0.749761 w_k1_MSFT_s104
                                 + 0.922156 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.881814 w_k1_ADM_s104
                                 + 1.43419 w_k1_AXP_s104
                                 + 1.34094 w_k1_DIS_s104
                                 + 1.00652 w_k1_FDX_s104
                                 + 1.75999 w_k1_IBM_s104
                                 + 1.06809 w_k1_INTC_s104
                                 + 1.46294 w_k1_KO_s104 + 2.09639 w_k1_MAT_s104
                                 + 1.14912 w_k1_MSFT_s104
                                 + 1.17275 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.3472 w_k1_ADM_s104
                                 + 1.58375 w_k1_AXP_s104
                                 + 1.86446 w_k1_DIS_s104
                                 + 1.51051 w_k1_FDX_s104
                                 + 0.884975 w_k1_IBM_s104
                                 + 1.25855 w_k1_INTC_s104
                                 + 1.22418 w_k1_KO_s104 + 1.07061 w_k1_MAT_s104
                                 + 1.27807 w_k1_MSFT_s104
                                 + 1.11514 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.946613 w_k1_ADM_s104
                                 + 1.07547 w_k1_AXP_s104
                                 + 0.960121 w_k1_DIS_s104
                                 + 1.05485 w_k1_FDX_s104
                                 + 0.628225 w_k1_IBM_s104
                                 + 0.842551 w_k1_INTC_s104
                                 + 0.820058 w_k1_KO_s104
                                 + 0.572677 w_k1_MAT_s104
                                 + 0.96707 w_k1_MSFT_s104
                                 + 0.883629 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.08417 w_k1_ADM_s105
                                 + 1.2183 w_k1_AXP_s105
                                 + 0.991343 w_k1_DIS_s105
                                 + 1.17942 w_k1_FDX_s105
                                 + 1.00532 w_k1_IBM_s105
                                 + 0.844561 w_k1_INTC_s105
                                 + 0.906132 w_k1_KO_s105
                                 + 0.981516 w_k1_MAT_s105
                                 + 0.995689 w_k1_MSFT_s105
                                 + 1.04875 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.946378 w_k1_ADM_s105
                                 + 0.834311 w_k1_AXP_s105
                                 + 0.947717 w_k1_DIS_s105
                                 + 0.935952 w_k1_FDX_s105
                                 + 0.980791 w_k1_IBM_s105
                                 + 0.887347 w_k1_INTC_s105
                                 + 1.09582 w_k1_KO_s105
                                 + 0.947274 w_k1_MAT_s105
                                 + 1.05418 w_k1_MSFT_s105
                                 + 1.01296 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.51587 w_k1_ADM_s105
                                 + 1.24382 w_k1_AXP_s105
                                 + 1.16608 w_k1_DIS_s105
                                 + 1.38552 w_k1_FDX_s105
                                 + 0.713915 w_k1_IBM_s105
                                 + 1.13616 w_k1_INTC_s105
                                 + 0.920247 w_k1_KO_s105
                                 + 0.630799 w_k1_MAT_s105
                                 + 1.13978 w_k1_MSFT_s105
                                 + 0.999227 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.561391 w_k1_ADM_s105
                                 + 0.623685 w_k1_AXP_s105
                                 + 0.669534 w_k1_DIS_s105
                                 + 0.673617 w_k1_FDX_s105
                                 + 1.11661 w_k1_IBM_s105
                                 + 0.729263 w_k1_INTC_s105
                                 + 0.741942 w_k1_KO_s105
                                 + 0.755564 w_k1_MAT_s105
                                 + 0.739655 w_k1_MSFT_s105
                                 + 0.770346 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.90083 w_k1_ADM_s105
                                 + 0.517275 w_k1_AXP_s105
                                 + 0.792032 w_k1_DIS_s105
                                 + 0.8147 w_k1_FDX_s105
                                 + 0.732689 w_k1_IBM_s105
                                 + 0.863667 w_k1_INTC_s105
                                 + 0.782994 w_k1_KO_s105
                                 + 0.281926 w_k1_MAT_s105
                                 + 0.898551 w_k1_MSFT_s105
                                 + 0.915813 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.02512 w_k1_ADM_s105
                                 + 0.750883 w_k1_AXP_s105
                                 + 0.698988 w_k1_DIS_s105
                                 + 0.891203 w_k1_FDX_s105
                                 + 0.665665 w_k1_IBM_s105
                                 + 0.74589 w_k1_INTC_s105
                                 + 0.742644 w_k1_KO_s105
                                 + 0.601764 w_k1_MAT_s105
                                 + 0.805036 w_k1_MSFT_s105
                                 + 0.991685 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.774568 w_k1_ADM_s105
                                 + 0.815657 w_k1_AXP_s105
                                 + 1.03012 w_k1_DIS_s105
                                 + 0.892351 w_k1_FDX_s105
                                 + 1.35461 w_k1_IBM_s105
                                 + 0.877285 w_k1_INTC_s105
                                 + 1.15727 w_k1_KO_s105 + 0.83823 w_k1_MAT_s105
                                 + 1.04316 w_k1_MSFT_s105
                                 + 0.926496 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.45511 w_k1_ADM_s105
                                 + 1.09542 w_k1_AXP_s105
                                 + 1.46616 w_k1_DIS_s105
                                 + 1.29653 w_k1_FDX_s105
                                 + 1.13033 w_k1_IBM_s105
                                 + 0.980391 w_k1_INTC_s105
                                 + 1.09823 w_k1_KO_s105
                                 + 0.919513 w_k1_MAT_s105
                                 + 1.2497 w_k1_MSFT_s105 + 1.13369 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.08847 w_k1_ADM_s106
                                 + 0.836307 w_k1_AXP_s106
                                 + 1.06848 w_k1_DIS_s106
                                 + 0.958918 w_k1_FDX_s106
                                 + 1.18491 w_k1_IBM_s106
                                 + 1.11572 w_k1_INTC_s106
                                 + 1.04665 w_k1_KO_s106 + 1.04893 w_k1_MAT_s106
                                 + 1.03577 w_k1_MSFT_s106
                                 + 0.869515 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.4996 w_k1_ADM_s106
                                 + 1.4331 w_k1_AXP_s106 + 1.51115 w_k1_DIS_s106
                                 + 1.05764 w_k1_FDX_s106
                                 + 1.93062 w_k1_IBM_s106
                                 + 0.964956 w_k1_INTC_s106
                                 + 1.32053 w_k1_KO_s106 + 1.81771 w_k1_MAT_s106
                                 + 1.17577 w_k1_MSFT_s106
                                 + 1.03025 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.01751 w_k1_ADM_s106
                                 + 0.941629 w_k1_AXP_s106
                                 + 1.03049 w_k1_DIS_s106
                                 + 1.06039 w_k1_FDX_s106
                                 + 0.737829 w_k1_IBM_s106
                                 + 1.13767 w_k1_INTC_s106
                                 + 0.888779 w_k1_KO_s106
                                 + 0.858522 w_k1_MAT_s106
                                 + 1.16701 w_k1_MSFT_s106
                                 + 0.965921 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.57569 w_k1_ADM_s106
                                 + 1.27406 w_k1_AXP_s106
                                 + 1.73468 w_k1_DIS_s106
                                 + 1.34567 w_k1_FDX_s106
                                 + 1.33493 w_k1_IBM_s106
                                 + 1.29566 w_k1_INTC_s106
                                 + 1.33486 w_k1_KO_s106 + 1.4568 w_k1_MAT_s106
                                 + 1.30159 w_k1_MSFT_s106
                                 + 1.08817 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.32471 w_k1_ADM_s106
                                 + 1.20675 w_k1_AXP_s106
                                 + 1.44827 w_k1_DIS_s106
                                 + 1.24838 w_k1_FDX_s106
                                 + 1.14038 w_k1_IBM_s106
                                 + 1.24078 w_k1_INTC_s106
                                 + 1.14901 w_k1_KO_s106
                                 + 0.831414 w_k1_MAT_s106
                                 + 1.26166 w_k1_MSFT_s106
                                 + 1.16886 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.919017 w_k1_ADM_s106
                                 + 1.05516 w_k1_AXP_s106
                                 + 0.780644 w_k1_DIS_s106
                                 + 0.874288 w_k1_FDX_s106
                                 + 0.944251 w_k1_IBM_s106
                                 + 1.02677 w_k1_INTC_s106
                                 + 0.888304 w_k1_KO_s106
                                 + 1.28639 w_k1_MAT_s106
                                 + 0.895251 w_k1_MSFT_s106
                                 + 0.916792 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.29218 w_k1_ADM_s106
                                 + 1.31217 w_k1_AXP_s106
                                 + 1.30702 w_k1_DIS_s106
                                 + 1.36717 w_k1_FDX_s106
                                 + 0.720911 w_k1_IBM_s106
                                 + 1.04391 w_k1_INTC_s106
                                 + 0.943646 w_k1_KO_s106
                                 + 0.928007 w_k1_MAT_s106
                                 + 1.38537 w_k1_MSFT_s106
                                 + 1.08137 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.931508 w_k1_ADM_s106
                                 + 0.624387 w_k1_AXP_s106
                                 + 0.741255 w_k1_DIS_s106
                                 + 0.748268 w_k1_FDX_s106
                                 + 0.711606 w_k1_IBM_s106
                                 + 0.871163 w_k1_INTC_s106
                                 + 0.815672 w_k1_KO_s106
                                 + 0.57747 w_k1_MAT_s106
                                 + 0.75992 w_k1_MSFT_s106
                                 + 0.786433 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.19327 w_k1_ADM_s107
                                 + 1.39366 w_k1_AXP_s107
                                 + 1.09022 w_k1_DIS_s107
                                 + 1.26853 w_k1_FDX_s107
                                 + 0.822324 w_k1_IBM_s107
                                 + 0.997985 w_k1_INTC_s107
                                 + 0.886204 w_k1_KO_s107
                                 + 0.66886 w_k1_MAT_s107
                                 + 1.04568 w_k1_MSFT_s107
                                 + 1.03708 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.14036 w_k1_ADM_s107
                                 + 1.43509 w_k1_AXP_s107 + 1.6305 w_k1_DIS_s107
                                 + 1.21509 w_k1_FDX_s107
                                 + 1.40502 w_k1_IBM_s107
                                 + 0.951399 w_k1_INTC_s107
                                 + 1.3015 w_k1_KO_s107 + 1.73849 w_k1_MAT_s107
                                 + 1.22388 w_k1_MSFT_s107
                                 + 1.11641 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.0814 w_k1_ADM_s107
                                 + 1.27081 w_k1_AXP_s107
                                 + 1.26462 w_k1_DIS_s107
                                 + 1.10801 w_k1_FDX_s107
                                 + 0.927426 w_k1_IBM_s107
                                 + 0.995719 w_k1_INTC_s107
                                 + 1.05142 w_k1_KO_s107 + 1.17576 w_k1_MAT_s107
                                 + 1.08744 w_k1_MSFT_s107
                                 + 1.02532 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.10501 w_k1_ADM_s107
                                 + 1.02384 w_k1_AXP_s107
                                 + 1.21145 w_k1_DIS_s107
                                 + 0.995647 w_k1_FDX_s107
                                 + 1.30178 w_k1_IBM_s107
                                 + 0.889875 w_k1_INTC_s107
                                 + 1.23862 w_k1_KO_s107 + 1.47032 w_k1_MAT_s107
                                 + 1.03113 w_k1_MSFT_s107
                                 + 1.08318 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.801777 w_k1_ADM_s107
                                 + 0.76118 w_k1_AXP_s107
                                 + 0.832065 w_k1_DIS_s107
                                 + 0.711048 w_k1_FDX_s107
                                 + 1.18049 w_k1_IBM_s107
                                 + 0.713625 w_k1_INTC_s107
                                 + 0.959489 w_k1_KO_s107
                                 + 1.42063 w_k1_MAT_s107
                                 + 0.788665 w_k1_MSFT_s107
                                 + 0.946165 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.27965 w_k1_ADM_s107
                                 + 1.48598 w_k1_AXP_s107
                                 + 1.77247 w_k1_DIS_s107
                                 + 1.47771 w_k1_FDX_s107
                                 + 1.21955 w_k1_IBM_s107
                                 + 1.09627 w_k1_INTC_s107
                                 + 1.51481 w_k1_KO_s107 + 1.70641 w_k1_MAT_s107
                                 + 1.43007 w_k1_MSFT_s107
                                 + 1.18283 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.897917 w_k1_ADM_s107
                                 + 0.953371 w_k1_AXP_s107
                                 + 0.830686 w_k1_DIS_s107
                                 + 0.973211 w_k1_FDX_s107
                                 + 0.791844 w_k1_IBM_s107
                                 + 1.07609 w_k1_INTC_s107
                                 + 0.862712 w_k1_KO_s107
                                 + 0.891052 w_k1_MAT_s107
                                 + 1.10642 w_k1_MSFT_s107
                                 + 0.929752 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.842026 w_k1_ADM_s107
                                 + 0.815922 w_k1_AXP_s107
                                 + 0.689083 w_k1_DIS_s107
                                 + 0.809317 w_k1_FDX_s107
                                 + 0.777497 w_k1_IBM_s107
                                 + 0.812144 w_k1_INTC_s107
                                 + 0.803692 w_k1_KO_s107
                                 + 0.813692 w_k1_MAT_s107
                                 + 0.842293 w_k1_MSFT_s107
                                 + 0.776281 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.05628 w_k1_ADM_s110
                                 + 0.88636 w_k1_AXP_s110
                                 + 1.24578 w_k1_DIS_s110
                                 + 1.06598 w_k1_FDX_s110
                                 + 1.47654 w_k1_IBM_s110
                                 + 1.06868 w_k1_INTC_s110
                                 + 1.23027 w_k1_KO_s110 + 1.26645 w_k1_MAT_s110
                                 + 1.05737 w_k1_MSFT_s110
                                 + 1.02165 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.823595 w_k1_ADM_s110
                                 + 0.763124 w_k1_AXP_s110
                                 + 1.0377 w_k1_DIS_s110
                                 + 0.882249 w_k1_FDX_s110
                                 + 1.23526 w_k1_IBM_s110
                                 + 0.87646 w_k1_INTC_s110
                                 + 0.91157 w_k1_KO_s110
                                 + 0.807493 w_k1_MAT_s110
                                 + 0.843457 w_k1_MSFT_s110
                                 + 0.971235 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.1163 w_k1_ADM_s110
                                 + 1.43848 w_k1_AXP_s110 + 1.8428 w_k1_DIS_s110
                                 + 1.23381 w_k1_FDX_s110
                                 + 1.68304 w_k1_IBM_s110
                                 + 1.18337 w_k1_INTC_s110
                                 + 1.27716 w_k1_KO_s110 + 1.34014 w_k1_MAT_s110
                                 + 1.1753 w_k1_MSFT_s110 + 1.09158 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.11156 w_k1_ADM_s110
                                 + 1.19214 w_k1_AXP_s110
                                 + 1.06521 w_k1_DIS_s110
                                 + 1.15836 w_k1_FDX_s110 + 1.1595 w_k1_IBM_s110
                                 + 0.977764 w_k1_INTC_s110
                                 + 0.974847 w_k1_KO_s110
                                 + 1.32115 w_k1_MAT_s110
                                 + 1.02314 w_k1_MSFT_s110
                                 + 1.00244 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.25466 w_k1_ADM_s110
                                 + 1.11142 w_k1_AXP_s110
                                 + 1.40908 w_k1_DIS_s110
                                 + 1.31059 w_k1_FDX_s110
                                 + 0.925201 w_k1_IBM_s110
                                 + 1.27946 w_k1_INTC_s110
                                 + 1.02256 w_k1_KO_s110
                                 + 0.631033 w_k1_MAT_s110
                                 + 1.09009 w_k1_MSFT_s110
                                 + 1.05691 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.00925 w_k1_ADM_s110
                                 + 0.653994 w_k1_AXP_s110
                                 + 0.762762 w_k1_DIS_s110
                                 + 0.734317 w_k1_FDX_s110
                                 + 1.08169 w_k1_IBM_s110
                                 + 0.652758 w_k1_INTC_s110
                                 + 0.844969 w_k1_KO_s110 + 1.0016 w_k1_MAT_s110
                                 + 0.719181 w_k1_MSFT_s110
                                 + 0.92428 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.1344 w_k1_ADM_s110
                                 + 1.11648 w_k1_AXP_s110
                                 + 0.905647 w_k1_DIS_s110
                                 + 0.984078 w_k1_FDX_s110
                                 + 1.15145 w_k1_IBM_s110
                                 + 1.17833 w_k1_INTC_s110
                                 + 0.905049 w_k1_KO_s110
                                 + 1.23031 w_k1_MAT_s110
                                 + 0.931606 w_k1_MSFT_s110
                                 + 1.00095 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.03504 w_k1_ADM_s110
                                 + 0.78503 w_k1_AXP_s110
                                 + 0.803512 w_k1_DIS_s110
                                 + 0.905113 w_k1_FDX_s110
                                 + 0.646895 w_k1_IBM_s110
                                 + 0.915361 w_k1_INTC_s110
                                 + 0.679006 w_k1_KO_s110
                                 + 0.653635 w_k1_MAT_s110
                                 + 0.956269 w_k1_MSFT_s110
                                 + 0.847531 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.31661 w_k1_ADM_s111
                                 + 1.49181 w_k1_AXP_s111
                                 + 1.02227 w_k1_DIS_s111
                                 + 1.17383 w_k1_FDX_s111
                                 + 0.823465 w_k1_IBM_s111
                                 + 0.90187 w_k1_INTC_s111
                                 + 0.942997 w_k1_KO_s111 + 1.1704 w_k1_MAT_s111
                                 + 1.0604 w_k1_MSFT_s111 + 1.04316 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.09566 w_k1_ADM_s111
                                 + 0.781599 w_k1_AXP_s111
                                 + 0.878705 w_k1_DIS_s111
                                 + 0.928322 w_k1_FDX_s111
                                 + 0.708918 w_k1_IBM_s111
                                 + 0.828245 w_k1_INTC_s111
                                 + 0.937072 w_k1_KO_s111
                                 + 0.771536 w_k1_MAT_s111
                                 + 0.958218 w_k1_MSFT_s111
                                 + 0.914748 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.31618 w_k1_ADM_s111
                                 + 1.16632 w_k1_AXP_s111
                                 + 1.28793 w_k1_DIS_s111
                                 + 1.05956 w_k1_FDX_s111
                                 + 0.904423 w_k1_IBM_s111
                                 + 1.01214 w_k1_INTC_s111
                                 + 1.16514 w_k1_KO_s111
                                 + 0.882738 w_k1_MAT_s111
                                 + 1.14901 w_k1_MSFT_s111
                                 + 1.04137 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.1324 w_k1_ADM_s111
                                 + 0.992334 w_k1_AXP_s111
                                 + 1.31531 w_k1_DIS_s111
                                 + 1.21021 w_k1_FDX_s111
                                 + 0.633967 w_k1_IBM_s111
                                 + 0.892381 w_k1_INTC_s111
                                 + 1.08913 w_k1_KO_s111
                                 + 0.788055 w_k1_MAT_s111
                                 + 1.25477 w_k1_MSFT_s111
                                 + 1.09058 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.775923 w_k1_ADM_s111
                                 + 0.62909 w_k1_AXP_s111
                                 + 0.682332 w_k1_DIS_s111
                                 + 0.655307 w_k1_FDX_s111
                                 + 0.970244 w_k1_IBM_s111
                                 + 0.694167 w_k1_INTC_s111
                                 + 1.06975 w_k1_KO_s111
                                 + 0.727282 w_k1_MAT_s111
                                 + 0.809253 w_k1_MSFT_s111
                                 + 0.752561 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.991476 w_k1_ADM_s111
                                 + 0.927913 w_k1_AXP_s111
                                 + 0.985913 w_k1_DIS_s111
                                 + 0.902253 w_k1_FDX_s111
                                 + 0.931814 w_k1_IBM_s111
                                 + 0.697261 w_k1_INTC_s111
                                 + 0.9485 w_k1_KO_s111 + 0.979592 w_k1_MAT_s111
                                 + 0.976034 w_k1_MSFT_s111
                                 + 0.869241 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.903942 w_k1_ADM_s111
                                 + 1.16561 w_k1_AXP_s111
                                 + 1.02861 w_k1_DIS_s111
                                 + 0.985511 w_k1_FDX_s111
                                 + 1.25089 w_k1_IBM_s111
                                 + 0.977055 w_k1_INTC_s111
                                 + 1.24073 w_k1_KO_s111 + 1.35132 w_k1_MAT_s111
                                 + 1.1674 w_k1_MSFT_s111 + 1.01323 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.28985 w_k1_ADM_s111
                                 + 1.33227 w_k1_AXP_s111
                                 + 1.64639 w_k1_DIS_s111
                                 + 1.32512 w_k1_FDX_s111
                                 + 1.53311 w_k1_IBM_s111
                                 + 0.861362 w_k1_INTC_s111
                                 + 1.43939 w_k1_KO_s111 + 1.74661 w_k1_MAT_s111
                                 + 1.17502 w_k1_MSFT_s111
                                 + 1.01573 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.0485 w_k1_ADM_s112
                                 + 1.06782 w_k1_AXP_s112
                                 + 1.01143 w_k1_DIS_s112
                                 + 1.03859 w_k1_FDX_s112
                                 + 0.983256 w_k1_IBM_s112
                                 + 0.792494 w_k1_INTC_s112
                                 + 1.04611 w_k1_KO_s112 + 1.2033 w_k1_MAT_s112
                                 + 0.863015 w_k1_MSFT_s112
                                 + 1.01907 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.27546 w_k1_ADM_s112
                                 + 1.24432 w_k1_AXP_s112
                                 + 1.72082 w_k1_DIS_s112
                                 + 1.34323 w_k1_FDX_s112
                                 + 1.00264 w_k1_IBM_s112
                                 + 0.981419 w_k1_INTC_s112
                                 + 1.30852 w_k1_KO_s112
                                 + 0.988536 w_k1_MAT_s112
                                 + 1.13755 w_k1_MSFT_s112
                                 + 1.11433 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.749034 w_k1_ADM_s112
                                 + 0.920786 w_k1_AXP_s112
                                 + 0.880525 w_k1_DIS_s112
                                 + 0.803148 w_k1_FDX_s112
                                 + 1.29724 w_k1_IBM_s112
                                 + 0.823059 w_k1_INTC_s112
                                 + 0.968413 w_k1_KO_s112
                                 + 1.18059 w_k1_MAT_s112
                                 + 0.782899 w_k1_MSFT_s112
                                 + 0.968284 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.941752 w_k1_ADM_s112
                                 + 0.871087 w_k1_AXP_s112
                                 + 0.84572 w_k1_DIS_s112
                                 + 0.947472 w_k1_FDX_s112
                                 + 0.808179 w_k1_IBM_s112
                                 + 0.809675 w_k1_INTC_s112
                                 + 0.983238 w_k1_KO_s112
                                 + 0.698391 w_k1_MAT_s112
                                 + 0.883655 w_k1_MSFT_s112
                                 + 1.01466 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.826126 w_k1_ADM_s112
                                 + 0.826396 w_k1_AXP_s112
                                 + 1.12495 w_k1_DIS_s112
                                 + 1.09045 w_k1_FDX_s112
                                 + 0.827089 w_k1_IBM_s112
                                 + 1.04179 w_k1_INTC_s112
                                 + 1.08255 w_k1_KO_s112
                                 + 0.799875 w_k1_MAT_s112
                                 + 1.15014 w_k1_MSFT_s112
                                 + 1.09467 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.939718 w_k1_ADM_s112
                                 + 0.998666 w_k1_AXP_s112
                                 + 0.8415 w_k1_DIS_s112
                                 + 0.894452 w_k1_FDX_s112
                                 + 0.974498 w_k1_IBM_s112
                                 + 0.930153 w_k1_INTC_s112
                                 + 0.976204 w_k1_KO_s112
                                 + 0.790319 w_k1_MAT_s112
                                 + 0.792536 w_k1_MSFT_s112
                                 + 0.946761 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.811643 w_k1_ADM_s112
                                 + 0.746646 w_k1_AXP_s112
                                 + 0.636974 w_k1_DIS_s112
                                 + 0.721101 w_k1_FDX_s112
                                 + 0.78661 w_k1_IBM_s112
                                 + 0.699227 w_k1_INTC_s112
                                 + 0.785833 w_k1_KO_s112
                                 + 0.677147 w_k1_MAT_s112
                                 + 0.756058 w_k1_MSFT_s112
                                 + 0.851279 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.784486 w_k1_ADM_s112
                                 + 1.13826 w_k1_AXP_s112
                                 + 1.21972 w_k1_DIS_s112
                                 + 1.09526 w_k1_FDX_s112
                                 + 1.48963 w_k1_IBM_s112
                                 + 0.993688 w_k1_INTC_s112
                                 + 1.27197 w_k1_KO_s112 + 1.26616 w_k1_MAT_s112
                                 + 1.02863 w_k1_MSFT_s112
                                 + 1.13129 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.774291 w_k1_ADM_s113
                                 + 0.87961 w_k1_AXP_s113
                                 + 0.994936 w_k1_DIS_s113
                                 + 0.930651 w_k1_FDX_s113
                                 + 1.18677 w_k1_IBM_s113
                                 + 1.12869 w_k1_INTC_s113
                                 + 1.06891 w_k1_KO_s113 + 1.15438 w_k1_MAT_s113
                                 + 1.0483 w_k1_MSFT_s113 + 1.01798 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.03719 w_k1_ADM_s113
                                 + 1.06394 w_k1_AXP_s113
                                 + 0.887871 w_k1_DIS_s113
                                 + 0.892977 w_k1_FDX_s113
                                 + 0.94461 w_k1_IBM_s113
                                 + 0.912785 w_k1_INTC_s113
                                 + 0.883694 w_k1_KO_s113
                                 + 0.968652 w_k1_MAT_s113
                                 + 0.814656 w_k1_MSFT_s113
                                 + 1.0369 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.09437 w_k1_ADM_s113
                                 + 1.04577 w_k1_AXP_s113
                                 + 1.33454 w_k1_DIS_s113
                                 + 1.00404 w_k1_FDX_s113
                                 + 1.58513 w_k1_IBM_s113
                                 + 1.36995 w_k1_INTC_s113
                                 + 1.30551 w_k1_KO_s113 + 1.25388 w_k1_MAT_s113
                                 + 1.1017 w_k1_MSFT_s113 + 1.07772 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.38009 w_k1_ADM_s113
                                 + 1.36825 w_k1_AXP_s113
                                 + 1.34718 w_k1_DIS_s113
                                 + 1.33153 w_k1_FDX_s113
                                 + 0.624184 w_k1_IBM_s113
                                 + 1.33343 w_k1_INTC_s113
                                 + 1.07288 w_k1_KO_s113
                                 + 0.691946 w_k1_MAT_s113
                                 + 1.29119 w_k1_MSFT_s113
                                 + 1.12388 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.38695 w_k1_ADM_s113
                                 + 1.43513 w_k1_AXP_s113
                                 + 1.53211 w_k1_DIS_s113
                                 + 1.28947 w_k1_FDX_s113
                                 + 1.10162 w_k1_IBM_s113
                                 + 1.2383 w_k1_INTC_s113 + 1.38034 w_k1_KO_s113
                                 + 1.19618 w_k1_MAT_s113
                                 + 1.2427 w_k1_MSFT_s113 + 1.211 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.815588 w_k1_ADM_s113
                                 + 0.719619 w_k1_AXP_s113
                                 + 0.927501 w_k1_DIS_s113
                                 + 0.809263 w_k1_FDX_s113
                                 + 0.854612 w_k1_IBM_s113
                                 + 1.05605 w_k1_INTC_s113
                                 + 0.970481 w_k1_KO_s113
                                 + 0.680519 w_k1_MAT_s113
                                 + 0.980925 w_k1_MSFT_s113
                                 + 0.974674 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.29626 w_k1_ADM_s113
                                 + 1.19248 w_k1_AXP_s113
                                 + 1.42448 w_k1_DIS_s113
                                 + 1.32374 w_k1_FDX_s113
                                 + 0.986842 w_k1_IBM_s113
                                 + 1.35606 w_k1_INTC_s113
                                 + 1.19655 w_k1_KO_s113 + 1.0413 w_k1_MAT_s113
                                 + 1.36883 w_k1_MSFT_s113
                                 + 1.17263 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.837728 w_k1_ADM_s113
                                 + 0.753018 w_k1_AXP_s113
                                 + 0.733495 w_k1_DIS_s113
                                 + 0.874219 w_k1_FDX_s113
                                 + 0.552036 w_k1_IBM_s113
                                 + 0.915306 w_k1_INTC_s113
                                 + 0.637225 w_k1_KO_s113
                                 + 0.704875 w_k1_MAT_s113
                                 + 1.01669 w_k1_MSFT_s113
                                 + 0.817163 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.43211 w_k1_ADM_s114
                                 + 1.54854 w_k1_AXP_s114
                                 + 1.61718 w_k1_DIS_s114
                                 + 1.39287 w_k1_FDX_s114
                                 + 1.34465 w_k1_IBM_s114
                                 + 1.15613 w_k1_INTC_s114
                                 + 1.24689 w_k1_KO_s114 + 1.13909 w_k1_MAT_s114
                                 + 1.13866 w_k1_MSFT_s114
                                 + 1.25413 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.962327 w_k1_ADM_s114
                                 + 0.683931 w_k1_AXP_s114
                                 + 0.560675 w_k1_DIS_s114
                                 + 0.761508 w_k1_FDX_s114
                                 + 0.610755 w_k1_IBM_s114
                                 + 0.629711 w_k1_INTC_s114
                                 + 0.605635 w_k1_KO_s114
                                 + 0.581251 w_k1_MAT_s114
                                 + 0.701519 w_k1_MSFT_s114
                                 + 0.817633 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.958548 w_k1_ADM_s114
                                 + 0.870017 w_k1_AXP_s114
                                 + 0.994661 w_k1_DIS_s114
                                 + 0.911948 w_k1_FDX_s114
                                 + 1.04278 w_k1_IBM_s114
                                 + 0.818812 w_k1_INTC_s114
                                 + 1.00953 w_k1_KO_s114 + 1.00164 w_k1_MAT_s114
                                 + 0.9563 w_k1_MSFT_s114 + 0.96435 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.962308 w_k1_ADM_s114
                                 + 1.11925 w_k1_AXP_s114
                                 + 1.19275 w_k1_DIS_s114
                                 + 1.01925 w_k1_FDX_s114
                                 + 1.33372 w_k1_IBM_s114
                                 + 0.970152 w_k1_INTC_s114
                                 + 0.986726 w_k1_KO_s114 + 1.1827 w_k1_MAT_s114
                                 + 0.948263 w_k1_MSFT_s114
                                 + 1.06611 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.906209 w_k1_ADM_s114
                                 + 1.46194 w_k1_AXP_s114
                                 + 1.15602 w_k1_DIS_s114
                                 + 1.07203 w_k1_FDX_s114
                                 + 1.33564 w_k1_IBM_s114
                                 + 0.952968 w_k1_INTC_s114
                                 + 1.11069 w_k1_KO_s114 + 1.48977 w_k1_MAT_s114
                                 + 0.9939 w_k1_MSFT_s114 + 1.14672 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.04908 w_k1_ADM_s114
                                 + 1.08834 w_k1_AXP_s114
                                 + 0.938476 w_k1_DIS_s114
                                 + 0.988985 w_k1_FDX_s114
                                 + 0.805712 w_k1_IBM_s114
                                 + 0.896851 w_k1_INTC_s114
                                 + 0.890061 w_k1_KO_s114
                                 + 0.802901 w_k1_MAT_s114
                                 + 0.962125 w_k1_MSFT_s114
                                 + 0.97372 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.626802 w_k1_ADM_s114
                                 + 0.815385 w_k1_AXP_s114
                                 + 0.733043 w_k1_DIS_s114
                                 + 0.758908 w_k1_FDX_s114
                                 + 1.18307 w_k1_IBM_s114
                                 + 0.867739 w_k1_INTC_s114
                                 + 0.795886 w_k1_KO_s114
                                 + 1.15198 w_k1_MAT_s114
                                 + 0.803388 w_k1_MSFT_s114
                                 + 0.917141 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.10937 w_k1_ADM_s114
                                 + 1.10439 w_k1_AXP_s114
                                 + 1.26881 w_k1_DIS_s114
                                 + 1.14775 w_k1_FDX_s114
                                 + 0.802315 w_k1_IBM_s114
                                 + 0.928384 w_k1_INTC_s114
                                 + 0.966013 w_k1_KO_s114
                                 + 1.11622 w_k1_MAT_s114
                                 + 1.18312 w_k1_MSFT_s114
                                 + 0.981636 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.764047 w_k1_ADM_s115
                                 + 0.836357 w_k1_AXP_s115
                                 + 1.01354 w_k1_DIS_s115
                                 + 0.900416 w_k1_FDX_s115
                                 + 1.14936 w_k1_IBM_s115
                                 + 1.05615 w_k1_INTC_s115
                                 + 1.10435 w_k1_KO_s115 + 1.42272 w_k1_MAT_s115
                                 + 1.06143 w_k1_MSFT_s115
                                 + 0.937529 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.0643 w_k1_ADM_s115
                                 + 0.987974 w_k1_AXP_s115
                                 + 0.919513 w_k1_DIS_s115
                                 + 0.976223 w_k1_FDX_s115
                                 + 1.05136 w_k1_IBM_s115
                                 + 0.894955 w_k1_INTC_s115
                                 + 1.05216 w_k1_KO_s115 + 1.24786 w_k1_MAT_s115
                                 + 0.963668 w_k1_MSFT_s115
                                 + 0.986518 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.07923 w_k1_ADM_s115
                                 + 1.07057 w_k1_AXP_s115
                                 + 1.17298 w_k1_DIS_s115
                                 + 1.05814 w_k1_FDX_s115
                                 + 1.13729 w_k1_IBM_s115
                                 + 0.907138 w_k1_INTC_s115 + 1.127 w_k1_KO_s115
                                 + 1.88247 w_k1_MAT_s115
                                 + 1.04922 w_k1_MSFT_s115
                                 + 0.984497 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.948919 w_k1_ADM_s115
                                 + 0.694902 w_k1_AXP_s115
                                 + 0.601581 w_k1_DIS_s115
                                 + 0.747644 w_k1_FDX_s115
                                 + 1.00377 w_k1_IBM_s115
                                 + 0.883727 w_k1_INTC_s115
                                 + 0.890057 w_k1_KO_s115
                                 + 1.14609 w_k1_MAT_s115
                                 + 0.75601 w_k1_MSFT_s115
                                 + 0.834365 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.879267 w_k1_ADM_s115
                                 + 0.766648 w_k1_AXP_s115
                                 + 0.79829 w_k1_DIS_s115
                                 + 0.88218 w_k1_FDX_s115
                                 + 0.707314 w_k1_IBM_s115
                                 + 0.845067 w_k1_INTC_s115
                                 + 0.829367 w_k1_KO_s115
                                 + 0.994515 w_k1_MAT_s115
                                 + 0.872333 w_k1_MSFT_s115
                                 + 0.882734 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.08796 w_k1_ADM_s115
                                 + 1.02321 w_k1_AXP_s115
                                 + 0.915669 w_k1_DIS_s115
                                 + 1.06844 w_k1_FDX_s115
                                 + 0.688032 w_k1_IBM_s115
                                 + 0.9812 w_k1_INTC_s115
                                 + 0.914071 w_k1_KO_s115
                                 + 0.843113 w_k1_MAT_s115
                                 + 1.01261 w_k1_MSFT_s115
                                 + 0.986865 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.946204 w_k1_ADM_s115
                                 + 1.18035 w_k1_AXP_s115
                                 + 1.38672 w_k1_DIS_s115
                                 + 1.18165 w_k1_FDX_s115
                                 + 1.36243 w_k1_IBM_s115
                                 + 1.24375 w_k1_INTC_s115
                                 + 1.36045 w_k1_KO_s115 + 1.56585 w_k1_MAT_s115
                                 + 1.19608 w_k1_MSFT_s115
                                 + 1.05287 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.39459 w_k1_ADM_s115
                                 + 1.33839 w_k1_AXP_s115
                                 + 1.55412 w_k1_DIS_s115
                                 + 1.36718 w_k1_FDX_s115
                                 + 0.971736 w_k1_IBM_s115
                                 + 1.21352 w_k1_INTC_s115
                                 + 0.991267 w_k1_KO_s115 + 1.0169 w_k1_MAT_s115
                                 + 1.12606 w_k1_MSFT_s115
                                 + 1.11138 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.31395 w_k1_ADM_s116
                                 + 0.98038 w_k1_AXP_s116
                                 + 1.96796 w_k1_DIS_s116
                                 + 1.39697 w_k1_FDX_s116
                                 + 1.00783 w_k1_IBM_s116
                                 + 1.38078 w_k1_INTC_s116 + 1.4032 w_k1_KO_s116
                                 + 0.91235 w_k1_MAT_s116
                                 + 1.3787 w_k1_MSFT_s116 + 1.23082 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.796208 w_k1_ADM_s116
                                 + 0.935124 w_k1_AXP_s116
                                 + 0.815091 w_k1_DIS_s116
                                 + 0.827482 w_k1_FDX_s116
                                 + 1.27276 w_k1_IBM_s116
                                 + 0.893664 w_k1_INTC_s116
                                 + 0.978409 w_k1_KO_s116
                                 + 1.06069 w_k1_MAT_s116
                                 + 0.796224 w_k1_MSFT_s116
                                 + 0.956633 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.38476 w_k1_ADM_s116
                                 + 1.46793 w_k1_AXP_s116 + 1.4275 w_k1_DIS_s116
                                 + 1.28338 w_k1_FDX_s116
                                 + 0.94858 w_k1_IBM_s116
                                 + 1.03557 w_k1_INTC_s116
                                 + 1.22235 w_k1_KO_s116 + 1.43391 w_k1_MAT_s116
                                 + 1.09264 w_k1_MSFT_s116
                                 + 1.11205 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.865218 w_k1_ADM_s116
                                 + 0.7456 w_k1_AXP_s116
                                 + 0.963516 w_k1_DIS_s116
                                 + 0.868525 w_k1_FDX_s116
                                 + 1.04395 w_k1_IBM_s116
                                 + 0.918985 w_k1_INTC_s116
                                 + 1.0192 w_k1_KO_s116 + 0.770993 w_k1_MAT_s116
                                 + 0.872558 w_k1_MSFT_s116
                                 + 1.01774 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.90916 w_k1_ADM_s116
                                 + 0.736114 w_k1_AXP_s116
                                 + 0.766556 w_k1_DIS_s116
                                 + 0.897683 w_k1_FDX_s116
                                 + 0.64119 w_k1_IBM_s116
                                 + 0.958795 w_k1_INTC_s116
                                 + 0.74244 w_k1_KO_s116 + 0.54814 w_k1_MAT_s116
                                 + 0.922177 w_k1_MSFT_s116
                                 + 0.869667 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.12732 w_k1_ADM_s116
                                 + 1.21718 w_k1_AXP_s116
                                 + 1.41699 w_k1_DIS_s116 + 1.1015 w_k1_FDX_s116
                                 + 1.55843 w_k1_IBM_s116
                                 + 0.960924 w_k1_INTC_s116
                                 + 1.45928 w_k1_KO_s116 + 1.71139 w_k1_MAT_s116
                                 + 1.01469 w_k1_MSFT_s116
                                 + 1.08434 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.01713 w_k1_ADM_s116
                                 + 0.981836 w_k1_AXP_s116
                                 + 1.19611 w_k1_DIS_s116
                                 + 1.09652 w_k1_FDX_s116
                                 + 0.987997 w_k1_IBM_s116
                                 + 1.05736 w_k1_INTC_s116
                                 + 1.00657 w_k1_KO_s116 + 0.88862 w_k1_MAT_s116
                                 + 0.990364 w_k1_MSFT_s116
                                 + 0.940709 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.15193 w_k1_ADM_s116
                                 + 0.898128 w_k1_AXP_s116
                                 + 1.1784 w_k1_DIS_s116 + 1.21291 w_k1_FDX_s116
                                 + 0.676644 w_k1_IBM_s116
                                 + 1.30407 w_k1_INTC_s116
                                 + 0.962933 w_k1_KO_s116
                                 + 0.503399 w_k1_MAT_s116
                                 + 1.30865 w_k1_MSFT_s116
                                 + 0.978259 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.789318 w_k1_ADM_s117
                                 + 0.890967 w_k1_AXP_s117
                                 + 0.868951 w_k1_DIS_s117
                                 + 0.87674 w_k1_FDX_s117
                                 + 1.37282 w_k1_IBM_s117
                                 + 1.01001 w_k1_INTC_s117
                                 + 1.10453 w_k1_KO_s117 + 1.08814 w_k1_MAT_s117
                                 + 1.02397 w_k1_MSFT_s117
                                 + 1.03117 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.03171 w_k1_ADM_s117
                                 + 1.03389 w_k1_AXP_s117
                                 + 0.941284 w_k1_DIS_s117
                                 + 1.11452 w_k1_FDX_s117
                                 + 0.918332 w_k1_IBM_s117
                                 + 1.16217 w_k1_INTC_s117
                                 + 0.991779 w_k1_KO_s117
                                 + 0.781954 w_k1_MAT_s117
                                 + 1.06944 w_k1_MSFT_s117
                                 + 1.07045 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.857759 w_k1_ADM_s117
                                 + 1.12729 w_k1_AXP_s117
                                 + 0.812378 w_k1_DIS_s117
                                 + 0.963746 w_k1_FDX_s117
                                 + 1.08861 w_k1_IBM_s117
                                 + 1.12216 w_k1_INTC_s117
                                 + 0.860659 w_k1_KO_s117
                                 + 1.16682 w_k1_MAT_s117
                                 + 0.983592 w_k1_MSFT_s117
                                 + 1.04881 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.24239 w_k1_ADM_s117
                                 + 1.40449 w_k1_AXP_s117
                                 + 1.24824 w_k1_DIS_s117
                                 + 1.09037 w_k1_FDX_s117
                                 + 1.70357 w_k1_IBM_s117
                                 + 1.3845 w_k1_INTC_s117 + 1.16552 w_k1_KO_s117
                                 + 1.22762 w_k1_MAT_s117
                                 + 1.1252 w_k1_MSFT_s117 + 1.20008 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.79803 w_k1_ADM_s117
                                 + 0.775048 w_k1_AXP_s117
                                 + 0.676767 w_k1_DIS_s117
                                 + 0.801975 w_k1_FDX_s117
                                 + 0.748079 w_k1_IBM_s117
                                 + 0.764526 w_k1_INTC_s117
                                 + 0.70139 w_k1_KO_s117
                                 + 0.521801 w_k1_MAT_s117
                                 + 0.818385 w_k1_MSFT_s117
                                 + 0.883085 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.13096 w_k1_ADM_s117
                                 + 1.32602 w_k1_AXP_s117
                                 + 1.44167 w_k1_DIS_s117
                                 + 1.38892 w_k1_FDX_s117
                                 + 0.824504 w_k1_IBM_s117
                                 + 1.2339 w_k1_INTC_s117 + 1.04885 w_k1_KO_s117
                                 + 0.936031 w_k1_MAT_s117
                                 + 1.29653 w_k1_MSFT_s117
                                 + 1.20385 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.885215 w_k1_ADM_s117
                                 + 0.710022 w_k1_AXP_s117
                                 + 0.981293 w_k1_DIS_s117
                                 + 0.970215 w_k1_FDX_s117
                                 + 0.861769 w_k1_IBM_s117
                                 + 1.02718 w_k1_INTC_s117
                                 + 0.898192 w_k1_KO_s117
                                 + 0.62966 w_k1_MAT_s117
                                 + 1.08287 w_k1_MSFT_s117
                                 + 1.00955 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.0754 w_k1_ADM_s117
                                 + 1.36894 w_k1_AXP_s117
                                 + 1.19024 w_k1_DIS_s117
                                 + 1.26641 w_k1_FDX_s117
                                 + 1.12285 w_k1_IBM_s117
                                 + 1.13135 w_k1_INTC_s117
                                 + 1.09373 w_k1_KO_s117 + 1.31369 w_k1_MAT_s117
                                 + 1.13854 w_k1_MSFT_s117
                                 + 1.24762 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0113153101751043 w_k1_ADM_s000
                                 + 0.00757041949685104 w_k1_ADM_s001
                                 + 0.016317581174981 w_k1_ADM_s002
                                 + 0.0186788828855334 w_k1_ADM_s003
                                 + 0.00637681285561345 w_k1_ADM_s004
                                 + 0.0191861631254371 w_k1_ADM_s005
                                 + 0.0106952233250341 w_k1_ADM_s006
                                 + 0.0172020190710158 w_k1_ADM_s007
                                 + 0.0239947022810322 w_k1_ADM_s010
                                 + 0.0385922381831293 w_k1_ADM_s011
                                 + 0.031176007478171 w_k1_ADM_s012
                                 + 0.0239968568631242 w_k1_ADM_s013
                                 + 0.0313476092848351 w_k1_ADM_s014
                                 + 0.0423657602926529 w_k1_ADM_s015
                                 + 0.0142642961594366 w_k1_ADM_s016
                                 + 0.02894325576036 w_k1_ADM_s017
                                 + 0.029374175227749 w_k1_ADM_s100
                                 + 0.0606818213462372 w_k1_ADM_s101
                                 + 0.0438523927185496 w_k1_ADM_s102
                                 + 0.0842273494809646 w_k1_ADM_s103
                                 + 0.0205329397542409 w_k1_ADM_s104
                                 + 0.0259299717162772 w_k1_ADM_s105
                                 + 0.0255855176020664 w_k1_ADM_s106
                                 + 0.0404623397292236 w_k1_ADM_s107
                                 + 0.0256290698449503 w_k1_ADM_s110
                                 + 0.0300857690360115 w_k1_ADM_s111
                                 + 0.0242739517169319 w_k1_ADM_s112
                                 + 0.0435534177142847 w_k1_ADM_s113
                                 + 0.0417779188789485 w_k1_ADM_s114
                                 + 0.0410782537751468 w_k1_ADM_s115
                                 + 0.0800039317587474 w_k1_ADM_s116
                                 + 0.0611281423018912 w_k1_ADM_s117
                                 + 0.0118300660331585 w_k1_AXP_s000
                                 + 0.00732899292114571 w_k1_AXP_s001
                                 + 0.016703527116731 w_k1_AXP_s002
                                 + 0.0194105925311381 w_k1_AXP_s003
                                 + 0.00756522676823228 w_k1_AXP_s004
                                 + 0.0156050899288748 w_k1_AXP_s005
                                 + 0.0121208538037105 w_k1_AXP_s006
                                 + 0.0181685263590052 w_k1_AXP_s007
                                 + 0.02347535566214 w_k1_AXP_s010
                                 + 0.0376542171159206 w_k1_AXP_s011
                                 + 0.0283995494920116 w_k1_AXP_s012
                                 + 0.0251726400121906 w_k1_AXP_s013
                                 + 0.0318052755741061 w_k1_AXP_s014
                                 + 0.0426760975838972 w_k1_AXP_s015
                                 + 0.0155367784036611 w_k1_AXP_s016
                                 + 0.0329291693789913 w_k1_AXP_s017
                                 + 0.0285232815431065 w_k1_AXP_s100
                                 + 0.0448636788233381 w_k1_AXP_s101
                                 + 0.041218785748425 w_k1_AXP_s102
                                 + 0.0966963722341965 w_k1_AXP_s103
                                 + 0.0217522611735249 w_k1_AXP_s104
                                 + 0.0227836039012578 w_k1_AXP_s105
                                 + 0.0225656364855061 w_k1_AXP_s106
                                 + 0.0426969534268124 w_k1_AXP_s107
                                 + 0.0232372077940235 w_k1_AXP_s110
                                 + 0.0303862173967131 w_k1_AXP_s111
                                 + 0.0261285496976411 w_k1_AXP_s112
                                 + 0.0427476435040022 w_k1_AXP_s113
                                 + 0.0455948513845717 w_k1_AXP_s114
                                 + 0.0396447734534849 w_k1_AXP_s115
                                 + 0.0769667196707815 w_k1_AXP_s116
                                 + 0.0673413652740524 w_k1_AXP_s117
                                 + 0.0113289814694016 w_k1_DIS_s000
                                 + 0.00787435954760883 w_k1_DIS_s001
                                 + 0.0173353830002771 w_k1_DIS_s002
                                 + 0.0200571110291281 w_k1_DIS_s003
                                 + 0.00693696563041763 w_k1_DIS_s004
                                 + 0.0180647173286591 w_k1_DIS_s005
                                 + 0.0112784684886918 w_k1_DIS_s006
                                 + 0.0198415856664253 w_k1_DIS_s007
                                 + 0.0247708344788629 w_k1_DIS_s010
                                 + 0.0365065813929625 w_k1_DIS_s011
                                 + 0.0314395418801952 w_k1_DIS_s012
                                 + 0.0240770491582721 w_k1_DIS_s013
                                 + 0.0317028643102976 w_k1_DIS_s014
                                 + 0.0448090314341881 w_k1_DIS_s015
                                 + 0.0158402023092027 w_k1_DIS_s016
                                 + 0.0330651414732266 w_k1_DIS_s017
                                 + 0.0299229892523016 w_k1_DIS_s100
                                 + 0.0493547542464198 w_k1_DIS_s101
                                 + 0.0416803903721468 w_k1_DIS_s102
                                 + 0.102725830346337 w_k1_DIS_s103
                                 + 0.0223820183190633 w_k1_DIS_s104
                                 + 0.0247216718563136 w_k1_DIS_s105
                                 + 0.0247196028949763 w_k1_DIS_s106
                                 + 0.0430984631274282 w_k1_DIS_s107
                                 + 0.02581762301383 w_k1_DIS_s110
                                 + 0.0299943767813159 w_k1_DIS_s111
                                 + 0.0276446790886388 w_k1_DIS_s112
                                 + 0.0463828308557513 w_k1_DIS_s113
                                 + 0.0437234369638155 w_k1_DIS_s114
                                 + 0.0427293438159657 w_k1_DIS_s115
                                 + 0.0893547335324141 w_k1_DIS_s116
                                 + 0.0634030733757361 w_k1_DIS_s117
                                 + 0.0115872427210282 w_k1_FDX_s000
                                 + 0.00767121589703841 w_k1_FDX_s001
                                 + 0.0170083102273348 w_k1_FDX_s002
                                 + 0.0179827985028659 w_k1_FDX_s003
                                 + 0.00688235354581213 w_k1_FDX_s004
                                 + 0.018002001943176 w_k1_FDX_s005
                                 + 0.0109784453730447 w_k1_FDX_s006
                                 + 0.0192251346013627 w_k1_FDX_s007
                                 + 0.0238930837250908 w_k1_FDX_s010
                                 + 0.0366401356983061 w_k1_FDX_s011
                                 + 0.0297822994394039 w_k1_FDX_s012
                                 + 0.0237569451954076 w_k1_FDX_s013
                                 + 0.0335046782208037 w_k1_FDX_s014
                                 + 0.043416816137349 w_k1_FDX_s015
                                 + 0.0152199414238864 w_k1_FDX_s016
                                 + 0.0319628237386085 w_k1_FDX_s017
                                 + 0.0286744662704233 w_k1_FDX_s100
                                 + 0.047246832840739 w_k1_FDX_s101
                                 + 0.0422488676367084 w_k1_FDX_s102
                                 + 0.0969217077709168 w_k1_FDX_s103
                                 + 0.0209996033684502 w_k1_FDX_s104
                                 + 0.0254489281853331 w_k1_FDX_s105
                                 + 0.0229508070677197 w_k1_FDX_s106
                                 + 0.0405904093866823 w_k1_FDX_s107
                                 + 0.0244609463685181 w_k1_FDX_s110
                                 + 0.0284870649694346 w_k1_FDX_s111
                                 + 0.0262451825141531 w_k1_FDX_s112
                                 + 0.0424049237078094 w_k1_FDX_s113
                                 + 0.0418329873683287 w_k1_FDX_s114
                                 + 0.0411073263759455 w_k1_FDX_s115
                                 + 0.0809365074795024 w_k1_FDX_s116
                                 + 0.0667817794092452 w_k1_FDX_s117
                                 + 0.0113081487113819 w_k1_IBM_s000
                                 + 0.00799739976684868 w_k1_IBM_s001
                                 + 0.0183253878654635 w_k1_IBM_s002
                                 + 0.0182461099928029 w_k1_IBM_s003
                                 + 0.00685725133525953 w_k1_IBM_s004
                                 + 0.0194371691405612 w_k1_IBM_s005
                                 + 0.0118643270692544 w_k1_IBM_s006
                                 + 0.019745046133676 w_k1_IBM_s007
                                 + 0.0233170397004608 w_k1_IBM_s010
                                 + 0.0358838712135513 w_k1_IBM_s011
                                 + 0.031154857436175 w_k1_IBM_s012
                                 + 0.025494721885547 w_k1_IBM_s013
                                 + 0.0320010509051664 w_k1_IBM_s014
                                 + 0.0449199534539857 w_k1_IBM_s015
                                 + 0.0152702698200701 w_k1_IBM_s016
                                 + 0.0329175832157176 w_k1_IBM_s017
                                 + 0.0318774030101989 w_k1_IBM_s100
                                 + 0.0483037277282222 w_k1_IBM_s101
                                 + 0.0429570883430971 w_k1_IBM_s102
                                 + 0.0958417404648206 w_k1_IBM_s103
                                 + 0.0205106030836208 w_k1_IBM_s104
                                 + 0.0250041698313013 w_k1_IBM_s105
                                 + 0.0234519146323062 w_k1_IBM_s106
                                 + 0.0413273637543035 w_k1_IBM_s107
                                 + 0.0247004272723464 w_k1_IBM_s110
                                 + 0.0284590163943469 w_k1_IBM_s111
                                 + 0.0275890131664571 w_k1_IBM_s112
                                 + 0.0410813463556166 w_k1_IBM_s113
                                 + 0.0424429658660147 w_k1_IBM_s114
                                 + 0.0394205672521644 w_k1_IBM_s115
                                 + 0.0822309100361837 w_k1_IBM_s116
                                 + 0.0662560770151596 w_k1_IBM_s117
                                 + 0.0117727439309912 w_k1_INTC_s000
                                 + 0.00594082337704119 w_k1_INTC_s001
                                 + 0.0175532608369819 w_k1_INTC_s002
                                 + 0.0182165506446938 w_k1_INTC_s003
                                 + 0.00642604200558283 w_k1_INTC_s004
                                 + 0.0165015514524276 w_k1_INTC_s005
                                 + 0.0114306073913544 w_k1_INTC_s006
                                 + 0.0215381149447139 w_k1_INTC_s007
                                 + 0.0199174832107158 w_k1_INTC_s010
                                 + 0.0391218350296289 w_k1_INTC_s011
                                 + 0.0286619005393779 w_k1_INTC_s012
                                 + 0.0253490381516665 w_k1_INTC_s013
                                 + 0.0332737914978704 w_k1_INTC_s014
                                 + 0.0455188612880296 w_k1_INTC_s015
                                 + 0.0143215813818056 w_k1_INTC_s016
                                 + 0.0324220037816976 w_k1_INTC_s017
                                 + 0.0271155394731923 w_k1_INTC_s100
                                 + 0.0434713697550802 w_k1_INTC_s101
                                 + 0.0405187696326863 w_k1_INTC_s102
                                 + 0.103124729318856 w_k1_INTC_s103
                                 + 0.0192924002254763 w_k1_INTC_s104
                                 + 0.0218535721277104 w_k1_INTC_s105
                                 + 0.0241720066314744 w_k1_INTC_s106
                                 + 0.0378643290609064 w_k1_INTC_s107
                                 + 0.0245755173591431 w_k1_INTC_s110
                                 + 0.0238804984254009 w_k1_INTC_s111
                                 + 0.0233891812866614 w_k1_INTC_s112
                                 + 0.0473530573575861 w_k1_INTC_s113
                                 + 0.0374144729563765 w_k1_INTC_s114
                                 + 0.0398410187592641 w_k1_INTC_s115
                                 + 0.0788543894364243 w_k1_INTC_s116
                                 + 0.0685231652902105 w_k1_INTC_s117
                                 + 0.0118571629044078 w_k1_KO_s000
                                 + 0.00768409281994701 w_k1_KO_s001
                                 + 0.0164237590549658 w_k1_KO_s002
                                 + 0.0196370798551559 w_k1_KO_s003
                                 + 0.0063503458555408 w_k1_KO_s004
                                 + 0.0185952932822409 w_k1_KO_s005
                                 + 0.0118393932843072 w_k1_KO_s006
                                 + 0.0192725142038101 w_k1_KO_s007
                                 + 0.0236958526104119 w_k1_KO_s010
                                 + 0.0376450176642591 w_k1_KO_s011
                                 + 0.0296765479137333 w_k1_KO_s012
                                 + 0.0245080025524876 w_k1_KO_s013
                                 + 0.0308995022341896 w_k1_KO_s014
                                 + 0.0464574093262866 w_k1_KO_s015
                                 + 0.0139901352322195 w_k1_KO_s016
                                 + 0.0326876092797121 w_k1_KO_s017
                                 + 0.0301942894196178 w_k1_KO_s100
                                 + 0.0526545430709366 w_k1_KO_s101
                                 + 0.0459745018376262 w_k1_KO_s102
                                 + 0.0967430144887191 w_k1_KO_s103
                                 + 0.020743796892521 w_k1_KO_s104
                                 + 0.0241417771240848 w_k1_KO_s105
                                 + 0.0227642070367692 w_k1_KO_s106
                                 + 0.0417982627515981 w_k1_KO_s107
                                 + 0.0218094275919128 w_k1_KO_s110
                                 + 0.0308804267985541 w_k1_KO_s111
                                 + 0.0279707823345856 w_k1_KO_s112
                                 + 0.0436806432482908 w_k1_KO_s113
                                 + 0.0395272123808873 w_k1_KO_s114
                                 + 0.0403080033969512 w_k1_KO_s115
                                 + 0.0840527599074177 w_k1_KO_s116
                                 + 0.0612742636889406 w_k1_KO_s117
                                 + 0.0109662913454326 w_k1_MAT_s000
                                 + 0.0108492608591737 w_k1_MAT_s001
                                 + 0.0150979248160061 w_k1_MAT_s002
                                 + 0.0199919752736963 w_k1_MAT_s003
                                 + 0.00744179344921439 w_k1_MAT_s004
                                 + 0.017590413863746 w_k1_MAT_s005
                                 + 0.0117150896840318 w_k1_MAT_s006
                                 + 0.0202218533784763 w_k1_MAT_s007
                                 + 0.0260827312851197 w_k1_MAT_s010
                                 + 0.0344664584491416 w_k1_MAT_s011
                                 + 0.0272380857279835 w_k1_MAT_s012
                                 + 0.0253973699726292 w_k1_MAT_s013
                                 + 0.032530272103566 w_k1_MAT_s014
                                 + 0.0412239985359692 w_k1_MAT_s015
                                 + 0.0138222666119128 w_k1_MAT_s016
                                 + 0.0347072404506711 w_k1_MAT_s017
                                 + 0.0294233774628747 w_k1_MAT_s100
                                 + 0.0479906550959572 w_k1_MAT_s101
                                 + 0.0465696901762581 w_k1_MAT_s102
                                 + 0.101567197678866 w_k1_MAT_s103
                                 + 0.0197596605802283 w_k1_MAT_s104
                                 + 0.020282909262561 w_k1_MAT_s105
                                 + 0.02356495429426 w_k1_MAT_s106
                                 + 0.047867776211259 w_k1_MAT_s107
                                 + 0.0219987207792521 w_k1_MAT_s110
                                 + 0.0312174509072833 w_k1_MAT_s111
                                 + 0.0257997022983321 w_k1_MAT_s112
                                 + 0.03957510955139 w_k1_MAT_s113
                                 + 0.0426325971743972 w_k1_MAT_s114
                                 + 0.0497403796380049 w_k1_MAT_s115
                                 + 0.0789027862731714 w_k1_MAT_s116
                                 + 0.0592930522336802 w_k1_MAT_s117
                                 + 0.01186622212422 w_k1_MSFT_s000
                                 + 0.00714987634758129 w_k1_MSFT_s001
                                 + 0.0160287433993118 w_k1_MSFT_s002
                                 + 0.0179510957722961 w_k1_MSFT_s003
                                 + 0.00674110809655784 w_k1_MSFT_s004
                                 + 0.018547026451691 w_k1_MSFT_s005
                                 + 0.0117263522153567 w_k1_MSFT_s006
                                 + 0.0206874077471927 w_k1_MSFT_s007
                                 + 0.0236476511670428 w_k1_MSFT_s010
                                 + 0.0374106414015405 w_k1_MSFT_s011
                                 + 0.0287243565170548 w_k1_MSFT_s012
                                 + 0.0249522946821275 w_k1_MSFT_s013
                                 + 0.0303585956385734 w_k1_MSFT_s014
                                 + 0.0430435263939154 w_k1_MSFT_s015
                                 + 0.0154944381514039 w_k1_MSFT_s016
                                 + 0.0325432361138078 w_k1_MSFT_s017
                                 + 0.029711781580366 w_k1_MSFT_s100
                                 + 0.0449371413355207 w_k1_MSFT_s101
                                 + 0.0433037465514013 w_k1_MSFT_s102
                                 + 0.0964730193007673 w_k1_MSFT_s103
                                 + 0.0204649839552867 w_k1_MSFT_s104
                                 + 0.02517732117163 w_k1_MSFT_s105
                                 + 0.0238488086782602 w_k1_MSFT_s106
                                 + 0.0419077041334981 w_k1_MSFT_s107
                                 + 0.0230116633624807 w_k1_MSFT_s110
                                 + 0.0296912372991945 w_k1_MSFT_s111
                                 + 0.0243236461114037 w_k1_MSFT_s112
                                 + 0.0445445655507013 w_k1_MSFT_s113
                                 + 0.0399782489836796 w_k1_MSFT_s114
                                 + 0.0397852761520512 w_k1_MSFT_s115
                                 + 0.0768966646790852 w_k1_MSFT_s116
                                 + 0.0671925547836809 w_k1_MSFT_s117
                                 + 0.0112750651604408 w_k1_PG_s000
                                 + 0.00723473000028465 w_k1_PG_s001
                                 + 0.0161713699426945 w_k1_PG_s002
                                 + 0.0164875556825975 w_k1_PG_s003
                                 + 0.00669530135362372 w_k1_PG_s004
                                 + 0.0178063280384255 w_k1_PG_s005
                                 + 0.0111968226821103 w_k1_PG_s006
                                 + 0.0179917768750513 w_k1_PG_s007
                                 + 0.0226630314294316 w_k1_PG_s010
                                 + 0.0372843240671839 w_k1_PG_s011
                                 + 0.0266349959427751 w_k1_PG_s012
                                 + 0.024946806492501 w_k1_PG_s013
                                 + 0.0314293034349328 w_k1_PG_s014
                                 + 0.0432342027530826 w_k1_PG_s015
                                 + 0.0139146156101537 w_k1_PG_s016
                                 + 0.0303581675608573 w_k1_PG_s017
                                 + 0.0290697086422219 w_k1_PG_s100
                                 + 0.0481921376895753 w_k1_PG_s101
                                 + 0.0426550377693157 w_k1_PG_s102
                                 + 0.0890602161847032 w_k1_PG_s103
                                 + 0.0198394022131192 w_k1_PG_s104
                                 + 0.0250788259875965 w_k1_PG_s105
                                 + 0.0214445776997307 w_k1_PG_s106
                                 + 0.03986394745823 w_k1_PG_s107
                                 + 0.022855945441866 w_k1_PG_s110
                                 + 0.0268966759784658 w_k1_PG_s111
                                 + 0.0268320866209311 w_k1_PG_s112
                                 + 0.0432871144227718 w_k1_PG_s113
                                 + 0.0418296314275345 w_k1_PG_s114
                                 + 0.0384894177417797 w_k1_PG_s115
                                 + 0.0781328414208934 w_k1_PG_s116
                                 + 0.0686427553409794 w_k1_PG_s117 >= 1.229
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
