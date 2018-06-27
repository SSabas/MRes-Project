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
 w_k1_ADM_s0_return_constraint:  1.13866 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.976259 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.12075 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.0634 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.90377 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.13178 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.08664 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.42467 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.823694 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.867009 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.1051 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.2717 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.849768 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.01247 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.721271 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.84443 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.3399 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.952218 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.1173 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.10645 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.886873 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.756598 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.89256 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.967433 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.743397 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.01544 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.987598 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.853168 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.09235 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.99974 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.09604 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.604023 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.785991 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.920537 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.737642 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.812376 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.41535 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.06846 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19546 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.848906 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.06912 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.09515 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.819435 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.20014 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.15204 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.84528 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.703329 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.833999 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.24775 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.5915 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.867094 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.99856 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.684626 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.841861 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.57164 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.20385 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.16598 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.13358 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.869931 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.914164 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.12729 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.935572 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.642002 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.954561 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.20231 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.17891 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.66298 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.16292 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.15945 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.613942 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.86378 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.910683 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.02004 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.34294 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.69169 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.25881 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.2766 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.851005 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.11552 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.04103 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.821033 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.29484 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.00654 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.89011 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.631925 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.02236 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.37771 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.76332 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.706058 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.14743 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.562851 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.949416 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.82015 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.27093 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.01292 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.56874 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.822063 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.918054 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.972291 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.846163 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.544096 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.734548 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.36793 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.869849 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.51212 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.03374 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.14049 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.49905 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.735491 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.824767 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.822803 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.37719 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.80535 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.0904 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.14519 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.907943 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.14504 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.967965 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.834482 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.18032 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.10806 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.59348 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.813668 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.957326 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.2382 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.29995 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.817684 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.04581 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.737495 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.02713 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.5149 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.24248 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.15457 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.32255 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.922427 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.880097 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.957895 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.00594 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.696903 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.887606 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.27814 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.992385 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.22472 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.967524 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.17897 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.683324 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.78981 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.967208 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.842886 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.04132 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.51388 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.10712 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18719 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.902328 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.799209 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.18732 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.971032 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.11441 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.832031 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.08974 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.601435 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.949279 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.09313 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.77903 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.05056 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.35906 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.723723 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.645591 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.59906 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.08033 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.834991 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.887724 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.942901 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.41146 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.1622 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.540134 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.633621 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.750188 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.843793 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.917475 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.67025 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.43539 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.725189 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.686824 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.825898 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.56359 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.29019 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.87423 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.40064 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.13457 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08737 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.903067 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.953205 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.0582 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.902406 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.20643 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.08542 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.34524 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.00066 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.18958 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.30332 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.31294 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.02001 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.34046 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.792998 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.857352 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.22484 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.10478 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.975741 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.30576 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.956877 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.821729 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.0312 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.0356 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.715824 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.860476 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.15986 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.933041 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.37276 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.01045 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.04012 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.743143 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.866871 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.837093 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.895901 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.0473 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.38787 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.994928 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15399 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.908881 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.91259 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.06662 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.884279 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.18715 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.0582 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.37754 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.733526 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.0902 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.1856 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.74452 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.882489 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.17469 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.688843 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.81332 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.47912 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.18456 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.892321 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.26151 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.931494 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.22362 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.0139 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.789538 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.642531 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.828024 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.9787 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.837139 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.36397 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.13937 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.909733 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.605163 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.774807 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.679926 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.929081 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.34247 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.31071 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.06097 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.38678 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.883118 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.73487 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.06865 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.999213 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.993161 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.933982 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.72816 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.615947 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.902253 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.00162 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.97907 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.09335 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.05459 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.63443 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.689857 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.79641 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.04394 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.954225 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.946794 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.946285 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.2412 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.983862 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.511716 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.475048 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.611181 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.674705 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.937799 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.61429 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.24815 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.885044 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.637049 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.792091 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.688258 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.57421 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 2.53018 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.64406 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.24997 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.17751 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.922808 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.0479 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.943554 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.890662 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.1387 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.00232 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.36638 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.773766 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.07308 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.09179 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.3215 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.846474 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.12141 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.788017 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.986036 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.35413 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.16006 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.07662 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.35915 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.00851 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.872148 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.967116 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.953373 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.727023 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.840222 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.17631 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.912698 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.15817 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.943954 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.07388 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.729414 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.821065 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.937217 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.975451 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.10214 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.40378 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.0524 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11037 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.946615 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.02534 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.961936 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.893634 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.09107 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.989406 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.25835 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.857404 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.936279 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.09307 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.26736 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.909295 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.00868 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.814097 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.973344 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.35877 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.12845 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.05383 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.298 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.05909 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.988798 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.97093 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.849681 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.768447 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.904597 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.04966 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.921846 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.22316 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.972249 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.04398 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.817704 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.816126 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.857934 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.960379 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.12416 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.29926 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.06415 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.658998978524782 z_k1_s0000
                                 - 0.19999980726613 z_k1_s0001
                                 - 0.18399990220476 z_k1_s0002
                                 - 0.352998868353007 z_k1_s0003
                                 - 0.229999691381136 z_k1_s0004
                                 - 0.151999843582268 z_k1_s0005
                                 - 0.145999817059317 z_k1_s0006
                                 - 0.561000616147841 z_k1_s0007
                                 - 0.0709999354230112 z_k1_s0010
                                 - 0.144999669526167 z_k1_s0011
                                 - 0.0859998312223754 z_k1_s0012
                                 - 0.228999446962063 z_k1_s0013
                                 - 0.0909999092886602 z_k1_s0014
                                 - 0.142999728539517 z_k1_s0015
                                 - 0.117999902207244 z_k1_s0016
                                 - 0.245000102853562 z_k1_s0017
                                 - 0.29999954862788 z_k1_s0020
                                 - 0.263999617896503 z_k1_s0021
                                 - 0.677998254908533 z_k1_s0022
                                 - 0.286999563152682 z_k1_s0023
                                 - 0.174999484966792 z_k1_s0024
                                 - 0.622997154515908 z_k1_s0025
                                 - 0.684998944193712 z_k1_s0026
                                 - 0.764000250140059 z_k1_s0027
                                 - 0.0959998776677719 z_k1_s0030
                                 - 0.350998942345274 z_k1_s0031
                                 - 0.129999862297997 z_k1_s0032
                                 - 0.18900046272335 z_k1_s0033
                                 - 0.19200035173495 z_k1_s0034
                                 - 0.0929998395563209 z_k1_s0035
                                 - 0.157999621605468 z_k1_s0036
                                 - 0.282000004079968 z_k1_s0037
                                 - 0.173000049993916 z_k1_s0040
                                 - 0.241999271119165 z_k1_s0041
                                 - 0.14999992111908 z_k1_s0042
                                 - 0.20999965646686 z_k1_s0043
                                 - 0.110999984918092 z_k1_s0044
                                 - 0.332999788254381 z_k1_s0045
                                 - 0.244999599211337 z_k1_s0046
                                 - 0.201000337189286 z_k1_s0047
                                 - 0.0819998299101102 z_k1_s0050
                                 - 0.068999847926813 z_k1_s0051
                                 - 0.199999346512809 z_k1_s0052
                                 - 0.218999847010712 z_k1_s0053
                                 - 0.146000305673218 z_k1_s0054
                                 - 0.214999419801093 z_k1_s0055
                                 - 0.0719998625844697 z_k1_s0056
                                 - 0.220000259562674 z_k1_s0057
                                 - 0.393000708470768 z_k1_s0060
                                 - 0.199999402448637 z_k1_s0061
                                 - 0.637996643205624 z_k1_s0062
                                 - 0.565998125160209 z_k1_s0063
                                 - 0.444999638170234 z_k1_s0064
                                 - 0.211999155456206 z_k1_s0065
                                 - 0.540997662230787 z_k1_s0066
                                 - 0.914995829281944 z_k1_s0067
                                 - 0.155999537232149 z_k1_s0070
                                 - 0.116999924723985 z_k1_s0071
                                 - 0.104000008587951 z_k1_s0072
                                 - 0.181999913103962 z_k1_s0073
                                 - 0.113999933623367 z_k1_s0074
                                 - 0.324000397866137 z_k1_s0075
                                 - 0.236999976448524 z_k1_s0076
                                 - 0.124999946292279 z_k1_s0077
                                 - 0.829003732195442 z_k1_s0100
                                 - 0.699998083772861 z_k1_s0101
                                 - 1.17700289460261 z_k1_s0102
                                 - 0.95500285039416 z_k1_s0103
                                 - 0.595000384508421 z_k1_s0104
                                 - 0.778000094469196 z_k1_s0105
                                 - 1.00100007841728 z_k1_s0106
                                 - 0.67599953002167 z_k1_s0107
                                 - 0.664998021847304 z_k1_s0110
                                 - 0.518998692456387 z_k1_s0111
                                 - 0.294999530439123 z_k1_s0112
                                 - 0.256999558762271 z_k1_s0113
                                 - 0.168999686773671 z_k1_s0114
                                 - 0.262999523079719 z_k1_s0115
                                 - 0.173999854704559 z_k1_s0116
                                 - 0.622998864624211 z_k1_s0117
                                 - 0.21599942178767 z_k1_s0120
                                 - 0.150999808652947 z_k1_s0121
                                 - 0.551997615948272 z_k1_s0122
                                 - 0.252999618008054 z_k1_s0123
                                 - 0.626999166176232 z_k1_s0124
                                 - 0.654998212291391 z_k1_s0125
                                 - 0.516997881306372 z_k1_s0126
                                 - 0.736996743054287 z_k1_s0127
                                 - 0.347999372574752 z_k1_s0130
                                 - 0.249000141852761 z_k1_s0131
                                 - 0.19200010615514 z_k1_s0132
                                 - 0.154000082356727 z_k1_s0133
                                 - 0.450001127191939 z_k1_s0134
                                 - 0.615999258626658 z_k1_s0135
                                 - 0.124000007210597 z_k1_s0136
                                 - 0.544001326956469 z_k1_s0137
                                 - 0.130000129618849 z_k1_s0140
                                 - 0.349999920358058 z_k1_s0141
                                 - 0.122000163279265 z_k1_s0142
                                 - 0.162000472577613 z_k1_s0143
                                 - 0.278000064101657 z_k1_s0144
                                 - 0.154000187837744 z_k1_s0145
                                 - 0.242000135973456 z_k1_s0146
                                 - 0.154000187837744 z_k1_s0147
                                 - 0.27099939260499 z_k1_s0150
                                 - 0.219999855177602 z_k1_s0151
                                 - 0.058999961161266 z_k1_s0152
                                 - 0.248000694745954 z_k1_s0153
                                 - 0.138000052156908 z_k1_s0154
                                 - 0.13200005610661 z_k1_s0155
                                 - 0.226999707569568 z_k1_s0156
                                 - 0.135000054131759 z_k1_s0157
                                 - 0.494001466575118 z_k1_s0160
                                 - 0.667003309870271 z_k1_s0161
                                 - 1.04400387657616 z_k1_s0162
                                 - 0.653001074408583 z_k1_s0163
                                 - 0.360000923038978 z_k1_s0164
                                 - 0.658000087425966 z_k1_s0165
                                 - 0.67300212549113 z_k1_s0166
                                 - 0.450001153798722 z_k1_s0167
                                 - 0.529002565050105 z_k1_s0170
                                 - 0.799003074588982 z_k1_s0171
                                 - 0.28800086397591 z_k1_s0172
                                 - 0.603001778905708 z_k1_s0173
                                 - 0.879001445186938 z_k1_s0174
                                 - 0.871002569530467 z_k1_s0175
                                 - 0.183000719233198 z_k1_s0176
                                 - 0.655004084706013 z_k1_s0177
                                 - 0.199999553282853 z_k1_s1000
                                 - 0.230999392491959 z_k1_s1001
                                 - 0.377997874008288 z_k1_s1002
                                 - 0.176999330006117 z_k1_s1003
                                 - 0.585998874218231 z_k1_s1004
                                 - 0.822999168806036 z_k1_s1005
                                 - 0.868997418165844 z_k1_s1006
                                 - 0.397999660331293 z_k1_s1007
                                 - 0.465001545702818 z_k1_s1010
                                 - 0.298000846492971 z_k1_s1011
                                 - 0.26200096456833 z_k1_s1012
                                 - 0.34200256329554 z_k1_s1013
                                 - 0.25400080469561 z_k1_s1014
                                 - 0.63500117423642 z_k1_s1015
                                 - 0.849003189574674 z_k1_s1016
                                 - 0.245000666713929 z_k1_s1017
                                 - 0.192000013591461 z_k1_s1020
                                 - 0.161999960692767 z_k1_s1021
                                 - 0.449999439479661 z_k1_s1022
                                 - 0.441000750530783 z_k1_s1023
                                 - 0.467000272829363 z_k1_s1024
                                 - 0.201000056541084 z_k1_s1025
                                 - 0.365000688734132 z_k1_s1026
                                 - 0.430000848481326 z_k1_s1027
                                 - 0.415999189524479 z_k1_s1030
                                 - 0.307998834147579 z_k1_s1031
                                 - 0.303000063029055 z_k1_s1032
                                 - 0.383000014937675 z_k1_s1033
                                 - 0.647999330259655 z_k1_s1034
                                 - 0.188000027085346 z_k1_s1035
                                 - 0.2480001311169 z_k1_s1036
                                 - 0.307998834147579 z_k1_s1037
                                 - 0.314000415655066 z_k1_s1040
                                 - 0.437998977815558 z_k1_s1041
                                 - 0.837000809992387 z_k1_s1042
                                 - 0.316000060179949 z_k1_s1043
                                 - 0.295000144465535 z_k1_s1044
                                 - 0.38500045769932 z_k1_s1045
                                 - 0.721000825931426 z_k1_s1046
                                 - 0.98600201052002 z_k1_s1047
                                 - 0.218999615992899 z_k1_s1050
                                 - 0.518998721708364 z_k1_s1051
                                 - 0.491997185003326 z_k1_s1052
                                 - 0.431998985317873 z_k1_s1053
                                 - 0.708000944656052 z_k1_s1054
                                 - 0.543000561778819 z_k1_s1055
                                 - 0.394999455462775 z_k1_s1056
                                 - 0.958998320383056 z_k1_s1057
                                 - 0.421998510651825 z_k1_s1060
                                 - 0.418997110371199 z_k1_s1061
                                 - 0.612000197981891 z_k1_s1062
                                 - 0.35299892811352 z_k1_s1063
                                 - 0.709998363938038 z_k1_s1064
                                 - 0.828998175879544 z_k1_s1065
                                 - 0.190999571363583 z_k1_s1066
                                 - 0.541997430023702 z_k1_s1067
                                 - 0.261000041087502 z_k1_s1070
                                 - 0.219000041094111 z_k1_s1071
                                 - 0.583999341851028 z_k1_s1072
                                 - 0.740000822454927 z_k1_s1073
                                 - 0.266999876572287 z_k1_s1074
                                 - 0.305001233809049 z_k1_s1075
                                 - 0.217999876579996 z_k1_s1076
                                 - 0.28499995882659 z_k1_s1077
                                 - 0.408996493701841 z_k1_s1100
                                 - 0.608998052257091 z_k1_s1101
                                 - 0.214999291706279 z_k1_s1102
                                 - 0.776996462814913 z_k1_s1103
                                 - 0.430000109006552 z_k1_s1104
                                 - 0.203999377615065 z_k1_s1105
                                 - 0.584996644755266 z_k1_s1106
                                 - 0.584996644755266 z_k1_s1107
                                 - 0.237000624177458 z_k1_s1110
                                 - 0.218000141597771 z_k1_s1111
                                 - 0.0969999667696375 z_k1_s1112
                                 - 0.12700046524279 z_k1_s1113
                                 - 0.122000024563883 z_k1_s1114
                                 - 0.177000104031217 z_k1_s1115
                                 - 0.131999713921536 z_k1_s1116
                                 - 0.0819999559330935 z_k1_s1117
                                 - 0.245000764597933 z_k1_s1120
                                 - 0.67000279643298 z_k1_s1121
                                 - 0.494000020952557 z_k1_s1122
                                 - 0.730003218845351 z_k1_s1123
                                 - 0.618003198544325 z_k1_s1124
                                 - 0.699002424447409 z_k1_s1125
                                 - 0.203000468909273 z_k1_s1126
                                 - 0.18200051311546 z_k1_s1127
                                 - 0.290999917525987 z_k1_s1130
                                 - 0.279999729273277 z_k1_s1131
                                 - 0.593999839255101 z_k1_s1132
                                 - 0.350999569741935 z_k1_s1133
                                 - 0.458001266818408 z_k1_s1134
                                 - 0.241999715963375 z_k1_s1135
                                 - 0.901999209535987 z_k1_s1136
                                 - 0.5699983556501 z_k1_s1137
                                 - 0.528998727858661 z_k1_s1140
                                 - 0.207000164866588 z_k1_s1141
                                 - 0.373999986367578 z_k1_s1142
                                 - 0.661999722999237 z_k1_s1143
                                 - 1.01699861811794 z_k1_s1144
                                 - 0.337000206104527 z_k1_s1145
                                 - 0.543001095137146 z_k1_s1146
                                 - 0.239000048758974 z_k1_s1147
                                 - 0.331998729118027 z_k1_s1150
                                 - 0.402999702888725 z_k1_s1151
                                 - 0.346999034689178 z_k1_s1152
                                 - 0.292000277258719 z_k1_s1153
                                 - 0.266999767973467 z_k1_s1154
                                 - 0.14499988195824 z_k1_s1155
                                 - 0.304999124288793 z_k1_s1156
                                 - 0.271999869830517 z_k1_s1157
                                 - 0.232999616584643 z_k1_s1160
                                 - 0.208999897537087 z_k1_s1161
                                 - 0.280999054679755 z_k1_s1162
                                 - 0.426000348089462 z_k1_s1163
                                 - 0.441000414993973 z_k1_s1164
                                 - 0.129999868506381 z_k1_s1165
                                 - 0.102999942078093 z_k1_s1166
                                 - 0.116999939855692 z_k1_s1167
                                 - 0.313999573614556 z_k1_s1170
                                 - 0.627001454955648 z_k1_s1171
                                 - 0.474999386999671 z_k1_s1172
                                 - 0.959995798699199 z_k1_s1173
                                 - 1.10599954603544 z_k1_s1174
                                 - 0.997999092933387 z_k1_s1175
                                 - 0.587998352451657 z_k1_s1176
                                 - 0.221999755822931 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.02185 w_k1_ADM_s000
                                 + 0.883155 w_k1_AXP_s000
                                 + 0.864971 w_k1_DIS_s000
                                 + 0.970657 w_k1_FDX_s000
                                 + 0.78204 w_k1_IBM_s000
                                 + 1.06025 w_k1_INTC_s000
                                 + 0.846779 w_k1_KO_s000
                                 + 0.628659 w_k1_MAT_s000
                                 + 1.02918 w_k1_MSFT_s000
                                 + 0.949841 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.971674 w_k1_ADM_s000
                                 + 0.71285 w_k1_AXP_s000
                                 + 0.602065 w_k1_DIS_s000
                                 + 0.701235 w_k1_FDX_s000
                                 + 0.785835 w_k1_IBM_s000
                                 + 0.965815 w_k1_INTC_s000
                                 + 0.811074 w_k1_KO_s000
                                 + 0.76798 w_k1_MAT_s000
                                 + 0.773112 w_k1_MSFT_s000
                                 + 0.862365 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.753294 w_k1_ADM_s000
                                 + 0.793831 w_k1_AXP_s000
                                 + 0.662138 w_k1_DIS_s000
                                 + 0.848904 w_k1_FDX_s000
                                 + 0.874387 w_k1_IBM_s000
                                 + 1.05789 w_k1_INTC_s000
                                 + 0.768932 w_k1_KO_s000
                                 + 0.964031 w_k1_MAT_s000
                                 + 0.919048 w_k1_MSFT_s000
                                 + 0.851353 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.15548 w_k1_ADM_s000
                                 + 1.17871 w_k1_AXP_s000
                                 + 0.916601 w_k1_DIS_s000
                                 + 1.00968 w_k1_FDX_s000
                                 + 1.23729 w_k1_IBM_s000
                                 + 1.07212 w_k1_INTC_s000
                                 + 0.975257 w_k1_KO_s000
                                 + 1.06189 w_k1_MAT_s000
                                 + 0.958496 w_k1_MSFT_s000
                                 + 1.05304 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.839642 w_k1_ADM_s000
                                 + 1.0377 w_k1_AXP_s000
                                 + 0.746689 w_k1_DIS_s000
                                 + 0.805202 w_k1_FDX_s000
                                 + 1.34646 w_k1_IBM_s000
                                 + 0.973292 w_k1_INTC_s000
                                 + 0.850314 w_k1_KO_s000 + 1.2393 w_k1_MAT_s000
                                 + 0.820472 w_k1_MSFT_s000
                                 + 0.926457 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.10792 w_k1_ADM_s000
                                 + 1.19358 w_k1_AXP_s000
                                 + 1.54801 w_k1_DIS_s000
                                 + 1.15758 w_k1_FDX_s000
                                 + 1.35764 w_k1_IBM_s000
                                 + 1.07414 w_k1_INTC_s000
                                 + 1.23074 w_k1_KO_s000 + 1.08863 w_k1_MAT_s000
                                 + 1.17886 w_k1_MSFT_s000
                                 + 1.12296 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.47295 w_k1_ADM_s000
                                 + 1.83734 w_k1_AXP_s000 + 1.4475 w_k1_DIS_s000
                                 + 1.2943 w_k1_FDX_s000 + 1.66987 w_k1_IBM_s000
                                 + 1.43506 w_k1_INTC_s000
                                 + 1.33173 w_k1_KO_s000 + 1.55909 w_k1_MAT_s000
                                 + 1.24897 w_k1_MSFT_s000
                                 + 1.29422 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.14121 w_k1_ADM_s000
                                 + 1.18794 w_k1_AXP_s000
                                 + 1.24482 w_k1_DIS_s000
                                 + 1.19419 w_k1_FDX_s000
                                 + 1.06868 w_k1_IBM_s000
                                 + 1.40866 w_k1_INTC_s000
                                 + 1.02709 w_k1_KO_s000 + 1.02955 w_k1_MAT_s000
                                 + 1.06148 w_k1_MSFT_s000
                                 + 1.05668 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.10462 w_k1_ADM_s001
                                 + 1.00556 w_k1_AXP_s001
                                 + 0.620988 w_k1_DIS_s001
                                 + 0.8776 w_k1_FDX_s001 + 0.84052 w_k1_IBM_s001
                                 + 0.897865 w_k1_INTC_s001
                                 + 0.853348 w_k1_KO_s001
                                 + 0.633721 w_k1_MAT_s001
                                 + 0.828252 w_k1_MSFT_s001
                                 + 0.955403 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.989615 w_k1_ADM_s001
                                 + 0.820397 w_k1_AXP_s001
                                 + 1.05529 w_k1_DIS_s001
                                 + 1.02334 w_k1_FDX_s001
                                 + 0.760518 w_k1_IBM_s001
                                 + 1.30318 w_k1_INTC_s001
                                 + 0.985971 w_k1_KO_s001
                                 + 0.571609 w_k1_MAT_s001
                                 + 1.08333 w_k1_MSFT_s001
                                 + 0.910321 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.977014 w_k1_ADM_s001
                                 + 1.52538 w_k1_AXP_s001
                                 + 0.88288 w_k1_DIS_s001
                                 + 0.98107 w_k1_FDX_s001
                                 + 1.61185 w_k1_IBM_s001
                                 + 0.942844 w_k1_INTC_s001
                                 + 1.25051 w_k1_KO_s001 + 1.79131 w_k1_MAT_s001
                                 + 0.920455 w_k1_MSFT_s001
                                 + 1.08323 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.974864 w_k1_ADM_s001
                                 + 1.13406 w_k1_AXP_s001
                                 + 0.997631 w_k1_DIS_s001
                                 + 0.977536 w_k1_FDX_s001
                                 + 1.02618 w_k1_IBM_s001
                                 + 1.06386 w_k1_INTC_s001
                                 + 1.05497 w_k1_KO_s001 + 1.39498 w_k1_MAT_s001
                                 + 1.00201 w_k1_MSFT_s001
                                 + 1.08569 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.684955 w_k1_ADM_s001
                                 + 0.665985 w_k1_AXP_s001
                                 + 0.582097 w_k1_DIS_s001
                                 + 0.655944 w_k1_FDX_s001
                                 + 0.784225 w_k1_IBM_s001
                                 + 0.8875 w_k1_INTC_s001
                                 + 0.720005 w_k1_KO_s001
                                 + 0.563879 w_k1_MAT_s001
                                 + 0.76283 w_k1_MSFT_s001
                                 + 0.906037 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.00895 w_k1_ADM_s001
                                 + 1.32405 w_k1_AXP_s001
                                 + 1.65818 w_k1_DIS_s001 + 1.3018 w_k1_FDX_s001
                                 + 1.72643 w_k1_IBM_s001
                                 + 1.37371 w_k1_INTC_s001
                                 + 1.61395 w_k1_KO_s001 + 1.18547 w_k1_MAT_s001
                                 + 1.24168 w_k1_MSFT_s001
                                 + 1.30137 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.21834 w_k1_ADM_s001
                                 + 1.15893 w_k1_AXP_s001
                                 + 1.40967 w_k1_DIS_s001 + 1.3257 w_k1_FDX_s001
                                 + 1.08287 w_k1_IBM_s001
                                 + 1.12602 w_k1_INTC_s001 + 1.131 w_k1_KO_s001
                                 + 0.569029 w_k1_MAT_s001
                                 + 1.2257 w_k1_MSFT_s001 + 1.21052 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.884364 w_k1_ADM_s001
                                 + 0.843108 w_k1_AXP_s001
                                 + 0.832432 w_k1_DIS_s001
                                 + 0.867498 w_k1_FDX_s001
                                 + 1.14631 w_k1_IBM_s001
                                 + 1.02366 w_k1_INTC_s001
                                 + 1.12275 w_k1_KO_s001 + 1.02136 w_k1_MAT_s001
                                 + 0.955556 w_k1_MSFT_s001
                                 + 0.95077 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.665058 w_k1_ADM_s002
                                 + 0.614453 w_k1_AXP_s002
                                 + 0.569338 w_k1_DIS_s002
                                 + 0.620139 w_k1_FDX_s002
                                 + 1.03714 w_k1_IBM_s002
                                 + 0.793265 w_k1_INTC_s002
                                 + 0.730663 w_k1_KO_s002
                                 + 0.679559 w_k1_MAT_s002
                                 + 0.730455 w_k1_MSFT_s002
                                 + 0.846391 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.1232 w_k1_ADM_s002
                                 + 1.04343 w_k1_AXP_s002
                                 + 1.18045 w_k1_DIS_s002
                                 + 1.22269 w_k1_FDX_s002
                                 + 0.83647 w_k1_IBM_s002
                                 + 1.23997 w_k1_INTC_s002
                                 + 1.03069 w_k1_KO_s002
                                 + 0.703943 w_k1_MAT_s002
                                 + 1.38376 w_k1_MSFT_s002
                                 + 0.990233 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.15053 w_k1_ADM_s002
                                 + 1.35843 w_k1_AXP_s002
                                 + 1.50298 w_k1_DIS_s002
                                 + 1.34636 w_k1_FDX_s002
                                 + 1.30692 w_k1_IBM_s002
                                 + 1.21797 w_k1_INTC_s002
                                 + 1.28042 w_k1_KO_s002 + 1.30085 w_k1_MAT_s002
                                 + 1.28465 w_k1_MSFT_s002
                                 + 1.18829 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.893704 w_k1_ADM_s002
                                 + 0.797482 w_k1_AXP_s002
                                 + 0.625301 w_k1_DIS_s002
                                 + 0.767263 w_k1_FDX_s002
                                 + 1.03953 w_k1_IBM_s002
                                 + 0.816089 w_k1_INTC_s002
                                 + 0.749612 w_k1_KO_s002
                                 + 0.923478 w_k1_MAT_s002
                                 + 0.789641 w_k1_MSFT_s002
                                 + 0.821148 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.619449 w_k1_ADM_s002
                                 + 0.418978 w_k1_AXP_s002
                                 + 0.491068 w_k1_DIS_s002
                                 + 0.566238 w_k1_FDX_s002
                                 + 0.679605 w_k1_IBM_s002
                                 + 0.683399 w_k1_INTC_s002
                                 + 0.589047 w_k1_KO_s002
                                 + 0.362913 w_k1_MAT_s002
                                 + 0.663911 w_k1_MSFT_s002
                                 + 0.704034 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.977432 w_k1_ADM_s002
                                 + 0.717115 w_k1_AXP_s002
                                 + 0.853033 w_k1_DIS_s002
                                 + 0.870714 w_k1_FDX_s002
                                 + 0.789792 w_k1_IBM_s002
                                 + 0.809302 w_k1_INTC_s002
                                 + 0.829245 w_k1_KO_s002
                                 + 0.821557 w_k1_MAT_s002
                                 + 1.01237 w_k1_MSFT_s002
                                 + 0.936735 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.939676 w_k1_ADM_s002
                                 + 0.905121 w_k1_AXP_s002
                                 + 0.821646 w_k1_DIS_s002
                                 + 0.911032 w_k1_FDX_s002
                                 + 1.23983 w_k1_IBM_s002
                                 + 0.892658 w_k1_INTC_s002
                                 + 1.02481 w_k1_KO_s002 + 1.09309 w_k1_MAT_s002
                                 + 0.904843 w_k1_MSFT_s002
                                 + 0.91284 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.26947 w_k1_ADM_s002
                                 + 1.17088 w_k1_AXP_s002
                                 + 1.13293 w_k1_DIS_s002
                                 + 1.11299 w_k1_FDX_s002 + 1.1554 w_k1_IBM_s002
                                 + 1.04496 w_k1_INTC_s002 + 1.2252 w_k1_KO_s002
                                 + 1.28131 w_k1_MAT_s002
                                 + 1.1687 w_k1_MSFT_s002 + 1.09535 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.24757 w_k1_ADM_s003
                                 + 1.26534 w_k1_AXP_s003
                                 + 1.48998 w_k1_DIS_s003
                                 + 1.12004 w_k1_FDX_s003
                                 + 1.28835 w_k1_IBM_s003
                                 + 0.930835 w_k1_INTC_s003
                                 + 1.30433 w_k1_KO_s003 + 1.48062 w_k1_MAT_s003
                                 + 1.07541 w_k1_MSFT_s003
                                 + 1.25033 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.987416 w_k1_ADM_s003
                                 + 1.0236 w_k1_AXP_s003
                                 + 0.932797 w_k1_DIS_s003
                                 + 1.0484 w_k1_FDX_s003
                                 + 0.973237 w_k1_IBM_s003
                                 + 0.965151 w_k1_INTC_s003
                                 + 1.03934 w_k1_KO_s003 + 1.47038 w_k1_MAT_s003
                                 + 0.929996 w_k1_MSFT_s003
                                 + 1.09076 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.06232 w_k1_ADM_s003
                                 + 0.853734 w_k1_AXP_s003
                                 + 0.913834 w_k1_DIS_s003
                                 + 0.919697 w_k1_FDX_s003
                                 + 1.16307 w_k1_IBM_s003
                                 + 0.798583 w_k1_INTC_s003
                                 + 1.05032 w_k1_KO_s003 + 1.28869 w_k1_MAT_s003
                                 + 0.85256 w_k1_MSFT_s003
                                 + 0.943239 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.68275 w_k1_ADM_s003
                                 + 1.25521 w_k1_AXP_s003
                                 + 1.55362 w_k1_DIS_s003
                                 + 1.39038 w_k1_FDX_s003
                                 + 1.00536 w_k1_IBM_s003
                                 + 1.12024 w_k1_INTC_s003
                                 + 1.33714 w_k1_KO_s003 + 1.43785 w_k1_MAT_s003
                                 + 1.3252 w_k1_MSFT_s003 + 1.35447 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.865862 w_k1_ADM_s003
                                 + 0.787128 w_k1_AXP_s003
                                 + 0.70318 w_k1_DIS_s003
                                 + 0.896005 w_k1_FDX_s003
                                 + 0.650515 w_k1_IBM_s003
                                 + 0.785478 w_k1_INTC_s003
                                 + 0.732659 w_k1_KO_s003
                                 + 0.776245 w_k1_MAT_s003
                                 + 0.857678 w_k1_MSFT_s003
                                 + 0.982537 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.25957 w_k1_ADM_s003
                                 + 1.14239 w_k1_AXP_s003
                                 + 1.14626 w_k1_DIS_s003
                                 + 1.13268 w_k1_FDX_s003
                                 + 1.43577 w_k1_IBM_s003
                                 + 0.962541 w_k1_INTC_s003
                                 + 1.23038 w_k1_KO_s003 + 1.04436 w_k1_MAT_s003
                                 + 0.942967 w_k1_MSFT_s003
                                 + 1.12114 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.904989 w_k1_ADM_s003
                                 + 0.655344 w_k1_AXP_s003
                                 + 0.642723 w_k1_DIS_s003
                                 + 0.779568 w_k1_FDX_s003
                                 + 0.860954 w_k1_IBM_s003
                                 + 0.66216 w_k1_INTC_s003
                                 + 0.791582 w_k1_KO_s003
                                 + 0.890584 w_k1_MAT_s003
                                 + 0.720408 w_k1_MSFT_s003
                                 + 0.843523 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.937203 w_k1_ADM_s003
                                 + 0.878681 w_k1_AXP_s003
                                 + 1.06825 w_k1_DIS_s003
                                 + 1.04832 w_k1_FDX_s003
                                 + 0.792183 w_k1_IBM_s003
                                 + 0.977879 w_k1_INTC_s003
                                 + 1.05834 w_k1_KO_s003 + 0.75462 w_k1_MAT_s003
                                 + 0.972653 w_k1_MSFT_s003
                                 + 1.12092 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.898794 w_k1_ADM_s004
                                 + 1.30484 w_k1_AXP_s004
                                 + 1.12996 w_k1_DIS_s004
                                 + 1.02773 w_k1_FDX_s004
                                 + 1.46052 w_k1_IBM_s004
                                 + 1.09817 w_k1_INTC_s004
                                 + 1.20267 w_k1_KO_s004 + 1.60247 w_k1_MAT_s004
                                 + 1.20085 w_k1_MSFT_s004 + 1.1014 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.11031 w_k1_ADM_s004
                                 + 0.852619 w_k1_AXP_s004
                                 + 0.762515 w_k1_DIS_s004
                                 + 0.918939 w_k1_FDX_s004
                                 + 0.70607 w_k1_IBM_s004
                                 + 1.01312 w_k1_INTC_s004
                                 + 0.839505 w_k1_KO_s004
                                 + 0.779652 w_k1_MAT_s004
                                 + 0.987571 w_k1_MSFT_s004
                                 + 0.905908 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.44528 w_k1_ADM_s004
                                 + 0.881391 w_k1_AXP_s004
                                 + 1.05528 w_k1_DIS_s004
                                 + 1.13067 w_k1_FDX_s004
                                 + 1.13755 w_k1_IBM_s004
                                 + 1.08756 w_k1_INTC_s004
                                 + 1.15233 w_k1_KO_s004
                                 + 0.886139 w_k1_MAT_s004
                                 + 1.1486 w_k1_MSFT_s004 + 1.17035 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.21291 w_k1_ADM_s004
                                 + 1.20261 w_k1_AXP_s004
                                 + 0.878023 w_k1_DIS_s004
                                 + 0.956536 w_k1_FDX_s004
                                 + 0.999034 w_k1_IBM_s004
                                 + 0.810711 w_k1_INTC_s004
                                 + 0.946058 w_k1_KO_s004
                                 + 1.79287 w_k1_MAT_s004
                                 + 1.02776 w_k1_MSFT_s004
                                 + 1.10754 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.980559 w_k1_ADM_s004
                                 + 1.05999 w_k1_AXP_s004
                                 + 0.86918 w_k1_DIS_s004
                                 + 0.882427 w_k1_FDX_s004
                                 + 1.46662 w_k1_IBM_s004
                                 + 0.914357 w_k1_INTC_s004
                                 + 1.17797 w_k1_KO_s004 + 1.81309 w_k1_MAT_s004
                                 + 0.907948 w_k1_MSFT_s004
                                 + 1.24508 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.03077 w_k1_ADM_s004
                                 + 1.06839 w_k1_AXP_s004
                                 + 1.18527 w_k1_DIS_s004
                                 + 1.10587 w_k1_FDX_s004
                                 + 0.887086 w_k1_IBM_s004
                                 + 1.07696 w_k1_INTC_s004
                                 + 1.13234 w_k1_KO_s004 + 1.48878 w_k1_MAT_s004
                                 + 1.17833 w_k1_MSFT_s004
                                 + 1.00705 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.799271 w_k1_ADM_s004
                                 + 0.784465 w_k1_AXP_s004
                                 + 0.687152 w_k1_DIS_s004
                                 + 0.719257 w_k1_FDX_s004
                                 + 0.78177 w_k1_IBM_s004
                                 + 0.837142 w_k1_INTC_s004
                                 + 0.937333 w_k1_KO_s004
                                 + 1.17059 w_k1_MAT_s004
                                 + 1.02001 w_k1_MSFT_s004
                                 + 0.938063 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.940863 w_k1_ADM_s004
                                 + 1.25797 w_k1_AXP_s004
                                 + 1.61567 w_k1_DIS_s004
                                 + 1.34326 w_k1_FDX_s004
                                 + 1.05307 w_k1_IBM_s004
                                 + 1.37726 w_k1_INTC_s004
                                 + 1.47108 w_k1_KO_s004 + 1.36737 w_k1_MAT_s004
                                 + 1.53799 w_k1_MSFT_s004
                                 + 1.25677 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.17819 w_k1_ADM_s005
                                 + 1.0599 w_k1_AXP_s005
                                 + 0.845151 w_k1_DIS_s005
                                 + 1.08574 w_k1_FDX_s005
                                 + 0.733336 w_k1_IBM_s005
                                 + 1.06255 w_k1_INTC_s005
                                 + 0.810656 w_k1_KO_s005
                                 + 0.954484 w_k1_MAT_s005
                                 + 1.01019 w_k1_MSFT_s005
                                 + 0.962291 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.988578 w_k1_ADM_s005
                                 + 1.02588 w_k1_AXP_s005
                                 + 1.22617 w_k1_DIS_s005
                                 + 1.27208 w_k1_FDX_s005
                                 + 0.762542 w_k1_IBM_s005
                                 + 1.34585 w_k1_INTC_s005
                                 + 0.897863 w_k1_KO_s005
                                 + 0.762468 w_k1_MAT_s005
                                 + 1.0696 w_k1_MSFT_s005 + 1.06917 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.9422 w_k1_ADM_s005
                                 + 1.0337 w_k1_AXP_s005 + 1.05301 w_k1_DIS_s005
                                 + 0.9806 w_k1_FDX_s005 + 1.22274 w_k1_IBM_s005
                                 + 1.21131 w_k1_INTC_s005
                                 + 1.24409 w_k1_KO_s005 + 1.22673 w_k1_MAT_s005
                                 + 1.11141 w_k1_MSFT_s005 + 1.1917 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.776493 w_k1_ADM_s005
                                 + 0.954085 w_k1_AXP_s005
                                 + 0.691702 w_k1_DIS_s005
                                 + 0.863072 w_k1_FDX_s005
                                 + 1.13209 w_k1_IBM_s005
                                 + 0.986801 w_k1_INTC_s005
                                 + 0.790548 w_k1_KO_s005
                                 + 0.978538 w_k1_MAT_s005
                                 + 0.819252 w_k1_MSFT_s005
                                 + 0.82148 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.942147 w_k1_ADM_s005
                                 + 1.56386 w_k1_AXP_s005 + 1.8775 w_k1_DIS_s005
                                 + 1.40493 w_k1_FDX_s005
                                 + 1.43761 w_k1_IBM_s005
                                 + 1.71142 w_k1_INTC_s005
                                 + 1.31953 w_k1_KO_s005 + 1.48631 w_k1_MAT_s005
                                 + 1.35193 w_k1_MSFT_s005
                                 + 1.18779 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.966677 w_k1_ADM_s005
                                 + 0.788694 w_k1_AXP_s005
                                 + 0.674384 w_k1_DIS_s005
                                 + 0.893296 w_k1_FDX_s005
                                 + 0.72118 w_k1_IBM_s005
                                 + 0.985634 w_k1_INTC_s005
                                 + 0.692997 w_k1_KO_s005
                                 + 0.563211 w_k1_MAT_s005
                                 + 0.89294 w_k1_MSFT_s005
                                 + 0.933391 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.921403 w_k1_ADM_s005
                                 + 0.788075 w_k1_AXP_s005
                                 + 1.3866 w_k1_DIS_s005 + 1.05584 w_k1_FDX_s005
                                 + 0.906873 w_k1_IBM_s005
                                 + 1.33788 w_k1_INTC_s005
                                 + 0.989039 w_k1_KO_s005
                                 + 0.568089 w_k1_MAT_s005
                                 + 1.07307 w_k1_MSFT_s005
                                 + 0.883134 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.31005 w_k1_ADM_s005
                                 + 1.15697 w_k1_AXP_s005
                                 + 1.08864 w_k1_DIS_s005
                                 + 1.14666 w_k1_FDX_s005
                                 + 0.924778 w_k1_IBM_s005
                                 + 1.40054 w_k1_INTC_s005
                                 + 0.951772 w_k1_KO_s005
                                 + 0.857393 w_k1_MAT_s005
                                 + 1.19691 w_k1_MSFT_s005
                                 + 0.981472 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.03782 w_k1_ADM_s006
                                 + 0.972728 w_k1_AXP_s006
                                 + 1.20775 w_k1_DIS_s006
                                 + 1.02316 w_k1_FDX_s006
                                 + 1.28597 w_k1_IBM_s006
                                 + 0.988328 w_k1_INTC_s006
                                 + 1.17604 w_k1_KO_s006 + 1.18707 w_k1_MAT_s006
                                 + 1.01491 w_k1_MSFT_s006
                                 + 1.00721 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.636175 w_k1_ADM_s006
                                 + 0.677745 w_k1_AXP_s006
                                 + 0.627396 w_k1_DIS_s006
                                 + 0.687881 w_k1_FDX_s006
                                 + 1.11507 w_k1_IBM_s006
                                 + 0.738752 w_k1_INTC_s006
                                 + 0.860444 w_k1_KO_s006
                                 + 1.02921 w_k1_MAT_s006
                                 + 0.76183 w_k1_MSFT_s006
                                 + 0.887074 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.14968 w_k1_ADM_s006
                                 + 1.31504 w_k1_AXP_s006
                                 + 1.23792 w_k1_DIS_s006
                                 + 1.22101 w_k1_FDX_s006
                                 + 1.05542 w_k1_IBM_s006
                                 + 1.2317 w_k1_INTC_s006 + 1.07472 w_k1_KO_s006
                                 + 0.937789 w_k1_MAT_s006
                                 + 1.17412 w_k1_MSFT_s006 + 1.0016 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.979796 w_k1_ADM_s006
                                 + 0.852932 w_k1_AXP_s006
                                 + 1.23828 w_k1_DIS_s006
                                 + 0.956398 w_k1_FDX_s006
                                 + 0.903562 w_k1_IBM_s006
                                 + 0.938419 w_k1_INTC_s006
                                 + 1.22802 w_k1_KO_s006 + 0.90708 w_k1_MAT_s006
                                 + 0.985259 w_k1_MSFT_s006
                                 + 1.09233 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.17004 w_k1_ADM_s006
                                 + 1.41803 w_k1_AXP_s006
                                 + 1.63406 w_k1_DIS_s006 + 1.2844 w_k1_FDX_s006
                                 + 1.61783 w_k1_IBM_s006
                                 + 1.23511 w_k1_INTC_s006
                                 + 1.65911 w_k1_KO_s006 + 1.60876 w_k1_MAT_s006
                                 + 1.24165 w_k1_MSFT_s006
                                 + 1.09753 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.25645 w_k1_ADM_s006
                                 + 1.18908 w_k1_AXP_s006
                                 + 1.25344 w_k1_DIS_s006
                                 + 0.982819 w_k1_FDX_s006
                                 + 1.71808 w_k1_IBM_s006
                                 + 0.859211 w_k1_INTC_s006
                                 + 1.26126 w_k1_KO_s006 + 1.44862 w_k1_MAT_s006
                                 + 0.979268 w_k1_MSFT_s006
                                 + 1.06798 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.680459 w_k1_ADM_s006
                                 + 0.793976 w_k1_AXP_s006
                                 + 0.756178 w_k1_DIS_s006
                                 + 0.786673 w_k1_FDX_s006
                                 + 0.967696 w_k1_IBM_s006
                                 + 0.766194 w_k1_INTC_s006
                                 + 0.867382 w_k1_KO_s006
                                 + 0.650965 w_k1_MAT_s006
                                 + 0.803227 w_k1_MSFT_s006
                                 + 0.858472 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.95583 w_k1_ADM_s006
                                 + 0.978954 w_k1_AXP_s006
                                 + 0.865418 w_k1_DIS_s006
                                 + 1.00373 w_k1_FDX_s006
                                 + 0.687938 w_k1_IBM_s006
                                 + 0.928445 w_k1_INTC_s006
                                 + 0.782896 w_k1_KO_s006
                                 + 0.551766 w_k1_MAT_s006
                                 + 0.879201 w_k1_MSFT_s006
                                 + 0.916397 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.908473 w_k1_ADM_s007
                                 + 0.973945 w_k1_AXP_s007
                                 + 0.973037 w_k1_DIS_s007
                                 + 0.870669 w_k1_FDX_s007
                                 + 1.04061 w_k1_IBM_s007
                                 + 0.875785 w_k1_INTC_s007
                                 + 1.17145 w_k1_KO_s007 + 1.44324 w_k1_MAT_s007
                                 + 1.00006 w_k1_MSFT_s007 + 1.1615 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.09718 w_k1_ADM_s007
                                 + 1.51229 w_k1_AXP_s007
                                 + 1.34203 w_k1_DIS_s007
                                 + 1.27941 w_k1_FDX_s007
                                 + 1.05996 w_k1_IBM_s007
                                 + 1.14067 w_k1_INTC_s007
                                 + 1.22813 w_k1_KO_s007 + 1.24968 w_k1_MAT_s007
                                 + 1.04304 w_k1_MSFT_s007
                                 + 1.14873 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.2798 w_k1_ADM_s007
                                 + 1.78903 w_k1_AXP_s007
                                 + 1.44647 w_k1_DIS_s007
                                 + 1.30655 w_k1_FDX_s007
                                 + 1.42307 w_k1_IBM_s007
                                 + 1.01128 w_k1_INTC_s007
                                 + 1.31192 w_k1_KO_s007 + 1.81631 w_k1_MAT_s007
                                 + 1.10576 w_k1_MSFT_s007
                                 + 1.20383 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.17988 w_k1_ADM_s007
                                 + 1.11499 w_k1_AXP_s007
                                 + 1.37428 w_k1_DIS_s007
                                 + 1.16474 w_k1_FDX_s007
                                 + 1.21428 w_k1_IBM_s007
                                 + 0.777436 w_k1_INTC_s007
                                 + 1.24187 w_k1_KO_s007 + 1.47629 w_k1_MAT_s007
                                 + 0.923648 w_k1_MSFT_s007
                                 + 1.14808 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.15624 w_k1_ADM_s007
                                 + 0.907432 w_k1_AXP_s007
                                 + 1.33093 w_k1_DIS_s007
                                 + 1.04165 w_k1_FDX_s007
                                 + 1.00288 w_k1_IBM_s007
                                 + 0.995441 w_k1_INTC_s007
                                 + 1.27363 w_k1_KO_s007 + 1.15926 w_k1_MAT_s007
                                 + 1.15956 w_k1_MSFT_s007
                                 + 1.14691 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.961863 w_k1_ADM_s007
                                 + 1.00287 w_k1_AXP_s007
                                 + 0.939606 w_k1_DIS_s007
                                 + 0.988732 w_k1_FDX_s007
                                 + 0.745379 w_k1_IBM_s007
                                 + 0.816696 w_k1_INTC_s007
                                 + 0.897152 w_k1_KO_s007
                                 + 1.01156 w_k1_MAT_s007
                                 + 0.94312 w_k1_MSFT_s007
                                 + 0.972695 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.813863 w_k1_ADM_s007
                                 + 0.913038 w_k1_AXP_s007
                                 + 0.752263 w_k1_DIS_s007
                                 + 0.778719 w_k1_FDX_s007
                                 + 0.92768 w_k1_IBM_s007
                                 + 0.728584 w_k1_INTC_s007
                                 + 0.877779 w_k1_KO_s007
                                 + 1.11475 w_k1_MAT_s007
                                 + 0.78924 w_k1_MSFT_s007
                                 + 0.919627 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.19781 w_k1_ADM_s007
                                 + 1.28521 w_k1_AXP_s007
                                 + 0.967328 w_k1_DIS_s007
                                 + 0.900646 w_k1_FDX_s007
                                 + 1.37227 w_k1_IBM_s007
                                 + 0.828497 w_k1_INTC_s007
                                 + 1.20176 w_k1_KO_s007 + 1.80182 w_k1_MAT_s007
                                 + 0.928219 w_k1_MSFT_s007
                                 + 0.998347 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.547202 w_k1_ADM_s010
                                 + 0.529615 w_k1_AXP_s010
                                 + 0.618954 w_k1_DIS_s010
                                 + 0.696256 w_k1_FDX_s010
                                 + 0.750619 w_k1_IBM_s010
                                 + 0.769854 w_k1_INTC_s010
                                 + 0.756743 w_k1_KO_s010
                                 + 0.810564 w_k1_MAT_s010
                                 + 0.702959 w_k1_MSFT_s010
                                 + 0.797902 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.07635 w_k1_ADM_s010
                                 + 0.768182 w_k1_AXP_s010
                                 + 1.26136 w_k1_DIS_s010
                                 + 1.06874 w_k1_FDX_s010
                                 + 1.02883 w_k1_IBM_s010
                                 + 1.01957 w_k1_INTC_s010
                                 + 1.31205 w_k1_KO_s010 + 1.14089 w_k1_MAT_s010
                                 + 1.19038 w_k1_MSFT_s010
                                 + 1.04335 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.22834 w_k1_ADM_s010
                                 + 1.12917 w_k1_AXP_s010
                                 + 1.44431 w_k1_DIS_s010
                                 + 1.32994 w_k1_FDX_s010
                                 + 1.00295 w_k1_IBM_s010
                                 + 1.22585 w_k1_INTC_s010
                                 + 1.22738 w_k1_KO_s010 + 1.24606 w_k1_MAT_s010
                                 + 1.24417 w_k1_MSFT_s010
                                 + 1.15714 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.03256 w_k1_ADM_s010
                                 + 0.890107 w_k1_AXP_s010
                                 + 0.906533 w_k1_DIS_s010
                                 + 1.01558 w_k1_FDX_s010
                                 + 0.598861 w_k1_IBM_s010
                                 + 1.00977 w_k1_INTC_s010
                                 + 0.856766 w_k1_KO_s010
                                 + 0.793766 w_k1_MAT_s010
                                 + 1.06653 w_k1_MSFT_s010
                                 + 0.907272 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.843317 w_k1_ADM_s010
                                 + 0.864521 w_k1_AXP_s010
                                 + 1.00483 w_k1_DIS_s010
                                 + 0.931496 w_k1_FDX_s010
                                 + 1.24184 w_k1_IBM_s010
                                 + 0.902124 w_k1_INTC_s010
                                 + 1.22552 w_k1_KO_s010 + 1.13881 w_k1_MAT_s010
                                 + 0.916239 w_k1_MSFT_s010
                                 + 1.12112 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.831719 w_k1_ADM_s010
                                 + 1.0591 w_k1_AXP_s010 + 0.90679 w_k1_DIS_s010
                                 + 0.890436 w_k1_FDX_s010
                                 + 1.16099 w_k1_IBM_s010
                                 + 0.878513 w_k1_INTC_s010
                                 + 1.0393 w_k1_KO_s010 + 1.23807 w_k1_MAT_s010
                                 + 0.841427 w_k1_MSFT_s010
                                 + 1.00762 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.834698 w_k1_ADM_s010
                                 + 0.739553 w_k1_AXP_s010
                                 + 0.932292 w_k1_DIS_s010
                                 + 0.907776 w_k1_FDX_s010
                                 + 0.771596 w_k1_IBM_s010
                                 + 0.849257 w_k1_INTC_s010
                                 + 0.909193 w_k1_KO_s010
                                 + 0.970043 w_k1_MAT_s010
                                 + 1.01924 w_k1_MSFT_s010
                                 + 0.953692 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.933121 w_k1_ADM_s010
                                 + 1.12852 w_k1_AXP_s010
                                 + 1.43353 w_k1_DIS_s010
                                 + 1.02315 w_k1_FDX_s010
                                 + 1.67763 w_k1_IBM_s010
                                 + 1.03194 w_k1_INTC_s010
                                 + 1.76333 w_k1_KO_s010 + 1.76304 w_k1_MAT_s010
                                 + 1.07219 w_k1_MSFT_s010
                                 + 1.10346 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.960704 w_k1_ADM_s011
                                 + 0.874258 w_k1_AXP_s011
                                 + 0.90207 w_k1_DIS_s011
                                 + 0.838215 w_k1_FDX_s011
                                 + 1.16394 w_k1_IBM_s011
                                 + 1.04854 w_k1_INTC_s011
                                 + 1.20438 w_k1_KO_s011 + 1.1262 w_k1_MAT_s011
                                 + 1.0418 w_k1_MSFT_s011
                                 + 0.975267 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.23493 w_k1_ADM_s011
                                 + 1.00696 w_k1_AXP_s011
                                 + 0.777045 w_k1_DIS_s011
                                 + 0.958459 w_k1_FDX_s011
                                 + 0.99289 w_k1_IBM_s011
                                 + 0.91685 w_k1_INTC_s011
                                 + 0.94553 w_k1_KO_s011
                                 + 0.791207 w_k1_MAT_s011
                                 + 0.895869 w_k1_MSFT_s011
                                 + 0.950334 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.18729 w_k1_ADM_s011
                                 + 0.96613 w_k1_AXP_s011
                                 + 1.10274 w_k1_DIS_s011
                                 + 1.04433 w_k1_FDX_s011
                                 + 0.894031 w_k1_IBM_s011
                                 + 1.02252 w_k1_INTC_s011
                                 + 1.06486 w_k1_KO_s011
                                 + 0.841547 w_k1_MAT_s011
                                 + 0.982948 w_k1_MSFT_s011
                                 + 0.988383 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.40217 w_k1_ADM_s011
                                 + 1.25734 w_k1_AXP_s011
                                 + 1.26964 w_k1_DIS_s011
                                 + 1.25295 w_k1_FDX_s011
                                 + 1.11735 w_k1_IBM_s011
                                 + 1.12751 w_k1_INTC_s011
                                 + 1.02617 w_k1_KO_s011 + 1.17565 w_k1_MAT_s011
                                 + 1.26387 w_k1_MSFT_s011
                                 + 1.05278 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.08232 w_k1_ADM_s011
                                 + 1.19575 w_k1_AXP_s011
                                 + 1.21072 w_k1_DIS_s011
                                 + 1.08445 w_k1_FDX_s011
                                 + 1.60318 w_k1_IBM_s011
                                 + 1.40303 w_k1_INTC_s011
                                 + 1.37634 w_k1_KO_s011 + 1.51455 w_k1_MAT_s011
                                 + 1.22896 w_k1_MSFT_s011
                                 + 1.22295 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.21365 w_k1_ADM_s011
                                 + 1.37603 w_k1_AXP_s011 + 1.5574 w_k1_DIS_s011
                                 + 1.33707 w_k1_FDX_s011
                                 + 1.41011 w_k1_IBM_s011
                                 + 1.55408 w_k1_INTC_s011
                                 + 1.44728 w_k1_KO_s011 + 1.19177 w_k1_MAT_s011
                                 + 1.26885 w_k1_MSFT_s011
                                 + 1.37462 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.34048 w_k1_ADM_s011
                                 + 1.19103 w_k1_AXP_s011
                                 + 1.04531 w_k1_DIS_s011
                                 + 1.26948 w_k1_FDX_s011
                                 + 0.798337 w_k1_IBM_s011
                                 + 1.31812 w_k1_INTC_s011
                                 + 1.03779 w_k1_KO_s011 + 1.02428 w_k1_MAT_s011
                                 + 1.28932 w_k1_MSFT_s011
                                 + 1.09702 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.12858 w_k1_ADM_s011
                                 + 0.822469 w_k1_AXP_s011
                                 + 0.788656 w_k1_DIS_s011
                                 + 0.886263 w_k1_FDX_s011
                                 + 0.688905 w_k1_IBM_s011
                                 + 0.97539 w_k1_INTC_s011
                                 + 0.863515 w_k1_KO_s011
                                 + 0.661684 w_k1_MAT_s011
                                 + 0.961774 w_k1_MSFT_s011
                                 + 0.908537 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.42274 w_k1_ADM_s012
                                 + 1.61489 w_k1_AXP_s012
                                 + 1.40263 w_k1_DIS_s012
                                 + 1.43695 w_k1_FDX_s012
                                 + 0.824926 w_k1_IBM_s012
                                 + 1.12336 w_k1_INTC_s012
                                 + 1.21648 w_k1_KO_s012 + 1.02193 w_k1_MAT_s012
                                 + 1.14778 w_k1_MSFT_s012
                                 + 1.07585 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.14705 w_k1_ADM_s012
                                 + 1.32341 w_k1_AXP_s012
                                 + 1.05515 w_k1_DIS_s012
                                 + 1.20837 w_k1_FDX_s012
                                 + 0.852689 w_k1_IBM_s012
                                 + 1.25247 w_k1_INTC_s012
                                 + 0.897065 w_k1_KO_s012
                                 + 0.817095 w_k1_MAT_s012
                                 + 1.27635 w_k1_MSFT_s012
                                 + 1.04438 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.904558 w_k1_ADM_s012
                                 + 1.14227 w_k1_AXP_s012
                                 + 1.31022 w_k1_DIS_s012
                                 + 1.11897 w_k1_FDX_s012
                                 + 0.984351 w_k1_IBM_s012
                                 + 1.2097 w_k1_INTC_s012 + 1.17655 w_k1_KO_s012
                                 + 1.08393 w_k1_MAT_s012
                                 + 1.11848 w_k1_MSFT_s012
                                 + 1.18124 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.735398 w_k1_ADM_s012
                                 + 1.0106 w_k1_AXP_s012
                                 + 0.786685 w_k1_DIS_s012
                                 + 0.94699 w_k1_FDX_s012
                                 + 0.679147 w_k1_IBM_s012
                                 + 0.932246 w_k1_INTC_s012
                                 + 0.909862 w_k1_KO_s012
                                 + 0.762567 w_k1_MAT_s012
                                 + 0.927344 w_k1_MSFT_s012
                                 + 0.925515 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.652323 w_k1_ADM_s012
                                 + 0.738241 w_k1_AXP_s012
                                 + 0.650232 w_k1_DIS_s012
                                 + 0.663661 w_k1_FDX_s012
                                 + 0.859172 w_k1_IBM_s012
                                 + 0.858252 w_k1_INTC_s012
                                 + 0.882919 w_k1_KO_s012
                                 + 0.743304 w_k1_MAT_s012
                                 + 0.810442 w_k1_MSFT_s012
                                 + 0.905242 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.721709 w_k1_ADM_s012
                                 + 0.856349 w_k1_AXP_s012
                                 + 0.979183 w_k1_DIS_s012
                                 + 0.802135 w_k1_FDX_s012
                                 + 1.18079 w_k1_IBM_s012
                                 + 0.950092 w_k1_INTC_s012
                                 + 1.0636 w_k1_KO_s012 + 1.05298 w_k1_MAT_s012
                                 + 0.960085 w_k1_MSFT_s012
                                 + 1.09901 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.989039 w_k1_ADM_s012
                                 + 0.73252 w_k1_AXP_s012
                                 + 0.961036 w_k1_DIS_s012
                                 + 0.966204 w_k1_FDX_s012
                                 + 0.74581 w_k1_IBM_s012
                                 + 1.06532 w_k1_INTC_s012
                                 + 1.00094 w_k1_KO_s012
                                 + 0.558319 w_k1_MAT_s012
                                 + 1.03203 w_k1_MSFT_s012
                                 + 0.932238 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.01135 w_k1_ADM_s012
                                 + 1.27947 w_k1_AXP_s012 + 1.4314 w_k1_DIS_s012
                                 + 1.14006 w_k1_FDX_s012
                                 + 1.59299 w_k1_IBM_s012
                                 + 1.22325 w_k1_INTC_s012
                                 + 1.31989 w_k1_KO_s012 + 1.56904 w_k1_MAT_s012
                                 + 1.21263 w_k1_MSFT_s012
                                 + 1.08226 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.972371 w_k1_ADM_s013
                                 + 0.654948 w_k1_AXP_s013
                                 + 0.675959 w_k1_DIS_s013
                                 + 0.766021 w_k1_FDX_s013
                                 + 0.747932 w_k1_IBM_s013
                                 + 0.851812 w_k1_INTC_s013
                                 + 0.786979 w_k1_KO_s013
                                 + 0.52214 w_k1_MAT_s013
                                 + 0.88789 w_k1_MSFT_s013
                                 + 0.827386 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.39773 w_k1_ADM_s013
                                 + 0.98367 w_k1_AXP_s013
                                 + 0.952335 w_k1_DIS_s013
                                 + 1.04448 w_k1_FDX_s013
                                 + 0.710118 w_k1_IBM_s013
                                 + 0.888793 w_k1_INTC_s013
                                 + 0.945543 w_k1_KO_s013
                                 + 0.917227 w_k1_MAT_s013
                                 + 1.01425 w_k1_MSFT_s013
                                 + 1.14202 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.28152 w_k1_ADM_s013
                                 + 1.16236 w_k1_AXP_s013
                                 + 1.23506 w_k1_DIS_s013 + 1.0016 w_k1_FDX_s013
                                 + 1.62462 w_k1_IBM_s013
                                 + 1.22422 w_k1_INTC_s013
                                 + 1.37608 w_k1_KO_s013 + 1.51734 w_k1_MAT_s013
                                 + 1.11774 w_k1_MSFT_s013
                                 + 1.17313 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.3196 w_k1_ADM_s013
                                 + 1.40575 w_k1_AXP_s013
                                 + 1.52294 w_k1_DIS_s013 + 1.3516 w_k1_FDX_s013
                                 + 0.98342 w_k1_IBM_s013
                                 + 1.48525 w_k1_INTC_s013 + 1.0993 w_k1_KO_s013
                                 + 0.974552 w_k1_MAT_s013
                                 + 1.25217 w_k1_MSFT_s013
                                 + 1.11593 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15869 w_k1_ADM_s013
                                 + 1.00104 w_k1_AXP_s013
                                 + 1.34637 w_k1_DIS_s013
                                 + 1.14227 w_k1_FDX_s013
                                 + 1.24668 w_k1_IBM_s013
                                 + 0.962212 w_k1_INTC_s013
                                 + 1.26666 w_k1_KO_s013 + 1.52941 w_k1_MAT_s013
                                 + 1.2364 w_k1_MSFT_s013 + 1.27539 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.952814 w_k1_ADM_s013
                                 + 1.00078 w_k1_AXP_s013
                                 + 0.880847 w_k1_DIS_s013
                                 + 0.937135 w_k1_FDX_s013
                                 + 1.06281 w_k1_IBM_s013
                                 + 1.00056 w_k1_INTC_s013
                                 + 0.945826 w_k1_KO_s013
                                 + 1.16402 w_k1_MAT_s013
                                 + 0.968896 w_k1_MSFT_s013
                                 + 0.953379 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.60795 w_k1_ADM_s013
                                 + 1.52502 w_k1_AXP_s013
                                 + 1.26357 w_k1_DIS_s013
                                 + 1.28384 w_k1_FDX_s013
                                 + 1.42052 w_k1_IBM_s013
                                 + 1.20645 w_k1_INTC_s013
                                 + 1.04283 w_k1_KO_s013 + 1.35376 w_k1_MAT_s013
                                 + 1.16149 w_k1_MSFT_s013 + 1.0491 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.03282 w_k1_ADM_s013
                                 + 0.658399 w_k1_AXP_s013
                                 + 0.963403 w_k1_DIS_s013
                                 + 0.948343 w_k1_FDX_s013
                                 + 0.935657 w_k1_IBM_s013
                                 + 1.04403 w_k1_INTC_s013 + 1.0026 w_k1_KO_s013
                                 + 0.79657 w_k1_MAT_s013
                                 + 1.05797 w_k1_MSFT_s013
                                 + 0.914014 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.867174 w_k1_ADM_s014
                                 + 1.48685 w_k1_AXP_s014
                                 + 1.00347 w_k1_DIS_s014
                                 + 1.06068 w_k1_FDX_s014
                                 + 0.970604 w_k1_IBM_s014
                                 + 1.02614 w_k1_INTC_s014
                                 + 1.05623 w_k1_KO_s014 + 1.53032 w_k1_MAT_s014
                                 + 0.871307 w_k1_MSFT_s014
                                 + 1.14928 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.777249 w_k1_ADM_s014
                                 + 0.819375 w_k1_AXP_s014
                                 + 0.849052 w_k1_DIS_s014
                                 + 0.844089 w_k1_FDX_s014
                                 + 0.873279 w_k1_IBM_s014
                                 + 0.899322 w_k1_INTC_s014
                                 + 0.847643 w_k1_KO_s014
                                 + 0.512958 w_k1_MAT_s014
                                 + 0.837075 w_k1_MSFT_s014
                                 + 0.876938 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.839142 w_k1_ADM_s014
                                 + 1.48539 w_k1_AXP_s014
                                 + 1.36205 w_k1_DIS_s014
                                 + 1.28937 w_k1_FDX_s014 + 1.3455 w_k1_IBM_s014
                                 + 1.33578 w_k1_INTC_s014
                                 + 1.43743 w_k1_KO_s014 + 1.69679 w_k1_MAT_s014
                                 + 1.24581 w_k1_MSFT_s014
                                 + 1.36857 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.768236 w_k1_ADM_s014
                                 + 1.00405 w_k1_AXP_s014
                                 + 0.731972 w_k1_DIS_s014
                                 + 0.91701 w_k1_FDX_s014
                                 + 0.808655 w_k1_IBM_s014
                                 + 1.17338 w_k1_INTC_s014
                                 + 0.897807 w_k1_KO_s014
                                 + 0.753701 w_k1_MAT_s014
                                 + 0.90876 w_k1_MSFT_s014
                                 + 0.899038 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.761256 w_k1_ADM_s014
                                 + 1.25024 w_k1_AXP_s014
                                 + 1.48492 w_k1_DIS_s014
                                 + 1.18959 w_k1_FDX_s014
                                 + 1.33545 w_k1_IBM_s014
                                 + 1.29983 w_k1_INTC_s014
                                 + 1.42615 w_k1_KO_s014 + 1.46037 w_k1_MAT_s014
                                 + 1.29646 w_k1_MSFT_s014
                                 + 1.36803 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.796529 w_k1_ADM_s014
                                 + 0.93281 w_k1_AXP_s014
                                 + 0.89636 w_k1_DIS_s014
                                 + 0.938367 w_k1_FDX_s014
                                 + 1.24386 w_k1_IBM_s014
                                 + 1.08203 w_k1_INTC_s014 + 1.1846 w_k1_KO_s014
                                 + 1.05662 w_k1_MAT_s014
                                 + 0.787536 w_k1_MSFT_s014
                                 + 0.927411 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.25893 w_k1_ADM_s014
                                 + 1.2064 w_k1_AXP_s014 + 1.18537 w_k1_DIS_s014
                                 + 1.10803 w_k1_FDX_s014
                                 + 0.947167 w_k1_IBM_s014
                                 + 1.12997 w_k1_INTC_s014 + 1.2351 w_k1_KO_s014
                                 + 1.05349 w_k1_MAT_s014
                                 + 1.03782 w_k1_MSFT_s014
                                 + 0.987626 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.654783 w_k1_ADM_s014
                                 + 0.816239 w_k1_AXP_s014
                                 + 0.833635 w_k1_DIS_s014
                                 + 0.954393 w_k1_FDX_s014
                                 + 0.656718 w_k1_IBM_s014
                                 + 1.02807 w_k1_INTC_s014
                                 + 0.836758 w_k1_KO_s014
                                 + 0.728849 w_k1_MAT_s014
                                 + 0.933846 w_k1_MSFT_s014
                                 + 0.965729 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.03736 w_k1_ADM_s015
                                 + 1.04817 w_k1_AXP_s015
                                 + 0.965245 w_k1_DIS_s015
                                 + 1.18742 w_k1_FDX_s015
                                 + 0.810262 w_k1_IBM_s015
                                 + 0.911872 w_k1_INTC_s015
                                 + 0.778454 w_k1_KO_s015
                                 + 0.712607 w_k1_MAT_s015
                                 + 1.01466 w_k1_MSFT_s015
                                 + 0.928683 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.837282 w_k1_ADM_s015
                                 + 0.756613 w_k1_AXP_s015
                                 + 0.79348 w_k1_DIS_s015
                                 + 0.748904 w_k1_FDX_s015
                                 + 1.0558 w_k1_IBM_s015
                                 + 0.745516 w_k1_INTC_s015
                                 + 0.851636 w_k1_KO_s015
                                 + 0.845147 w_k1_MAT_s015
                                 + 0.819888 w_k1_MSFT_s015
                                 + 0.781013 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.02749 w_k1_ADM_s015
                                 + 0.765595 w_k1_AXP_s015
                                 + 1.11936 w_k1_DIS_s015
                                 + 0.970705 w_k1_FDX_s015
                                 + 0.817638 w_k1_IBM_s015
                                 + 0.893503 w_k1_INTC_s015
                                 + 0.854927 w_k1_KO_s015
                                 + 0.573159 w_k1_MAT_s015
                                 + 1.07385 w_k1_MSFT_s015
                                 + 0.812318 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.804677 w_k1_ADM_s015
                                 + 1.00963 w_k1_AXP_s015
                                 + 1.21527 w_k1_DIS_s015
                                 + 1.08789 w_k1_FDX_s015
                                 + 1.15546 w_k1_IBM_s015
                                 + 1.02851 w_k1_INTC_s015
                                 + 1.13036 w_k1_KO_s015 + 1.16308 w_k1_MAT_s015
                                 + 1.03002 w_k1_MSFT_s015
                                 + 1.01026 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.893686 w_k1_ADM_s015
                                 + 0.814815 w_k1_AXP_s015
                                 + 0.78984 w_k1_DIS_s015
                                 + 0.816376 w_k1_FDX_s015
                                 + 0.804293 w_k1_IBM_s015
                                 + 0.667585 w_k1_INTC_s015
                                 + 0.798632 w_k1_KO_s015
                                 + 0.987513 w_k1_MAT_s015
                                 + 0.791192 w_k1_MSFT_s015
                                 + 0.862675 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.7733 w_k1_ADM_s015
                                 + 1.33193 w_k1_AXP_s015 + 1.6443 w_k1_DIS_s015
                                 + 1.364 w_k1_FDX_s015 + 1.41991 w_k1_IBM_s015
                                 + 1.17698 w_k1_INTC_s015
                                 + 1.45883 w_k1_KO_s015 + 1.26334 w_k1_MAT_s015
                                 + 1.38511 w_k1_MSFT_s015
                                 + 1.25606 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.03123 w_k1_ADM_s015
                                 + 1.39923 w_k1_AXP_s015
                                 + 0.962576 w_k1_DIS_s015
                                 + 1.06427 w_k1_FDX_s015
                                 + 1.12291 w_k1_IBM_s015
                                 + 0.878915 w_k1_INTC_s015
                                 + 1.00348 w_k1_KO_s015 + 1.46619 w_k1_MAT_s015
                                 + 0.949101 w_k1_MSFT_s015
                                 + 0.972397 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.7328 w_k1_ADM_s015
                                 + 0.588213 w_k1_AXP_s015
                                 + 0.610985 w_k1_DIS_s015
                                 + 0.724487 w_k1_FDX_s015
                                 + 0.691613 w_k1_IBM_s015
                                 + 0.790042 w_k1_INTC_s015
                                 + 0.617683 w_k1_KO_s015
                                 + 0.409539 w_k1_MAT_s015
                                 + 0.737045 w_k1_MSFT_s015
                                 + 0.704219 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.916293 w_k1_ADM_s016
                                 + 0.759974 w_k1_AXP_s016
                                 + 0.897754 w_k1_DIS_s016
                                 + 0.977456 w_k1_FDX_s016
                                 + 0.70806 w_k1_IBM_s016
                                 + 1.03428 w_k1_INTC_s016
                                 + 0.807223 w_k1_KO_s016
                                 + 0.597161 w_k1_MAT_s016
                                 + 0.867895 w_k1_MSFT_s016
                                 + 0.880834 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.832837 w_k1_ADM_s016
                                 + 0.718353 w_k1_AXP_s016
                                 + 0.645714 w_k1_DIS_s016
                                 + 0.799247 w_k1_FDX_s016
                                 + 0.795571 w_k1_IBM_s016
                                 + 0.935061 w_k1_INTC_s016
                                 + 0.754522 w_k1_KO_s016
                                 + 0.623208 w_k1_MAT_s016
                                 + 0.770421 w_k1_MSFT_s016
                                 + 0.80939 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.29161 w_k1_ADM_s016
                                 + 1.11142 w_k1_AXP_s016
                                 + 1.07475 w_k1_DIS_s016
                                 + 1.05276 w_k1_FDX_s016
                                 + 1.09236 w_k1_IBM_s016
                                 + 0.949629 w_k1_INTC_s016
                                 + 1.05424 w_k1_KO_s016 + 1.04201 w_k1_MAT_s016
                                 + 0.861115 w_k1_MSFT_s016
                                 + 0.943675 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.3465 w_k1_ADM_s016
                                 + 1.21139 w_k1_AXP_s016
                                 + 1.55956 w_k1_DIS_s016
                                 + 1.33417 w_k1_FDX_s016
                                 + 0.994196 w_k1_IBM_s016
                                 + 1.39501 w_k1_INTC_s016
                                 + 1.21223 w_k1_KO_s016
                                 + 0.809209 w_k1_MAT_s016
                                 + 1.1845 w_k1_MSFT_s016 + 1.08211 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.06827 w_k1_ADM_s016
                                 + 1.74215 w_k1_AXP_s016
                                 + 1.45469 w_k1_DIS_s016
                                 + 1.27971 w_k1_FDX_s016
                                 + 1.39906 w_k1_IBM_s016
                                 + 1.15799 w_k1_INTC_s016
                                 + 1.38509 w_k1_KO_s016 + 1.49839 w_k1_MAT_s016
                                 + 1.18927 w_k1_MSFT_s016
                                 + 1.20293 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.05104 w_k1_ADM_s016
                                 + 1.0161 w_k1_AXP_s016 + 1.12542 w_k1_DIS_s016
                                 + 0.968097 w_k1_FDX_s016
                                 + 1.47156 w_k1_IBM_s016
                                 + 0.924083 w_k1_INTC_s016
                                 + 1.31338 w_k1_KO_s016 + 1.42148 w_k1_MAT_s016
                                 + 1.06002 w_k1_MSFT_s016
                                 + 1.04589 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.05115 w_k1_ADM_s016
                                 + 1.06569 w_k1_AXP_s016
                                 + 0.981705 w_k1_DIS_s016
                                 + 1.03533 w_k1_FDX_s016
                                 + 0.887299 w_k1_IBM_s016
                                 + 1.05475 w_k1_INTC_s016
                                 + 0.985438 w_k1_KO_s016
                                 + 0.967436 w_k1_MAT_s016
                                 + 0.984988 w_k1_MSFT_s016
                                 + 0.94449 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.777561 w_k1_ADM_s016
                                 + 0.762688 w_k1_AXP_s016
                                 + 0.849016 w_k1_DIS_s016
                                 + 0.821737 w_k1_FDX_s016
                                 + 1.02036 w_k1_IBM_s016
                                 + 0.777272 w_k1_INTC_s016
                                 + 0.919695 w_k1_KO_s016
                                 + 0.944022 w_k1_MAT_s016
                                 + 0.853116 w_k1_MSFT_s016
                                 + 0.883953 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.23436 w_k1_ADM_s017
                                 + 1.22057 w_k1_AXP_s017
                                 + 1.43741 w_k1_DIS_s017
                                 + 1.28391 w_k1_FDX_s017
                                 + 0.842346 w_k1_IBM_s017
                                 + 1.34692 w_k1_INTC_s017
                                 + 1.22124 w_k1_KO_s017 + 1.07575 w_k1_MAT_s017
                                 + 1.48609 w_k1_MSFT_s017
                                 + 1.23612 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.987893 w_k1_ADM_s017
                                 + 1.01991 w_k1_AXP_s017
                                 + 0.973658 w_k1_DIS_s017
                                 + 1.00013 w_k1_FDX_s017
                                 + 0.913693 w_k1_IBM_s017
                                 + 1.22688 w_k1_INTC_s017
                                 + 0.81648 w_k1_KO_s017 + 1.10064 w_k1_MAT_s017
                                 + 1.12887 w_k1_MSFT_s017
                                 + 0.895069 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.876547 w_k1_ADM_s017
                                 + 0.568394 w_k1_AXP_s017
                                 + 0.562422 w_k1_DIS_s017
                                 + 0.701881 w_k1_FDX_s017
                                 + 0.802604 w_k1_IBM_s017
                                 + 0.772466 w_k1_INTC_s017
                                 + 0.684717 w_k1_KO_s017
                                 + 0.644602 w_k1_MAT_s017
                                 + 0.739463 w_k1_MSFT_s017
                                 + 0.850065 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.987008 w_k1_ADM_s017
                                 + 0.941067 w_k1_AXP_s017
                                 + 0.718983 w_k1_DIS_s017
                                 + 0.876434 w_k1_FDX_s017
                                 + 1.10957 w_k1_IBM_s017
                                 + 1.0157 w_k1_INTC_s017
                                 + 0.788345 w_k1_KO_s017
                                 + 1.10992 w_k1_MAT_s017
                                 + 0.908361 w_k1_MSFT_s017
                                 + 0.989898 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.25984 w_k1_ADM_s017
                                 + 1.21109 w_k1_AXP_s017 + 1.2186 w_k1_DIS_s017
                                 + 1.08944 w_k1_FDX_s017
                                 + 1.55455 w_k1_IBM_s017
                                 + 1.15887 w_k1_INTC_s017
                                 + 1.24584 w_k1_KO_s017 + 1.77331 w_k1_MAT_s017
                                 + 1.12564 w_k1_MSFT_s017
                                 + 1.31291 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.3721 w_k1_ADM_s017
                                 + 1.4918 w_k1_AXP_s017 + 1.3089 w_k1_DIS_s017
                                 + 1.22555 w_k1_FDX_s017
                                 + 1.15054 w_k1_IBM_s017
                                 + 1.36287 w_k1_INTC_s017
                                 + 1.08097 w_k1_KO_s017 + 1.25578 w_k1_MAT_s017
                                 + 1.21868 w_k1_MSFT_s017
                                 + 1.20996 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.09214 w_k1_ADM_s017
                                 + 1.00875 w_k1_AXP_s017
                                 + 0.735533 w_k1_DIS_s017
                                 + 0.904638 w_k1_FDX_s017
                                 + 0.70668 w_k1_IBM_s017
                                 + 0.968317 w_k1_INTC_s017
                                 + 0.725259 w_k1_KO_s017
                                 + 0.840328 w_k1_MAT_s017
                                 + 0.904778 w_k1_MSFT_s017
                                 + 1.04931 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.920598 w_k1_ADM_s017
                                 + 0.787678 w_k1_AXP_s017
                                 + 0.842478 w_k1_DIS_s017
                                 + 0.919905 w_k1_FDX_s017
                                 + 0.67749 w_k1_IBM_s017
                                 + 0.873934 w_k1_INTC_s017
                                 + 0.651254 w_k1_KO_s017
                                 + 0.731882 w_k1_MAT_s017
                                 + 0.884206 w_k1_MSFT_s017
                                 + 0.936268 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.44081 w_k1_ADM_s100
                                 + 1.13463 w_k1_AXP_s100
                                 + 1.16908 w_k1_DIS_s100
                                 + 1.29044 w_k1_FDX_s100
                                 + 0.688992 w_k1_IBM_s100
                                 + 1.15123 w_k1_INTC_s100
                                 + 0.919198 w_k1_KO_s100
                                 + 0.687899 w_k1_MAT_s100
                                 + 0.983395 w_k1_MSFT_s100
                                 + 1.04411 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.848023 w_k1_ADM_s100
                                 + 0.58316 w_k1_AXP_s100
                                 + 0.570438 w_k1_DIS_s100
                                 + 0.698965 w_k1_FDX_s100
                                 + 1.11124 w_k1_IBM_s100
                                 + 0.777893 w_k1_INTC_s100
                                 + 0.847478 w_k1_KO_s100
                                 + 0.890921 w_k1_MAT_s100
                                 + 0.78508 w_k1_MSFT_s100
                                 + 0.801636 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.69038 w_k1_ADM_s100
                                 + 0.592171 w_k1_AXP_s100
                                 + 0.614825 w_k1_DIS_s100
                                 + 0.725131 w_k1_FDX_s100
                                 + 0.791696 w_k1_IBM_s100
                                 + 0.855773 w_k1_INTC_s100
                                 + 0.787631 w_k1_KO_s100
                                 + 0.752804 w_k1_MAT_s100
                                 + 0.665058 w_k1_MSFT_s100
                                 + 0.806716 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.923004 w_k1_ADM_s100
                                 + 0.890581 w_k1_AXP_s100
                                 + 1.16774 w_k1_DIS_s100
                                 + 0.956958 w_k1_FDX_s100
                                 + 0.877046 w_k1_IBM_s100
                                 + 0.919524 w_k1_INTC_s100
                                 + 1.11348 w_k1_KO_s100
                                 + 0.845296 w_k1_MAT_s100
                                 + 1.07971 w_k1_MSFT_s100
                                 + 0.986222 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.982867 w_k1_ADM_s100
                                 + 0.956713 w_k1_AXP_s100
                                 + 0.808677 w_k1_DIS_s100
                                 + 0.900481 w_k1_FDX_s100
                                 + 1.02534 w_k1_IBM_s100
                                 + 0.942213 w_k1_INTC_s100
                                 + 1.02574 w_k1_KO_s100 + 1.39312 w_k1_MAT_s100
                                 + 0.98093 w_k1_MSFT_s100
                                 + 0.998013 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.942384 w_k1_ADM_s100
                                 + 1.00977 w_k1_AXP_s100
                                 + 1.05957 w_k1_DIS_s100
                                 + 1.00852 w_k1_FDX_s100
                                 + 1.42387 w_k1_IBM_s100
                                 + 1.11009 w_k1_INTC_s100
                                 + 1.42582 w_k1_KO_s100 + 1.50419 w_k1_MAT_s100
                                 + 0.909301 w_k1_MSFT_s100
                                 + 1.08029 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.29048 w_k1_ADM_s100
                                 + 1.12484 w_k1_AXP_s100
                                 + 1.32309 w_k1_DIS_s100 + 1.1812 w_k1_FDX_s100
                                 + 0.993119 w_k1_IBM_s100
                                 + 0.924622 w_k1_INTC_s100
                                 + 1.13593 w_k1_KO_s100 + 1.35374 w_k1_MAT_s100
                                 + 1.15494 w_k1_MSFT_s100
                                 + 1.17216 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.82414 w_k1_ADM_s100
                                 + 0.845911 w_k1_AXP_s100
                                 + 0.794829 w_k1_DIS_s100
                                 + 0.879698 w_k1_FDX_s100
                                 + 0.791605 w_k1_IBM_s100
                                 + 0.885237 w_k1_INTC_s100
                                 + 0.923724 w_k1_KO_s100
                                 + 0.831149 w_k1_MAT_s100
                                 + 0.833561 w_k1_MSFT_s100
                                 + 0.997443 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.813221 w_k1_ADM_s101
                                 + 0.818306 w_k1_AXP_s101
                                 + 0.869311 w_k1_DIS_s101
                                 + 0.939834 w_k1_FDX_s101
                                 + 0.705075 w_k1_IBM_s101
                                 + 0.887781 w_k1_INTC_s101
                                 + 0.741467 w_k1_KO_s101
                                 + 0.738562 w_k1_MAT_s101
                                 + 0.915372 w_k1_MSFT_s101
                                 + 0.96765 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.771242 w_k1_ADM_s101
                                 + 0.745975 w_k1_AXP_s101
                                 + 0.937096 w_k1_DIS_s101
                                 + 0.893243 w_k1_FDX_s101
                                 + 1.18221 w_k1_IBM_s101
                                 + 0.915936 w_k1_INTC_s101
                                 + 1.02638 w_k1_KO_s101 + 1.10775 w_k1_MAT_s101
                                 + 0.892582 w_k1_MSFT_s101
                                 + 1.00503 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.782319 w_k1_ADM_s101
                                 + 1.07811 w_k1_AXP_s101
                                 + 1.20554 w_k1_DIS_s101
                                 + 0.954731 w_k1_FDX_s101
                                 + 1.71987 w_k1_IBM_s101
                                 + 0.984137 w_k1_INTC_s101
                                 + 1.17259 w_k1_KO_s101 + 1.74161 w_k1_MAT_s101
                                 + 1.01549 w_k1_MSFT_s101
                                 + 1.11115 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.773413 w_k1_ADM_s101
                                 + 0.61669 w_k1_AXP_s101
                                 + 0.575035 w_k1_DIS_s101
                                 + 0.72006 w_k1_FDX_s101
                                 + 0.792109 w_k1_IBM_s101
                                 + 0.795537 w_k1_INTC_s101
                                 + 0.700946 w_k1_KO_s101
                                 + 0.589057 w_k1_MAT_s101
                                 + 0.763992 w_k1_MSFT_s101
                                 + 0.825905 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.900548 w_k1_ADM_s101
                                 + 1.14731 w_k1_AXP_s101
                                 + 0.79616 w_k1_DIS_s101
                                 + 0.938194 w_k1_FDX_s101
                                 + 0.980509 w_k1_IBM_s101
                                 + 0.917494 w_k1_INTC_s101
                                 + 0.830171 w_k1_KO_s101
                                 + 1.03044 w_k1_MAT_s101
                                 + 0.92645 w_k1_MSFT_s101
                                 + 0.999022 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.954906 w_k1_ADM_s101
                                 + 1.04053 w_k1_AXP_s101
                                 + 1.33801 w_k1_DIS_s101
                                 + 1.09563 w_k1_FDX_s101
                                 + 1.16544 w_k1_IBM_s101
                                 + 1.05342 w_k1_INTC_s101
                                 + 1.17939 w_k1_KO_s101 + 1.10988 w_k1_MAT_s101
                                 + 1.24367 w_k1_MSFT_s101
                                 + 1.17216 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.02868 w_k1_ADM_s101
                                 + 1.19767 w_k1_AXP_s101
                                 + 1.19621 w_k1_DIS_s101
                                 + 1.24631 w_k1_FDX_s101
                                 + 1.07891 w_k1_IBM_s101
                                 + 1.10731 w_k1_INTC_s101
                                 + 1.04259 w_k1_KO_s101 + 1.22284 w_k1_MAT_s101
                                 + 1.17565 w_k1_MSFT_s101
                                 + 1.10407 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.799317 w_k1_ADM_s101
                                 + 0.737133 w_k1_AXP_s101
                                 + 1.0229 w_k1_DIS_s101
                                 + 0.870388 w_k1_FDX_s101
                                 + 1.40907 w_k1_IBM_s101
                                 + 0.797327 w_k1_INTC_s101
                                 + 1.15107 w_k1_KO_s101 + 0.95815 w_k1_MAT_s101
                                 + 0.948515 w_k1_MSFT_s101
                                 + 1.0166 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.03077 w_k1_ADM_s102
                                 + 0.877763 w_k1_AXP_s102
                                 + 1.01726 w_k1_DIS_s102
                                 + 1.06219 w_k1_FDX_s102
                                 + 0.687988 w_k1_IBM_s102
                                 + 0.920738 w_k1_INTC_s102
                                 + 0.906653 w_k1_KO_s102
                                 + 0.647409 w_k1_MAT_s102
                                 + 0.97781 w_k1_MSFT_s102
                                 + 0.979002 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.10248 w_k1_ADM_s102
                                 + 1.19458 w_k1_AXP_s102 + 1.0025 w_k1_DIS_s102
                                 + 1.19058 w_k1_FDX_s102
                                 + 0.514654 w_k1_IBM_s102
                                 + 1.27451 w_k1_INTC_s102
                                 + 0.822078 w_k1_KO_s102
                                 + 0.53263 w_k1_MAT_s102
                                 + 1.12481 w_k1_MSFT_s102
                                 + 0.898266 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.69043 w_k1_ADM_s102
                                 + 0.81188 w_k1_AXP_s102
                                 + 0.744222 w_k1_DIS_s102
                                 + 0.804496 w_k1_FDX_s102
                                 + 1.2057 w_k1_IBM_s102
                                 + 0.805967 w_k1_INTC_s102
                                 + 0.930746 w_k1_KO_s102
                                 + 1.45626 w_k1_MAT_s102
                                 + 0.803766 w_k1_MSFT_s102
                                 + 0.88086 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.742407 w_k1_ADM_s102
                                 + 0.735119 w_k1_AXP_s102
                                 + 0.626526 w_k1_DIS_s102
                                 + 0.777002 w_k1_FDX_s102
                                 + 0.677229 w_k1_IBM_s102
                                 + 0.807328 w_k1_INTC_s102
                                 + 0.681151 w_k1_KO_s102
                                 + 0.854204 w_k1_MAT_s102
                                 + 0.741883 w_k1_MSFT_s102
                                 + 0.741896 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.857144 w_k1_ADM_s102
                                 + 1.1039 w_k1_AXP_s102 + 1.34561 w_k1_DIS_s102
                                 + 1.04746 w_k1_FDX_s102
                                 + 1.70015 w_k1_IBM_s102
                                 + 1.26465 w_k1_INTC_s102
                                 + 1.45409 w_k1_KO_s102 + 1.49535 w_k1_MAT_s102
                                 + 1.22471 w_k1_MSFT_s102 + 1.1943 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.37426 w_k1_ADM_s102
                                 + 1.52528 w_k1_AXP_s102
                                 + 1.40347 w_k1_DIS_s102 + 1.3893 w_k1_FDX_s102
                                 + 1.0788 w_k1_IBM_s102
                                 + 1.49336 w_k1_INTC_s102
                                 + 1.29688 w_k1_KO_s102 + 1.23778 w_k1_MAT_s102
                                 + 1.30097 w_k1_MSFT_s102
                                 + 1.17996 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.96519 w_k1_ADM_s102
                                 + 1.05401 w_k1_AXP_s102
                                 + 1.18338 w_k1_DIS_s102
                                 + 1.12175 w_k1_FDX_s102
                                 + 1.08366 w_k1_IBM_s102
                                 + 1.03877 w_k1_INTC_s102
                                 + 1.09661 w_k1_KO_s102 + 1.14533 w_k1_MAT_s102
                                 + 1.06708 w_k1_MSFT_s102
                                 + 1.09548 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.828692 w_k1_ADM_s102
                                 + 1.05353 w_k1_AXP_s102
                                 + 0.981968 w_k1_DIS_s102
                                 + 1.0039 w_k1_FDX_s102
                                 + 0.932761 w_k1_IBM_s102
                                 + 1.0114 w_k1_INTC_s102
                                 + 0.993424 w_k1_KO_s102
                                 + 1.13366 w_k1_MAT_s102
                                 + 0.961568 w_k1_MSFT_s102
                                 + 0.923185 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.875585 w_k1_ADM_s103
                                 + 0.827173 w_k1_AXP_s103
                                 + 0.790461 w_k1_DIS_s103
                                 + 0.931148 w_k1_FDX_s103
                                 + 0.751568 w_k1_IBM_s103
                                 + 0.853049 w_k1_INTC_s103
                                 + 0.760309 w_k1_KO_s103
                                 + 0.780475 w_k1_MAT_s103
                                 + 0.851937 w_k1_MSFT_s103
                                 + 0.942178 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.07519 w_k1_ADM_s103
                                 + 1.22802 w_k1_AXP_s103
                                 + 1.51671 w_k1_DIS_s103
                                 + 1.16984 w_k1_FDX_s103
                                 + 1.52356 w_k1_IBM_s103
                                 + 1.03281 w_k1_INTC_s103
                                 + 1.40691 w_k1_KO_s103 + 1.73648 w_k1_MAT_s103
                                 + 1.22419 w_k1_MSFT_s103
                                 + 1.24332 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.702906 w_k1_ADM_s103
                                 + 0.880445 w_k1_AXP_s103
                                 + 0.770593 w_k1_DIS_s103
                                 + 0.786526 w_k1_FDX_s103
                                 + 1.36493 w_k1_IBM_s103
                                 + 1.00725 w_k1_INTC_s103
                                 + 1.06209 w_k1_KO_s103 + 1.26515 w_k1_MAT_s103
                                 + 0.857947 w_k1_MSFT_s103
                                 + 1.01315 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.681924 w_k1_ADM_s103
                                 + 0.740919 w_k1_AXP_s103
                                 + 0.980821 w_k1_DIS_s103
                                 + 0.971926 w_k1_FDX_s103
                                 + 0.856351 w_k1_IBM_s103
                                 + 1.01726 w_k1_INTC_s103
                                 + 0.841374 w_k1_KO_s103
                                 + 1.06638 w_k1_MAT_s103
                                 + 1.01048 w_k1_MSFT_s103
                                 + 0.884795 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.843032 w_k1_ADM_s103
                                 + 0.900951 w_k1_AXP_s103
                                 + 0.918212 w_k1_DIS_s103
                                 + 0.974616 w_k1_FDX_s103
                                 + 1.15501 w_k1_IBM_s103
                                 + 0.881608 w_k1_INTC_s103
                                 + 1.06728 w_k1_KO_s103 + 1.49448 w_k1_MAT_s103
                                 + 0.968598 w_k1_MSFT_s103
                                 + 1.05398 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.602347 w_k1_ADM_s103
                                 + 0.692503 w_k1_AXP_s103
                                 + 0.689829 w_k1_DIS_s103
                                 + 0.713716 w_k1_FDX_s103
                                 + 0.936169 w_k1_IBM_s103
                                 + 0.840283 w_k1_INTC_s103
                                 + 0.768196 w_k1_KO_s103
                                 + 1.08031 w_k1_MAT_s103
                                 + 0.789566 w_k1_MSFT_s103
                                 + 0.819152 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.11874 w_k1_ADM_s103
                                 + 1.13882 w_k1_AXP_s103
                                 + 1.20624 w_k1_DIS_s103
                                 + 1.30143 w_k1_FDX_s103
                                 + 0.824663 w_k1_IBM_s103
                                 + 1.20956 w_k1_INTC_s103
                                 + 0.861292 w_k1_KO_s103
                                 + 0.807152 w_k1_MAT_s103
                                 + 1.21035 w_k1_MSFT_s103
                                 + 1.16002 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.840256 w_k1_ADM_s103
                                 + 1.37584 w_k1_AXP_s103
                                 + 1.32405 w_k1_DIS_s103
                                 + 1.21715 w_k1_FDX_s103
                                 + 1.17119 w_k1_IBM_s103
                                 + 1.21789 w_k1_INTC_s103 + 1.0221 w_k1_KO_s103
                                 + 1.20546 w_k1_MAT_s103
                                 + 1.11795 w_k1_MSFT_s103
                                 + 1.12184 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.968942 w_k1_ADM_s104
                                 + 0.869276 w_k1_AXP_s104
                                 + 1.11153 w_k1_DIS_s104
                                 + 0.909677 w_k1_FDX_s104
                                 + 1.15696 w_k1_IBM_s104
                                 + 0.97305 w_k1_INTC_s104
                                 + 1.18757 w_k1_KO_s104 + 1.14877 w_k1_MAT_s104
                                 + 0.957963 w_k1_MSFT_s104
                                 + 1.01364 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.919031 w_k1_ADM_s104
                                 + 0.901767 w_k1_AXP_s104
                                 + 0.749047 w_k1_DIS_s104
                                 + 0.840384 w_k1_FDX_s104
                                 + 1.07397 w_k1_IBM_s104
                                 + 0.875818 w_k1_INTC_s104
                                 + 0.973798 w_k1_KO_s104 + 1.2741 w_k1_MAT_s104
                                 + 0.958603 w_k1_MSFT_s104
                                 + 1.00424 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.997414 w_k1_ADM_s104
                                 + 0.759202 w_k1_AXP_s104
                                 + 0.677743 w_k1_DIS_s104
                                 + 0.836417 w_k1_FDX_s104
                                 + 0.755279 w_k1_IBM_s104
                                 + 0.925911 w_k1_INTC_s104
                                 + 0.834722 w_k1_KO_s104
                                 + 0.84342 w_k1_MAT_s104
                                 + 0.809031 w_k1_MSFT_s104
                                 + 0.823911 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.18087 w_k1_ADM_s104
                                 + 1.08147 w_k1_AXP_s104
                                 + 0.904679 w_k1_DIS_s104
                                 + 0.902593 w_k1_FDX_s104
                                 + 1.5445 w_k1_IBM_s104
                                 + 0.926771 w_k1_INTC_s104
                                 + 1.28804 w_k1_KO_s104 + 1.76856 w_k1_MAT_s104
                                 + 0.917503 w_k1_MSFT_s104
                                 + 1.08553 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.28468 w_k1_ADM_s104
                                 + 1.31046 w_k1_AXP_s104
                                 + 0.850843 w_k1_DIS_s104
                                 + 1.13348 w_k1_FDX_s104
                                 + 0.75287 w_k1_IBM_s104
                                 + 1.3044 w_k1_INTC_s104 + 1.01422 w_k1_KO_s104
                                 + 1.31168 w_k1_MAT_s104
                                 + 1.07884 w_k1_MSFT_s104
                                 + 1.09615 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.30232 w_k1_ADM_s104
                                 + 1.00556 w_k1_AXP_s104
                                 + 1.23823 w_k1_DIS_s104
                                 + 1.13737 w_k1_FDX_s104
                                 + 1.12302 w_k1_IBM_s104
                                 + 1.03749 w_k1_INTC_s104
                                 + 1.28388 w_k1_KO_s104 + 1.11914 w_k1_MAT_s104
                                 + 1.11091 w_k1_MSFT_s104
                                 + 1.20599 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.25103 w_k1_ADM_s104
                                 + 1.00399 w_k1_AXP_s104
                                 + 0.973733 w_k1_DIS_s104
                                 + 1.04472 w_k1_FDX_s104
                                 + 0.719707 w_k1_IBM_s104
                                 + 1.00295 w_k1_INTC_s104
                                 + 0.862542 w_k1_KO_s104
                                 + 0.826468 w_k1_MAT_s104
                                 + 1.02328 w_k1_MSFT_s104
                                 + 0.972719 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.33751 w_k1_ADM_s104
                                 + 1.58456 w_k1_AXP_s104
                                 + 1.47586 w_k1_DIS_s104
                                 + 1.33185 w_k1_FDX_s104
                                 + 1.24199 w_k1_IBM_s104
                                 + 1.0589 w_k1_INTC_s104 + 1.22086 w_k1_KO_s104
                                 + 1.88597 w_k1_MAT_s104
                                 + 1.23134 w_k1_MSFT_s104
                                 + 1.36846 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.41296 w_k1_ADM_s105
                                 + 1.42116 w_k1_AXP_s105
                                 + 1.38991 w_k1_DIS_s105
                                 + 1.42077 w_k1_FDX_s105
                                 + 0.922388 w_k1_IBM_s105
                                 + 1.43093 w_k1_INTC_s105
                                 + 1.03738 w_k1_KO_s105
                                 + 0.777456 w_k1_MAT_s105
                                 + 1.35419 w_k1_MSFT_s105
                                 + 1.22028 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.24312 w_k1_ADM_s105
                                 + 0.936462 w_k1_AXP_s105
                                 + 1.01858 w_k1_DIS_s105 + 1.0976 w_k1_FDX_s105
                                 + 1.03811 w_k1_IBM_s105
                                 + 1.02493 w_k1_INTC_s105
                                 + 1.16265 w_k1_KO_s105
                                 + 0.956326 w_k1_MAT_s105
                                 + 1.05121 w_k1_MSFT_s105
                                 + 1.06116 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.980573 w_k1_ADM_s105
                                 + 1.47459 w_k1_AXP_s105
                                 + 1.53905 w_k1_DIS_s105
                                 + 1.37677 w_k1_FDX_s105
                                 + 1.29566 w_k1_IBM_s105
                                 + 1.3059 w_k1_INTC_s105 + 1.38795 w_k1_KO_s105
                                 + 1.69412 w_k1_MAT_s105
                                 + 1.23043 w_k1_MSFT_s105 + 1.1848 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.13725 w_k1_ADM_s105
                                 + 0.880669 w_k1_AXP_s105
                                 + 1.13819 w_k1_DIS_s105
                                 + 1.13423 w_k1_FDX_s105
                                 + 0.774035 w_k1_IBM_s105
                                 + 1.15907 w_k1_INTC_s105
                                 + 1.04071 w_k1_KO_s105
                                 + 0.888972 w_k1_MAT_s105
                                 + 0.977463 w_k1_MSFT_s105
                                 + 1.02116 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.750016 w_k1_ADM_s105
                                 + 0.643706 w_k1_AXP_s105
                                 + 0.737883 w_k1_DIS_s105
                                 + 0.771601 w_k1_FDX_s105
                                 + 0.795766 w_k1_IBM_s105
                                 + 0.908289 w_k1_INTC_s105
                                 + 0.878588 w_k1_KO_s105
                                 + 0.66673 w_k1_MAT_s105
                                 + 0.78792 w_k1_MSFT_s105
                                 + 0.866021 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.94744 w_k1_ADM_s105
                                 + 1.16178 w_k1_AXP_s105
                                 + 0.986484 w_k1_DIS_s105
                                 + 0.987363 w_k1_FDX_s105
                                 + 1.04106 w_k1_IBM_s105
                                 + 1.16519 w_k1_INTC_s105
                                 + 1.14488 w_k1_KO_s105 + 1.38672 w_k1_MAT_s105
                                 + 0.936223 w_k1_MSFT_s105
                                 + 1.04852 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.9403 w_k1_ADM_s105
                                 + 0.747131 w_k1_AXP_s105
                                 + 0.96729 w_k1_DIS_s105
                                 + 0.936421 w_k1_FDX_s105
                                 + 0.818021 w_k1_IBM_s105
                                 + 0.86549 w_k1_INTC_s105
                                 + 0.85885 w_k1_KO_s105
                                 + 0.583014 w_k1_MAT_s105
                                 + 0.908544 w_k1_MSFT_s105
                                 + 0.903278 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.03977 w_k1_ADM_s105
                                 + 0.725411 w_k1_AXP_s105
                                 + 1.11291 w_k1_DIS_s105
                                 + 0.903228 w_k1_FDX_s105
                                 + 1.36766 w_k1_IBM_s105
                                 + 1.10071 w_k1_INTC_s105
                                 + 1.42885 w_k1_KO_s105 + 1.23557 w_k1_MAT_s105
                                 + 1.10794 w_k1_MSFT_s105
                                 + 1.07933 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.05126 w_k1_ADM_s106
                                 + 0.81452 w_k1_AXP_s106
                                 + 0.741349 w_k1_DIS_s106
                                 + 0.966239 w_k1_FDX_s106
                                 + 0.572678 w_k1_IBM_s106
                                 + 0.756044 w_k1_INTC_s106
                                 + 0.802971 w_k1_KO_s106
                                 + 0.606605 w_k1_MAT_s106
                                 + 0.889327 w_k1_MSFT_s106
                                 + 0.885594 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.54827 w_k1_ADM_s106
                                 + 1.13439 w_k1_AXP_s106
                                 + 1.33638 w_k1_DIS_s106
                                 + 1.25118 w_k1_FDX_s106
                                 + 0.877952 w_k1_IBM_s106
                                 + 0.905348 w_k1_INTC_s106
                                 + 1.27225 w_k1_KO_s106 + 1.02022 w_k1_MAT_s106
                                 + 1.08996 w_k1_MSFT_s106
                                 + 1.16031 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.709012 w_k1_ADM_s106
                                 + 0.717361 w_k1_AXP_s106
                                 + 0.677052 w_k1_DIS_s106
                                 + 0.729743 w_k1_FDX_s106
                                 + 0.935911 w_k1_IBM_s106
                                 + 0.716652 w_k1_INTC_s106
                                 + 0.804626 w_k1_KO_s106
                                 + 0.838055 w_k1_MAT_s106
                                 + 0.811076 w_k1_MSFT_s106
                                 + 0.849895 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.14352 w_k1_ADM_s106
                                 + 1.46972 w_k1_AXP_s106
                                 + 1.26684 w_k1_DIS_s106
                                 + 1.30543 w_k1_FDX_s106
                                 + 1.33974 w_k1_IBM_s106
                                 + 1.20997 w_k1_INTC_s106
                                 + 1.46935 w_k1_KO_s106 + 1.85222 w_k1_MAT_s106
                                 + 1.31413 w_k1_MSFT_s106
                                 + 1.33722 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.15122 w_k1_ADM_s106
                                 + 1.00723 w_k1_AXP_s106
                                 + 0.922559 w_k1_DIS_s106
                                 + 1.1316 w_k1_FDX_s106
                                 + 0.807449 w_k1_IBM_s106
                                 + 1.02232 w_k1_INTC_s106
                                 + 0.920195 w_k1_KO_s106
                                 + 0.853085 w_k1_MAT_s106
                                 + 1.00434 w_k1_MSFT_s106
                                 + 0.954967 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.886488 w_k1_ADM_s106
                                 + 0.867576 w_k1_AXP_s106 + 1.01 w_k1_DIS_s106
                                 + 0.910608 w_k1_FDX_s106
                                 + 1.28405 w_k1_IBM_s106
                                 + 1.07132 w_k1_INTC_s106
                                 + 1.21124 w_k1_KO_s106 + 1.10983 w_k1_MAT_s106
                                 + 0.97971 w_k1_MSFT_s106
                                 + 1.01205 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.645448 w_k1_ADM_s106
                                 + 0.402517 w_k1_AXP_s106
                                 + 0.567838 w_k1_DIS_s106
                                 + 0.646843 w_k1_FDX_s106
                                 + 0.985839 w_k1_IBM_s106
                                 + 0.659277 w_k1_INTC_s106
                                 + 0.875506 w_k1_KO_s106 + 0.7363 w_k1_MAT_s106
                                 + 0.75812 w_k1_MSFT_s106
                                 + 0.765517 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.16104 w_k1_ADM_s106
                                 + 1.2783 w_k1_AXP_s106 + 1.53285 w_k1_DIS_s106
                                 + 1.14434 w_k1_FDX_s106
                                 + 1.57998 w_k1_IBM_s106
                                 + 0.928964 w_k1_INTC_s106
                                 + 1.67427 w_k1_KO_s106 + 1.43325 w_k1_MAT_s106
                                 + 1.14444 w_k1_MSFT_s106
                                 + 1.29842 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.951273 w_k1_ADM_s107
                                 + 1.25566 w_k1_AXP_s107
                                 + 1.39688 w_k1_DIS_s107
                                 + 1.30696 w_k1_FDX_s107 + 1.1649 w_k1_IBM_s107
                                 + 1.28589 w_k1_INTC_s107
                                 + 1.30518 w_k1_KO_s107 + 1.49193 w_k1_MAT_s107
                                 + 1.34225 w_k1_MSFT_s107
                                 + 1.28899 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.874532 w_k1_ADM_s107
                                 + 1.37078 w_k1_AXP_s107 + 1.305 w_k1_DIS_s107
                                 + 1.01821 w_k1_FDX_s107
                                 + 1.75754 w_k1_IBM_s107
                                 + 1.13476 w_k1_INTC_s107 + 1.3956 w_k1_KO_s107
                                 + 1.91892 w_k1_MAT_s107
                                 + 1.02554 w_k1_MSFT_s107
                                 + 1.27872 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.902065 w_k1_ADM_s107
                                 + 0.928652 w_k1_AXP_s107
                                 + 1.04989 w_k1_DIS_s107
                                 + 1.13932 w_k1_FDX_s107
                                 + 0.838626 w_k1_IBM_s107
                                 + 1.14676 w_k1_INTC_s107 + 1.0181 w_k1_KO_s107
                                 + 0.805132 w_k1_MAT_s107
                                 + 1.18233 w_k1_MSFT_s107
                                 + 1.10462 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.695539 w_k1_ADM_s107
                                 + 0.749271 w_k1_AXP_s107
                                 + 0.735037 w_k1_DIS_s107
                                 + 0.802385 w_k1_FDX_s107
                                 + 0.806017 w_k1_IBM_s107
                                 + 0.886121 w_k1_INTC_s107
                                 + 0.897401 w_k1_KO_s107
                                 + 0.519892 w_k1_MAT_s107
                                 + 0.920568 w_k1_MSFT_s107
                                 + 0.948673 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.628167 w_k1_ADM_s107
                                 + 1.03022 w_k1_AXP_s107 + 1.1206 w_k1_DIS_s107
                                 + 0.982012 w_k1_FDX_s107
                                 + 1.34146 w_k1_IBM_s107
                                 + 1.09672 w_k1_INTC_s107
                                 + 1.17061 w_k1_KO_s107 + 1.51143 w_k1_MAT_s107
                                 + 0.961219 w_k1_MSFT_s107
                                 + 1.15754 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.910931 w_k1_ADM_s107
                                 + 0.916038 w_k1_AXP_s107
                                 + 0.741108 w_k1_DIS_s107
                                 + 0.766192 w_k1_FDX_s107
                                 + 1.19271 w_k1_IBM_s107
                                 + 0.893572 w_k1_INTC_s107
                                 + 1.01965 w_k1_KO_s107 + 1.10944 w_k1_MAT_s107
                                 + 0.857859 w_k1_MSFT_s107
                                 + 1.07264 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.915073 w_k1_ADM_s107
                                 + 0.860746 w_k1_AXP_s107
                                 + 1.20503 w_k1_DIS_s107
                                 + 0.969442 w_k1_FDX_s107
                                 + 1.12078 w_k1_IBM_s107
                                 + 0.958628 w_k1_INTC_s107
                                 + 1.15337 w_k1_KO_s107 + 1.06776 w_k1_MAT_s107
                                 + 1.14227 w_k1_MSFT_s107
                                 + 1.08913 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.01874 w_k1_ADM_s107
                                 + 1.18894 w_k1_AXP_s107
                                 + 1.48516 w_k1_DIS_s107
                                 + 1.38652 w_k1_FDX_s107
                                 + 0.938208 w_k1_IBM_s107
                                 + 1.27354 w_k1_INTC_s107 + 1.2204 w_k1_KO_s107
                                 + 1.07747 w_k1_MAT_s107
                                 + 1.23056 w_k1_MSFT_s107
                                 + 1.14712 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.37796 w_k1_ADM_s110
                                 + 1.49441 w_k1_AXP_s110
                                 + 1.01099 w_k1_DIS_s110
                                 + 1.06508 w_k1_FDX_s110
                                 + 1.32909 w_k1_IBM_s110
                                 + 0.933556 w_k1_INTC_s110
                                 + 1.04113 w_k1_KO_s110 + 1.37187 w_k1_MAT_s110
                                 + 0.944479 w_k1_MSFT_s110
                                 + 1.04008 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.8446 w_k1_ADM_s110
                                 + 0.909432 w_k1_AXP_s110
                                 + 0.677152 w_k1_DIS_s110
                                 + 0.81469 w_k1_FDX_s110
                                 + 1.05292 w_k1_IBM_s110
                                 + 0.867193 w_k1_INTC_s110
                                 + 0.762049 w_k1_KO_s110 + 1.2081 w_k1_MAT_s110
                                 + 0.856748 w_k1_MSFT_s110
                                 + 0.895613 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.970071 w_k1_ADM_s110
                                 + 1.43903 w_k1_AXP_s110
                                 + 1.71258 w_k1_DIS_s110
                                 + 1.27447 w_k1_FDX_s110
                                 + 1.70063 w_k1_IBM_s110
                                 + 1.45498 w_k1_INTC_s110
                                 + 1.62836 w_k1_KO_s110 + 1.84635 w_k1_MAT_s110
                                 + 1.51498 w_k1_MSFT_s110
                                 + 1.20777 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.967129 w_k1_ADM_s110
                                 + 0.790577 w_k1_AXP_s110
                                 + 0.928021 w_k1_DIS_s110
                                 + 0.964451 w_k1_FDX_s110
                                 + 1.0244 w_k1_IBM_s110
                                 + 0.879122 w_k1_INTC_s110
                                 + 0.938759 w_k1_KO_s110
                                 + 1.22199 w_k1_MAT_s110
                                 + 1.09954 w_k1_MSFT_s110
                                 + 1.00931 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.841265 w_k1_ADM_s110
                                 + 0.711314 w_k1_AXP_s110
                                 + 0.580861 w_k1_DIS_s110
                                 + 0.813791 w_k1_FDX_s110
                                 + 0.711189 w_k1_IBM_s110
                                 + 0.826693 w_k1_INTC_s110
                                 + 0.680574 w_k1_KO_s110
                                 + 0.72417 w_k1_MAT_s110
                                 + 0.851592 w_k1_MSFT_s110
                                 + 0.839708 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.857208 w_k1_ADM_s110
                                 + 0.940868 w_k1_AXP_s110
                                 + 1.04334 w_k1_DIS_s110
                                 + 0.889519 w_k1_FDX_s110
                                 + 1.52607 w_k1_IBM_s110
                                 + 1.07876 w_k1_INTC_s110
                                 + 1.39967 w_k1_KO_s110 + 1.68992 w_k1_MAT_s110
                                 + 1.01179 w_k1_MSFT_s110
                                 + 1.04206 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.25295 w_k1_ADM_s110
                                 + 1.31732 w_k1_AXP_s110
                                 + 1.43029 w_k1_DIS_s110
                                 + 1.23925 w_k1_FDX_s110
                                 + 1.00971 w_k1_IBM_s110
                                 + 1.07116 w_k1_INTC_s110
                                 + 1.23943 w_k1_KO_s110 + 0.99586 w_k1_MAT_s110
                                 + 1.27968 w_k1_MSFT_s110
                                 + 1.20493 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.15894 w_k1_ADM_s110
                                 + 1.16778 w_k1_AXP_s110
                                 + 1.14503 w_k1_DIS_s110
                                 + 1.27531 w_k1_FDX_s110 + 1.058 w_k1_IBM_s110
                                 + 1.16776 w_k1_INTC_s110 + 1.0626 w_k1_KO_s110
                                 + 1.2505 w_k1_MAT_s110
                                 + 1.21736 w_k1_MSFT_s110
                                 + 1.04443 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.26903 w_k1_ADM_s111
                                 + 1.18773 w_k1_AXP_s111
                                 + 1.09586 w_k1_DIS_s111
                                 + 1.10163 w_k1_FDX_s111
                                 + 0.859193 w_k1_IBM_s111
                                 + 1.06899 w_k1_INTC_s111
                                 + 1.07308 w_k1_KO_s111 + 1.30851 w_k1_MAT_s111
                                 + 1.06888 w_k1_MSFT_s111
                                 + 1.00881 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.817425 w_k1_ADM_s111
                                 + 0.733227 w_k1_AXP_s111
                                 + 0.850404 w_k1_DIS_s111
                                 + 0.770434 w_k1_FDX_s111
                                 + 0.931266 w_k1_IBM_s111
                                 + 0.807134 w_k1_INTC_s111
                                 + 0.945329 w_k1_KO_s111
                                 + 1.14308 w_k1_MAT_s111
                                 + 0.794557 w_k1_MSFT_s111
                                 + 0.877957 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.33774 w_k1_ADM_s111
                                 + 1.10846 w_k1_AXP_s111 + 1.387 w_k1_DIS_s111
                                 + 1.07669 w_k1_FDX_s111
                                 + 1.08985 w_k1_IBM_s111
                                 + 1.02323 w_k1_INTC_s111
                                 + 1.20513 w_k1_KO_s111 + 1.48496 w_k1_MAT_s111
                                 + 1.15724 w_k1_MSFT_s111
                                 + 1.00651 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.932946 w_k1_ADM_s111
                                 + 0.832276 w_k1_AXP_s111
                                 + 0.914306 w_k1_DIS_s111
                                 + 0.84172 w_k1_FDX_s111 + 1.5022 w_k1_IBM_s111
                                 + 0.773627 w_k1_INTC_s111
                                 + 1.24557 w_k1_KO_s111 + 2.20374 w_k1_MAT_s111
                                 + 0.988918 w_k1_MSFT_s111
                                 + 0.980351 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.56343 w_k1_ADM_s111
                                 + 1.15097 w_k1_AXP_s111 + 1.3162 w_k1_DIS_s111
                                 + 1.33457 w_k1_FDX_s111
                                 + 0.70378 w_k1_IBM_s111
                                 + 1.17843 w_k1_INTC_s111
                                 + 1.22977 w_k1_KO_s111 + 1.2216 w_k1_MAT_s111
                                 + 1.33077 w_k1_MSFT_s111
                                 + 1.15743 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.905498 w_k1_ADM_s111
                                 + 0.69383 w_k1_AXP_s111
                                 + 0.860368 w_k1_DIS_s111
                                 + 0.922698 w_k1_FDX_s111
                                 + 0.609174 w_k1_IBM_s111
                                 + 0.95933 w_k1_INTC_s111
                                 + 0.823815 w_k1_KO_s111
                                 + 0.678462 w_k1_MAT_s111
                                 + 1.03289 w_k1_MSFT_s111
                                 + 0.796662 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.977053 w_k1_ADM_s111
                                 + 1.5767 w_k1_AXP_s111 + 1.6064 w_k1_DIS_s111
                                 + 1.19812 w_k1_FDX_s111
                                 + 1.53677 w_k1_IBM_s111
                                 + 1.20422 w_k1_INTC_s111
                                 + 1.45963 w_k1_KO_s111 + 2.21606 w_k1_MAT_s111
                                 + 1.38661 w_k1_MSFT_s111
                                 + 1.12279 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.50593 w_k1_ADM_s111
                                 + 1.23232 w_k1_AXP_s111
                                 + 1.05606 w_k1_DIS_s111
                                 + 1.07208 w_k1_FDX_s111
                                 + 1.21693 w_k1_IBM_s111
                                 + 0.89446 w_k1_INTC_s111
                                 + 1.04964 w_k1_KO_s111 + 1.58399 w_k1_MAT_s111
                                 + 0.911151 w_k1_MSFT_s111
                                 + 0.912501 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.13318 w_k1_ADM_s112
                                 + 1.03597 w_k1_AXP_s112
                                 + 0.80591 w_k1_DIS_s112
                                 + 1.04826 w_k1_FDX_s112
                                 + 0.753151 w_k1_IBM_s112
                                 + 0.843324 w_k1_INTC_s112
                                 + 0.835993 w_k1_KO_s112
                                 + 0.71676 w_k1_MAT_s112
                                 + 0.90037 w_k1_MSFT_s112
                                 + 0.993716 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.883748 w_k1_ADM_s112
                                 + 0.859554 w_k1_AXP_s112
                                 + 0.90869 w_k1_DIS_s112
                                 + 0.928136 w_k1_FDX_s112
                                 + 0.968178 w_k1_IBM_s112
                                 + 0.812959 w_k1_INTC_s112
                                 + 0.986963 w_k1_KO_s112
                                 + 1.30732 w_k1_MAT_s112
                                 + 0.919177 w_k1_MSFT_s112
                                 + 1.05343 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.891907 w_k1_ADM_s112
                                 + 0.655612 w_k1_AXP_s112
                                 + 0.671393 w_k1_DIS_s112
                                 + 0.811392 w_k1_FDX_s112
                                 + 0.664922 w_k1_IBM_s112
                                 + 0.743553 w_k1_INTC_s112
                                 + 0.748963 w_k1_KO_s112
                                 + 1.00388 w_k1_MAT_s112
                                 + 0.792922 w_k1_MSFT_s112
                                 + 0.904993 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.10818 w_k1_ADM_s112
                                 + 0.880813 w_k1_AXP_s112
                                 + 1.28789 w_k1_DIS_s112
                                 + 1.22399 w_k1_FDX_s112
                                 + 0.955863 w_k1_IBM_s112
                                 + 1.18055 w_k1_INTC_s112
                                 + 1.06441 w_k1_KO_s112
                                 + 0.926902 w_k1_MAT_s112
                                 + 1.21347 w_k1_MSFT_s112
                                 + 1.14828 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.23107 w_k1_ADM_s112
                                 + 1.17763 w_k1_AXP_s112
                                 + 1.43197 w_k1_DIS_s112
                                 + 1.24107 w_k1_FDX_s112
                                 + 1.40252 w_k1_IBM_s112
                                 + 1.21296 w_k1_INTC_s112 + 1.4167 w_k1_KO_s112
                                 + 1.3909 w_k1_MAT_s112
                                 + 1.34418 w_k1_MSFT_s112 + 1.2619 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.920165 w_k1_ADM_s112
                                 + 0.861665 w_k1_AXP_s112
                                 + 0.787218 w_k1_DIS_s112
                                 + 0.835053 w_k1_FDX_s112
                                 + 1.32864 w_k1_IBM_s112
                                 + 0.858931 w_k1_INTC_s112
                                 + 1.13692 w_k1_KO_s112 + 1.5721 w_k1_MAT_s112
                                 + 0.881048 w_k1_MSFT_s112
                                 + 1.10356 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.869932 w_k1_ADM_s112
                                 + 0.611603 w_k1_AXP_s112
                                 + 0.980479 w_k1_DIS_s112
                                 + 0.989445 w_k1_FDX_s112
                                 + 0.75018 w_k1_IBM_s112
                                 + 0.875995 w_k1_INTC_s112
                                 + 0.970617 w_k1_KO_s112
                                 + 0.836697 w_k1_MAT_s112
                                 + 1.03914 w_k1_MSFT_s112
                                 + 0.990645 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.08361 w_k1_ADM_s112
                                 + 0.954125 w_k1_AXP_s112
                                 + 0.872113 w_k1_DIS_s112
                                 + 1.13695 w_k1_FDX_s112
                                 + 0.677641 w_k1_IBM_s112
                                 + 1.03138 w_k1_INTC_s112
                                 + 0.824817 w_k1_KO_s112
                                 + 0.85291 w_k1_MAT_s112
                                 + 1.18598 w_k1_MSFT_s112
                                 + 1.07691 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.24517 w_k1_ADM_s113
                                 + 1.34182 w_k1_AXP_s113
                                 + 1.60459 w_k1_DIS_s113
                                 + 1.48966 w_k1_FDX_s113 + 1.0783 w_k1_IBM_s113
                                 + 1.32987 w_k1_INTC_s113
                                 + 1.28837 w_k1_KO_s113 + 1.46832 w_k1_MAT_s113
                                 + 1.31589 w_k1_MSFT_s113
                                 + 1.25518 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.991558 w_k1_ADM_s113
                                 + 0.863093 w_k1_AXP_s113
                                 + 1.06513 w_k1_DIS_s113
                                 + 1.00118 w_k1_FDX_s113
                                 + 1.10215 w_k1_IBM_s113
                                 + 1.09966 w_k1_INTC_s113
                                 + 1.13831 w_k1_KO_s113 + 1.0937 w_k1_MAT_s113
                                 + 1.09166 w_k1_MSFT_s113
                                 + 1.11848 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.13784 w_k1_ADM_s113
                                 + 1.16039 w_k1_AXP_s113
                                 + 1.17938 w_k1_DIS_s113
                                 + 1.00804 w_k1_FDX_s113
                                 + 1.49799 w_k1_IBM_s113
                                 + 1.17526 w_k1_INTC_s113
                                 + 1.31477 w_k1_KO_s113 + 1.67637 w_k1_MAT_s113
                                 + 1.20431 w_k1_MSFT_s113
                                 + 1.15468 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.19745 w_k1_ADM_s113
                                 + 1.239 w_k1_AXP_s113 + 1.55689 w_k1_DIS_s113
                                 + 1.17607 w_k1_FDX_s113
                                 + 1.29469 w_k1_IBM_s113
                                 + 1.28751 w_k1_INTC_s113
                                 + 1.57589 w_k1_KO_s113 + 1.34407 w_k1_MAT_s113
                                 + 1.40905 w_k1_MSFT_s113
                                 + 1.32723 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.729428 w_k1_ADM_s113
                                 + 0.498936 w_k1_AXP_s113
                                 + 0.618934 w_k1_DIS_s113
                                 + 0.733584 w_k1_FDX_s113
                                 + 0.723043 w_k1_IBM_s113
                                 + 0.904735 w_k1_INTC_s113
                                 + 0.714946 w_k1_KO_s113
                                 + 0.48609 w_k1_MAT_s113
                                 + 0.819975 w_k1_MSFT_s113
                                 + 0.786344 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.20013 w_k1_ADM_s113
                                 + 1.07698 w_k1_AXP_s113 + 1.1495 w_k1_DIS_s113
                                 + 1.23969 w_k1_FDX_s113
                                 + 0.982703 w_k1_IBM_s113
                                 + 1.27672 w_k1_INTC_s113
                                 + 1.10446 w_k1_KO_s113
                                 + 0.873161 w_k1_MAT_s113
                                 + 1.16817 w_k1_MSFT_s113
                                 + 1.01143 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.90341 w_k1_ADM_s113
                                 + 0.875434 w_k1_AXP_s113
                                 + 0.788794 w_k1_DIS_s113
                                 + 0.875373 w_k1_FDX_s113
                                 + 0.886357 w_k1_IBM_s113
                                 + 1.00902 w_k1_INTC_s113
                                 + 0.938268 w_k1_KO_s113
                                 + 1.07844 w_k1_MAT_s113
                                 + 0.908738 w_k1_MSFT_s113
                                 + 0.92504 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.05015 w_k1_ADM_s113
                                 + 0.891506 w_k1_AXP_s113
                                 + 0.978816 w_k1_DIS_s113
                                 + 1.01601 w_k1_FDX_s113
                                 + 0.78578 w_k1_IBM_s113
                                 + 0.904364 w_k1_INTC_s113
                                 + 1.03368 w_k1_KO_s113 + 0.88357 w_k1_MAT_s113
                                 + 1.0281 w_k1_MSFT_s113 + 1.05487 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.89267 w_k1_ADM_s114
                                 + 0.70955 w_k1_AXP_s114
                                 + 0.711935 w_k1_DIS_s114
                                 + 0.771434 w_k1_FDX_s114
                                 + 0.865693 w_k1_IBM_s114
                                 + 0.749024 w_k1_INTC_s114
                                 + 0.793517 w_k1_KO_s114
                                 + 0.769896 w_k1_MAT_s114
                                 + 0.6573 w_k1_MSFT_s114
                                 + 0.876872 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.38132 w_k1_ADM_s114
                                 + 0.95947 w_k1_AXP_s114
                                 + 1.30441 w_k1_DIS_s114 + 1.0387 w_k1_FDX_s114
                                 + 0.715308 w_k1_IBM_s114
                                 + 1.15915 w_k1_INTC_s114
                                 + 1.03293 w_k1_KO_s114
                                 + 0.767349 w_k1_MAT_s114
                                 + 0.998853 w_k1_MSFT_s114
                                 + 1.07006 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.04617 w_k1_ADM_s114
                                 + 0.727364 w_k1_AXP_s114
                                 + 0.966135 w_k1_DIS_s114
                                 + 0.902172 w_k1_FDX_s114
                                 + 0.93997 w_k1_IBM_s114
                                 + 1.08458 w_k1_INTC_s114
                                 + 1.09686 w_k1_KO_s114
                                 + 0.698704 w_k1_MAT_s114
                                 + 0.874172 w_k1_MSFT_s114
                                 + 0.93045 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.986435 w_k1_ADM_s114
                                 + 1.0694 w_k1_AXP_s114
                                 + 0.947725 w_k1_DIS_s114
                                 + 0.988656 w_k1_FDX_s114
                                 + 1.1096 w_k1_IBM_s114
                                 + 1.12458 w_k1_INTC_s114
                                 + 1.09258 w_k1_KO_s114
                                 + 0.873361 w_k1_MAT_s114
                                 + 0.826939 w_k1_MSFT_s114
                                 + 1.02858 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.42752 w_k1_ADM_s114
                                 + 1.29682 w_k1_AXP_s114
                                 + 1.44018 w_k1_DIS_s114 + 1.2131 w_k1_FDX_s114
                                 + 0.999019 w_k1_IBM_s114
                                 + 1.206 w_k1_INTC_s114 + 1.35283 w_k1_KO_s114
                                 + 1.0542 w_k1_MAT_s114
                                 + 1.10215 w_k1_MSFT_s114
                                 + 1.10325 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.976764 w_k1_ADM_s114
                                 + 0.762507 w_k1_AXP_s114
                                 + 0.797706 w_k1_DIS_s114
                                 + 0.963095 w_k1_FDX_s114
                                 + 0.635636 w_k1_IBM_s114
                                 + 0.810248 w_k1_INTC_s114
                                 + 0.7351 w_k1_KO_s114 + 0.600591 w_k1_MAT_s114
                                 + 0.733688 w_k1_MSFT_s114
                                 + 0.797044 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.27064 w_k1_ADM_s114
                                 + 1.37216 w_k1_AXP_s114
                                 + 1.46553 w_k1_DIS_s114
                                 + 1.12241 w_k1_FDX_s114
                                 + 1.53543 w_k1_IBM_s114
                                 + 1.04385 w_k1_INTC_s114
                                 + 1.49463 w_k1_KO_s114 + 1.65573 w_k1_MAT_s114
                                 + 0.990279 w_k1_MSFT_s114
                                 + 1.18798 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.11869 w_k1_ADM_s114
                                 + 1.03775 w_k1_AXP_s114
                                 + 1.03713 w_k1_DIS_s114
                                 + 1.10483 w_k1_FDX_s114
                                 + 0.747967 w_k1_IBM_s114
                                 + 1.10794 w_k1_INTC_s114
                                 + 1.00417 w_k1_KO_s114
                                 + 0.833231 w_k1_MAT_s114
                                 + 1.03549 w_k1_MSFT_s114
                                 + 0.978391 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.36903 w_k1_ADM_s115
                                 + 1.47043 w_k1_AXP_s115
                                 + 1.43747 w_k1_DIS_s115
                                 + 1.23243 w_k1_FDX_s115
                                 + 1.65291 w_k1_IBM_s115
                                 + 1.05321 w_k1_INTC_s115
                                 + 1.36873 w_k1_KO_s115 + 1.84013 w_k1_MAT_s115
                                 + 1.16358 w_k1_MSFT_s115
                                 + 1.42373 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.59931 w_k1_ADM_s115
                                 + 0.941792 w_k1_AXP_s115
                                 + 1.35691 w_k1_DIS_s115
                                 + 1.25584 w_k1_FDX_s115
                                 + 0.923547 w_k1_IBM_s115
                                 + 0.900674 w_k1_INTC_s115
                                 + 1.17118 w_k1_KO_s115
                                 + 0.887723 w_k1_MAT_s115
                                 + 1.21448 w_k1_MSFT_s115
                                 + 1.24618 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.61843 w_k1_ADM_s115
                                 + 1.23268 w_k1_AXP_s115
                                 + 1.10155 w_k1_DIS_s115
                                 + 1.08748 w_k1_FDX_s115
                                 + 1.18002 w_k1_IBM_s115
                                 + 1.0591 w_k1_INTC_s115 + 1.11323 w_k1_KO_s115
                                 + 0.78743 w_k1_MAT_s115
                                 + 0.963567 w_k1_MSFT_s115
                                 + 1.03153 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.19613 w_k1_ADM_s115
                                 + 0.897692 w_k1_AXP_s115
                                 + 0.786618 w_k1_DIS_s115
                                 + 0.935945 w_k1_FDX_s115
                                 + 0.624319 w_k1_IBM_s115
                                 + 0.980992 w_k1_INTC_s115
                                 + 0.907137 w_k1_KO_s115
                                 + 0.518184 w_k1_MAT_s115
                                 + 0.91582 w_k1_MSFT_s115
                                 + 0.914167 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.801542 w_k1_ADM_s115
                                 + 0.676026 w_k1_AXP_s115
                                 + 0.633929 w_k1_DIS_s115
                                 + 0.785771 w_k1_FDX_s115
                                 + 0.7922 w_k1_IBM_s115
                                 + 0.928399 w_k1_INTC_s115
                                 + 0.78276 w_k1_KO_s115 + 0.5888 w_k1_MAT_s115
                                 + 0.827719 w_k1_MSFT_s115
                                 + 0.814016 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.30632 w_k1_ADM_s115
                                 + 0.911982 w_k1_AXP_s115
                                 + 1.23694 w_k1_DIS_s115
                                 + 1.19706 w_k1_FDX_s115
                                 + 0.705354 w_k1_IBM_s115
                                 + 1.31327 w_k1_INTC_s115
                                 + 1.16313 w_k1_KO_s115
                                 + 0.542995 w_k1_MAT_s115
                                 + 1.23221 w_k1_MSFT_s115
                                 + 1.11081 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.12379 w_k1_ADM_s115
                                 + 0.687428 w_k1_AXP_s115
                                 + 0.874226 w_k1_DIS_s115
                                 + 0.922335 w_k1_FDX_s115
                                 + 0.913054 w_k1_IBM_s115
                                 + 0.846465 w_k1_INTC_s115
                                 + 0.891181 w_k1_KO_s115
                                 + 0.790653 w_k1_MAT_s115
                                 + 0.9996 w_k1_MSFT_s115 + 0.9912 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.935689 w_k1_ADM_s115
                                 + 0.774589 w_k1_AXP_s115
                                 + 0.810176 w_k1_DIS_s115
                                 + 0.817822 w_k1_FDX_s115
                                 + 1.20631 w_k1_IBM_s115
                                 + 0.876903 w_k1_INTC_s115
                                 + 1.20787 w_k1_KO_s115
                                 + 0.992827 w_k1_MAT_s115
                                 + 0.926178 w_k1_MSFT_s115
                                 + 0.975551 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.04848 w_k1_ADM_s116
                                 + 1.05005 w_k1_AXP_s116
                                 + 0.90366 w_k1_DIS_s116
                                 + 1.11557 w_k1_FDX_s116
                                 + 0.649111 w_k1_IBM_s116
                                 + 1.01841 w_k1_INTC_s116
                                 + 0.812362 w_k1_KO_s116
                                 + 0.580229 w_k1_MAT_s116
                                 + 1.19237 w_k1_MSFT_s116
                                 + 0.926937 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.838215 w_k1_ADM_s116
                                 + 0.575199 w_k1_AXP_s116
                                 + 0.680411 w_k1_DIS_s116
                                 + 0.800607 w_k1_FDX_s116
                                 + 0.669024 w_k1_IBM_s116
                                 + 0.691027 w_k1_INTC_s116
                                 + 0.700507 w_k1_KO_s116
                                 + 0.597273 w_k1_MAT_s116
                                 + 0.753818 w_k1_MSFT_s116
                                 + 0.762436 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.77049 w_k1_ADM_s116
                                 + 0.738749 w_k1_AXP_s116
                                 + 0.744679 w_k1_DIS_s116
                                 + 0.773678 w_k1_FDX_s116
                                 + 0.970858 w_k1_IBM_s116
                                 + 0.746998 w_k1_INTC_s116
                                 + 0.864273 w_k1_KO_s116
                                 + 1.01866 w_k1_MAT_s116
                                 + 0.783404 w_k1_MSFT_s116
                                 + 0.8496 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.09316 w_k1_ADM_s116
                                 + 1.37685 w_k1_AXP_s116
                                 + 1.16812 w_k1_DIS_s116
                                 + 1.06287 w_k1_FDX_s116
                                 + 1.33025 w_k1_IBM_s116
                                 + 1.01018 w_k1_INTC_s116
                                 + 1.11941 w_k1_KO_s116 + 1.23444 w_k1_MAT_s116
                                 + 1.2151 w_k1_MSFT_s116
                                 + 0.984333 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.00566 w_k1_ADM_s116
                                 + 0.923992 w_k1_AXP_s116
                                 + 1.0683 w_k1_DIS_s116 + 1.02615 w_k1_FDX_s116
                                 + 0.936232 w_k1_IBM_s116
                                 + 0.81508 w_k1_INTC_s116
                                 + 1.20386 w_k1_KO_s116 + 1.09153 w_k1_MAT_s116
                                 + 1.08692 w_k1_MSFT_s116
                                 + 1.10297 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.35044 w_k1_ADM_s116
                                 + 1.57894 w_k1_AXP_s116
                                 + 1.83845 w_k1_DIS_s116
                                 + 1.45845 w_k1_FDX_s116
                                 + 1.23705 w_k1_IBM_s116
                                 + 1.21128 w_k1_INTC_s116
                                 + 1.43316 w_k1_KO_s116 + 1.47955 w_k1_MAT_s116
                                 + 1.33955 w_k1_MSFT_s116
                                 + 1.41008 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.05515 w_k1_ADM_s116
                                 + 0.926579 w_k1_AXP_s116
                                 + 1.47085 w_k1_DIS_s116
                                 + 1.20282 w_k1_FDX_s116
                                 + 0.624327 w_k1_IBM_s116
                                 + 0.990557 w_k1_INTC_s116
                                 + 1.12952 w_k1_KO_s116
                                 + 0.884422 w_k1_MAT_s116
                                 + 1.34562 w_k1_MSFT_s116
                                 + 1.14098 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.74478 w_k1_ADM_s116
                                 + 0.759346 w_k1_AXP_s116
                                 + 0.890673 w_k1_DIS_s116
                                 + 0.960411 w_k1_FDX_s116
                                 + 0.549739 w_k1_IBM_s116
                                 + 0.962236 w_k1_INTC_s116
                                 + 0.825117 w_k1_KO_s116
                                 + 0.803571 w_k1_MAT_s116
                                 + 1.07148 w_k1_MSFT_s116
                                 + 0.994844 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.3163 w_k1_ADM_s117
                                 + 1.52173 w_k1_AXP_s117
                                 + 1.71121 w_k1_DIS_s117
                                 + 1.36219 w_k1_FDX_s117
                                 + 1.48816 w_k1_IBM_s117
                                 + 1.11827 w_k1_INTC_s117
                                 + 1.30204 w_k1_KO_s117 + 1.88518 w_k1_MAT_s117
                                 + 1.23588 w_k1_MSFT_s117
                                 + 1.13273 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.818098 w_k1_ADM_s117
                                 + 0.986023 w_k1_AXP_s117
                                 + 0.917765 w_k1_DIS_s117
                                 + 0.878836 w_k1_FDX_s117
                                 + 1.1441 w_k1_IBM_s117
                                 + 0.981544 w_k1_INTC_s117
                                 + 0.8779 w_k1_KO_s117 + 1.14267 w_k1_MAT_s117
                                 + 0.989173 w_k1_MSFT_s117
                                 + 0.990757 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.758061 w_k1_ADM_s117
                                 + 0.788623 w_k1_AXP_s117
                                 + 0.734726 w_k1_DIS_s117
                                 + 0.84516 w_k1_FDX_s117
                                 + 0.672244 w_k1_IBM_s117
                                 + 0.859193 w_k1_INTC_s117
                                 + 0.647956 w_k1_KO_s117
                                 + 0.645037 w_k1_MAT_s117
                                 + 0.79378 w_k1_MSFT_s117
                                 + 0.85424 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.947132 w_k1_ADM_s117
                                 + 1.22919 w_k1_AXP_s117
                                 + 1.20709 w_k1_DIS_s117
                                 + 1.16776 w_k1_FDX_s117
                                 + 0.887866 w_k1_IBM_s117
                                 + 1.0962 w_k1_INTC_s117 + 1.03386 w_k1_KO_s117
                                 + 1.40819 w_k1_MAT_s117
                                 + 1.25579 w_k1_MSFT_s117
                                 + 1.09002 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.88734 w_k1_ADM_s117
                                 + 0.951564 w_k1_AXP_s117
                                 + 1.2983 w_k1_DIS_s117 + 1.02961 w_k1_FDX_s117
                                 + 1.38512 w_k1_IBM_s117
                                 + 1.14987 w_k1_INTC_s117
                                 + 1.18566 w_k1_KO_s117 + 1.48731 w_k1_MAT_s117
                                 + 1.13752 w_k1_MSFT_s117
                                 + 1.07997 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.905564 w_k1_ADM_s117
                                 + 0.760252 w_k1_AXP_s117
                                 + 1.05619 w_k1_DIS_s117
                                 + 0.924517 w_k1_FDX_s117
                                 + 0.770999 w_k1_IBM_s117
                                 + 1.10247 w_k1_INTC_s117
                                 + 0.890761 w_k1_KO_s117
                                 + 0.845419 w_k1_MAT_s117
                                 + 1.14006 w_k1_MSFT_s117
                                 + 0.913003 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.737979 w_k1_ADM_s117
                                 + 0.678824 w_k1_AXP_s117
                                 + 0.666283 w_k1_DIS_s117
                                 + 0.69071 w_k1_FDX_s117
                                 + 0.871154 w_k1_IBM_s117
                                 + 0.783328 w_k1_INTC_s117
                                 + 0.892891 w_k1_KO_s117
                                 + 0.827049 w_k1_MAT_s117
                                 + 0.807789 w_k1_MSFT_s117
                                 + 0.877313 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.910777 w_k1_ADM_s117
                                 + 0.919239 w_k1_AXP_s117
                                 + 0.78448 w_k1_DIS_s117
                                 + 0.819827 w_k1_FDX_s117
                                 + 1.06832 w_k1_IBM_s117
                                 + 0.915076 w_k1_INTC_s117
                                 + 0.815192 w_k1_KO_s117
                                 + 1.02127 w_k1_MAT_s117
                                 + 0.782108 w_k1_MSFT_s117
                                 + 0.88602 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0263101288493093 w_k1_ADM_s000
                                 + 0.0109623184338215 w_k1_ADM_s001
                                 + 0.0386348453656589 w_k1_ADM_s002
                                 + 0.0161315530724566 w_k1_ADM_s003
                                 + 0.0173271126783119 w_k1_ADM_s004
                                 + 0.0122325531730401 w_k1_ADM_s005
                                 + 0.0385289420218212 w_k1_ADM_s006
                                 + 0.014039955889066 w_k1_ADM_s007
                                 + 0.062541064372353 w_k1_ADM_s010
                                 + 0.0342884887470726 w_k1_ADM_s011
                                 + 0.0330430142592989 w_k1_ADM_s012
                                 + 0.0300527573667621 w_k1_ADM_s013
                                 + 0.0135139264515611 w_k1_ADM_s014
                                 + 0.0141592953600568 w_k1_ADM_s015
                                 + 0.0536935904441256 w_k1_ADM_s016
                                 + 0.0539528468590244 w_k1_ADM_s017
                                 + 0.0370935884849793 w_k1_ADM_s100
                                 + 0.0298174949885503 w_k1_ADM_s101
                                 + 0.0239975082531867 w_k1_ADM_s102
                                 + 0.0236532834011025 w_k1_ADM_s103
                                 + 0.0501592994030749 w_k1_ADM_s104
                                 + 0.0434237355040166 w_k1_ADM_s105
                                 + 0.0423474986799422 w_k1_ADM_s106
                                 + 0.0241669188387825 w_k1_ADM_s107
                                 + 0.0398552245109709 w_k1_ADM_s110
                                 + 0.0133067362816677 w_k1_ADM_s111
                                 + 0.0389713758497675 w_k1_ADM_s112
                                 + 0.0377413049699489 w_k1_ADM_s113
                                 + 0.0454072343734847 w_k1_ADM_s114
                                 + 0.0301059164020793 w_k1_ADM_s115
                                 + 0.0191654889312642 w_k1_ADM_s116
                                 + 0.0471685992169236 w_k1_ADM_s117
                                 + 0.0264149725684376 w_k1_AXP_s000
                                 + 0.0117449206285807 w_k1_AXP_s001
                                 + 0.0364433197991696 w_k1_AXP_s002
                                 + 0.0143767866502443 w_k1_AXP_s003
                                 + 0.0173530535167085 w_k1_AXP_s004
                                 + 0.012825487596816 w_k1_AXP_s005
                                 + 0.0404796518634281 w_k1_AXP_s006
                                 + 0.0152327863617293 w_k1_AXP_s007
                                 + 0.0599741164835143 w_k1_AXP_s010
                                 + 0.0299574739483775 w_k1_AXP_s011
                                 + 0.037803217601331 w_k1_AXP_s012
                                 + 0.0252673560884458 w_k1_AXP_s013
                                 + 0.0173281591321036 w_k1_AXP_s014
                                 + 0.0143136013536576 w_k1_AXP_s015
                                 + 0.0516212947125525 w_k1_AXP_s016
                                 + 0.0525620470121199 w_k1_AXP_s017
                                 + 0.0344893467638787 w_k1_AXP_s100
                                 + 0.0324576931856826 w_k1_AXP_s101
                                 + 0.0271142213387915 w_k1_AXP_s102
                                 + 0.0269306975711327 w_k1_AXP_s103
                                 + 0.0470510723815436 w_k1_AXP_s104
                                 + 0.0398057423611881 w_k1_AXP_s105
                                 + 0.0398093348039364 w_k1_AXP_s106
                                 + 0.0280567320684547 w_k1_AXP_s107
                                 + 0.0404029347597728 w_k1_AXP_s110
                                 + 0.0122695684864131 w_k1_AXP_s111
                                 + 0.0342447765884909 w_k1_AXP_s112
                                 + 0.0354323516026854 w_k1_AXP_s113
                                 + 0.0412287397280317 w_k1_AXP_s114
                                 + 0.0229067698308941 w_k1_AXP_s115
                                 + 0.0195602799686175 w_k1_AXP_s116
                                 + 0.0506505148972515 w_k1_AXP_s117
                                 + 0.0245253537200053 w_k1_DIS_s000
                                 + 0.011618681992445 w_k1_DIS_s001
                                 + 0.03726677647648 w_k1_DIS_s002
                                 + 0.0152729114978812 w_k1_DIS_s003
                                 + 0.0170696346113636 w_k1_DIS_s004
                                 + 0.0127443556048378 w_k1_DIS_s005
                                 + 0.0428460711471408 w_k1_DIS_s006
                                 + 0.0146472337451469 w_k1_DIS_s007
                                 + 0.0716741032209398 w_k1_DIS_s010
                                 + 0.0294218714524621 w_k1_DIS_s011
                                 + 0.0398541278813689 w_k1_DIS_s012
                                 + 0.0277327371059579 w_k1_DIS_s013
                                 + 0.0167845660607963 w_k1_DIS_s014
                                 + 0.0143060942666961 w_k1_DIS_s015
                                 + 0.0532158662213348 w_k1_DIS_s016
                                 + 0.0503151138620679 w_k1_DIS_s017
                                 + 0.0361669055663523 w_k1_DIS_s100
                                 + 0.0351406293594133 w_k1_DIS_s101
                                 + 0.0273359618818737 w_k1_DIS_s102
                                 + 0.0283676239854381 w_k1_DIS_s103
                                 + 0.0441523390989638 w_k1_DIS_s104
                                 + 0.0458937773814427 w_k1_DIS_s105
                                 + 0.0416589776585689 w_k1_DIS_s106
                                 + 0.0301864969077166 w_k1_DIS_s107
                                 + 0.0388431269878761 w_k1_DIS_s110
                                 + 0.0130726680945255 w_k1_DIS_s111
                                 + 0.0387109744991246 w_k1_DIS_s112
                                 + 0.038432572244077 w_k1_DIS_s113
                                 + 0.0441250063045802 w_k1_DIS_s114
                                 + 0.0247162298162198 w_k1_DIS_s115
                                 + 0.0202545703828639 w_k1_DIS_s116
                                 + 0.0567647847623453 w_k1_DIS_s117
                                 + 0.0251258006617699 w_k1_FDX_s000
                                 + 0.0113373802363557 w_k1_FDX_s001
                                 + 0.0375778928849015 w_k1_FDX_s002
                                 + 0.0155402396757274 w_k1_FDX_s003
                                 + 0.0168306980098414 w_k1_FDX_s004
                                 + 0.0128739998072847 w_k1_FDX_s005
                                 + 0.039839095009596 w_k1_FDX_s006
                                 + 0.0136961330331292 w_k1_FDX_s007
                                 + 0.0670787233567979 w_k1_FDX_s010
                                 + 0.0299288540238517 w_k1_FDX_s011
                                 + 0.0363135870093109 w_k1_FDX_s012
                                 + 0.0269349758563756 w_k1_FDX_s013
                                 + 0.0162951392659414 w_k1_FDX_s014
                                 + 0.0144572067351705 w_k1_FDX_s015
                                 + 0.0515053163702741 w_k1_FDX_s016
                                 + 0.0500209969450357 w_k1_FDX_s017
                                 + 0.0359729804464375 w_k1_FDX_s100
                                 + 0.0340500941444572 w_k1_FDX_s101
                                 + 0.0271102648057415 w_k1_FDX_s102
                                 + 0.0282159600377263 w_k1_FDX_s103
                                 + 0.0447774293279425 w_k1_FDX_s104
                                 + 0.0436666707545948 w_k1_FDX_s105
                                 + 0.0414151306907754 w_k1_FDX_s106
                                 + 0.029256153030162 w_k1_FDX_s107
                                 + 0.0395754447568562 w_k1_FDX_s110
                                 + 0.0121257666930175 w_k1_FDX_s111
                                 + 0.0393149510530886 w_k1_FDX_s112
                                 + 0.0373009354061825 w_k1_FDX_s113
                                 + 0.0404680950991696 w_k1_FDX_s114
                                 + 0.0245305212856805 w_k1_FDX_s115
                                 + 0.0197582924347749 w_k1_FDX_s116
                                 + 0.0515080676148902 w_k1_FDX_s117
                                 + 0.026295568070979 w_k1_IBM_s000
                                 + 0.0127043325203937 w_k1_IBM_s001
                                 + 0.0405938025106678 w_k1_IBM_s002
                                 + 0.0142435054899148 w_k1_IBM_s003
                                 + 0.0166536267524731 w_k1_IBM_s004
                                 + 0.0123891435949264 w_k1_IBM_s005
                                 + 0.0415031621808044 w_k1_IBM_s006
                                 + 0.0140257322579826 w_k1_IBM_s007
                                 + 0.0664342435134887 w_k1_IBM_s010
                                 + 0.0305011632165399 w_k1_IBM_s011
                                 + 0.0389384884995437 w_k1_IBM_s012
                                 + 0.0280131404004135 w_k1_IBM_s013
                                 + 0.0162013903549785 w_k1_IBM_s014
                                 + 0.014333405596235 w_k1_IBM_s015
                                 + 0.0519833708972512 w_k1_IBM_s016
                                 + 0.0501752669871919 w_k1_IBM_s017
                                 + 0.0379975809660041 w_k1_IBM_s100
                                 + 0.0365199841713446 w_k1_IBM_s101
                                 + 0.0286412449721426 w_k1_IBM_s102
                                 + 0.0301314989494079 w_k1_IBM_s103
                                 + 0.0435188886206268 w_k1_IBM_s104
                                 + 0.0447602452509389 w_k1_IBM_s105
                                 + 0.0433763561799581 w_k1_IBM_s106
                                 + 0.0300881737410635 w_k1_IBM_s107
                                 + 0.0417314089100696 w_k1_IBM_s110
                                 + 0.0119946735522324 w_k1_IBM_s111
                                 + 0.039305614157242 w_k1_IBM_s112
                                 + 0.0378298344295171 w_k1_IBM_s113
                                 + 0.0393483775592998 w_k1_IBM_s114
                                 + 0.0243311060429305 w_k1_IBM_s115
                                 + 0.0183288581748559 w_k1_IBM_s116
                                 + 0.0540709937753666 w_k1_IBM_s117
                                 + 0.0285187888113881 w_k1_INTC_s000
                                 + 0.0123828643462197 w_k1_INTC_s001
                                 + 0.0365893627425336 w_k1_INTC_s002
                                 + 0.0136437970700553 w_k1_INTC_s003
                                 + 0.0171059095621272 w_k1_INTC_s004
                                 + 0.0150458776201165 w_k1_INTC_s005
                                 + 0.0384893539780237 w_k1_INTC_s006
                                 + 0.0117107328018423 w_k1_INTC_s007
                                 + 0.0652701571458268 w_k1_INTC_s010
                                 + 0.0324738694978185 w_k1_INTC_s011
                                 + 0.0394810853744101 w_k1_INTC_s012
                                 + 0.0274841283379765 w_k1_INTC_s013
                                 + 0.0176097596732524 w_k1_INTC_s014
                                 + 0.0127257535861712 w_k1_INTC_s015
                                 + 0.0512153045723575 w_k1_INTC_s016
                                 + 0.0548308953322454 w_k1_INTC_s017
                                 + 0.0351773086337352 w_k1_INTC_s100
                                 + 0.0325311229674728 w_k1_INTC_s101
                                 + 0.0280678119364055 w_k1_INTC_s102
                                 + 0.0277211373622681 w_k1_INTC_s103
                                 + 0.0430843106256641 w_k1_INTC_s104
                                 + 0.0466174208269858 w_k1_INTC_s105
                                 + 0.0380748249500005 w_k1_INTC_s106
                                 + 0.0304687148390335 w_k1_INTC_s107
                                 + 0.0379113946902132 w_k1_INTC_s110
                                 + 0.0117268326327951 w_k1_INTC_s111
                                 + 0.0369596742189493 w_k1_INTC_s112
                                 + 0.0399387315854919 w_k1_INTC_s113
                                 + 0.0411744502157268 w_k1_INTC_s114
                                 + 0.0230158724768309 w_k1_INTC_s115
                                 + 0.0175348220193174 w_k1_INTC_s116
                                 + 0.0546279134211214 w_k1_INTC_s117
                                 + 0.0235926387414946 w_k1_KO_s000
                                 + 0.0125753087399817 w_k1_KO_s001
                                 + 0.0383230824994137 w_k1_KO_s002
                                 + 0.0155790176435305 w_k1_KO_s003
                                 + 0.0181590025015386 w_k1_KO_s004
                                 + 0.0117261963803651 w_k1_KO_s005
                                 + 0.0420628522259356 w_k1_KO_s006
                                 + 0.0148302157698534 w_k1_KO_s007
                                 + 0.0744849192699926 w_k1_KO_s010
                                 + 0.0320127674088594 w_k1_KO_s011
                                 + 0.0401834721934851 w_k1_KO_s012
                                 + 0.0277016153645485 w_k1_KO_s013
                                 + 0.0176145054343211 w_k1_KO_s014
                                 + 0.0134304488743607 w_k1_KO_s015
                                 + 0.0523415966949329 w_k1_KO_s016
                                 + 0.0456690140472114 w_k1_KO_s017
                                 + 0.0400370738557165 w_k1_KO_s100
                                 + 0.0332454373945674 w_k1_KO_s101
                                 + 0.0279364722075176 w_k1_KO_s102
                                 + 0.02758097890368 w_k1_KO_s103
                                 + 0.0452425705296582 w_k1_KO_s104
                                 + 0.0491627449190482 w_k1_KO_s105
                                 + 0.0461515588903968 w_k1_KO_s106
                                 + 0.0312773139995866 w_k1_KO_s107
                                 + 0.0389427207447023 w_k1_KO_s110
                                 + 0.0131007651218169 w_k1_KO_s111
                                 + 0.0403048514917644 w_k1_KO_s112
                                 + 0.0405799001079104 w_k1_KO_s113
                                 + 0.0444223694407458 w_k1_KO_s114
                                 + 0.0255557616186961 w_k1_KO_s115
                                 + 0.0198550737648522 w_k1_KO_s116
                                 + 0.0516587349502112 w_k1_KO_s117
                                 + 0.0237582431524401 w_k1_MAT_s000
                                 + 0.011395927698096 w_k1_MAT_s001
                                 + 0.0383974380654186 w_k1_MAT_s002
                                 + 0.0169720440514792 w_k1_MAT_s003
                                 + 0.0223397776934644 w_k1_MAT_s004
                                 + 0.0115814333737653 w_k1_MAT_s005
                                 + 0.036641105158174 w_k1_MAT_s006
                                 + 0.0177826267785067 w_k1_MAT_s007
                                 + 0.0749888470969243 w_k1_MAT_s010
                                 + 0.0286980004090867 w_k1_MAT_s011
                                 + 0.0373615152892815 w_k1_MAT_s012
                                 + 0.0285797764258208 w_k1_MAT_s013
                                 + 0.0164347587097547 w_k1_MAT_s014
                                 + 0.0139245957335205 w_k1_MAT_s015
                                 + 0.0487761025520597 w_k1_MAT_s016
                                 + 0.0558911490817591 w_k1_MAT_s017
                                 + 0.0433906429559362 w_k1_MAT_s100
                                 + 0.0357075555473249 w_k1_MAT_s101
                                 + 0.0309525307338409 w_k1_MAT_s102
                                 + 0.0339424929342498 w_k1_MAT_s103
                                 + 0.0545684656883101 w_k1_MAT_s104
                                 + 0.0452436678612501 w_k1_MAT_s105
                                 + 0.0429336393985995 w_k1_MAT_s106
                                 + 0.0294633706198563 w_k1_MAT_s107
                                 + 0.0461351528665459 w_k1_MAT_s110
                                 + 0.016747558353483 w_k1_MAT_s111
                                 + 0.0450763528975939 w_k1_MAT_s112
                                 + 0.0411136822945487 w_k1_MAT_s113
                                 + 0.0377832008753036 w_k1_MAT_s114
                                 + 0.02140361209883 w_k1_MAT_s115
                                 + 0.0193095715893795 w_k1_MAT_s116
                                 + 0.061683668520719 w_k1_MAT_s117
                                 + 0.0248603873308908 w_k1_MSFT_s000
                                 + 0.0115022820315339 w_k1_MSFT_s001
                                 + 0.0394165666614996 w_k1_MSFT_s002
                                 + 0.0143144553538198 w_k1_MSFT_s003
                                 + 0.0188706213312688 w_k1_MSFT_s004
                                 + 0.0128828016027309 w_k1_MSFT_s005
                                 + 0.0385711222635985 w_k1_MSFT_s006
                                 + 0.0130198560875088 w_k1_MSFT_s007
                                 + 0.0684380233376103 w_k1_MSFT_s010
                                 + 0.0313745941216152 w_k1_MSFT_s011
                                 + 0.0385692649421804 w_k1_MSFT_s012
                                 + 0.0284175650216965 w_k1_MSFT_s013
                                 + 0.0158211647470782 w_k1_MSFT_s014
                                 + 0.0138111616818917 w_k1_MSFT_s015
                                 + 0.0478753568296487 w_k1_MSFT_s016
                                 + 0.0524447146913692 w_k1_MSFT_s017
                                 + 0.0347910473633455 w_k1_MSFT_s100
                                 + 0.0347455453138873 w_k1_MSFT_s101
                                 + 0.0269521885685755 w_k1_MSFT_s102
                                 + 0.0279900958447999 w_k1_MSFT_s103
                                 + 0.0438560677494812 w_k1_MSFT_s104
                                 + 0.043573795603707 w_k1_MSFT_s105
                                 + 0.0408259160746979 w_k1_MSFT_s106
                                 + 0.0306463930241758 w_k1_MSFT_s107
                                 + 0.0412145398524881 w_k1_MSFT_s110
                                 + 0.0126730612862058 w_k1_MSFT_s111
                                 + 0.0398734009270484 w_k1_MSFT_s112
                                 + 0.0396246694016547 w_k1_MSFT_s113
                                 + 0.0358219008067502 w_k1_MSFT_s114
                                 + 0.0243398717874284 w_k1_MSFT_s115
                                 + 0.0209057381652316 w_k1_MSFT_s116
                                 + 0.0563535867635428 w_k1_MSFT_s117
                                 + 0.0249231646235884 w_k1_PG_s000
                                 + 0.0118593463316457 w_k1_PG_s001
                                 + 0.0372559665234344 w_k1_PG_s002
                                 + 0.0162379257935748 w_k1_PG_s003
                                 + 0.017738943087571 w_k1_PG_s004
                                 + 0.0122452932921394 w_k1_PG_s005
                                 + 0.0384826477237974 w_k1_PG_s006
                                 + 0.0143838970619502 w_k1_PG_s007
                                 + 0.0677179070662667 w_k1_PG_s010
                                 + 0.0302901065697035 w_k1_PG_s011
                                 + 0.0384330349390398 w_k1_PG_s012
                                 + 0.0275790827853493 w_k1_PG_s013
                                 + 0.016798070930387 w_k1_PG_s014
                                 + 0.0132261981136892 w_k1_PG_s015
                                 + 0.0482149371683959 w_k1_PG_s016
                                 + 0.0522401514855749 w_k1_PG_s017
                                 + 0.0376299478414534 w_k1_PG_s100
                                 + 0.0350754453286554 w_k1_PG_s101
                                 + 0.0264878239342958 w_k1_PG_s102
                                 + 0.0289093686342726 w_k1_PG_s103
                                 + 0.0462908781854699 w_k1_PG_s104
                                 + 0.0441640049396986 w_k1_PG_s105
                                 + 0.042190253545694 w_k1_PG_s106
                                 + 0.0316416124765466 w_k1_PG_s107
                                 + 0.0390423781599129 w_k1_PG_s110
                                 + 0.0115786856553932 w_k1_PG_s111
                                 + 0.0418292580660532 w_k1_PG_s112
                                 + 0.038707401021032 w_k1_PG_s113
                                 + 0.0398379238672965 w_k1_PG_s114
                                 + 0.0254583793107959 w_k1_PG_s115
                                 + 0.0193702601335601 w_k1_PG_s116
                                 + 0.0524723823219263 w_k1_PG_s117 >= 1.09
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
