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
 w_k1_ADM_s0_return_constraint:  1.09093 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.903206 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.06199 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.24182 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.965123 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.15897 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.19703 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.825865 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.0034 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.16613 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.687249 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.06461 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.753995 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.45845 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.600114 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.82124 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.06823 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.89714 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.04638 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.767843 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.04596 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.755331 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.12062 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.735905 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.01913 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.795539 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.09981 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.872053 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.904393 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.74959 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.17284 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.3775 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.927048 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.755166 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.32273 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.15925 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.943168 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.76106 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24165 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.894184 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.03146 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.32024 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.984014 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.12264 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.2681 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.85436 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.082 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.01065 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.646067 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.0323 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.655699 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.84914 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.880701 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.763176 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.43811 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.849211 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.45921 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.01006 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.03532 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.627191 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.3314 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.535293 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.00102 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.673794 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.16875 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.965622 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.825507 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.64905 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.991467 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.72802 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.858245 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.899159 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.4371 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.0868 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.04529 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.670441 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.33017 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.882687 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.970169 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.36048 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.963179 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.13776 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.55696 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.817335 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.30991 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.09144 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.576029 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.97287 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.688366 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.97713 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.553157 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.939965 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.3386 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.699075 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.60042 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.940802 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.12258 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.644361 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.68572 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.447263 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.03597 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.583664 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.19161 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.817255 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.783226 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.89453 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.22674 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.37844 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.03269 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.845475 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.72894 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.901426 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.03799 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.585736 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17009 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.94419 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.0266 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.11207 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.926656 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.17224 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.31428 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.85113 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.10151 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.16876 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.673322 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.908537 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.75319 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.53039 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.681809 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.873408 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.37378 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.840732 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.36106 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.892295 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.16823 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.83552 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.35188 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.632432 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.07572 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.670204 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.06189 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.836067 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.90841 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.439 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.17291 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.28847 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.961595 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.783418 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.30877 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.971683 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.955565 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.71639 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.26776 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.921145 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.943757 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.56594 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.15344 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.82455 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.859737 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.844034 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.17801 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.661252 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.14142 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.42066 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.617356 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.40509 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.03057 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.02902 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.923649 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.781328 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.34846 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.3594 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.688681 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.595556 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.10277 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.490426 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.835817 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.746563 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.16469 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.32446 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.741452 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.82995 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.763768 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.06975 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.863705 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.18609 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.56411 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.02169 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.24287 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.701687 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.17325 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.960028 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.980242 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.893604 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.942599 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.07869 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.34952 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.933053 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.14489 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.01817 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.70044 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.04901 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.765869 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.41482 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.969217 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.26278 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.51471 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.10267 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.60923 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.22239 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.20579 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.08213 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.3947 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.717579 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.92011 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.757998 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.977078 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.779097 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.814123 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.38102 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.03312 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.08427 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.975317 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.874554 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.11321 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.00872 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.918921 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.83176 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.22537 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.934918 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.947248 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.39949 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.957831 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.997209 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.15737 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.88955 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.2043 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.920169 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.85709 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.16486 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.670264 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.58827 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.760296 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.938497 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.09942 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.73673 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.35206 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.14866 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.883279 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.705668 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.34587 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.55226 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.02308 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.768005 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.12147 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.964917 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.83873 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.67942 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.979132 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.08336 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.924134 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.918428 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.39991 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.04106 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.01298 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.685243 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.17898 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.913479 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.995244 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 2.02623 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.936953 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.861379 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.913388 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.01892 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.20569 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.748699 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.0781 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.56504 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.692153 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.95446 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.876166 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.741374 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.811373 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.675812 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.17305 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.24721 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.723751 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.457754 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.26134 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.48383 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.13535 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.761603 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.31305 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.41287 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.973843 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 2.76379 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.775969 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.35614 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.770513 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.21267 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.65249 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.00036 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.18212 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.708004 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15609 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.976778 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.970429 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.12395 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.946231 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.05852 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.38996 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.924003 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.14635 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.06867 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.697242 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.07167 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.857844 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.48082 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.651676 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.973217 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.26125 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.80233 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.50846 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.867683 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.985009 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.871925 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.26107 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.64099 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.99496 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.689178 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.999927 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.749048 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.849559 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.33047 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.23605 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.15119 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.986045 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.832251 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.21375 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.979209 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.06322 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.816582 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10061 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.953099 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.951032 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.1705 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.968871 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.06319 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.09531 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.961821 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.12285 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.14986 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.885384 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.11042 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.879443 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.5653 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.820127 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.929929 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.07605 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.856435 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.27985 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.99843 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.995301 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.805076 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.22398 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.703306 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.01187 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.804861 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.07397 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.966535 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.897158 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.30876 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.04341 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.05838 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.884672 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.93624 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.24507 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.96888 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.01253 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.789593 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.161999918749123 z_k1_s0000
                                 - 0.231000044436741 z_k1_s0001
                                 - 0.415000641903769 z_k1_s0002
                                 - 0.691000620054148 z_k1_s0003
                                 - 0.240999971417539 z_k1_s0004
                                 - 0.44800061078044 z_k1_s0005
                                 - 0.288000205287392 z_k1_s0006
                                 - 0.147000028277926 z_k1_s0007
                                 - 1.13400255548456 z_k1_s0010
                                 - 0.285000919077584 z_k1_s0011
                                 - 0.561002772923841 z_k1_s0012
                                 - 0.744002950014605 z_k1_s0013
                                 - 0.477001207802609 z_k1_s0014
                                 - 0.686000835389903 z_k1_s0015
                                 - 0.513001654339651 z_k1_s0016
                                 - 0.83200020473884 z_k1_s0017
                                 - 0.717001303685584 z_k1_s0020
                                 - 0.328000664302603 z_k1_s0021
                                 - 0.237000648258002 z_k1_s0022
                                 - 0.841001851751782 z_k1_s0023
                                 - 0.274000347086525 z_k1_s0024
                                 - 0.638000005464138 z_k1_s0025
                                 - 0.246000498226953 z_k1_s0026
                                 - 0.777003008817201 z_k1_s0027
                                 - 0.208000311012271 z_k1_s0030
                                 - 0.177000182638254 z_k1_s0031
                                 - 0.444001366930666 z_k1_s0032
                                 - 0.487001737178668 z_k1_s0033
                                 - 0.148000184843315 z_k1_s0034
                                 - 0.195000274648989 z_k1_s0035
                                 - 0.372000998887725 z_k1_s0036
                                 - 0.677000602178999 z_k1_s0037
                                 - 0.561000229299059 z_k1_s0040
                                 - 0.441999169416813 z_k1_s0041
                                 - 0.35599948202359 z_k1_s0042
                                 - 0.344001316574485 z_k1_s0043
                                 - 0.252000236155275 z_k1_s0044
                                 - 0.236000115221541 z_k1_s0045
                                 - 0.135000154015343 z_k1_s0046
                                 - 0.241000120925813 z_k1_s0047
                                 - 0.379999305890542 z_k1_s0050
                                 - 0.281000944575972 z_k1_s0051
                                 - 0.692003628109775 z_k1_s0052
                                 - 0.440000566087808 z_k1_s0053
                                 - 0.35200119647243 z_k1_s0054
                                 - 0.543000622553595 z_k1_s0055
                                 - 0.244000787122811 z_k1_s0056
                                 - 0.786002380142696 z_k1_s0057
                                 - 0.279999508465655 z_k1_s0060
                                 - 0.38699800577373 z_k1_s0061
                                 - 0.671998820317573 z_k1_s0062
                                 - 0.575001047168107 z_k1_s0063
                                 - 0.328999469647089 z_k1_s0064
                                 - 0.3009997076003 z_k1_s0065
                                 - 0.393998701484533 z_k1_s0066
                                 - 0.83799913576435 z_k1_s0067
                                 - 0.206000069139334 z_k1_s0070
                                 - 0.433000578445789 z_k1_s0071
                                 - 0.167000046328975 z_k1_s0072
                                 - 0.157999921257353 z_k1_s0073
                                 - 0.490999653907342 z_k1_s0074
                                 - 0.483999803851637 z_k1_s0075
                                 - 0.118999898446993 z_k1_s0076
                                 - 0.167000046328975 z_k1_s0077
                                 - 0.197000072687906 z_k1_s0100
                                 - 0.134000081422226 z_k1_s0101
                                 - 0.0749999596020041 z_k1_s0102
                                 - 0.0790000090474372 z_k1_s0103
                                 - 0.138999580729095 z_k1_s0104
                                 - 0.0420000241771158 z_k1_s0105
                                 - 0.138999580729095 z_k1_s0106
                                 - 0.0950003868291445 z_k1_s0107
                                 - 0.160000005970304 z_k1_s0110
                                 - 0.0329999517064589 z_k1_s0111
                                 - 0.0670001369248381 z_k1_s0112
                                 - 0.0660000166127267 z_k1_s0113
                                 - 0.0670001369248381 z_k1_s0114
                                 - 0.0670001369248381 z_k1_s0115
                                 - 0.0620001013633258 z_k1_s0116
                                 - 0.043999916741977 z_k1_s0117
                                 - 0.101999533565869 z_k1_s0120
                                 - 0.114999949881892 z_k1_s0121
                                 - 0.120999208799325 z_k1_s0122
                                 - 0.0869997686763521 z_k1_s0123
                                 - 0.217999359934 z_k1_s0124
                                 - 0.0869997686763521 z_k1_s0125
                                 - 0.214999730475283 z_k1_s0126
                                 - 0.0659998349716891 z_k1_s0127
                                 - 0.0959997389015499 z_k1_s0130
                                 - 0.067999580264909 z_k1_s0131
                                 - 0.154999469424986 z_k1_s0132
                                 - 0.0669996148490899 z_k1_s0133
                                 - 0.105999393059741 z_k1_s0134
                                 - 0.127999115205064 z_k1_s0135
                                 - 0.135999160529818 z_k1_s0136
                                 - 0.0489997543670436 z_k1_s0137
                                 - 0.231999612893321 z_k1_s0140
                                 - 0.0699998295627479 z_k1_s0141
                                 - 0.12499972832627 z_k1_s0142
                                 - 0.0479998883573023 z_k1_s0143
                                 - 0.110000163526307 z_k1_s0144
                                 - 0.163999694803963 z_k1_s0145
                                 - 0.0779998414555702 z_k1_s0146
                                 - 0.087999856321598 z_k1_s0147
                                 - 0.169999896834901 z_k1_s0150
                                 - 0.097000018923356 z_k1_s0151
                                 - 0.0359999331180297 z_k1_s0152
                                 - 0.0399999798199721 z_k1_s0153
                                 - 0.0989996246747968 z_k1_s0154
                                 - 0.0519999111260341 z_k1_s0155
                                 - 0.120000009059838 z_k1_s0156
                                 - 0.0819998437910719 z_k1_s0157
                                 - 0.0969998169297119 z_k1_s0160
                                 - 0.074999650987325 z_k1_s0161
                                 - 0.0329999213142662 z_k1_s0162
                                 - 0.0389999027532326 z_k1_s0163
                                 - 0.0879997277715072 z_k1_s0164
                                 - 0.0259999663687564 z_k1_s0165
                                 - 0.0479998515117312 z_k1_s0166
                                 - 0.0620000422021629 z_k1_s0167
                                 - 0.0929997065682655 z_k1_s0170
                                 - 0.0219999298183012 z_k1_s0171
                                 - 0.0219999298183012 z_k1_s0172
                                 - 0.0499998729506072 z_k1_s0173
                                 - 0.020999939499272 z_k1_s0174
                                 - 0.0199999491802429 z_k1_s0175
                                 - 0.0319999401083376 z_k1_s0176
                                 - 0.096999882043872 z_k1_s0177
                                 - 0.947002711600547 z_k1_s1000
                                 - 0.976004249195765 z_k1_s1001
                                 - 0.591002430799455 z_k1_s1002
                                 - 0.57800192563665 z_k1_s1003
                                 - 0.69100220973142 z_k1_s1004
                                 - 0.531001495636978 z_k1_s1005
                                 - 0.35800134418303 z_k1_s1006
                                 - 0.667003330591047 z_k1_s1007
                                 - 0.271000836028047 z_k1_s1010
                                 - 0.513000266930544 z_k1_s1011
                                 - 0.149000073185672 z_k1_s1012
                                 - 0.579000561904219 z_k1_s1013
                                 - 0.236000079359835 z_k1_s1014
                                 - 0.319000600372272 z_k1_s1015
                                 - 0.182000220672509 z_k1_s1016
                                 - 0.227000061645256 z_k1_s1017
                                 - 1.1869974787721 z_k1_s1020
                                 - 0.918998899044147 z_k1_s1021
                                 - 0.547001031720244 z_k1_s1022
                                 - 0.71100032385225 z_k1_s1023
                                 - 0.531999680086442 z_k1_s1024
                                 - 0.28499975144632 z_k1_s1025
                                 - 0.648002209789091 z_k1_s1026
                                 - 0.57300085361923 z_k1_s1027
                                 - 0.407998866638868 z_k1_s1030
                                 - 0.459997319528835 z_k1_s1031
                                 - 0.235999169095609 z_k1_s1032
                                 - 0.416998512889338 z_k1_s1033
                                 - 0.563996907899931 z_k1_s1034
                                 - 0.730997149714006 z_k1_s1035
                                 - 0.594998719763788 z_k1_s1036
                                 - 1.05999460082475 z_k1_s1037
                                 - 0.644000425431567 z_k1_s1040
                                 - 0.733998869367262 z_k1_s1041
                                 - 0.699000961699617 z_k1_s1042
                                 - 1.35800390120714 z_k1_s1043
                                 - 1.13200393390839 z_k1_s1044
                                 - 0.623000812629934 z_k1_s1045
                                 - 0.457000772622813 z_k1_s1046
                                 - 1.58800482832649 z_k1_s1047
                                 - 0.500001961898582 z_k1_s1050
                                 - 0.923002245445683 z_k1_s1051
                                 - 0.753004305470271 z_k1_s1052
                                 - 0.365001421615926 z_k1_s1053
                                 - 0.972004947039043 z_k1_s1054
                                 - 0.826000995980138 z_k1_s1055
                                 - 0.371001504350928 z_k1_s1056
                                 - 0.574999560823381 z_k1_s1057
                                 - 0.679999774275367 z_k1_s1060
                                 - 1.22000048375887 z_k1_s1061
                                 - 1.35599911201376 z_k1_s1062
                                 - 1.50400347188017 z_k1_s1063
                                 - 0.901999681074088 z_k1_s1064
                                 - 0.361000336456543 z_k1_s1065
                                 - 0.531999836409554 z_k1_s1066
                                 - 0.814996819395822 z_k1_s1067
                                 - 0.414000282351454 z_k1_s1070
                                 - 0.371000033157745 z_k1_s1071
                                 - 0.47100018930552 z_k1_s1072
                                 - 0.598999660974007 z_k1_s1073
                                 - 0.207000141175727 z_k1_s1074
                                 - 0.618999692203562 z_k1_s1075
                                 - 0.404002087238336 z_k1_s1076
                                 - 0.555999411780298 z_k1_s1077
                                 - 0.316000012095224 z_k1_s1100
                                 - 0.105999920930689 z_k1_s1101
                                 - 0.100000071840251 z_k1_s1102
                                 - 0.413000557390195 z_k1_s1103
                                 - 0.319000931640293 z_k1_s1104
                                 - 0.401000859209318 z_k1_s1105
                                 - 0.224000336042135 z_k1_s1106
                                 - 0.110999994172691 z_k1_s1107
                                 - 0.297000175554399 z_k1_s1110
                                 - 0.449999383143458 z_k1_s1111
                                 - 0.154999880016035 z_k1_s1112
                                 - 0.258000173521993 z_k1_s1113
                                 - 0.334000337559424 z_k1_s1114
                                 - 0.10599984544713 z_k1_s1115
                                 - 0.332000497532043 z_k1_s1116
                                 - 0.148999943734095 z_k1_s1117
                                 - 0.445999894550511 z_k1_s1120
                                 - 0.487998470273488 z_k1_s1121
                                 - 0.352998092031482 z_k1_s1122
                                 - 0.62099921671853 z_k1_s1123
                                 - 0.176999584763656 z_k1_s1124
                                 - 0.295999439237245 z_k1_s1125
                                 - 0.53299964768528 z_k1_s1126
                                 - 0.239999340744143 z_k1_s1127
                                 - 0.221999812040438 z_k1_s1130
                                 - 0.348000007219819 z_k1_s1131
                                 - 0.186999744335069 z_k1_s1132
                                 - 0.198999878542504 z_k1_s1133
                                 - 0.249000842406378 z_k1_s1134
                                 - 0.560998504605995 z_k1_s1135
                                 - 0.422000511099352 z_k1_s1136
                                 - 0.239999770551853 z_k1_s1137
                                 - 0.290000611328798 z_k1_s1140
                                 - 0.39700077933416 z_k1_s1141
                                 - 0.314000259044342 z_k1_s1142
                                 - 0.431000566623298 z_k1_s1143
                                 - 1.36800299212222 z_k1_s1144
                                 - 1.40700014924392 z_k1_s1145
                                 - 0.306000540106084 z_k1_s1146
                                 - 0.396000814466878 z_k1_s1147
                                 - 0.131999963489151 z_k1_s1150
                                 - 0.172999927273663 z_k1_s1151
                                 - 0.366000744515244 z_k1_s1152
                                 - 0.479999469233335 z_k1_s1153
                                 - 0.134999987534355 z_k1_s1154
                                 - 0.326999337427755 z_k1_s1155
                                 - 0.389000564028531 z_k1_s1156
                                 - 0.186999893551305 z_k1_s1157
                                 - 0.836997253618123 z_k1_s1160
                                 - 0.319998790615936 z_k1_s1161
                                 - 0.36499731574278 z_k1_s1162
                                 - 0.219999147754774 z_k1_s1163
                                 - 0.563997839349432 z_k1_s1164
                                 - 0.278998973639739 z_k1_s1165
                                 - 0.371997300724028 z_k1_s1166
                                 - 0.369997780299252 z_k1_s1167
                                 - 0.920998909595608 z_k1_s1170
                                 - 0.349999643036958 z_k1_s1171
                                 - 0.414999568060968 z_k1_s1172
                                 - 0.882002479819554 z_k1_s1173
                                 - 0.493999551742423 z_k1_s1174
                                 - 1.0579969655732 z_k1_s1175
                                 - 1.03299816325573 z_k1_s1176
                                 - 0.92499823172663 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.0158 w_k1_ADM_s000
                                 + 0.708781 w_k1_AXP_s000
                                 + 0.971725 w_k1_DIS_s000
                                 + 0.792109 w_k1_FDX_s000
                                 + 1.43969 w_k1_IBM_s000
                                 + 0.782266 w_k1_INTC_s000
                                 + 1.13849 w_k1_KO_s000 + 1.03643 w_k1_MAT_s000
                                 + 0.929781 w_k1_MSFT_s000
                                 + 1.15792 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.09672 w_k1_ADM_s000
                                 + 0.934577 w_k1_AXP_s000
                                 + 0.750482 w_k1_DIS_s000
                                 + 0.984234 w_k1_FDX_s000
                                 + 0.847807 w_k1_IBM_s000
                                 + 0.885754 w_k1_INTC_s000
                                 + 0.916844 w_k1_KO_s000
                                 + 1.05068 w_k1_MAT_s000
                                 + 0.901441 w_k1_MSFT_s000
                                 + 1.07217 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.933931 w_k1_ADM_s000
                                 + 1.14085 w_k1_AXP_s000
                                 + 1.47046 w_k1_DIS_s000
                                 + 1.21507 w_k1_FDX_s000
                                 + 1.10088 w_k1_IBM_s000
                                 + 1.18083 w_k1_INTC_s000
                                 + 1.29123 w_k1_KO_s000
                                 + 0.910066 w_k1_MAT_s000
                                 + 1.15361 w_k1_MSFT_s000
                                 + 1.16644 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.46811 w_k1_ADM_s000
                                 + 1.28245 w_k1_AXP_s000
                                 + 1.15948 w_k1_DIS_s000
                                 + 1.22293 w_k1_FDX_s000
                                 + 1.28366 w_k1_IBM_s000
                                 + 1.01487 w_k1_INTC_s000
                                 + 1.27196 w_k1_KO_s000 + 1.68474 w_k1_MAT_s000
                                 + 1.10522 w_k1_MSFT_s000
                                 + 1.42254 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.990548 w_k1_ADM_s000
                                 + 0.62979 w_k1_AXP_s000
                                 + 0.682561 w_k1_DIS_s000
                                 + 0.786204 w_k1_FDX_s000
                                 + 0.81209 w_k1_IBM_s000
                                 + 0.806023 w_k1_INTC_s000
                                 + 0.846669 w_k1_KO_s000
                                 + 0.800317 w_k1_MAT_s000
                                 + 0.841402 w_k1_MSFT_s000
                                 + 0.911688 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.982211 w_k1_ADM_s000
                                 + 0.835802 w_k1_AXP_s000
                                 + 0.869452 w_k1_DIS_s000
                                 + 0.947833 w_k1_FDX_s000
                                 + 0.889917 w_k1_IBM_s000
                                 + 0.800689 w_k1_INTC_s000
                                 + 0.931163 w_k1_KO_s000
                                 + 0.618565 w_k1_MAT_s000
                                 + 0.83775 w_k1_MSFT_s000
                                 + 1.08132 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.893343 w_k1_ADM_s000
                                 + 1.16311 w_k1_AXP_s000
                                 + 0.814252 w_k1_DIS_s000
                                 + 0.944519 w_k1_FDX_s000
                                 + 1.25813 w_k1_IBM_s000
                                 + 0.809722 w_k1_INTC_s000
                                 + 0.948985 w_k1_KO_s000
                                 + 1.59329 w_k1_MAT_s000
                                 + 0.835237 w_k1_MSFT_s000
                                 + 1.05345 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.843053 w_k1_ADM_s000
                                 + 0.676675 w_k1_AXP_s000
                                 + 0.532558 w_k1_DIS_s000
                                 + 0.703811 w_k1_FDX_s000
                                 + 0.825592 w_k1_IBM_s000
                                 + 0.656723 w_k1_INTC_s000
                                 + 0.758891 w_k1_KO_s000
                                 + 0.714783 w_k1_MAT_s000
                                 + 0.658082 w_k1_MSFT_s000
                                 + 0.864104 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.99055 w_k1_ADM_s001
                                 + 0.858401 w_k1_AXP_s001
                                 + 0.715017 w_k1_DIS_s001
                                 + 0.794479 w_k1_FDX_s001
                                 + 0.994367 w_k1_IBM_s001
                                 + 0.689522 w_k1_INTC_s001
                                 + 0.868381 w_k1_KO_s001
                                 + 0.949738 w_k1_MAT_s001
                                 + 0.773883 w_k1_MSFT_s001
                                 + 0.898781 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.90763 w_k1_ADM_s001
                                 + 1.03833 w_k1_AXP_s001
                                 + 0.986174 w_k1_DIS_s001
                                 + 0.908694 w_k1_FDX_s001
                                 + 1.64422 w_k1_IBM_s001
                                 + 1.1651 w_k1_INTC_s001 + 1.1206 w_k1_KO_s001
                                 + 1.12039 w_k1_MAT_s001
                                 + 0.936556 w_k1_MSFT_s001
                                 + 0.982811 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.35838 w_k1_ADM_s001
                                 + 1.51701 w_k1_AXP_s001
                                 + 1.20667 w_k1_DIS_s001 + 1.3279 w_k1_FDX_s001
                                 + 1.1003 w_k1_IBM_s001
                                 + 0.922293 w_k1_INTC_s001
                                 + 1.07181 w_k1_KO_s001 + 1.27022 w_k1_MAT_s001
                                 + 1.12351 w_k1_MSFT_s001 + 1.2251 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.20296 w_k1_ADM_s001
                                 + 1.02581 w_k1_AXP_s001
                                 + 1.28449 w_k1_DIS_s001
                                 + 1.07781 w_k1_FDX_s001
                                 + 1.09861 w_k1_IBM_s001
                                 + 1.06543 w_k1_INTC_s001
                                 + 1.07144 w_k1_KO_s001 + 1.13831 w_k1_MAT_s001
                                 + 1.18228 w_k1_MSFT_s001
                                 + 1.11263 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.26422 w_k1_ADM_s001
                                 + 1.08784 w_k1_AXP_s001
                                 + 1.01709 w_k1_DIS_s001
                                 + 1.13606 w_k1_FDX_s001
                                 + 0.76779 w_k1_IBM_s001
                                 + 0.972377 w_k1_INTC_s001
                                 + 0.884939 w_k1_KO_s001
                                 + 0.760483 w_k1_MAT_s001
                                 + 1.06253 w_k1_MSFT_s001
                                 + 1.02816 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.2155 w_k1_ADM_s001
                                 + 0.910393 w_k1_AXP_s001
                                 + 0.925892 w_k1_DIS_s001
                                 + 1.00224 w_k1_FDX_s001
                                 + 1.13683 w_k1_IBM_s001
                                 + 0.945236 w_k1_INTC_s001
                                 + 1.00818 w_k1_KO_s001
                                 + 0.862398 w_k1_MAT_s001
                                 + 1.00313 w_k1_MSFT_s001
                                 + 1.07652 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.06577 w_k1_ADM_s001
                                 + 1.4282 w_k1_AXP_s001 + 1.64125 w_k1_DIS_s001
                                 + 1.23828 w_k1_FDX_s001
                                 + 1.44836 w_k1_IBM_s001
                                 + 1.35289 w_k1_INTC_s001 + 1.4978 w_k1_KO_s001
                                 + 1.54582 w_k1_MAT_s001
                                 + 1.2547 w_k1_MSFT_s001 + 1.24698 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.04191 w_k1_ADM_s001
                                 + 0.931297 w_k1_AXP_s001
                                 + 0.79865 w_k1_DIS_s001
                                 + 0.910132 w_k1_FDX_s001
                                 + 0.721879 w_k1_IBM_s001
                                 + 0.952308 w_k1_INTC_s001
                                 + 0.828312 w_k1_KO_s001
                                 + 0.474799 w_k1_MAT_s001
                                 + 0.88428 w_k1_MSFT_s001
                                 + 0.950394 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.16444 w_k1_ADM_s002
                                 + 1.02074 w_k1_AXP_s002
                                 + 0.985566 w_k1_DIS_s002
                                 + 1.05537 w_k1_FDX_s002
                                 + 0.901637 w_k1_IBM_s002
                                 + 1.0154 w_k1_INTC_s002 + 0.90354 w_k1_KO_s002
                                 + 0.98773 w_k1_MAT_s002
                                 + 0.905996 w_k1_MSFT_s002
                                 + 1.02796 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.39794 w_k1_ADM_s002
                                 + 1.46641 w_k1_AXP_s002
                                 + 1.25562 w_k1_DIS_s002
                                 + 1.30037 w_k1_FDX_s002
                                 + 0.981981 w_k1_IBM_s002
                                 + 1.04028 w_k1_INTC_s002
                                 + 1.06169 w_k1_KO_s002 + 1.53272 w_k1_MAT_s002
                                 + 1.24942 w_k1_MSFT_s002
                                 + 1.20968 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.789953 w_k1_ADM_s002
                                 + 1.07065 w_k1_AXP_s002
                                 + 0.895961 w_k1_DIS_s002
                                 + 0.917455 w_k1_FDX_s002
                                 + 1.41208 w_k1_IBM_s002
                                 + 1.07084 w_k1_INTC_s002
                                 + 1.10143 w_k1_KO_s002 + 1.63202 w_k1_MAT_s002
                                 + 0.981292 w_k1_MSFT_s002
                                 + 1.01826 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.920754 w_k1_ADM_s002
                                 + 0.799567 w_k1_AXP_s002
                                 + 0.756177 w_k1_DIS_s002
                                 + 0.839177 w_k1_FDX_s002
                                 + 0.725641 w_k1_IBM_s002
                                 + 0.877682 w_k1_INTC_s002
                                 + 0.770608 w_k1_KO_s002
                                 + 0.672397 w_k1_MAT_s002
                                 + 0.987175 w_k1_MSFT_s002
                                 + 0.940747 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.879077 w_k1_ADM_s002
                                 + 1.2183 w_k1_AXP_s002 + 1.27147 w_k1_DIS_s002
                                 + 1.08094 w_k1_FDX_s002
                                 + 1.51865 w_k1_IBM_s002
                                 + 1.15425 w_k1_INTC_s002
                                 + 1.42642 w_k1_KO_s002 + 1.79786 w_k1_MAT_s002
                                 + 1.11724 w_k1_MSFT_s002
                                 + 1.27855 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.969286 w_k1_ADM_s002
                                 + 0.807567 w_k1_AXP_s002
                                 + 0.76728 w_k1_DIS_s002
                                 + 0.764133 w_k1_FDX_s002
                                 + 0.945842 w_k1_IBM_s002
                                 + 0.910254 w_k1_INTC_s002
                                 + 0.97031 w_k1_KO_s002
                                 + 0.964652 w_k1_MAT_s002
                                 + 0.80171 w_k1_MSFT_s002
                                 + 0.919766 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.35118 w_k1_ADM_s002
                                 + 1.45927 w_k1_AXP_s002
                                 + 1.50483 w_k1_DIS_s002
                                 + 1.13817 w_k1_FDX_s002
                                 + 1.85006 w_k1_IBM_s002
                                 + 1.14733 w_k1_INTC_s002
                                 + 1.73448 w_k1_KO_s002 + 1.83148 w_k1_MAT_s002
                                 + 1.31831 w_k1_MSFT_s002
                                 + 1.35705 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.26528 w_k1_ADM_s002
                                 + 1.17002 w_k1_AXP_s002
                                 + 1.23184 w_k1_DIS_s002 + 1.117 w_k1_FDX_s002
                                 + 1.23555 w_k1_IBM_s002
                                 + 0.930728 w_k1_INTC_s002
                                 + 1.20227 w_k1_KO_s002 + 1.36266 w_k1_MAT_s002
                                 + 1.09913 w_k1_MSFT_s002
                                 + 1.12774 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.13695 w_k1_ADM_s003
                                 + 1.54054 w_k1_AXP_s003
                                 + 1.15432 w_k1_DIS_s003
                                 + 1.11107 w_k1_FDX_s003
                                 + 1.28355 w_k1_IBM_s003
                                 + 0.997583 w_k1_INTC_s003
                                 + 1.17799 w_k1_KO_s003 + 1.55587 w_k1_MAT_s003
                                 + 1.13781 w_k1_MSFT_s003
                                 + 1.09569 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.00273 w_k1_ADM_s003
                                 + 1.32246 w_k1_AXP_s003
                                 + 1.49828 w_k1_DIS_s003
                                 + 1.24606 w_k1_FDX_s003
                                 + 1.63502 w_k1_IBM_s003
                                 + 1.42067 w_k1_INTC_s003
                                 + 1.39977 w_k1_KO_s003 + 1.09848 w_k1_MAT_s003
                                 + 1.30398 w_k1_MSFT_s003
                                 + 1.12946 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.844497 w_k1_ADM_s003
                                 + 0.954439 w_k1_AXP_s003
                                 + 0.90784 w_k1_DIS_s003
                                 + 0.860749 w_k1_FDX_s003
                                 + 1.22137 w_k1_IBM_s003
                                 + 0.974838 w_k1_INTC_s003
                                 + 1.05922 w_k1_KO_s003
                                 + 0.872648 w_k1_MAT_s003
                                 + 0.955572 w_k1_MSFT_s003
                                 + 0.906008 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.13286 w_k1_ADM_s003
                                 + 1.21715 w_k1_AXP_s003
                                 + 0.972834 w_k1_DIS_s003
                                 + 1.07404 w_k1_FDX_s003
                                 + 1.06118 w_k1_IBM_s003
                                 + 1.04104 w_k1_INTC_s003
                                 + 0.879683 w_k1_KO_s003 + 0.9683 w_k1_MAT_s003
                                 + 1.13614 w_k1_MSFT_s003
                                 + 0.976138 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.0755 w_k1_ADM_s003
                                 + 0.942663 w_k1_AXP_s003
                                 + 1.36164 w_k1_DIS_s003
                                 + 1.28335 w_k1_FDX_s003
                                 + 0.803318 w_k1_IBM_s003
                                 + 1.09518 w_k1_INTC_s003
                                 + 0.930929 w_k1_KO_s003
                                 + 0.569735 w_k1_MAT_s003
                                 + 1.13006 w_k1_MSFT_s003
                                 + 0.944572 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.47862 w_k1_ADM_s003
                                 + 1.47828 w_k1_AXP_s003 + 1.5046 w_k1_DIS_s003
                                 + 1.34714 w_k1_FDX_s003
                                 + 1.13204 w_k1_IBM_s003
                                 + 1.50914 w_k1_INTC_s003
                                 + 1.28133 w_k1_KO_s003 + 1.1129 w_k1_MAT_s003
                                 + 1.47198 w_k1_MSFT_s003 + 1.2261 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.860839 w_k1_ADM_s003
                                 + 0.919013 w_k1_AXP_s003
                                 + 1.12801 w_k1_DIS_s003
                                 + 0.99991 w_k1_FDX_s003
                                 + 0.731468 w_k1_IBM_s003
                                 + 1.06446 w_k1_INTC_s003
                                 + 0.807951 w_k1_KO_s003
                                 + 0.532938 w_k1_MAT_s003
                                 + 1.03727 w_k1_MSFT_s003
                                 + 0.872105 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.704503 w_k1_ADM_s003
                                 + 0.662555 w_k1_AXP_s003
                                 + 0.691055 w_k1_DIS_s003
                                 + 0.787141 w_k1_FDX_s003
                                 + 0.789319 w_k1_IBM_s003
                                 + 0.832232 w_k1_INTC_s003
                                 + 0.748592 w_k1_KO_s003
                                 + 0.59798 w_k1_MAT_s003
                                 + 0.816918 w_k1_MSFT_s003
                                 + 0.776679 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.10295 w_k1_ADM_s004
                                 + 1.16968 w_k1_AXP_s004 + 1.0329 w_k1_DIS_s004
                                 + 0.993261 w_k1_FDX_s004
                                 + 1.19825 w_k1_IBM_s004
                                 + 1.09743 w_k1_INTC_s004
                                 + 1.21603 w_k1_KO_s004 + 1.2573 w_k1_MAT_s004
                                 + 1.10727 w_k1_MSFT_s004
                                 + 1.02446 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.938506 w_k1_ADM_s004
                                 + 0.931669 w_k1_AXP_s004
                                 + 0.754766 w_k1_DIS_s004
                                 + 0.775793 w_k1_FDX_s004
                                 + 0.847723 w_k1_IBM_s004
                                 + 0.902656 w_k1_INTC_s004
                                 + 0.72357 w_k1_KO_s004 + 0.76302 w_k1_MAT_s004
                                 + 0.742274 w_k1_MSFT_s004
                                 + 0.866339 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.07562 w_k1_ADM_s004
                                 + 0.753239 w_k1_AXP_s004
                                 + 0.995217 w_k1_DIS_s004
                                 + 0.838337 w_k1_FDX_s004
                                 + 0.852576 w_k1_IBM_s004
                                 + 0.988687 w_k1_INTC_s004
                                 + 0.926002 w_k1_KO_s004
                                 + 0.60687 w_k1_MAT_s004
                                 + 1.0989 w_k1_MSFT_s004
                                 + 0.983468 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.0417 w_k1_ADM_s004
                                 + 1.20556 w_k1_AXP_s004
                                 + 1.22997 w_k1_DIS_s004
                                 + 1.06268 w_k1_FDX_s004
                                 + 1.03991 w_k1_IBM_s004
                                 + 1.16111 w_k1_INTC_s004
                                 + 1.12993 w_k1_KO_s004 + 1.0351 w_k1_MAT_s004
                                 + 1.1581 w_k1_MSFT_s004 + 1.09908 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.51005 w_k1_ADM_s004
                                 + 1.46294 w_k1_AXP_s004
                                 + 1.70856 w_k1_DIS_s004
                                 + 1.22167 w_k1_FDX_s004
                                 + 1.63844 w_k1_IBM_s004
                                 + 1.15629 w_k1_INTC_s004
                                 + 1.57178 w_k1_KO_s004 + 1.58914 w_k1_MAT_s004
                                 + 1.38534 w_k1_MSFT_s004
                                 + 1.36582 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.30192 w_k1_ADM_s004
                                 + 1.3855 w_k1_AXP_s004 + 1.26531 w_k1_DIS_s004
                                 + 1.15649 w_k1_FDX_s004
                                 + 0.838104 w_k1_IBM_s004
                                 + 1.16698 w_k1_INTC_s004
                                 + 1.03962 w_k1_KO_s004 + 1.14562 w_k1_MAT_s004
                                 + 1.22798 w_k1_MSFT_s004
                                 + 1.03804 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.58516 w_k1_ADM_s004
                                 + 1.32059 w_k1_AXP_s004
                                 + 1.90974 w_k1_DIS_s004
                                 + 1.48239 w_k1_FDX_s004
                                 + 0.911518 w_k1_IBM_s004
                                 + 1.37399 w_k1_INTC_s004
                                 + 1.15952 w_k1_KO_s004
                                 + 0.847804 w_k1_MAT_s004
                                 + 1.49039 w_k1_MSFT_s004
                                 + 1.15749 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.11811 w_k1_ADM_s004
                                 + 0.935615 w_k1_AXP_s004
                                 + 0.832216 w_k1_DIS_s004
                                 + 0.896827 w_k1_FDX_s004
                                 + 0.64014 w_k1_IBM_s004
                                 + 0.82552 w_k1_INTC_s004
                                 + 0.678313 w_k1_KO_s004
                                 + 0.631044 w_k1_MAT_s004
                                 + 0.910777 w_k1_MSFT_s004
                                 + 0.805165 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.908753 w_k1_ADM_s005
                                 + 1.48773 w_k1_AXP_s005
                                 + 1.14155 w_k1_DIS_s005
                                 + 1.14044 w_k1_FDX_s005
                                 + 1.15918 w_k1_IBM_s005
                                 + 0.954117 w_k1_INTC_s005
                                 + 1.14934 w_k1_KO_s005 + 1.52674 w_k1_MAT_s005
                                 + 0.950901 w_k1_MSFT_s005 + 1.04 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.608142 w_k1_ADM_s005
                                 + 0.758737 w_k1_AXP_s005
                                 + 0.725702 w_k1_DIS_s005
                                 + 0.714338 w_k1_FDX_s005
                                 + 1.21318 w_k1_IBM_s005
                                 + 0.692509 w_k1_INTC_s005
                                 + 0.921139 w_k1_KO_s005
                                 + 1.19268 w_k1_MAT_s005
                                 + 0.683674 w_k1_MSFT_s005
                                 + 0.962793 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.835775 w_k1_ADM_s005
                                 + 0.950766 w_k1_AXP_s005
                                 + 1.01027 w_k1_DIS_s005
                                 + 0.950752 w_k1_FDX_s005
                                 + 1.00166 w_k1_IBM_s005
                                 + 0.820702 w_k1_INTC_s005
                                 + 1.13487 w_k1_KO_s005 + 1.15658 w_k1_MAT_s005
                                 + 0.916666 w_k1_MSFT_s005
                                 + 1.01946 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.88644 w_k1_ADM_s005
                                 + 1.06432 w_k1_AXP_s005
                                 + 1.14221 w_k1_DIS_s005
                                 + 0.965378 w_k1_FDX_s005
                                 + 1.59545 w_k1_IBM_s005
                                 + 0.982271 w_k1_INTC_s005
                                 + 1.25586 w_k1_KO_s005 + 1.9284 w_k1_MAT_s005
                                 + 0.867263 w_k1_MSFT_s005
                                 + 1.02148 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.24797 w_k1_ADM_s005
                                 + 1.64399 w_k1_AXP_s005
                                 + 1.74375 w_k1_DIS_s005
                                 + 1.48054 w_k1_FDX_s005
                                 + 1.16866 w_k1_IBM_s005
                                 + 1.06141 w_k1_INTC_s005
                                 + 1.28299 w_k1_KO_s005 + 1.17174 w_k1_MAT_s005
                                 + 1.06817 w_k1_MSFT_s005
                                 + 1.17231 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.765512 w_k1_ADM_s005
                                 + 0.702842 w_k1_AXP_s005
                                 + 0.700337 w_k1_DIS_s005
                                 + 0.798385 w_k1_FDX_s005
                                 + 0.831396 w_k1_IBM_s005
                                 + 0.866192 w_k1_INTC_s005
                                 + 0.806183 w_k1_KO_s005
                                 + 0.818971 w_k1_MAT_s005
                                 + 0.828998 w_k1_MSFT_s005
                                 + 0.834256 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.751315 w_k1_ADM_s005
                                 + 0.729982 w_k1_AXP_s005
                                 + 0.653993 w_k1_DIS_s005
                                 + 0.856156 w_k1_FDX_s005
                                 + 0.586743 w_k1_IBM_s005
                                 + 0.671688 w_k1_INTC_s005
                                 + 0.690589 w_k1_KO_s005
                                 + 0.656015 w_k1_MAT_s005
                                 + 0.76853 w_k1_MSFT_s005
                                 + 0.801146 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.0928 w_k1_ADM_s005
                                 + 1.21277 w_k1_AXP_s005
                                 + 1.07956 w_k1_DIS_s005
                                 + 1.10809 w_k1_FDX_s005
                                 + 0.754152 w_k1_IBM_s005
                                 + 1.10284 w_k1_INTC_s005
                                 + 0.900159 w_k1_KO_s005
                                 + 0.631989 w_k1_MAT_s005
                                 + 0.925023 w_k1_MSFT_s005
                                 + 1.01356 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.983344 w_k1_ADM_s006
                                 + 0.848891 w_k1_AXP_s006
                                 + 1.0111 w_k1_DIS_s006
                                 + 0.927573 w_k1_FDX_s006
                                 + 1.51721 w_k1_IBM_s006
                                 + 1.02912 w_k1_INTC_s006
                                 + 1.36325 w_k1_KO_s006 + 1.12327 w_k1_MAT_s006
                                 + 0.959623 w_k1_MSFT_s006
                                 + 1.05712 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.24473 w_k1_ADM_s006
                                 + 1.11131 w_k1_AXP_s006
                                 + 1.45932 w_k1_DIS_s006
                                 + 1.10711 w_k1_FDX_s006
                                 + 1.49163 w_k1_IBM_s006
                                 + 1.17948 w_k1_INTC_s006
                                 + 1.52018 w_k1_KO_s006 + 1.08077 w_k1_MAT_s006
                                 + 1.10453 w_k1_MSFT_s006
                                 + 1.04071 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.07916 w_k1_ADM_s006
                                 + 0.932371 w_k1_AXP_s006
                                 + 0.905266 w_k1_DIS_s006
                                 + 1.1428 w_k1_FDX_s006
                                 + 0.849275 w_k1_IBM_s006
                                 + 1.0696 w_k1_INTC_s006
                                 + 0.927258 w_k1_KO_s006
                                 + 0.684004 w_k1_MAT_s006
                                 + 0.997587 w_k1_MSFT_s006
                                 + 1.01341 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.898065 w_k1_ADM_s006
                                 + 0.941055 w_k1_AXP_s006
                                 + 0.821553 w_k1_DIS_s006
                                 + 0.864591 w_k1_FDX_s006
                                 + 1.08261 w_k1_IBM_s006
                                 + 0.923827 w_k1_INTC_s006
                                 + 1.03639 w_k1_KO_s006 + 1.21457 w_k1_MAT_s006
                                 + 0.845298 w_k1_MSFT_s006
                                 + 0.924651 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.640179 w_k1_ADM_s006
                                 + 0.620572 w_k1_AXP_s006
                                 + 0.472107 w_k1_DIS_s006
                                 + 0.629776 w_k1_FDX_s006
                                 + 0.687608 w_k1_IBM_s006
                                 + 0.764338 w_k1_INTC_s006
                                 + 0.705961 w_k1_KO_s006
                                 + 0.65432 w_k1_MAT_s006
                                 + 0.697195 w_k1_MSFT_s006
                                 + 0.758474 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.08274 w_k1_ADM_s006
                                 + 1.4971 w_k1_AXP_s006 + 1.20386 w_k1_DIS_s006
                                 + 1.32572 w_k1_FDX_s006
                                 + 0.86939 w_k1_IBM_s006
                                 + 1.31387 w_k1_INTC_s006
                                 + 1.05909 w_k1_KO_s006 + 1.11546 w_k1_MAT_s006
                                 + 1.13245 w_k1_MSFT_s006
                                 + 1.08219 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.47484 w_k1_ADM_s006
                                 + 1.16653 w_k1_AXP_s006 + 1.7059 w_k1_DIS_s006
                                 + 1.38165 w_k1_FDX_s006
                                 + 1.07335 w_k1_IBM_s006
                                 + 1.25003 w_k1_INTC_s006
                                 + 1.48067 w_k1_KO_s006
                                 + 0.979747 w_k1_MAT_s006
                                 + 1.29141 w_k1_MSFT_s006
                                 + 1.25252 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.875284 w_k1_ADM_s006
                                 + 0.748152 w_k1_AXP_s006
                                 + 0.712668 w_k1_DIS_s006
                                 + 0.887412 w_k1_FDX_s006
                                 + 0.810107 w_k1_IBM_s006
                                 + 0.953018 w_k1_INTC_s006
                                 + 0.789147 w_k1_KO_s006
                                 + 0.561377 w_k1_MAT_s006
                                 + 0.758036 w_k1_MSFT_s006
                                 + 0.884507 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.06962 w_k1_ADM_s007
                                 + 1.03293 w_k1_AXP_s007
                                 + 1.33945 w_k1_DIS_s007
                                 + 1.30576 w_k1_FDX_s007
                                 + 0.921166 w_k1_IBM_s007
                                 + 1.25332 w_k1_INTC_s007
                                 + 1.17315 w_k1_KO_s007
                                 + 0.651915 w_k1_MAT_s007
                                 + 1.28296 w_k1_MSFT_s007
                                 + 1.02369 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.8339 w_k1_ADM_s007
                                 + 0.974357 w_k1_AXP_s007
                                 + 0.938475 w_k1_DIS_s007
                                 + 1.00806 w_k1_FDX_s007
                                 + 1.07917 w_k1_IBM_s007
                                 + 0.999274 w_k1_INTC_s007
                                 + 0.938267 w_k1_KO_s007
                                 + 1.05356 w_k1_MAT_s007
                                 + 0.901797 w_k1_MSFT_s007
                                 + 0.870403 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.768844 w_k1_ADM_s007
                                 + 0.536794 w_k1_AXP_s007
                                 + 0.51143 w_k1_DIS_s007
                                 + 0.686564 w_k1_FDX_s007
                                 + 0.761194 w_k1_IBM_s007
                                 + 0.804979 w_k1_INTC_s007
                                 + 0.754017 w_k1_KO_s007
                                 + 0.579374 w_k1_MAT_s007
                                 + 0.832632 w_k1_MSFT_s007
                                 + 0.762558 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.00565 w_k1_ADM_s007
                                 + 0.909779 w_k1_AXP_s007
                                 + 0.754648 w_k1_DIS_s007
                                 + 1.01646 w_k1_FDX_s007
                                 + 0.622161 w_k1_IBM_s007
                                 + 0.892518 w_k1_INTC_s007
                                 + 0.735627 w_k1_KO_s007
                                 + 0.678176 w_k1_MAT_s007
                                 + 0.978978 w_k1_MSFT_s007
                                 + 0.830816 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.752101 w_k1_ADM_s007
                                 + 0.656423 w_k1_AXP_s007
                                 + 0.795003 w_k1_DIS_s007
                                 + 0.788635 w_k1_FDX_s007
                                 + 0.82738 w_k1_IBM_s007
                                 + 0.892937 w_k1_INTC_s007
                                 + 0.860169 w_k1_KO_s007
                                 + 0.737745 w_k1_MAT_s007
                                 + 0.920339 w_k1_MSFT_s007
                                 + 0.808184 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.10415 w_k1_ADM_s007
                                 + 1.05045 w_k1_AXP_s007 + 1.2281 w_k1_DIS_s007
                                 + 1.04715 w_k1_FDX_s007
                                 + 1.39176 w_k1_IBM_s007
                                 + 1.09783 w_k1_INTC_s007
                                 + 1.18574 w_k1_KO_s007 + 1.59598 w_k1_MAT_s007
                                 + 1.20827 w_k1_MSFT_s007 + 1.0626 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.05472 w_k1_ADM_s007
                                 + 0.825176 w_k1_AXP_s007
                                 + 1.06173 w_k1_DIS_s007
                                 + 1.07416 w_k1_FDX_s007
                                 + 0.63377 w_k1_IBM_s007
                                 + 0.905459 w_k1_INTC_s007
                                 + 0.914932 w_k1_KO_s007
                                 + 0.637376 w_k1_MAT_s007
                                 + 1.17837 w_k1_MSFT_s007
                                 + 1.03437 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.809738 w_k1_ADM_s007
                                 + 1.11791 w_k1_AXP_s007
                                 + 1.68264 w_k1_DIS_s007
                                 + 1.33077 w_k1_FDX_s007
                                 + 0.872096 w_k1_IBM_s007
                                 + 1.18069 w_k1_INTC_s007
                                 + 1.18834 w_k1_KO_s007
                                 + 0.782872 w_k1_MAT_s007
                                 + 1.4459 w_k1_MSFT_s007
                                 + 0.986512 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.24022 w_k1_ADM_s010
                                 + 0.740119 w_k1_AXP_s010
                                 + 0.984622 w_k1_DIS_s010
                                 + 1.10556 w_k1_FDX_s010
                                 + 0.735829 w_k1_IBM_s010
                                 + 1.03834 w_k1_INTC_s010
                                 + 0.934776 w_k1_KO_s010
                                 + 0.72119 w_k1_MAT_s010
                                 + 1.18004 w_k1_MSFT_s010
                                 + 0.98748 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.778602 w_k1_ADM_s010
                                 + 0.548583 w_k1_AXP_s010
                                 + 0.683818 w_k1_DIS_s010
                                 + 0.75418 w_k1_FDX_s010
                                 + 0.832856 w_k1_IBM_s010
                                 + 0.832154 w_k1_INTC_s010
                                 + 0.844333 w_k1_KO_s010
                                 + 0.85252 w_k1_MAT_s010
                                 + 0.872391 w_k1_MSFT_s010
                                 + 0.842603 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.760064 w_k1_ADM_s010
                                 + 0.727073 w_k1_AXP_s010
                                 + 0.850005 w_k1_DIS_s010
                                 + 0.836448 w_k1_FDX_s010
                                 + 1.19699 w_k1_IBM_s010
                                 + 1.01256 w_k1_INTC_s010
                                 + 0.893174 w_k1_KO_s010
                                 + 1.06997 w_k1_MAT_s010
                                 + 0.970331 w_k1_MSFT_s010
                                 + 0.896168 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.41445 w_k1_ADM_s010
                                 + 1.14164 w_k1_AXP_s010
                                 + 1.67847 w_k1_DIS_s010
                                 + 1.19226 w_k1_FDX_s010
                                 + 1.67925 w_k1_IBM_s010
                                 + 1.26038 w_k1_INTC_s010
                                 + 1.52214 w_k1_KO_s010 + 1.99771 w_k1_MAT_s010
                                 + 1.37845 w_k1_MSFT_s010
                                 + 1.22889 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.30861 w_k1_ADM_s010
                                 + 0.992457 w_k1_AXP_s010
                                 + 0.920654 w_k1_DIS_s010
                                 + 0.967817 w_k1_FDX_s010
                                 + 1.01366 w_k1_IBM_s010
                                 + 0.927136 w_k1_INTC_s010
                                 + 0.985158 w_k1_KO_s010
                                 + 1.15911 w_k1_MAT_s010
                                 + 0.911071 w_k1_MSFT_s010
                                 + 1.01545 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 2.22853 w_k1_ADM_s010
                                 + 1.76527 w_k1_AXP_s010
                                 + 1.98821 w_k1_DIS_s010
                                 + 1.73989 w_k1_FDX_s010
                                 + 0.938928 w_k1_IBM_s010
                                 + 1.52461 w_k1_INTC_s010
                                 + 1.28725 w_k1_KO_s010 + 1.44202 w_k1_MAT_s010
                                 + 1.60589 w_k1_MSFT_s010
                                 + 1.25055 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.56741 w_k1_ADM_s010
                                 + 1.21332 w_k1_AXP_s010
                                 + 1.21989 w_k1_DIS_s010
                                 + 1.22764 w_k1_FDX_s010
                                 + 1.27413 w_k1_IBM_s010
                                 + 1.24096 w_k1_INTC_s010
                                 + 1.17832 w_k1_KO_s010 + 1.50283 w_k1_MAT_s010
                                 + 1.52759 w_k1_MSFT_s010
                                 + 1.12106 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.17038 w_k1_ADM_s010
                                 + 0.835485 w_k1_AXP_s010
                                 + 1.42776 w_k1_DIS_s010 + 1.0235 w_k1_FDX_s010
                                 + 1.02741 w_k1_IBM_s010
                                 + 0.958728 w_k1_INTC_s010
                                 + 1.19212 w_k1_KO_s010 + 1.06741 w_k1_MAT_s010
                                 + 1.29161 w_k1_MSFT_s010
                                 + 1.18091 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.04812 w_k1_ADM_s011
                                 + 0.990764 w_k1_AXP_s011
                                 + 0.75784 w_k1_DIS_s011
                                 + 0.93864 w_k1_FDX_s011
                                 + 0.773372 w_k1_IBM_s011
                                 + 0.993409 w_k1_INTC_s011
                                 + 0.886906 w_k1_KO_s011
                                 + 0.720405 w_k1_MAT_s011
                                 + 0.967444 w_k1_MSFT_s011
                                 + 0.946014 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.986687 w_k1_ADM_s011
                                 + 0.984489 w_k1_AXP_s011
                                 + 0.887435 w_k1_DIS_s011
                                 + 0.7833 w_k1_FDX_s011 + 1.13012 w_k1_IBM_s011
                                 + 0.78075 w_k1_INTC_s011
                                 + 1.00487 w_k1_KO_s011 + 1.03127 w_k1_MAT_s011
                                 + 0.882425 w_k1_MSFT_s011
                                 + 1.03109 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.10468 w_k1_ADM_s011
                                 + 1.38996 w_k1_AXP_s011
                                 + 1.31333 w_k1_DIS_s011
                                 + 1.19651 w_k1_FDX_s011
                                 + 1.51107 w_k1_IBM_s011
                                 + 1.13985 w_k1_INTC_s011
                                 + 1.38555 w_k1_KO_s011 + 1.54605 w_k1_MAT_s011
                                 + 1.10616 w_k1_MSFT_s011 + 1.3795 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.12436 w_k1_ADM_s011
                                 + 1.38534 w_k1_AXP_s011
                                 + 1.47273 w_k1_DIS_s011
                                 + 1.31649 w_k1_FDX_s011
                                 + 0.902954 w_k1_IBM_s011
                                 + 1.1419 w_k1_INTC_s011 + 1.13767 w_k1_KO_s011
                                 + 1.14325 w_k1_MAT_s011
                                 + 1.29123 w_k1_MSFT_s011
                                 + 1.06198 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.06601 w_k1_ADM_s011
                                 + 1.14937 w_k1_AXP_s011
                                 + 1.27029 w_k1_DIS_s011
                                 + 1.18945 w_k1_FDX_s011
                                 + 1.17887 w_k1_IBM_s011
                                 + 0.878494 w_k1_INTC_s011
                                 + 1.30181 w_k1_KO_s011 + 1.26244 w_k1_MAT_s011
                                 + 1.06387 w_k1_MSFT_s011
                                 + 1.22441 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.983785 w_k1_ADM_s011
                                 + 0.664174 w_k1_AXP_s011
                                 + 0.720855 w_k1_DIS_s011
                                 + 0.835161 w_k1_FDX_s011
                                 + 0.951033 w_k1_IBM_s011
                                 + 0.645383 w_k1_INTC_s011
                                 + 0.796161 w_k1_KO_s011
                                 + 0.922222 w_k1_MAT_s011
                                 + 0.797366 w_k1_MSFT_s011
                                 + 0.889584 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.853342 w_k1_ADM_s011
                                 + 1.15498 w_k1_AXP_s011
                                 + 0.982271 w_k1_DIS_s011
                                 + 1.07895 w_k1_FDX_s011
                                 + 0.98433 w_k1_IBM_s011
                                 + 1.01852 w_k1_INTC_s011
                                 + 1.06082 w_k1_KO_s011 + 1.06476 w_k1_MAT_s011
                                 + 1.03444 w_k1_MSFT_s011
                                 + 0.954395 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.21496 w_k1_ADM_s011
                                 + 1.10522 w_k1_AXP_s011
                                 + 0.928089 w_k1_DIS_s011
                                 + 0.985743 w_k1_FDX_s011
                                 + 1.47669 w_k1_IBM_s011
                                 + 0.868465 w_k1_INTC_s011
                                 + 1.28025 w_k1_KO_s011 + 1.29753 w_k1_MAT_s011
                                 + 0.918982 w_k1_MSFT_s011
                                 + 1.10528 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.851118 w_k1_ADM_s012
                                 + 0.879152 w_k1_AXP_s012
                                 + 1.05406 w_k1_DIS_s012
                                 + 0.924487 w_k1_FDX_s012
                                 + 1.54916 w_k1_IBM_s012
                                 + 0.932777 w_k1_INTC_s012
                                 + 1.13404 w_k1_KO_s012 + 1.68226 w_k1_MAT_s012
                                 + 0.97483 w_k1_MSFT_s012
                                 + 1.13786 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.09299 w_k1_ADM_s012
                                 + 0.828134 w_k1_AXP_s012
                                 + 0.655588 w_k1_DIS_s012
                                 + 0.753536 w_k1_FDX_s012
                                 + 1.05331 w_k1_IBM_s012
                                 + 0.758667 w_k1_INTC_s012
                                 + 0.895692 w_k1_KO_s012
                                 + 1.33242 w_k1_MAT_s012
                                 + 0.748618 w_k1_MSFT_s012
                                 + 0.951208 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.25419 w_k1_ADM_s012
                                 + 1.19047 w_k1_AXP_s012
                                 + 1.62095 w_k1_DIS_s012
                                 + 1.38239 w_k1_FDX_s012
                                 + 0.905881 w_k1_IBM_s012
                                 + 1.25109 w_k1_INTC_s012 + 1.1798 w_k1_KO_s012
                                 + 1.51324 w_k1_MAT_s012
                                 + 1.31101 w_k1_MSFT_s012
                                 + 1.39329 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.34054 w_k1_ADM_s012
                                 + 0.820669 w_k1_AXP_s012
                                 + 1.13031 w_k1_DIS_s012
                                 + 0.924406 w_k1_FDX_s012
                                 + 1.06778 w_k1_IBM_s012
                                 + 0.991437 w_k1_INTC_s012
                                 + 1.10891 w_k1_KO_s012
                                 + 0.824392 w_k1_MAT_s012
                                 + 1.02531 w_k1_MSFT_s012
                                 + 1.07706 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.15538 w_k1_ADM_s012
                                 + 0.688372 w_k1_AXP_s012
                                 + 0.744546 w_k1_DIS_s012
                                 + 0.761094 w_k1_FDX_s012
                                 + 0.796527 w_k1_IBM_s012
                                 + 0.683767 w_k1_INTC_s012
                                 + 0.694026 w_k1_KO_s012
                                 + 0.746548 w_k1_MAT_s012
                                 + 0.775573 w_k1_MSFT_s012
                                 + 0.93592 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.19524 w_k1_ADM_s012
                                 + 0.777189 w_k1_AXP_s012
                                 + 1.13916 w_k1_DIS_s012
                                 + 1.03343 w_k1_FDX_s012
                                 + 0.736645 w_k1_IBM_s012
                                 + 1.15746 w_k1_INTC_s012
                                 + 0.787695 w_k1_KO_s012
                                 + 0.621128 w_k1_MAT_s012
                                 + 1.07342 w_k1_MSFT_s012
                                 + 1.13494 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.13834 w_k1_ADM_s012
                                 + 1.15665 w_k1_AXP_s012
                                 + 0.92899 w_k1_DIS_s012
                                 + 1.06073 w_k1_FDX_s012
                                 + 1.02826 w_k1_IBM_s012
                                 + 0.904487 w_k1_INTC_s012
                                 + 0.772725 w_k1_KO_s012
                                 + 1.39999 w_k1_MAT_s012
                                 + 0.891119 w_k1_MSFT_s012
                                 + 0.979531 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.5049 w_k1_ADM_s012
                                 + 1.48359 w_k1_AXP_s012 + 1.6708 w_k1_DIS_s012
                                 + 1.29289 w_k1_FDX_s012
                                 + 1.48686 w_k1_IBM_s012
                                 + 1.14928 w_k1_INTC_s012
                                 + 1.45698 w_k1_KO_s012 + 1.38923 w_k1_MAT_s012
                                 + 1.17169 w_k1_MSFT_s012
                                 + 1.34866 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.982987 w_k1_ADM_s013
                                 + 0.865131 w_k1_AXP_s013
                                 + 1.26508 w_k1_DIS_s013
                                 + 1.07215 w_k1_FDX_s013
                                 + 1.00622 w_k1_IBM_s013
                                 + 0.826168 w_k1_INTC_s013
                                 + 1.08107 w_k1_KO_s013 + 1.02527 w_k1_MAT_s013
                                 + 1.02691 w_k1_MSFT_s013 + 1.0972 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.885234 w_k1_ADM_s013
                                 + 0.542515 w_k1_AXP_s013
                                 + 0.833235 w_k1_DIS_s013
                                 + 0.87134 w_k1_FDX_s013
                                 + 0.622647 w_k1_IBM_s013
                                 + 0.77833 w_k1_INTC_s013
                                 + 0.804149 w_k1_KO_s013
                                 + 0.367068 w_k1_MAT_s013
                                 + 0.946836 w_k1_MSFT_s013
                                 + 0.775543 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.726665 w_k1_ADM_s013
                                 + 0.614203 w_k1_AXP_s013
                                 + 0.695878 w_k1_DIS_s013
                                 + 0.727558 w_k1_FDX_s013
                                 + 0.863089 w_k1_IBM_s013
                                 + 0.671102 w_k1_INTC_s013
                                 + 0.863393 w_k1_KO_s013
                                 + 0.79663 w_k1_MAT_s013
                                 + 0.793148 w_k1_MSFT_s013
                                 + 0.88797 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.800433 w_k1_ADM_s013
                                 + 0.865078 w_k1_AXP_s013
                                 + 1.38715 w_k1_DIS_s013 + 1.0324 w_k1_FDX_s013
                                 + 1.2442 w_k1_IBM_s013
                                 + 0.904863 w_k1_INTC_s013
                                 + 1.23704 w_k1_KO_s013 + 0.90521 w_k1_MAT_s013
                                 + 1.12592 w_k1_MSFT_s013
                                 + 1.05495 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.01477 w_k1_ADM_s013
                                 + 1.31191 w_k1_AXP_s013
                                 + 1.49054 w_k1_DIS_s013
                                 + 1.23953 w_k1_FDX_s013 + 1.4011 w_k1_IBM_s013
                                 + 1.35736 w_k1_INTC_s013
                                 + 1.57526 w_k1_KO_s013 + 1.60928 w_k1_MAT_s013
                                 + 1.62879 w_k1_MSFT_s013
                                 + 1.29885 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.918572 w_k1_ADM_s013
                                 + 0.643372 w_k1_AXP_s013
                                 + 0.877671 w_k1_DIS_s013
                                 + 0.848409 w_k1_FDX_s013
                                 + 1.14815 w_k1_IBM_s013
                                 + 0.915665 w_k1_INTC_s013
                                 + 1.04796 w_k1_KO_s013
                                 + 0.786414 w_k1_MAT_s013
                                 + 1.01606 w_k1_MSFT_s013
                                 + 0.881027 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.96012 w_k1_ADM_s013
                                 + 1.07468 w_k1_AXP_s013
                                 + 0.987754 w_k1_DIS_s013
                                 + 0.989917 w_k1_FDX_s013
                                 + 1.01361 w_k1_IBM_s013
                                 + 1.06926 w_k1_INTC_s013
                                 + 1.15695 w_k1_KO_s013 + 1.04074 w_k1_MAT_s013
                                 + 1.0316 w_k1_MSFT_s013
                                 + 0.983087 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.24254 w_k1_ADM_s013
                                 + 1.33185 w_k1_AXP_s013
                                 + 1.53266 w_k1_DIS_s013
                                 + 1.35877 w_k1_FDX_s013
                                 + 0.899852 w_k1_IBM_s013
                                 + 1.32181 w_k1_INTC_s013 + 1.381 w_k1_KO_s013
                                 + 1.17877 w_k1_MAT_s013
                                 + 1.54554 w_k1_MSFT_s013
                                 + 1.31111 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.850984 w_k1_ADM_s014
                                 + 0.653574 w_k1_AXP_s014
                                 + 0.702993 w_k1_DIS_s014
                                 + 0.782123 w_k1_FDX_s014
                                 + 0.858952 w_k1_IBM_s014
                                 + 0.70068 w_k1_INTC_s014
                                 + 0.75689 w_k1_KO_s014
                                 + 0.960551 w_k1_MAT_s014
                                 + 0.666506 w_k1_MSFT_s014
                                 + 0.857036 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.33363 w_k1_ADM_s014
                                 + 1.14734 w_k1_AXP_s014
                                 + 1.38813 w_k1_DIS_s014
                                 + 1.17078 w_k1_FDX_s014
                                 + 1.83838 w_k1_IBM_s014
                                 + 1.17869 w_k1_INTC_s014
                                 + 1.53396 w_k1_KO_s014 + 2.25732 w_k1_MAT_s014
                                 + 1.12193 w_k1_MSFT_s014
                                 + 1.27507 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.818275 w_k1_ADM_s014
                                 + 0.954275 w_k1_AXP_s014
                                 + 0.945769 w_k1_DIS_s014
                                 + 0.807494 w_k1_FDX_s014
                                 + 1.31089 w_k1_IBM_s014
                                 + 0.904216 w_k1_INTC_s014
                                 + 0.905226 w_k1_KO_s014
                                 + 1.56223 w_k1_MAT_s014
                                 + 0.724434 w_k1_MSFT_s014
                                 + 0.811705 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 2.05647 w_k1_ADM_s014
                                 + 1.92044 w_k1_AXP_s014
                                 + 1.92082 w_k1_DIS_s014
                                 + 1.53235 w_k1_FDX_s014
                                 + 1.24379 w_k1_IBM_s014
                                 + 1.29095 w_k1_INTC_s014
                                 + 1.56814 w_k1_KO_s014 + 1.6849 w_k1_MAT_s014
                                 + 1.25711 w_k1_MSFT_s014
                                 + 1.18081 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.04845 w_k1_ADM_s014
                                 + 1.13216 w_k1_AXP_s014
                                 + 1.22483 w_k1_DIS_s014
                                 + 1.09719 w_k1_FDX_s014
                                 + 0.94688 w_k1_IBM_s014
                                 + 1.12572 w_k1_INTC_s014
                                 + 1.06205 w_k1_KO_s014
                                 + 0.925724 w_k1_MAT_s014
                                 + 1.12224 w_k1_MSFT_s014
                                 + 0.90325 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.31677 w_k1_ADM_s014
                                 + 1.14243 w_k1_AXP_s014
                                 + 0.778324 w_k1_DIS_s014
                                 + 0.934879 w_k1_FDX_s014
                                 + 1.03092 w_k1_IBM_s014
                                 + 0.991016 w_k1_INTC_s014
                                 + 0.966099 w_k1_KO_s014
                                 + 1.27513 w_k1_MAT_s014
                                 + 0.743948 w_k1_MSFT_s014
                                 + 0.957712 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.04271 w_k1_ADM_s014
                                 + 0.928794 w_k1_AXP_s014
                                 + 1.01326 w_k1_DIS_s014
                                 + 0.930882 w_k1_FDX_s014
                                 + 0.71362 w_k1_IBM_s014
                                 + 1.05406 w_k1_INTC_s014
                                 + 0.828176 w_k1_KO_s014
                                 + 0.708638 w_k1_MAT_s014
                                 + 0.91489 w_k1_MSFT_s014
                                 + 0.894983 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.62721 w_k1_ADM_s014
                                 + 1.25773 w_k1_AXP_s014
                                 + 1.50289 w_k1_DIS_s014
                                 + 1.25679 w_k1_FDX_s014
                                 + 1.05274 w_k1_IBM_s014
                                 + 1.24137 w_k1_INTC_s014
                                 + 1.36058 w_k1_KO_s014 + 1.38091 w_k1_MAT_s014
                                 + 1.22088 w_k1_MSFT_s014 + 1.2899 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.25585 w_k1_ADM_s015
                                 + 0.961573 w_k1_AXP_s015
                                 + 1.07906 w_k1_DIS_s015
                                 + 0.950514 w_k1_FDX_s015
                                 + 1.25449 w_k1_IBM_s015
                                 + 0.973988 w_k1_INTC_s015
                                 + 1.1154 w_k1_KO_s015 + 1.31817 w_k1_MAT_s015
                                 + 1.20204 w_k1_MSFT_s015
                                 + 1.08806 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.12736 w_k1_ADM_s015
                                 + 1.09691 w_k1_AXP_s015
                                 + 0.950456 w_k1_DIS_s015
                                 + 1.07115 w_k1_FDX_s015 + 0.9508 w_k1_IBM_s015
                                 + 0.967782 w_k1_INTC_s015
                                 + 0.905426 w_k1_KO_s015
                                 + 1.12857 w_k1_MAT_s015
                                 + 1.06603 w_k1_MSFT_s015
                                 + 1.01951 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.97979 w_k1_ADM_s015
                                 + 1.78467 w_k1_AXP_s015
                                 + 1.57277 w_k1_DIS_s015
                                 + 1.45744 w_k1_FDX_s015 + 1.2583 w_k1_IBM_s015
                                 + 1.25043 w_k1_INTC_s015
                                 + 1.29699 w_k1_KO_s015 + 1.01065 w_k1_MAT_s015
                                 + 1.37621 w_k1_MSFT_s015
                                 + 1.41034 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.18967 w_k1_ADM_s015
                                 + 1.63083 w_k1_AXP_s015
                                 + 1.67053 w_k1_DIS_s015
                                 + 1.23946 w_k1_FDX_s015
                                 + 1.76544 w_k1_IBM_s015
                                 + 1.25718 w_k1_INTC_s015
                                 + 1.52226 w_k1_KO_s015 + 1.95845 w_k1_MAT_s015
                                 + 1.28256 w_k1_MSFT_s015
                                 + 1.27106 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.15846 w_k1_ADM_s015
                                 + 1.0557 w_k1_AXP_s015
                                 + 0.946667 w_k1_DIS_s015
                                 + 1.1261 w_k1_FDX_s015
                                 + 0.758253 w_k1_IBM_s015
                                 + 0.874641 w_k1_INTC_s015
                                 + 0.794215 w_k1_KO_s015
                                 + 0.842886 w_k1_MAT_s015
                                 + 0.981564 w_k1_MSFT_s015
                                 + 0.978651 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.834818 w_k1_ADM_s015
                                 + 0.781665 w_k1_AXP_s015
                                 + 0.760932 w_k1_DIS_s015
                                 + 0.757443 w_k1_FDX_s015
                                 + 1.11444 w_k1_IBM_s015
                                 + 0.755994 w_k1_INTC_s015
                                 + 0.780803 w_k1_KO_s015
                                 + 0.886492 w_k1_MAT_s015
                                 + 0.828037 w_k1_MSFT_s015
                                 + 0.841999 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.0284 w_k1_ADM_s015
                                 + 0.703302 w_k1_AXP_s015
                                 + 0.731823 w_k1_DIS_s015
                                 + 0.83875 w_k1_FDX_s015
                                 + 0.829915 w_k1_IBM_s015
                                 + 0.798519 w_k1_INTC_s015
                                 + 0.810575 w_k1_KO_s015
                                 + 0.584841 w_k1_MAT_s015
                                 + 0.990411 w_k1_MSFT_s015
                                 + 1.01939 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.23343 w_k1_ADM_s015
                                 + 1.18078 w_k1_AXP_s015
                                 + 1.25613 w_k1_DIS_s015 + 1.3017 w_k1_FDX_s015
                                 + 0.938238 w_k1_IBM_s015
                                 + 1.06773 w_k1_INTC_s015 + 1.0622 w_k1_KO_s015
                                 + 1.01499 w_k1_MAT_s015
                                 + 1.32571 w_k1_MSFT_s015
                                 + 1.21203 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.877514 w_k1_ADM_s016
                                 + 0.818054 w_k1_AXP_s016
                                 + 1.16071 w_k1_DIS_s016
                                 + 0.993037 w_k1_FDX_s016
                                 + 1.23579 w_k1_IBM_s016
                                 + 1.16713 w_k1_INTC_s016 + 1.3069 w_k1_KO_s016
                                 + 1.10537 w_k1_MAT_s016
                                 + 1.25193 w_k1_MSFT_s016
                                 + 1.12369 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.891061 w_k1_ADM_s016
                                 + 1.09255 w_k1_AXP_s016
                                 + 1.03403 w_k1_DIS_s016
                                 + 1.06135 w_k1_FDX_s016
                                 + 1.25682 w_k1_IBM_s016
                                 + 1.20061 w_k1_INTC_s016
                                 + 1.08026 w_k1_KO_s016 + 1.13552 w_k1_MAT_s016
                                 + 1.17725 w_k1_MSFT_s016
                                 + 1.11117 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.18223 w_k1_ADM_s016
                                 + 1.00338 w_k1_AXP_s016
                                 + 0.891635 w_k1_DIS_s016
                                 + 1.04799 w_k1_FDX_s016
                                 + 0.919971 w_k1_IBM_s016
                                 + 0.954 w_k1_INTC_s016 + 0.890579 w_k1_KO_s016
                                 + 0.75403 w_k1_MAT_s016
                                 + 1.06922 w_k1_MSFT_s016
                                 + 1.10586 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.698444 w_k1_ADM_s016
                                 + 0.563996 w_k1_AXP_s016
                                 + 0.565416 w_k1_DIS_s016
                                 + 0.698519 w_k1_FDX_s016
                                 + 0.737856 w_k1_IBM_s016
                                 + 0.772107 w_k1_INTC_s016
                                 + 0.677153 w_k1_KO_s016
                                 + 0.642235 w_k1_MAT_s016
                                 + 0.756712 w_k1_MSFT_s016
                                 + 0.767515 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.62705 w_k1_ADM_s016
                                 + 1.39636 w_k1_AXP_s016
                                 + 1.47915 w_k1_DIS_s016
                                 + 1.37074 w_k1_FDX_s016
                                 + 1.02663 w_k1_IBM_s016
                                 + 1.11865 w_k1_INTC_s016
                                 + 1.24282 w_k1_KO_s016 + 1.16738 w_k1_MAT_s016
                                 + 1.35842 w_k1_MSFT_s016
                                 + 1.25304 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.992852 w_k1_ADM_s016
                                 + 0.696738 w_k1_AXP_s016
                                 + 0.950134 w_k1_DIS_s016
                                 + 0.908697 w_k1_FDX_s016
                                 + 0.849099 w_k1_IBM_s016
                                 + 1.10421 w_k1_INTC_s016
                                 + 0.912105 w_k1_KO_s016
                                 + 0.524271 w_k1_MAT_s016
                                 + 1.13179 w_k1_MSFT_s016
                                 + 0.916854 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.16253 w_k1_ADM_s016
                                 + 1.04018 w_k1_AXP_s016 + 1.0446 w_k1_DIS_s016
                                 + 1.24322 w_k1_FDX_s016
                                 + 0.788315 w_k1_IBM_s016
                                 + 1.16847 w_k1_INTC_s016
                                 + 0.919019 w_k1_KO_s016
                                 + 0.661904 w_k1_MAT_s016
                                 + 1.32634 w_k1_MSFT_s016
                                 + 1.02389 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.08419 w_k1_ADM_s016
                                 + 0.70818 w_k1_AXP_s016
                                 + 0.735646 w_k1_DIS_s016
                                 + 0.839868 w_k1_FDX_s016
                                 + 0.858802 w_k1_IBM_s016
                                 + 0.800534 w_k1_INTC_s016
                                 + 0.921371 w_k1_KO_s016
                                 + 0.806392 w_k1_MAT_s016
                                 + 1.01269 w_k1_MSFT_s016
                                 + 0.981392 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.06043 w_k1_ADM_s017
                                 + 0.934835 w_k1_AXP_s017
                                 + 0.960222 w_k1_DIS_s017
                                 + 0.877107 w_k1_FDX_s017
                                 + 0.747259 w_k1_IBM_s017
                                 + 0.862802 w_k1_INTC_s017
                                 + 0.836759 w_k1_KO_s017
                                 + 0.784658 w_k1_MAT_s017
                                 + 0.915578 w_k1_MSFT_s017
                                 + 0.958106 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.15116 w_k1_ADM_s017
                                 + 1.7894 w_k1_AXP_s017 + 1.04246 w_k1_DIS_s017
                                 + 1.13339 w_k1_FDX_s017
                                 + 0.753406 w_k1_IBM_s017
                                 + 1.04353 w_k1_INTC_s017
                                 + 0.773766 w_k1_KO_s017
                                 + 0.791918 w_k1_MAT_s017
                                 + 1.09867 w_k1_MSFT_s017
                                 + 1.02769 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.88784 w_k1_ADM_s017
                                 + 1.01333 w_k1_AXP_s017
                                 + 1.26275 w_k1_DIS_s017
                                 + 1.11795 w_k1_FDX_s017
                                 + 1.38433 w_k1_IBM_s017
                                 + 1.03194 w_k1_INTC_s017
                                 + 1.11832 w_k1_KO_s017 + 1.42523 w_k1_MAT_s017
                                 + 1.15875 w_k1_MSFT_s017
                                 + 1.08179 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.2352 w_k1_ADM_s017
                                 + 1.45473 w_k1_AXP_s017
                                 + 1.39015 w_k1_DIS_s017
                                 + 1.29059 w_k1_FDX_s017
                                 + 0.967538 w_k1_IBM_s017
                                 + 1.14184 w_k1_INTC_s017
                                 + 1.20719 w_k1_KO_s017 + 1.44933 w_k1_MAT_s017
                                 + 1.24966 w_k1_MSFT_s017
                                 + 1.25785 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.20741 w_k1_ADM_s017
                                 + 1.71451 w_k1_AXP_s017
                                 + 1.04829 w_k1_DIS_s017
                                 + 1.11879 w_k1_FDX_s017
                                 + 1.10532 w_k1_IBM_s017
                                 + 1.09545 w_k1_INTC_s017
                                 + 1.07394 w_k1_KO_s017 + 1.51333 w_k1_MAT_s017
                                 + 0.932447 w_k1_MSFT_s017
                                 + 1.00587 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.593992 w_k1_ADM_s017
                                 + 0.735081 w_k1_AXP_s017
                                 + 0.646357 w_k1_DIS_s017
                                 + 0.64264 w_k1_FDX_s017
                                 + 0.811755 w_k1_IBM_s017
                                 + 0.631653 w_k1_INTC_s017
                                 + 0.61475 w_k1_KO_s017
                                 + 0.870072 w_k1_MAT_s017
                                 + 0.634512 w_k1_MSFT_s017
                                 + 0.875569 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.46496 w_k1_ADM_s017
                                 + 1.86505 w_k1_AXP_s017 + 1.6521 w_k1_DIS_s017
                                 + 1.37781 w_k1_FDX_s017
                                 + 1.43218 w_k1_IBM_s017
                                 + 1.02886 w_k1_INTC_s017
                                 + 1.44245 w_k1_KO_s017 + 2.64872 w_k1_MAT_s017
                                 + 1.39485 w_k1_MSFT_s017
                                 + 1.30741 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.9431 w_k1_ADM_s017
                                 + 1.18717 w_k1_AXP_s017
                                 + 0.781584 w_k1_DIS_s017
                                 + 0.931393 w_k1_FDX_s017
                                 + 1.1746 w_k1_IBM_s017
                                 + 0.887473 w_k1_INTC_s017
                                 + 0.890963 w_k1_KO_s017
                                 + 1.15282 w_k1_MAT_s017
                                 + 0.785158 w_k1_MSFT_s017
                                 + 0.902227 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.10907 w_k1_ADM_s100
                                 + 0.75641 w_k1_AXP_s100
                                 + 0.853327 w_k1_DIS_s100
                                 + 0.950897 w_k1_FDX_s100
                                 + 0.796373 w_k1_IBM_s100
                                 + 0.840003 w_k1_INTC_s100
                                 + 0.853898 w_k1_KO_s100
                                 + 0.718739 w_k1_MAT_s100
                                 + 0.933827 w_k1_MSFT_s100
                                 + 0.907073 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.42799 w_k1_ADM_s100
                                 + 1.32227 w_k1_AXP_s100
                                 + 1.40038 w_k1_DIS_s100
                                 + 1.15971 w_k1_FDX_s100
                                 + 1.36971 w_k1_IBM_s100
                                 + 0.99221 w_k1_INTC_s100
                                 + 1.37806 w_k1_KO_s100 + 1.67941 w_k1_MAT_s100
                                 + 1.22653 w_k1_MSFT_s100
                                 + 1.23711 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.37039 w_k1_ADM_s100
                                 + 1.4478 w_k1_AXP_s100 + 1.31351 w_k1_DIS_s100
                                 + 1.33481 w_k1_FDX_s100
                                 + 0.946047 w_k1_IBM_s100
                                 + 1.21239 w_k1_INTC_s100 + 1.0879 w_k1_KO_s100
                                 + 1.1039 w_k1_MAT_s100
                                 + 1.14368 w_k1_MSFT_s100
                                 + 1.01418 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.08561 w_k1_ADM_s100
                                 + 0.924414 w_k1_AXP_s100
                                 + 0.963772 w_k1_DIS_s100
                                 + 1.11013 w_k1_FDX_s100
                                 + 0.642297 w_k1_IBM_s100
                                 + 0.868465 w_k1_INTC_s100
                                 + 0.839196 w_k1_KO_s100
                                 + 0.846884 w_k1_MAT_s100
                                 + 1.03167 w_k1_MSFT_s100
                                 + 0.993321 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.12146 w_k1_ADM_s100
                                 + 1.15016 w_k1_AXP_s100
                                 + 1.02792 w_k1_DIS_s100
                                 + 0.952055 w_k1_FDX_s100
                                 + 1.38105 w_k1_IBM_s100
                                 + 0.82419 w_k1_INTC_s100
                                 + 1.12246 w_k1_KO_s100 + 1.37519 w_k1_MAT_s100
                                 + 0.97066 w_k1_MSFT_s100
                                 + 0.99746 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.846553 w_k1_ADM_s100
                                 + 0.653212 w_k1_AXP_s100
                                 + 0.652921 w_k1_DIS_s100
                                 + 0.675513 w_k1_FDX_s100
                                 + 0.983997 w_k1_IBM_s100
                                 + 0.682418 w_k1_INTC_s100
                                 + 0.823859 w_k1_KO_s100
                                 + 0.795948 w_k1_MAT_s100
                                 + 0.820549 w_k1_MSFT_s100
                                 + 0.86773 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.2056 w_k1_ADM_s100
                                 + 1.14764 w_k1_AXP_s100
                                 + 0.818942 w_k1_DIS_s100
                                 + 1.00988 w_k1_FDX_s100
                                 + 0.884976 w_k1_IBM_s100
                                 + 0.989006 w_k1_INTC_s100
                                 + 0.888476 w_k1_KO_s100
                                 + 0.931386 w_k1_MAT_s100
                                 + 0.94587 w_k1_MSFT_s100
                                 + 0.980849 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.04795 w_k1_ADM_s100
                                 + 0.832479 w_k1_AXP_s100
                                 + 0.829293 w_k1_DIS_s100
                                 + 0.897943 w_k1_FDX_s100
                                 + 1.04003 w_k1_IBM_s100
                                 + 0.849532 w_k1_INTC_s100
                                 + 0.996638 w_k1_KO_s100
                                 + 1.17038 w_k1_MAT_s100
                                 + 0.921503 w_k1_MSFT_s100
                                 + 0.980915 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.15296 w_k1_ADM_s101
                                 + 1.40582 w_k1_AXP_s101
                                 + 1.49411 w_k1_DIS_s101
                                 + 1.31861 w_k1_FDX_s101
                                 + 1.43497 w_k1_IBM_s101
                                 + 1.26151 w_k1_INTC_s101 + 1.2362 w_k1_KO_s101
                                 + 1.24983 w_k1_MAT_s101
                                 + 1.30524 w_k1_MSFT_s101
                                 + 1.13872 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.896167 w_k1_ADM_s101
                                 + 0.798341 w_k1_AXP_s101
                                 + 0.717959 w_k1_DIS_s101
                                 + 0.777087 w_k1_FDX_s101
                                 + 0.93836 w_k1_IBM_s101
                                 + 0.763261 w_k1_INTC_s101
                                 + 0.898278 w_k1_KO_s101
                                 + 0.796166 w_k1_MAT_s101
                                 + 0.812208 w_k1_MSFT_s101
                                 + 0.818572 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.833535 w_k1_ADM_s101
                                 + 0.770243 w_k1_AXP_s101
                                 + 1.10955 w_k1_DIS_s101
                                 + 1.02558 w_k1_FDX_s101
                                 + 1.04416 w_k1_IBM_s101
                                 + 0.877633 w_k1_INTC_s101
                                 + 1.07543 w_k1_KO_s101 + 1.05546 w_k1_MAT_s101
                                 + 1.11062 w_k1_MSFT_s101
                                 + 1.05109 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09757 w_k1_ADM_s101
                                 + 0.971216 w_k1_AXP_s101
                                 + 0.963421 w_k1_DIS_s101
                                 + 1.09196 w_k1_FDX_s101
                                 + 0.755816 w_k1_IBM_s101
                                 + 0.92319 w_k1_INTC_s101
                                 + 0.925513 w_k1_KO_s101
                                 + 0.88075 w_k1_MAT_s101
                                 + 1.04146 w_k1_MSFT_s101
                                 + 0.984038 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.930329 w_k1_ADM_s101
                                 + 1.003 w_k1_AXP_s101 + 1.0951 w_k1_DIS_s101
                                 + 0.959698 w_k1_FDX_s101
                                 + 1.70971 w_k1_IBM_s101
                                 + 0.967254 w_k1_INTC_s101
                                 + 1.47715 w_k1_KO_s101 + 1.82564 w_k1_MAT_s101
                                 + 0.946931 w_k1_MSFT_s101
                                 + 0.98094 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.10806 w_k1_ADM_s101
                                 + 0.813512 w_k1_AXP_s101
                                 + 0.970763 w_k1_DIS_s101
                                 + 0.931922 w_k1_FDX_s101
                                 + 1.25493 w_k1_IBM_s101
                                 + 0.811445 w_k1_INTC_s101
                                 + 1.19977 w_k1_KO_s101 + 1.03007 w_k1_MAT_s101
                                 + 0.967839 w_k1_MSFT_s101
                                 + 1.03137 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.22296 w_k1_ADM_s101
                                 + 1.2535 w_k1_AXP_s101 + 1.21957 w_k1_DIS_s101
                                 + 1.26184 w_k1_FDX_s101
                                 + 0.902865 w_k1_IBM_s101
                                 + 1.12658 w_k1_INTC_s101
                                 + 1.02625 w_k1_KO_s101 + 1.10864 w_k1_MAT_s101
                                 + 1.20869 w_k1_MSFT_s101
                                 + 1.07224 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.23134 w_k1_ADM_s101
                                 + 1.26711 w_k1_AXP_s101 + 1.7996 w_k1_DIS_s101
                                 + 1.40092 w_k1_FDX_s101
                                 + 1.11218 w_k1_IBM_s101
                                 + 1.44177 w_k1_INTC_s101
                                 + 1.38802 w_k1_KO_s101 + 1.22177 w_k1_MAT_s101
                                 + 1.54926 w_k1_MSFT_s101
                                 + 1.26727 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.15725 w_k1_ADM_s102
                                 + 1.09885 w_k1_AXP_s102
                                 + 1.28844 w_k1_DIS_s102 + 1.072 w_k1_FDX_s102
                                 + 0.992922 w_k1_IBM_s102
                                 + 1.13237 w_k1_INTC_s102
                                 + 1.14326 w_k1_KO_s102
                                 + 0.827068 w_k1_MAT_s102
                                 + 1.12972 w_k1_MSFT_s102
                                 + 1.03031 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.996515 w_k1_ADM_s102
                                 + 0.774707 w_k1_AXP_s102
                                 + 0.826802 w_k1_DIS_s102
                                 + 0.771616 w_k1_FDX_s102
                                 + 0.920738 w_k1_IBM_s102
                                 + 0.925854 w_k1_INTC_s102
                                 + 0.903842 w_k1_KO_s102
                                 + 0.788676 w_k1_MAT_s102
                                 + 0.823288 w_k1_MSFT_s102
                                 + 0.902928 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.868769 w_k1_ADM_s102
                                 + 0.595864 w_k1_AXP_s102
                                 + 0.845802 w_k1_DIS_s102
                                 + 0.783722 w_k1_FDX_s102
                                 + 0.693729 w_k1_IBM_s102
                                 + 0.873551 w_k1_INTC_s102
                                 + 0.796523 w_k1_KO_s102
                                 + 0.625818 w_k1_MAT_s102
                                 + 0.943358 w_k1_MSFT_s102
                                 + 0.849788 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.05713 w_k1_ADM_s102
                                 + 1.06337 w_k1_AXP_s102
                                 + 0.944555 w_k1_DIS_s102
                                 + 0.891988 w_k1_FDX_s102
                                 + 1.4474 w_k1_IBM_s102
                                 + 1.15214 w_k1_INTC_s102
                                 + 1.21692 w_k1_KO_s102 + 1.25189 w_k1_MAT_s102
                                 + 0.990775 w_k1_MSFT_s102
                                 + 0.985293 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.03906 w_k1_ADM_s102
                                 + 1.40866 w_k1_AXP_s102
                                 + 1.00669 w_k1_DIS_s102
                                 + 1.12611 w_k1_FDX_s102
                                 + 0.91506 w_k1_IBM_s102
                                 + 1.12628 w_k1_INTC_s102
                                 + 0.974201 w_k1_KO_s102
                                 + 1.17969 w_k1_MAT_s102
                                 + 0.904257 w_k1_MSFT_s102
                                 + 1.09811 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.21967 w_k1_ADM_s102
                                 + 1.52811 w_k1_AXP_s102
                                 + 1.79758 w_k1_DIS_s102
                                 + 1.25943 w_k1_FDX_s102
                                 + 1.47018 w_k1_IBM_s102
                                 + 1.60979 w_k1_INTC_s102
                                 + 1.41594 w_k1_KO_s102 + 1.094 w_k1_MAT_s102
                                 + 1.25833 w_k1_MSFT_s102
                                 + 1.32623 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.21808 w_k1_ADM_s102
                                 + 1.61424 w_k1_AXP_s102
                                 + 1.50433 w_k1_DIS_s102
                                 + 1.42036 w_k1_FDX_s102
                                 + 0.983151 w_k1_IBM_s102
                                 + 1.62435 w_k1_INTC_s102 + 1.1738 w_k1_KO_s102
                                 + 0.812596 w_k1_MAT_s102
                                 + 1.27571 w_k1_MSFT_s102
                                 + 1.18741 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.19352 w_k1_ADM_s102
                                 + 0.988759 w_k1_AXP_s102
                                 + 1.01996 w_k1_DIS_s102
                                 + 0.954215 w_k1_FDX_s102
                                 + 1.02189 w_k1_IBM_s102
                                 + 1.13265 w_k1_INTC_s102
                                 + 1.04395 w_k1_KO_s102
                                 + 0.961653 w_k1_MAT_s102
                                 + 1.04236 w_k1_MSFT_s102
                                 + 1.06658 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.20434 w_k1_ADM_s103
                                 + 1.20761 w_k1_AXP_s103
                                 + 1.09134 w_k1_DIS_s103
                                 + 1.03636 w_k1_FDX_s103
                                 + 1.19945 w_k1_IBM_s103
                                 + 1.08582 w_k1_INTC_s103
                                 + 1.14884 w_k1_KO_s103 + 1.37055 w_k1_MAT_s103
                                 + 1.11199 w_k1_MSFT_s103
                                 + 1.02177 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.961315 w_k1_ADM_s103
                                 + 0.879441 w_k1_AXP_s103
                                 + 1.04071 w_k1_DIS_s103
                                 + 0.970877 w_k1_FDX_s103
                                 + 1.36135 w_k1_IBM_s103
                                 + 0.962578 w_k1_INTC_s103
                                 + 1.10203 w_k1_KO_s103 + 1.20118 w_k1_MAT_s103
                                 + 1.12244 w_k1_MSFT_s103
                                 + 1.01161 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.20599 w_k1_ADM_s103
                                 + 1.56939 w_k1_AXP_s103
                                 + 1.51103 w_k1_DIS_s103
                                 + 1.20609 w_k1_FDX_s103
                                 + 1.70726 w_k1_IBM_s103
                                 + 1.29283 w_k1_INTC_s103 + 1.3899 w_k1_KO_s103
                                 + 1.70815 w_k1_MAT_s103
                                 + 1.13508 w_k1_MSFT_s103
                                 + 1.28766 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.836422 w_k1_ADM_s103
                                 + 0.954456 w_k1_AXP_s103
                                 + 1.03859 w_k1_DIS_s103
                                 + 0.954215 w_k1_FDX_s103
                                 + 0.972568 w_k1_IBM_s103
                                 + 0.995638 w_k1_INTC_s103
                                 + 0.977143 w_k1_KO_s103
                                 + 1.18805 w_k1_MAT_s103
                                 + 0.963539 w_k1_MSFT_s103
                                 + 0.996839 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.994528 w_k1_ADM_s103
                                 + 1.21456 w_k1_AXP_s103
                                 + 1.21461 w_k1_DIS_s103
                                 + 1.18742 w_k1_FDX_s103
                                 + 0.874776 w_k1_IBM_s103
                                 + 1.1913 w_k1_INTC_s103
                                 + 0.919461 w_k1_KO_s103
                                 + 1.11035 w_k1_MAT_s103
                                 + 1.02639 w_k1_MSFT_s103
                                 + 1.01095 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.871048 w_k1_ADM_s103
                                 + 0.843679 w_k1_AXP_s103
                                 + 0.924399 w_k1_DIS_s103
                                 + 1.07752 w_k1_FDX_s103
                                 + 0.637104 w_k1_IBM_s103
                                 + 1.16076 w_k1_INTC_s103
                                 + 0.896975 w_k1_KO_s103
                                 + 0.549875 w_k1_MAT_s103
                                 + 0.959665 w_k1_MSFT_s103
                                 + 0.941881 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.08346 w_k1_ADM_s103
                                 + 1.3509 w_k1_AXP_s103 + 1.67123 w_k1_DIS_s103
                                 + 1.35899 w_k1_FDX_s103
                                 + 1.04108 w_k1_IBM_s103
                                 + 1.44271 w_k1_INTC_s103
                                 + 1.22168 w_k1_KO_s103 + 1.20519 w_k1_MAT_s103
                                 + 1.32323 w_k1_MSFT_s103
                                 + 1.29239 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.739438 w_k1_ADM_s103
                                 + 0.751568 w_k1_AXP_s103
                                 + 0.644021 w_k1_DIS_s103
                                 + 0.77872 w_k1_FDX_s103
                                 + 0.931022 w_k1_IBM_s103
                                 + 0.990561 w_k1_INTC_s103
                                 + 0.852615 w_k1_KO_s103
                                 + 1.08283 w_k1_MAT_s103
                                 + 0.904509 w_k1_MSFT_s103
                                 + 0.873032 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.894058 w_k1_ADM_s104
                                 + 0.773959 w_k1_AXP_s104
                                 + 0.578236 w_k1_DIS_s104
                                 + 0.783527 w_k1_FDX_s104
                                 + 0.824626 w_k1_IBM_s104
                                 + 0.90753 w_k1_INTC_s104
                                 + 0.74551 w_k1_KO_s104
                                 + 0.724045 w_k1_MAT_s104
                                 + 0.768287 w_k1_MSFT_s104
                                 + 0.90537 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.16102 w_k1_ADM_s104
                                 + 1.25187 w_k1_AXP_s104
                                 + 0.827325 w_k1_DIS_s104
                                 + 1.10098 w_k1_FDX_s104
                                 + 0.938332 w_k1_IBM_s104
                                 + 1.2876 w_k1_INTC_s104
                                 + 0.975704 w_k1_KO_s104
                                 + 1.39012 w_k1_MAT_s104
                                 + 1.13826 w_k1_MSFT_s104
                                 + 1.01452 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.951582 w_k1_ADM_s104
                                 + 0.697596 w_k1_AXP_s104
                                 + 0.776646 w_k1_DIS_s104
                                 + 0.807761 w_k1_FDX_s104
                                 + 1.01538 w_k1_IBM_s104
                                 + 0.844379 w_k1_INTC_s104
                                 + 0.98183 w_k1_KO_s104 + 0.87116 w_k1_MAT_s104
                                 + 0.851597 w_k1_MSFT_s104
                                 + 0.874515 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.04719 w_k1_ADM_s104
                                 + 1.36974 w_k1_AXP_s104
                                 + 1.38751 w_k1_DIS_s104
                                 + 1.25546 w_k1_FDX_s104
                                 + 1.30143 w_k1_IBM_s104
                                 + 1.15575 w_k1_INTC_s104
                                 + 1.38274 w_k1_KO_s104 + 1.33226 w_k1_MAT_s104
                                 + 1.11645 w_k1_MSFT_s104
                                 + 1.16448 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.873544 w_k1_ADM_s104
                                 + 0.774563 w_k1_AXP_s104
                                 + 0.778333 w_k1_DIS_s104
                                 + 0.834433 w_k1_FDX_s104
                                 + 0.754999 w_k1_IBM_s104
                                 + 1.146 w_k1_INTC_s104 + 0.94029 w_k1_KO_s104
                                 + 0.656356 w_k1_MAT_s104
                                 + 0.961088 w_k1_MSFT_s104
                                 + 0.944383 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.1728 w_k1_ADM_s104
                                 + 0.970322 w_k1_AXP_s104
                                 + 1.15562 w_k1_DIS_s104
                                 + 1.26126 w_k1_FDX_s104
                                 + 0.927624 w_k1_IBM_s104
                                 + 1.25804 w_k1_INTC_s104 + 1.1625 w_k1_KO_s104
                                 + 1.02265 w_k1_MAT_s104
                                 + 1.23231 w_k1_MSFT_s104
                                 + 1.14412 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.16289 w_k1_ADM_s104
                                 + 0.975089 w_k1_AXP_s104
                                 + 1.0169 w_k1_DIS_s104 + 1.07955 w_k1_FDX_s104
                                 + 0.76657 w_k1_IBM_s104
                                 + 0.966342 w_k1_INTC_s104
                                 + 1.01901 w_k1_KO_s104
                                 + 0.761647 w_k1_MAT_s104
                                 + 1.04883 w_k1_MSFT_s104
                                 + 0.90661 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.881048 w_k1_ADM_s104
                                 + 0.850742 w_k1_AXP_s104
                                 + 0.945545 w_k1_DIS_s104
                                 + 0.953273 w_k1_FDX_s104
                                 + 1.35181 w_k1_IBM_s104
                                 + 1.20769 w_k1_INTC_s104
                                 + 1.30173 w_k1_KO_s104 + 1.36939 w_k1_MAT_s104
                                 + 1.01178 w_k1_MSFT_s104
                                 + 1.05599 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.875952 w_k1_ADM_s105
                                 + 1.10098 w_k1_AXP_s105
                                 + 0.971032 w_k1_DIS_s105
                                 + 0.992634 w_k1_FDX_s105
                                 + 1.28638 w_k1_IBM_s105
                                 + 1.30477 w_k1_INTC_s105
                                 + 1.22192 w_k1_KO_s105 + 1.68977 w_k1_MAT_s105
                                 + 1.1477 w_k1_MSFT_s105 + 1.04894 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.30595 w_k1_ADM_s105
                                 + 0.982384 w_k1_AXP_s105
                                 + 1.11056 w_k1_DIS_s105
                                 + 1.13214 w_k1_FDX_s105
                                 + 0.783485 w_k1_IBM_s105
                                 + 1.3924 w_k1_INTC_s105 + 1.15315 w_k1_KO_s105
                                 + 0.73559 w_k1_MAT_s105
                                 + 1.25794 w_k1_MSFT_s105
                                 + 0.962442 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.41709 w_k1_ADM_s105
                                 + 1.4922 w_k1_AXP_s105 + 1.66347 w_k1_DIS_s105
                                 + 1.23839 w_k1_FDX_s105
                                 + 1.49915 w_k1_IBM_s105
                                 + 1.27087 w_k1_INTC_s105
                                 + 1.68636 w_k1_KO_s105 + 1.84244 w_k1_MAT_s105
                                 + 1.34621 w_k1_MSFT_s105
                                 + 1.10583 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.797302 w_k1_ADM_s105
                                 + 0.640919 w_k1_AXP_s105
                                 + 0.704409 w_k1_DIS_s105
                                 + 0.690934 w_k1_FDX_s105
                                 + 0.736234 w_k1_IBM_s105
                                 + 0.9177 w_k1_INTC_s105 + 0.94579 w_k1_KO_s105
                                 + 0.757602 w_k1_MAT_s105
                                 + 0.85696 w_k1_MSFT_s105
                                 + 0.844356 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.02461 w_k1_ADM_s105
                                 + 0.757957 w_k1_AXP_s105
                                 + 0.74538 w_k1_DIS_s105
                                 + 0.823705 w_k1_FDX_s105
                                 + 0.969678 w_k1_IBM_s105
                                 + 1.19181 w_k1_INTC_s105 + 1.0476 w_k1_KO_s105
                                 + 0.994098 w_k1_MAT_s105
                                 + 0.974343 w_k1_MSFT_s105
                                 + 0.880045 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.940129 w_k1_ADM_s105
                                 + 0.796804 w_k1_AXP_s105
                                 + 1.07408 w_k1_DIS_s105
                                 + 0.917171 w_k1_FDX_s105
                                 + 0.964649 w_k1_IBM_s105
                                 + 1.29076 w_k1_INTC_s105
                                 + 1.10248 w_k1_KO_s105
                                 + 0.975279 w_k1_MAT_s105
                                 + 1.22471 w_k1_MSFT_s105
                                 + 0.957784 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.995396 w_k1_ADM_s105
                                 + 0.678429 w_k1_AXP_s105
                                 + 0.6644 w_k1_DIS_s105
                                 + 0.754883 w_k1_FDX_s105
                                 + 0.557145 w_k1_IBM_s105
                                 + 0.945657 w_k1_INTC_s105
                                 + 0.799547 w_k1_KO_s105
                                 + 0.690712 w_k1_MAT_s105
                                 + 0.92666 w_k1_MSFT_s105
                                 + 0.844668 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.926 w_k1_ADM_s105
                                 + 0.907805 w_k1_AXP_s105
                                 + 1.24471 w_k1_DIS_s105
                                 + 0.952219 w_k1_FDX_s105
                                 + 1.43289 w_k1_IBM_s105
                                 + 1.43859 w_k1_INTC_s105
                                 + 1.45667 w_k1_KO_s105 + 1.18806 w_k1_MAT_s105
                                 + 1.2885 w_k1_MSFT_s105 + 1.05967 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.37209 w_k1_ADM_s106
                                 + 1.13118 w_k1_AXP_s106
                                 + 1.10046 w_k1_DIS_s106
                                 + 1.25387 w_k1_FDX_s106
                                 + 0.747526 w_k1_IBM_s106
                                 + 1.0622 w_k1_INTC_s106
                                 + 0.906389 w_k1_KO_s106
                                 + 0.76312 w_k1_MAT_s106
                                 + 1.06705 w_k1_MSFT_s106
                                 + 1.03189 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.17764 w_k1_ADM_s106
                                 + 1.09544 w_k1_AXP_s106
                                 + 0.845983 w_k1_DIS_s106
                                 + 0.915207 w_k1_FDX_s106
                                 + 1.05125 w_k1_IBM_s106
                                 + 1.00204 w_k1_INTC_s106
                                 + 0.824028 w_k1_KO_s106
                                 + 0.802372 w_k1_MAT_s106
                                 + 0.853151 w_k1_MSFT_s106
                                 + 0.988474 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.24594 w_k1_ADM_s106
                                 + 1.23143 w_k1_AXP_s106
                                 + 1.25588 w_k1_DIS_s106
                                 + 1.07931 w_k1_FDX_s106
                                 + 1.44427 w_k1_IBM_s106
                                 + 1.29542 w_k1_INTC_s106
                                 + 1.20084 w_k1_KO_s106 + 1.00171 w_k1_MAT_s106
                                 + 0.991287 w_k1_MSFT_s106
                                 + 1.08817 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.939039 w_k1_ADM_s106
                                 + 0.739185 w_k1_AXP_s106
                                 + 0.823291 w_k1_DIS_s106
                                 + 0.794976 w_k1_FDX_s106
                                 + 0.939576 w_k1_IBM_s106
                                 + 0.880459 w_k1_INTC_s106
                                 + 0.972848 w_k1_KO_s106
                                 + 0.661605 w_k1_MAT_s106
                                 + 0.916635 w_k1_MSFT_s106
                                 + 0.96511 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.934162 w_k1_ADM_s106
                                 + 0.80989 w_k1_AXP_s106
                                 + 0.758275 w_k1_DIS_s106
                                 + 0.859629 w_k1_FDX_s106
                                 + 0.648161 w_k1_IBM_s106
                                 + 1.06167 w_k1_INTC_s106
                                 + 0.802794 w_k1_KO_s106
                                 + 0.435564 w_k1_MAT_s106
                                 + 0.781592 w_k1_MSFT_s106
                                 + 0.890095 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.06577 w_k1_ADM_s106
                                 + 0.888724 w_k1_AXP_s106
                                 + 1.01433 w_k1_DIS_s106
                                 + 1.05569 w_k1_FDX_s106
                                 + 0.674585 w_k1_IBM_s106
                                 + 1.22827 w_k1_INTC_s106
                                 + 0.918221 w_k1_KO_s106
                                 + 0.587964 w_k1_MAT_s106
                                 + 1.23455 w_k1_MSFT_s106
                                 + 1.03817 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.28379 w_k1_ADM_s106
                                 + 1.5219 w_k1_AXP_s106 + 1.62866 w_k1_DIS_s106
                                 + 1.40303 w_k1_FDX_s106
                                 + 1.11675 w_k1_IBM_s106
                                 + 1.47573 w_k1_INTC_s106
                                 + 1.15865 w_k1_KO_s106 + 1.09012 w_k1_MAT_s106
                                 + 1.29382 w_k1_MSFT_s106
                                 + 1.26627 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.949832 w_k1_ADM_s106
                                 + 0.957601 w_k1_AXP_s106
                                 + 1.30484 w_k1_DIS_s106
                                 + 1.06973 w_k1_FDX_s106
                                 + 1.09498 w_k1_IBM_s106
                                 + 1.41653 w_k1_INTC_s106
                                 + 1.09921 w_k1_KO_s106
                                 + 0.815744 w_k1_MAT_s106
                                 + 1.1456 w_k1_MSFT_s106
                                 + 0.963314 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.708934 w_k1_ADM_s107
                                 + 1.26088 w_k1_AXP_s107
                                 + 1.06033 w_k1_DIS_s107
                                 + 1.00715 w_k1_FDX_s107
                                 + 1.24372 w_k1_IBM_s107
                                 + 1.0271 w_k1_INTC_s107 + 1.18184 w_k1_KO_s107
                                 + 1.23845 w_k1_MAT_s107
                                 + 0.951503 w_k1_MSFT_s107
                                 + 1.10141 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.972137 w_k1_ADM_s107
                                 + 1.52504 w_k1_AXP_s107
                                 + 1.70373 w_k1_DIS_s107
                                 + 1.52503 w_k1_FDX_s107
                                 + 1.11882 w_k1_IBM_s107
                                 + 1.53331 w_k1_INTC_s107
                                 + 1.39102 w_k1_KO_s107 + 1.12219 w_k1_MAT_s107
                                 + 1.58844 w_k1_MSFT_s107
                                 + 1.41999 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.547095 w_k1_ADM_s107
                                 + 0.680361 w_k1_AXP_s107
                                 + 0.739742 w_k1_DIS_s107
                                 + 0.77584 w_k1_FDX_s107
                                 + 1.03875 w_k1_IBM_s107
                                 + 0.847674 w_k1_INTC_s107
                                 + 0.893376 w_k1_KO_s107
                                 + 0.596306 w_k1_MAT_s107
                                 + 0.808532 w_k1_MSFT_s107
                                 + 0.854159 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.02487 w_k1_ADM_s107
                                 + 1.2525 w_k1_AXP_s107 + 1.36919 w_k1_DIS_s107
                                 + 1.1478 w_k1_FDX_s107
                                 + 0.953018 w_k1_IBM_s107
                                 + 1.12758 w_k1_INTC_s107
                                 + 1.27628 w_k1_KO_s107
                                 + 0.870505 w_k1_MAT_s107
                                 + 1.13674 w_k1_MSFT_s107
                                 + 1.21805 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.562764 w_k1_ADM_s107
                                 + 0.860745 w_k1_AXP_s107
                                 + 0.538978 w_k1_DIS_s107
                                 + 0.68098 w_k1_FDX_s107
                                 + 0.941573 w_k1_IBM_s107
                                 + 0.915489 w_k1_INTC_s107
                                 + 0.830092 w_k1_KO_s107
                                 + 0.888895 w_k1_MAT_s107
                                 + 0.771913 w_k1_MSFT_s107
                                 + 0.925509 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.665587 w_k1_ADM_s107
                                 + 0.877653 w_k1_AXP_s107
                                 + 0.6605 w_k1_DIS_s107
                                 + 0.850903 w_k1_FDX_s107
                                 + 0.681944 w_k1_IBM_s107
                                 + 0.923067 w_k1_INTC_s107
                                 + 0.783642 w_k1_KO_s107
                                 + 0.641841 w_k1_MAT_s107
                                 + 0.871456 w_k1_MSFT_s107
                                 + 0.932278 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.777883 w_k1_ADM_s107
                                 + 0.905729 w_k1_AXP_s107
                                 + 1.06131 w_k1_DIS_s107
                                 + 0.994704 w_k1_FDX_s107
                                 + 0.924189 w_k1_IBM_s107
                                 + 0.9348 w_k1_INTC_s107
                                 + 0.977364 w_k1_KO_s107
                                 + 0.713091 w_k1_MAT_s107
                                 + 0.954952 w_k1_MSFT_s107
                                 + 1.03261 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.753541 w_k1_ADM_s107
                                 + 1.01265 w_k1_AXP_s107
                                 + 0.95801 w_k1_DIS_s107
                                 + 0.852873 w_k1_FDX_s107
                                 + 1.55992 w_k1_IBM_s107
                                 + 1.07916 w_k1_INTC_s107
                                 + 1.16092 w_k1_KO_s107 + 1.09493 w_k1_MAT_s107
                                 + 0.938401 w_k1_MSFT_s107
                                 + 0.995356 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.994887 w_k1_ADM_s110
                                 + 1.25677 w_k1_AXP_s110
                                 + 0.950958 w_k1_DIS_s110
                                 + 0.956478 w_k1_FDX_s110
                                 + 1.11077 w_k1_IBM_s110
                                 + 1.05747 w_k1_INTC_s110
                                 + 1.01035 w_k1_KO_s110 + 1.24808 w_k1_MAT_s110
                                 + 0.912557 w_k1_MSFT_s110
                                 + 1.10461 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.01952 w_k1_ADM_s110
                                 + 1.33349 w_k1_AXP_s110
                                 + 1.08572 w_k1_DIS_s110
                                 + 0.953307 w_k1_FDX_s110
                                 + 1.49477 w_k1_IBM_s110
                                 + 1.24737 w_k1_INTC_s110 + 1.1369 w_k1_KO_s110
                                 + 1.8298 w_k1_MAT_s110
                                 + 0.967511 w_k1_MSFT_s110
                                 + 1.10803 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.758881 w_k1_ADM_s110
                                 + 0.637821 w_k1_AXP_s110
                                 + 0.602515 w_k1_DIS_s110
                                 + 0.738434 w_k1_FDX_s110
                                 + 0.82346 w_k1_IBM_s110
                                 + 0.895368 w_k1_INTC_s110
                                 + 0.809813 w_k1_KO_s110
                                 + 0.886792 w_k1_MAT_s110
                                 + 0.763991 w_k1_MSFT_s110
                                 + 0.941429 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.768996 w_k1_ADM_s110
                                 + 0.845649 w_k1_AXP_s110
                                 + 0.738188 w_k1_DIS_s110
                                 + 0.774894 w_k1_FDX_s110
                                 + 0.888229 w_k1_IBM_s110
                                 + 0.989889 w_k1_INTC_s110
                                 + 0.861845 w_k1_KO_s110
                                 + 0.730959 w_k1_MAT_s110
                                 + 0.766352 w_k1_MSFT_s110
                                 + 0.822245 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.32299 w_k1_ADM_s110
                                 + 1.44519 w_k1_AXP_s110
                                 + 1.48998 w_k1_DIS_s110
                                 + 1.28666 w_k1_FDX_s110
                                 + 1.22112 w_k1_IBM_s110
                                 + 1.20923 w_k1_INTC_s110 + 1.3897 w_k1_KO_s110
                                 + 1.24139 w_k1_MAT_s110
                                 + 1.1886 w_k1_MSFT_s110 + 1.20022 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.03252 w_k1_ADM_s110
                                 + 1.03236 w_k1_AXP_s110
                                 + 1.21699 w_k1_DIS_s110
                                 + 1.13117 w_k1_FDX_s110
                                 + 0.866052 w_k1_IBM_s110
                                 + 1.23565 w_k1_INTC_s110
                                 + 0.959387 w_k1_KO_s110
                                 + 0.818385 w_k1_MAT_s110
                                 + 1.12335 w_k1_MSFT_s110
                                 + 0.960988 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.97066 w_k1_ADM_s110
                                 + 1.00086 w_k1_AXP_s110
                                 + 0.819567 w_k1_DIS_s110
                                 + 0.935593 w_k1_FDX_s110
                                 + 0.836926 w_k1_IBM_s110
                                 + 0.882875 w_k1_INTC_s110
                                 + 0.715262 w_k1_KO_s110
                                 + 0.825033 w_k1_MAT_s110
                                 + 0.846527 w_k1_MSFT_s110
                                 + 0.878252 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.984664 w_k1_ADM_s110
                                 + 1.2949 w_k1_AXP_s110
                                 + 0.810097 w_k1_DIS_s110
                                 + 1.10788 w_k1_FDX_s110
                                 + 0.663344 w_k1_IBM_s110
                                 + 1.25643 w_k1_INTC_s110
                                 + 0.775604 w_k1_KO_s110
                                 + 0.818163 w_k1_MAT_s110
                                 + 0.921421 w_k1_MSFT_s110
                                 + 0.891384 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.832696 w_k1_ADM_s111
                                 + 0.638265 w_k1_AXP_s111
                                 + 0.73037 w_k1_DIS_s111
                                 + 0.843798 w_k1_FDX_s111
                                 + 0.720245 w_k1_IBM_s111
                                 + 0.720276 w_k1_INTC_s111
                                 + 0.747317 w_k1_KO_s111
                                 + 0.617513 w_k1_MAT_s111
                                 + 0.827144 w_k1_MSFT_s111
                                 + 0.807264 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.01599 w_k1_ADM_s111
                                 + 0.948885 w_k1_AXP_s111
                                 + 0.978528 w_k1_DIS_s111
                                 + 1.03392 w_k1_FDX_s111
                                 + 1.17494 w_k1_IBM_s111
                                 + 0.977804 w_k1_INTC_s111
                                 + 1.10559 w_k1_KO_s111 + 1.13006 w_k1_MAT_s111
                                 + 1.07122 w_k1_MSFT_s111
                                 + 1.04823 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.35675 w_k1_ADM_s111
                                 + 1.46192 w_k1_AXP_s111
                                 + 1.58255 w_k1_DIS_s111
                                 + 1.58409 w_k1_FDX_s111
                                 + 1.12279 w_k1_IBM_s111
                                 + 1.52622 w_k1_INTC_s111
                                 + 1.28816 w_k1_KO_s111 + 1.23966 w_k1_MAT_s111
                                 + 1.33255 w_k1_MSFT_s111
                                 + 1.29721 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.807226 w_k1_ADM_s111
                                 + 0.860214 w_k1_AXP_s111
                                 + 0.978551 w_k1_DIS_s111
                                 + 0.940737 w_k1_FDX_s111
                                 + 1.05658 w_k1_IBM_s111
                                 + 1.08891 w_k1_INTC_s111
                                 + 0.926841 w_k1_KO_s111
                                 + 1.07363 w_k1_MAT_s111
                                 + 1.0869 w_k1_MSFT_s111
                                 + 0.983058 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.861508 w_k1_ADM_s111
                                 + 0.848527 w_k1_AXP_s111
                                 + 0.642968 w_k1_DIS_s111
                                 + 0.844558 w_k1_FDX_s111
                                 + 0.958557 w_k1_IBM_s111
                                 + 0.814733 w_k1_INTC_s111
                                 + 0.798873 w_k1_KO_s111
                                 + 0.983256 w_k1_MAT_s111
                                 + 0.893895 w_k1_MSFT_s111
                                 + 0.961964 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.766394 w_k1_ADM_s111
                                 + 0.803365 w_k1_AXP_s111
                                 + 0.790911 w_k1_DIS_s111
                                 + 0.791716 w_k1_FDX_s111
                                 + 1.37363 w_k1_IBM_s111
                                 + 1.06078 w_k1_INTC_s111
                                 + 0.999074 w_k1_KO_s111
                                 + 1.00815 w_k1_MAT_s111
                                 + 0.927251 w_k1_MSFT_s111
                                 + 1.01134 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.37968 w_k1_ADM_s111
                                 + 1.10715 w_k1_AXP_s111 + 1.2665 w_k1_DIS_s111
                                 + 1.22959 w_k1_FDX_s111
                                 + 1.06849 w_k1_IBM_s111
                                 + 1.07611 w_k1_INTC_s111
                                 + 1.19472 w_k1_KO_s111 + 1.04251 w_k1_MAT_s111
                                 + 1.12116 w_k1_MSFT_s111
                                 + 1.21674 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.957396 w_k1_ADM_s111
                                 + 1.50664 w_k1_AXP_s111 + 1.1632 w_k1_DIS_s111
                                 + 1.05219 w_k1_FDX_s111
                                 + 1.76033 w_k1_IBM_s111
                                 + 1.23357 w_k1_INTC_s111
                                 + 1.32341 w_k1_KO_s111 + 1.60337 w_k1_MAT_s111
                                 + 1.07024 w_k1_MSFT_s111
                                 + 1.21166 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.914007 w_k1_ADM_s112
                                 + 0.874583 w_k1_AXP_s112
                                 + 0.844783 w_k1_DIS_s112
                                 + 0.934547 w_k1_FDX_s112
                                 + 0.694021 w_k1_IBM_s112
                                 + 0.932041 w_k1_INTC_s112
                                 + 0.80563 w_k1_KO_s112
                                 + 0.968378 w_k1_MAT_s112
                                 + 0.938781 w_k1_MSFT_s112
                                 + 0.962548 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.15742 w_k1_ADM_s112
                                 + 0.897317 w_k1_AXP_s112
                                 + 1.01992 w_k1_DIS_s112
                                 + 1.10436 w_k1_FDX_s112
                                 + 0.769611 w_k1_IBM_s112
                                 + 1.06958 w_k1_INTC_s112
                                 + 0.931535 w_k1_KO_s112
                                 + 0.748026 w_k1_MAT_s112
                                 + 1.05117 w_k1_MSFT_s112
                                 + 1.00613 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.843134 w_k1_ADM_s112
                                 + 0.698458 w_k1_AXP_s112
                                 + 0.62335 w_k1_DIS_s112
                                 + 0.768746 w_k1_FDX_s112
                                 + 0.762572 w_k1_IBM_s112
                                 + 0.790334 w_k1_INTC_s112
                                 + 0.777561 w_k1_KO_s112
                                 + 0.749083 w_k1_MAT_s112
                                 + 0.815755 w_k1_MSFT_s112
                                 + 0.937143 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.93723 w_k1_ADM_s112
                                 + 0.954288 w_k1_AXP_s112
                                 + 0.790168 w_k1_DIS_s112
                                 + 0.901088 w_k1_FDX_s112
                                 + 1.06101 w_k1_IBM_s112
                                 + 0.902009 w_k1_INTC_s112
                                 + 1.0183 w_k1_KO_s112 + 1.10436 w_k1_MAT_s112
                                 + 1.00835 w_k1_MSFT_s112
                                 + 1.06561 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.36951 w_k1_ADM_s112
                                 + 1.49043 w_k1_AXP_s112
                                 + 1.78754 w_k1_DIS_s112
                                 + 1.27585 w_k1_FDX_s112
                                 + 1.54088 w_k1_IBM_s112
                                 + 1.23983 w_k1_INTC_s112 + 1.4875 w_k1_KO_s112
                                 + 1.66067 w_k1_MAT_s112
                                 + 1.31662 w_k1_MSFT_s112
                                 + 1.24188 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.04711 w_k1_ADM_s112
                                 + 0.900636 w_k1_AXP_s112 + 1.385 w_k1_DIS_s112
                                 + 1.149 w_k1_FDX_s112 + 1.03864 w_k1_IBM_s112
                                 + 1.36299 w_k1_INTC_s112
                                 + 1.11788 w_k1_KO_s112
                                 + 0.747733 w_k1_MAT_s112
                                 + 1.2027 w_k1_MSFT_s112 + 1.20077 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.27904 w_k1_ADM_s112
                                 + 1.22098 w_k1_AXP_s112 + 1.2753 w_k1_DIS_s112
                                 + 1.10204 w_k1_FDX_s112
                                 + 1.09634 w_k1_IBM_s112
                                 + 0.995951 w_k1_INTC_s112
                                 + 1.19519 w_k1_KO_s112 + 1.55321 w_k1_MAT_s112
                                 + 1.10032 w_k1_MSFT_s112 + 1.1506 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.37804 w_k1_ADM_s112
                                 + 1.64686 w_k1_AXP_s112
                                 + 1.36022 w_k1_DIS_s112
                                 + 1.26105 w_k1_FDX_s112 + 1.1824 w_k1_IBM_s112
                                 + 1.08883 w_k1_INTC_s112
                                 + 1.13936 w_k1_KO_s112 + 1.37032 w_k1_MAT_s112
                                 + 1.20918 w_k1_MSFT_s112
                                 + 1.10243 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.21833 w_k1_ADM_s113
                                 + 0.760228 w_k1_AXP_s113
                                 + 0.809653 w_k1_DIS_s113
                                 + 0.830614 w_k1_FDX_s113
                                 + 0.612281 w_k1_IBM_s113
                                 + 0.918829 w_k1_INTC_s113
                                 + 0.786729 w_k1_KO_s113
                                 + 0.475671 w_k1_MAT_s113
                                 + 0.927683 w_k1_MSFT_s113
                                 + 0.845302 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.34436 w_k1_ADM_s113
                                 + 1.52752 w_k1_AXP_s113
                                 + 1.53473 w_k1_DIS_s113
                                 + 1.29787 w_k1_FDX_s113
                                 + 1.30564 w_k1_IBM_s113
                                 + 1.20844 w_k1_INTC_s113
                                 + 1.22949 w_k1_KO_s113 + 1.89091 w_k1_MAT_s113
                                 + 1.33759 w_k1_MSFT_s113
                                 + 1.07499 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.993755 w_k1_ADM_s113
                                 + 0.662758 w_k1_AXP_s113
                                 + 1.03788 w_k1_DIS_s113
                                 + 0.895096 w_k1_FDX_s113
                                 + 0.932161 w_k1_IBM_s113
                                 + 0.936885 w_k1_INTC_s113
                                 + 1.10641 w_k1_KO_s113
                                 + 0.650703 w_k1_MAT_s113
                                 + 1.02588 w_k1_MSFT_s113
                                 + 0.891175 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.33231 w_k1_ADM_s113
                                 + 0.989627 w_k1_AXP_s113
                                 + 1.53139 w_k1_DIS_s113 + 1.2613 w_k1_FDX_s113
                                 + 0.996073 w_k1_IBM_s113
                                 + 1.06417 w_k1_INTC_s113
                                 + 1.19626 w_k1_KO_s113
                                 + 0.971676 w_k1_MAT_s113
                                 + 1.15729 w_k1_MSFT_s113
                                 + 1.01098 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.17872 w_k1_ADM_s113
                                 + 1.11382 w_k1_AXP_s113
                                 + 0.928325 w_k1_DIS_s113
                                 + 1.08098 w_k1_FDX_s113
                                 + 0.786522 w_k1_IBM_s113
                                 + 1.00256 w_k1_INTC_s113
                                 + 0.862584 w_k1_KO_s113
                                 + 0.845964 w_k1_MAT_s113
                                 + 1.06778 w_k1_MSFT_s113
                                 + 0.969922 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.970305 w_k1_ADM_s113
                                 + 0.937449 w_k1_AXP_s113
                                 + 0.77184 w_k1_DIS_s113
                                 + 0.813675 w_k1_FDX_s113
                                 + 0.956022 w_k1_IBM_s113
                                 + 0.95399 w_k1_INTC_s113
                                 + 0.929727 w_k1_KO_s113
                                 + 0.905885 w_k1_MAT_s113
                                 + 0.854725 w_k1_MSFT_s113
                                 + 0.877963 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.398 w_k1_ADM_s113
                                 + 1.16962 w_k1_AXP_s113
                                 + 1.07819 w_k1_DIS_s113
                                 + 1.00431 w_k1_FDX_s113 + 1.5349 w_k1_IBM_s113
                                 + 0.963899 w_k1_INTC_s113
                                 + 1.18242 w_k1_KO_s113 + 1.26753 w_k1_MAT_s113
                                 + 1.04145 w_k1_MSFT_s113
                                 + 1.01064 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.31949 w_k1_ADM_s113
                                 + 0.919927 w_k1_AXP_s113
                                 + 1.16196 w_k1_DIS_s113
                                 + 1.14495 w_k1_FDX_s113
                                 + 0.895318 w_k1_IBM_s113
                                 + 1.11076 w_k1_INTC_s113
                                 + 1.16571 w_k1_KO_s113 + 0.93747 w_k1_MAT_s113
                                 + 1.29578 w_k1_MSFT_s113
                                 + 1.10827 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.55008 w_k1_ADM_s114
                                 + 1.57131 w_k1_AXP_s114
                                 + 1.59158 w_k1_DIS_s114
                                 + 1.55557 w_k1_FDX_s114
                                 + 0.922838 w_k1_IBM_s114
                                 + 1.61749 w_k1_INTC_s114 + 1.2263 w_k1_KO_s114
                                 + 1.02565 w_k1_MAT_s114
                                 + 1.53219 w_k1_MSFT_s114 + 1.1719 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.959371 w_k1_ADM_s114
                                 + 1.08031 w_k1_AXP_s114
                                 + 0.853592 w_k1_DIS_s114
                                 + 0.807985 w_k1_FDX_s114
                                 + 1.50177 w_k1_IBM_s114
                                 + 1.10048 w_k1_INTC_s114
                                 + 1.02177 w_k1_KO_s114 + 1.59152 w_k1_MAT_s114
                                 + 0.950233 w_k1_MSFT_s114
                                 + 0.820317 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.13632 w_k1_ADM_s114
                                 + 1.14866 w_k1_AXP_s114
                                 + 1.62873 w_k1_DIS_s114
                                 + 1.25506 w_k1_FDX_s114
                                 + 1.26626 w_k1_IBM_s114
                                 + 1.11993 w_k1_INTC_s114
                                 + 1.51438 w_k1_KO_s114 + 1.33356 w_k1_MAT_s114
                                 + 1.46179 w_k1_MSFT_s114
                                 + 1.09386 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.24085 w_k1_ADM_s114
                                 + 1.23161 w_k1_AXP_s114
                                 + 0.967828 w_k1_DIS_s114
                                 + 1.04969 w_k1_FDX_s114
                                 + 1.13157 w_k1_IBM_s114
                                 + 1.18953 w_k1_INTC_s114
                                 + 0.978153 w_k1_KO_s114
                                 + 1.25092 w_k1_MAT_s114
                                 + 1.03196 w_k1_MSFT_s114
                                 + 1.02966 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.962449 w_k1_ADM_s114
                                 + 0.945313 w_k1_AXP_s114
                                 + 1.0836 w_k1_DIS_s114
                                 + 0.971254 w_k1_FDX_s114
                                 + 0.922472 w_k1_IBM_s114
                                 + 1.01477 w_k1_INTC_s114
                                 + 0.995588 w_k1_KO_s114
                                 + 1.07847 w_k1_MAT_s114
                                 + 1.10584 w_k1_MSFT_s114 + 1.0026 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.749803 w_k1_ADM_s114
                                 + 0.771367 w_k1_AXP_s114
                                 + 0.651687 w_k1_DIS_s114
                                 + 0.80796 w_k1_FDX_s114
                                 + 0.816761 w_k1_IBM_s114
                                 + 1.02398 w_k1_INTC_s114
                                 + 0.786334 w_k1_KO_s114
                                 + 0.745075 w_k1_MAT_s114
                                 + 0.826867 w_k1_MSFT_s114
                                 + 0.793038 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.01453 w_k1_ADM_s114
                                 + 1.12483 w_k1_AXP_s114
                                 + 1.64355 w_k1_DIS_s114
                                 + 1.31509 w_k1_FDX_s114
                                 + 0.832374 w_k1_IBM_s114
                                 + 1.36306 w_k1_INTC_s114
                                 + 1.12912 w_k1_KO_s114
                                 + 0.883007 w_k1_MAT_s114
                                 + 1.49419 w_k1_MSFT_s114
                                 + 1.22146 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.4709 w_k1_ADM_s114
                                 + 1.61898 w_k1_AXP_s114
                                 + 1.50197 w_k1_DIS_s114
                                 + 1.22381 w_k1_FDX_s114
                                 + 1.55796 w_k1_IBM_s114
                                 + 1.42266 w_k1_INTC_s114
                                 + 1.53772 w_k1_KO_s114 + 1.43546 w_k1_MAT_s114
                                 + 1.28608 w_k1_MSFT_s114
                                 + 1.22421 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.62444 w_k1_ADM_s115
                                 + 0.658739 w_k1_AXP_s115
                                 + 0.644857 w_k1_DIS_s115
                                 + 0.720205 w_k1_FDX_s115
                                 + 0.655368 w_k1_IBM_s115
                                 + 0.64245 w_k1_INTC_s115
                                 + 0.657073 w_k1_KO_s115
                                 + 0.856211 w_k1_MAT_s115
                                 + 0.768141 w_k1_MSFT_s115
                                 + 0.803512 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.804539 w_k1_ADM_s115
                                 + 0.951063 w_k1_AXP_s115
                                 + 0.839304 w_k1_DIS_s115
                                 + 0.819148 w_k1_FDX_s115
                                 + 0.861571 w_k1_IBM_s115
                                 + 0.829841 w_k1_INTC_s115
                                 + 0.693441 w_k1_KO_s115
                                 + 1.18913 w_k1_MAT_s115
                                 + 0.794969 w_k1_MSFT_s115
                                 + 0.808172 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.908264 w_k1_ADM_s115
                                 + 0.934608 w_k1_AXP_s115
                                 + 0.869772 w_k1_DIS_s115
                                 + 1.08104 w_k1_FDX_s115
                                 + 0.674691 w_k1_IBM_s115
                                 + 0.925321 w_k1_INTC_s115
                                 + 0.745517 w_k1_KO_s115
                                 + 0.721262 w_k1_MAT_s115
                                 + 0.952465 w_k1_MSFT_s115
                                 + 0.929329 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.0681 w_k1_ADM_s115
                                 + 0.945057 w_k1_AXP_s115
                                 + 1.01471 w_k1_DIS_s115
                                 + 0.951999 w_k1_FDX_s115
                                 + 1.03488 w_k1_IBM_s115
                                 + 0.860573 w_k1_INTC_s115
                                 + 0.875211 w_k1_KO_s115
                                 + 0.831031 w_k1_MAT_s115
                                 + 0.924301 w_k1_MSFT_s115
                                 + 1.01945 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.88561 w_k1_ADM_s115
                                 + 0.629551 w_k1_AXP_s115
                                 + 0.707416 w_k1_DIS_s115
                                 + 0.870019 w_k1_FDX_s115
                                 + 0.604683 w_k1_IBM_s115
                                 + 0.771483 w_k1_INTC_s115
                                 + 0.60383 w_k1_KO_s115
                                 + 0.489693 w_k1_MAT_s115
                                 + 0.803357 w_k1_MSFT_s115
                                 + 0.856523 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.01174 w_k1_ADM_s115
                                 + 1.28453 w_k1_AXP_s115
                                 + 1.18016 w_k1_DIS_s115
                                 + 1.15512 w_k1_FDX_s115
                                 + 1.33767 w_k1_IBM_s115
                                 + 1.09289 w_k1_INTC_s115
                                 + 1.19605 w_k1_KO_s115 + 1.62026 w_k1_MAT_s115
                                 + 1.05479 w_k1_MSFT_s115
                                 + 1.23704 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.07253 w_k1_ADM_s115
                                 + 1.47155 w_k1_AXP_s115
                                 + 1.44498 w_k1_DIS_s115
                                 + 1.32029 w_k1_FDX_s115
                                 + 1.08113 w_k1_IBM_s115
                                 + 1.20511 w_k1_INTC_s115
                                 + 1.03147 w_k1_KO_s115 + 1.2239 w_k1_MAT_s115
                                 + 1.22091 w_k1_MSFT_s115
                                 + 1.06464 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.908616 w_k1_ADM_s115
                                 + 0.781878 w_k1_AXP_s115
                                 + 0.814121 w_k1_DIS_s115
                                 + 0.82242 w_k1_FDX_s115 + 1.0035 w_k1_IBM_s115
                                 + 0.835481 w_k1_INTC_s115
                                 + 0.832414 w_k1_KO_s115
                                 + 1.06023 w_k1_MAT_s115
                                 + 0.879024 w_k1_MSFT_s115
                                 + 0.948566 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.16349 w_k1_ADM_s116
                                 + 1.37127 w_k1_AXP_s116 + 1.402 w_k1_DIS_s116
                                 + 1.21084 w_k1_FDX_s116
                                 + 1.38268 w_k1_IBM_s116
                                 + 1.21157 w_k1_INTC_s116
                                 + 1.28508 w_k1_KO_s116 + 1.51504 w_k1_MAT_s116
                                 + 1.10548 w_k1_MSFT_s116
                                 + 1.16396 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.06707 w_k1_ADM_s116
                                 + 1.23078 w_k1_AXP_s116
                                 + 0.933139 w_k1_DIS_s116
                                 + 1.09094 w_k1_FDX_s116
                                 + 0.783708 w_k1_IBM_s116
                                 + 1.18122 w_k1_INTC_s116
                                 + 0.967231 w_k1_KO_s116
                                 + 1.03026 w_k1_MAT_s116
                                 + 0.979528 w_k1_MSFT_s116
                                 + 0.993015 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.940273 w_k1_ADM_s116
                                 + 0.640124 w_k1_AXP_s116
                                 + 0.81723 w_k1_DIS_s116
                                 + 0.947333 w_k1_FDX_s116
                                 + 0.752186 w_k1_IBM_s116
                                 + 0.97706 w_k1_INTC_s116
                                 + 0.86818 w_k1_KO_s116
                                 + 0.765823 w_k1_MAT_s116
                                 + 1.0064 w_k1_MSFT_s116 + 1.00256 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00155 w_k1_ADM_s116
                                 + 0.812311 w_k1_AXP_s116
                                 + 0.616807 w_k1_DIS_s116
                                 + 0.813061 w_k1_FDX_s116
                                 + 0.883346 w_k1_IBM_s116
                                 + 0.995021 w_k1_INTC_s116
                                 + 0.826731 w_k1_KO_s116
                                 + 0.797919 w_k1_MAT_s116
                                 + 0.735521 w_k1_MSFT_s116
                                 + 0.86144 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.0807 w_k1_ADM_s116
                                 + 1.09613 w_k1_AXP_s116
                                 + 1.01763 w_k1_DIS_s116
                                 + 0.932323 w_k1_FDX_s116
                                 + 1.28608 w_k1_IBM_s116
                                 + 1.16247 w_k1_INTC_s116
                                 + 1.11725 w_k1_KO_s116 + 1.11636 w_k1_MAT_s116
                                 + 0.87988 w_k1_MSFT_s116
                                 + 1.00686 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.899975 w_k1_ADM_s116
                                 + 0.89077 w_k1_AXP_s116
                                 + 1.04632 w_k1_DIS_s116
                                 + 1.04666 w_k1_FDX_s116
                                 + 0.827971 w_k1_IBM_s116
                                 + 1.11083 w_k1_INTC_s116
                                 + 1.05731 w_k1_KO_s116
                                 + 0.936037 w_k1_MAT_s116
                                 + 1.02075 w_k1_MSFT_s116
                                 + 1.03103 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.830731 w_k1_ADM_s116
                                 + 0.682424 w_k1_AXP_s116
                                 + 0.631566 w_k1_DIS_s116
                                 + 0.784343 w_k1_FDX_s116
                                 + 0.686553 w_k1_IBM_s116
                                 + 0.825799 w_k1_INTC_s116
                                 + 0.704356 w_k1_KO_s116
                                 + 0.749202 w_k1_MAT_s116
                                 + 0.685877 w_k1_MSFT_s116
                                 + 0.847798 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.853954 w_k1_ADM_s116
                                 + 0.67621 w_k1_AXP_s116
                                 + 0.89173 w_k1_DIS_s116
                                 + 0.888147 w_k1_FDX_s116
                                 + 0.986902 w_k1_IBM_s116
                                 + 1.03819 w_k1_INTC_s116
                                 + 1.07462 w_k1_KO_s116
                                 + 0.985202 w_k1_MAT_s116
                                 + 0.891502 w_k1_MSFT_s116
                                 + 0.929166 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.26591 w_k1_ADM_s117
                                 + 1.36436 w_k1_AXP_s117
                                 + 1.30719 w_k1_DIS_s117
                                 + 1.24954 w_k1_FDX_s117
                                 + 1.20421 w_k1_IBM_s117
                                 + 1.10659 w_k1_INTC_s117
                                 + 1.20475 w_k1_KO_s117
                                 + 0.814583 w_k1_MAT_s117
                                 + 1.12466 w_k1_MSFT_s117
                                 + 1.21957 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.51177 w_k1_ADM_s117
                                 + 1.73724 w_k1_AXP_s117
                                 + 1.90858 w_k1_DIS_s117
                                 + 1.46801 w_k1_FDX_s117
                                 + 1.19258 w_k1_IBM_s117
                                 + 1.17313 w_k1_INTC_s117
                                 + 1.38046 w_k1_KO_s117 + 1.28793 w_k1_MAT_s117
                                 + 1.16337 w_k1_MSFT_s117 + 1.2222 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.849459 w_k1_ADM_s117
                                 + 0.556904 w_k1_AXP_s117
                                 + 0.631252 w_k1_DIS_s117
                                 + 0.626407 w_k1_FDX_s117
                                 + 0.79991 w_k1_IBM_s117
                                 + 0.692941 w_k1_INTC_s117
                                 + 0.710044 w_k1_KO_s117
                                 + 0.66757 w_k1_MAT_s117
                                 + 0.613079 w_k1_MSFT_s117
                                 + 0.759727 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.997733 w_k1_ADM_s117
                                 + 1.23111 w_k1_AXP_s117
                                 + 1.09325 w_k1_DIS_s117
                                 + 0.977943 w_k1_FDX_s117
                                 + 1.57359 w_k1_IBM_s117
                                 + 1.01363 w_k1_INTC_s117
                                 + 1.19396 w_k1_KO_s117 + 1.71052 w_k1_MAT_s117
                                 + 0.996354 w_k1_MSFT_s117
                                 + 1.10647 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.752862 w_k1_ADM_s117
                                 + 0.851119 w_k1_AXP_s117
                                 + 0.771857 w_k1_DIS_s117
                                 + 0.772123 w_k1_FDX_s117
                                 + 1.35482 w_k1_IBM_s117
                                 + 0.909244 w_k1_INTC_s117
                                 + 1.1058 w_k1_KO_s117 + 1.20737 w_k1_MAT_s117
                                 + 0.736777 w_k1_MSFT_s117
                                 + 0.872641 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.04748 w_k1_ADM_s117
                                 + 0.955237 w_k1_AXP_s117
                                 + 0.676646 w_k1_DIS_s117
                                 + 0.871387 w_k1_FDX_s117
                                 + 0.839576 w_k1_IBM_s117
                                 + 0.903427 w_k1_INTC_s117
                                 + 0.76507 w_k1_KO_s117
                                 + 0.588423 w_k1_MAT_s117
                                 + 0.78028 w_k1_MSFT_s117
                                 + 0.885276 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.03589 w_k1_ADM_s117
                                 + 0.993301 w_k1_AXP_s117
                                 + 1.16442 w_k1_DIS_s117
                                 + 1.11203 w_k1_FDX_s117
                                 + 0.912709 w_k1_IBM_s117
                                 + 0.972417 w_k1_INTC_s117
                                 + 0.997795 w_k1_KO_s117
                                 + 0.927004 w_k1_MAT_s117
                                 + 1.02647 w_k1_MSFT_s117
                                 + 1.01926 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.02512 w_k1_ADM_s117
                                 + 0.739374 w_k1_AXP_s117
                                 + 0.807243 w_k1_DIS_s117
                                 + 0.9319 w_k1_FDX_s117
                                 + 0.630387 w_k1_IBM_s117
                                 + 1.00419 w_k1_INTC_s117
                                 + 0.823556 w_k1_KO_s117
                                 + 0.570984 w_k1_MAT_s117
                                 + 0.901792 w_k1_MSFT_s117
                                 + 0.817794 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0287991372083636 w_k1_ADM_s000
                                 + 0.0588950193976114 w_k1_ADM_s001
                                 + 0.0442979463542266 w_k1_ADM_s002
                                 + 0.0258531938511023 w_k1_ADM_s003
                                 + 0.0294608812569077 w_k1_ADM_s004
                                 + 0.0338183269787965 w_k1_ADM_s005
                                 + 0.0384972168207049 w_k1_ADM_s006
                                 + 0.0203313815128147 w_k1_ADM_s007
                                 + 0.011219529344216 w_k1_ADM_s010
                                 + 0.00592183294554066 w_k1_ADM_s011
                                 + 0.0118081375064774 w_k1_ADM_s012
                                 + 0.00737424504088548 w_k1_ADM_s013
                                 + 0.0104758132484987 w_k1_ADM_s014
                                 + 0.00824354039005253 w_k1_ADM_s015
                                 + 0.00510215875228499 w_k1_ADM_s016
                                 + 0.00380831946585265 w_k1_ADM_s017
                                 + 0.0623644990489887 w_k1_ADM_s100
                                 + 0.0260700022571245 w_k1_ADM_s101
                                 + 0.0588987807681916 w_k1_ADM_s102
                                 + 0.0419307296900018 w_k1_ADM_s103
                                 + 0.0716526664207163 w_k1_ADM_s104
                                 + 0.056756802215477 w_k1_ADM_s105
                                 + 0.0815599699343401 w_k1_ADM_s106
                                 + 0.0278746545047582 w_k1_ADM_s107
                                 + 0.0197873958871598 w_k1_ADM_s110
                                 + 0.020927542716746 w_k1_ADM_s111
                                 + 0.0341691323933818 w_k1_ADM_s112
                                 + 0.0293374371855725 w_k1_ADM_s113
                                 + 0.0498653738035266 w_k1_ADM_s114
                                 + 0.021042454978205 w_k1_ADM_s115
                                 + 0.033644316520057 w_k1_ADM_s116
                                 + 0.0642599959494857 w_k1_ADM_s117
                                 + 0.0265100336184219 w_k1_AXP_s000
                                 + 0.0553454602447368 w_k1_AXP_s001
                                 + 0.0425617001721317 w_k1_AXP_s002
                                 + 0.0278923803793816 w_k1_AXP_s003
                                 + 0.0285025716007899 w_k1_AXP_s004
                                 + 0.0399646348471594 w_k1_AXP_s005
                                 + 0.0357678019486779 w_k1_AXP_s006
                                 + 0.019836783710248 w_k1_AXP_s007
                                 + 0.00824148538472141 w_k1_AXP_s010
                                 + 0.00617316470078426 w_k1_AXP_s011
                                 + 0.00964628455722355 w_k1_AXP_s012
                                 + 0.00705932785525359 w_k1_AXP_s013
                                 + 0.00938429341302194 w_k1_AXP_s014
                                 + 0.00725729205506943 w_k1_AXP_s015
                                 + 0.00451229346997398 w_k1_AXP_s016
                                 + 0.00446878586346167 w_k1_AXP_s017
                                 + 0.0550457242986082 w_k1_AXP_s100
                                 + 0.0247961863523994 w_k1_AXP_s101
                                 + 0.0589579017445708 w_k1_AXP_s102
                                 + 0.0456780230404485 w_k1_AXP_s103
                                 + 0.070429522159874 w_k1_AXP_s104
                                 + 0.0498338867581681 w_k1_AXP_s105
                                 + 0.0752863677246653 w_k1_AXP_s106
                                 + 0.0380888228593367 w_k1_AXP_s107
                                 + 0.0219444589613638 w_k1_AXP_s110
                                 + 0.0202572376391946 w_k1_AXP_s111
                                 + 0.0324354015721893 w_k1_AXP_s112
                                 + 0.0253883078564464 w_k1_AXP_s113
                                 + 0.0513988940066932 w_k1_AXP_s114
                                 + 0.0227085615483131 w_k1_AXP_s115
                                 + 0.0332474355729818 w_k1_AXP_s116
                                 + 0.063226524475749 w_k1_AXP_s117
                                 + 0.0260902698197997 w_k1_DIS_s000
                                 + 0.0535125793947801 w_k1_DIS_s001
                                 + 0.0413202607767669 w_k1_DIS_s002
                                 + 0.0276453468441281 w_k1_DIS_s003
                                 + 0.0287802129783732 w_k1_DIS_s004
                                 + 0.0384159175670742 w_k1_DIS_s005
                                 + 0.037156149657308 w_k1_DIS_s006
                                 + 0.0227902324583525 w_k1_DIS_s007
                                 + 0.00998629097995993 w_k1_DIS_s010
                                 + 0.00570876938893323 w_k1_DIS_s011
                                 + 0.0104879892334115 w_k1_DIS_s012
                                 + 0.00858677341584613 w_k1_DIS_s013
                                 + 0.00944347603062497 w_k1_DIS_s014
                                 + 0.00723184424239636 w_k1_DIS_s015
                                 + 0.00492234577789121 w_k1_DIS_s016
                                 + 0.00373143936429475 w_k1_DIS_s017
                                 + 0.054115500369285 w_k1_DIS_s100
                                 + 0.0269495153610506 w_k1_DIS_s101
                                 + 0.0603055436286453 w_k1_DIS_s102
                                 + 0.047514894152562 w_k1_DIS_s103
                                 + 0.0697404246652544 w_k1_DIS_s104
                                 + 0.0559418293721769 w_k1_DIS_s105
                                 + 0.0770164149084604 w_k1_DIS_s106
                                 + 0.0372146604420144 w_k1_DIS_s107
                                 + 0.0201753566186141 w_k1_DIS_s110
                                 + 0.0204740174000594 w_k1_DIS_s111
                                 + 0.0331776701351532 w_k1_DIS_s112
                                 + 0.0261068024082477 w_k1_DIS_s113
                                 + 0.0522599220003942 w_k1_DIS_s114
                                 + 0.0223146935698866 w_k1_DIS_s115
                                 + 0.0333679956693404 w_k1_DIS_s116
                                 + 0.0614486989735055 w_k1_DIS_s117
                                 + 0.026945667168638 w_k1_FDX_s000
                                 + 0.053286784114324 w_k1_FDX_s001
                                 + 0.0403800827167571 w_k1_FDX_s002
                                 + 0.0271437998852512 w_k1_FDX_s003
                                 + 0.0256118156276849 w_k1_FDX_s004
                                 + 0.0375132627985265 w_k1_FDX_s005
                                 + 0.0384752642579857 w_k1_FDX_s006
                                 + 0.0222483752238799 w_k1_FDX_s007
                                 + 0.0095125361925784 w_k1_FDX_s010
                                 + 0.00589002796992543 w_k1_FDX_s011
                                 + 0.00997858415973447 w_k1_FDX_s012
                                 + 0.00785310686924285 w_k1_FDX_s013
                                 + 0.0089511273117817 w_k1_FDX_s014
                                 + 0.00725794620012189 w_k1_FDX_s015
                                 + 0.00493748112514818 w_k1_FDX_s016
                                 + 0.00366411489928169 w_k1_FDX_s017
                                 + 0.0543995704890263 w_k1_FDX_s100
                                 + 0.0261248376593297 w_k1_FDX_s101
                                 + 0.0546966587612121 w_k1_FDX_s102
                                 + 0.0464337855971769 w_k1_FDX_s103
                                 + 0.073197598444849 w_k1_FDX_s104
                                 + 0.0511180933350174 w_k1_FDX_s105
                                 + 0.0740310217248498 w_k1_FDX_s106
                                 + 0.0357943019241414 w_k1_FDX_s107
                                 + 0.019937663337179 w_k1_FDX_s110
                                 + 0.0213512075880983 w_k1_FDX_s111
                                 + 0.0324264104381982 w_k1_FDX_s112
                                 + 0.0247867865066481 w_k1_FDX_s113
                                 + 0.0497091930128467 w_k1_FDX_s114
                                 + 0.0225196207363239 w_k1_FDX_s115
                                 + 0.0332544708513949 w_k1_FDX_s116
                                 + 0.0610121875638459 w_k1_FDX_s117
                                 + 0.0285104965225206 w_k1_IBM_s000
                                 + 0.0552057700113003 w_k1_IBM_s001
                                 + 0.0434819420244788 w_k1_IBM_s002
                                 + 0.027615796791717 w_k1_IBM_s003
                                 + 0.0259616720544894 w_k1_IBM_s004
                                 + 0.0377529286332632 w_k1_IBM_s005
                                 + 0.0378496584662212 w_k1_IBM_s006
                                 + 0.0218337581490536 w_k1_IBM_s007
                                 + 0.00934037302064691 w_k1_IBM_s010
                                 + 0.00590576895408609 w_k1_IBM_s011
                                 + 0.0103859015014383 w_k1_IBM_s012
                                 + 0.00833496419602521 w_k1_IBM_s013
                                 + 0.00973056228981675 w_k1_IBM_s014
                                 + 0.0073094978911011 w_k1_IBM_s015
                                 + 0.00476772217128334 w_k1_IBM_s016
                                 + 0.00364114134322671 w_k1_IBM_s017
                                 + 0.0550871370202878 w_k1_IBM_s100
                                 + 0.0268405648217145 w_k1_IBM_s101
                                 + 0.0556176589110603 w_k1_IBM_s102
                                 + 0.0448947467024171 w_k1_IBM_s103
                                 + 0.0762646896871306 w_k1_IBM_s104
                                 + 0.0553388940086383 w_k1_IBM_s105
                                 + 0.0747707988528549 w_k1_IBM_s106
                                 + 0.0384780935126129 w_k1_IBM_s107
                                 + 0.0195656286563372 w_k1_IBM_s110
                                 + 0.0227205651712877 w_k1_IBM_s111
                                 + 0.030614711554678 w_k1_IBM_s112
                                 + 0.0255759752303993 w_k1_IBM_s113
                                 + 0.0503192835125219 w_k1_IBM_s114
                                 + 0.0210650350557711 w_k1_IBM_s115
                                 + 0.0345386302555149 w_k1_IBM_s116
                                 + 0.0632983417187423 w_k1_IBM_s117
                                 + 0.0240535633185735 w_k1_INTC_s000
                                 + 0.0502267922255653 w_k1_INTC_s001
                                 + 0.0396360678187238 w_k1_INTC_s002
                                 + 0.0281454542208682 w_k1_INTC_s003
                                 + 0.0271725916144746 w_k1_INTC_s004
                                 + 0.0343197568933058 w_k1_INTC_s005
                                 + 0.0393266067225653 w_k1_INTC_s006
                                 + 0.0224102560706575 w_k1_INTC_s007
                                 + 0.0094805137243344 w_k1_INTC_s010
                                 + 0.00539906381860129 w_k1_INTC_s011
                                 + 0.00940101447842722 w_k1_INTC_s012
                                 + 0.00768154425713472 w_k1_INTC_s013
                                 + 0.00897870398249812 w_k1_INTC_s014
                                 + 0.00664032034813464 w_k1_INTC_s015
                                 + 0.00497720596769787 w_k1_INTC_s016
                                 + 0.00337638028417435 w_k1_INTC_s017
                                 + 0.04834974415009 w_k1_INTC_s100
                                 + 0.0241816142178365 w_k1_INTC_s101
                                 + 0.0625154650898743 w_k1_INTC_s102
                                 + 0.0503488468303044 w_k1_INTC_s103
                                 + 0.0812975443446779 w_k1_INTC_s104
                                 + 0.0663215132200174 w_k1_INTC_s105
                                 + 0.0836617425534289 w_k1_INTC_s106
                                 + 0.0380731042053004 w_k1_INTC_s107
                                 + 0.0218321345485052 w_k1_INTC_s110
                                 + 0.0209707071017622 w_k1_INTC_s111
                                 + 0.0319183055325029 w_k1_INTC_s112
                                 + 0.024696572140003 w_k1_INTC_s113
                                 + 0.0557973094463193 w_k1_INTC_s114
                                 + 0.0206665639638939 w_k1_INTC_s115
                                 + 0.0362447177600146 w_k1_INTC_s116
                                 + 0.0594972575619419 w_k1_INTC_s117
                                 + 0.0281708639057851 w_k1_KO_s000
                                 + 0.0527382013137633 w_k1_KO_s001
                                 + 0.0427594447416947 w_k1_KO_s002
                                 + 0.0258647728889845 w_k1_KO_s003
                                 + 0.0268181235241945 w_k1_KO_s004
                                 + 0.0379890586171749 w_k1_KO_s005
                                 + 0.0398479071196957 w_k1_KO_s006
                                 + 0.0219365927571637 w_k1_KO_s007
                                 + 0.00952567645568783 w_k1_KO_s010
                                 + 0.00605650090583877 w_k1_KO_s011
                                 + 0.00940028479032625 w_k1_KO_s012
                                 + 0.00901298338900357 w_k1_KO_s013
                                 + 0.0093099273059627 w_k1_KO_s014
                                 + 0.00688624061535441 w_k1_KO_s015
                                 + 0.00497906459208274 w_k1_KO_s016
                                 + 0.00347232710604918 w_k1_KO_s017
                                 + 0.0547757375472738 w_k1_KO_s100
                                 + 0.0272513300259647 w_k1_KO_s101
                                 + 0.0576923236077107 w_k1_KO_s102
                                 + 0.045160633488385 w_k1_KO_s103
                                 + 0.0808183312476614 w_k1_KO_s104
                                 + 0.0635351318237193 w_k1_KO_s105
                                 + 0.0728101590353248 w_k1_KO_s106
                                 + 0.0388785336334847 w_k1_KO_s107
                                 + 0.0195104731974088 w_k1_KO_s110
                                 + 0.021248190736791 w_k1_KO_s111
                                 + 0.032253998351332 w_k1_KO_s112
                                 + 0.0256258177704066 w_k1_KO_s113
                                 + 0.0508116056605176 w_k1_KO_s114
                                 + 0.0192918778826669 w_k1_KO_s115
                                 + 0.0346862520217598 w_k1_KO_s116
                                 + 0.0608869226647322 w_k1_KO_s117
                                 + 0.0298637790151007 w_k1_MAT_s000
                                 + 0.0509821857949363 w_k1_MAT_s001
                                 + 0.0478061195373314 w_k1_MAT_s002
                                 + 0.0228149627088334 w_k1_MAT_s003
                                 + 0.0255208642115626 w_k1_MAT_s004
                                 + 0.0407812767409145 w_k1_MAT_s005
                                 + 0.0329827633896643 w_k1_MAT_s006
                                 + 0.0213566775010596 w_k1_MAT_s007
                                 + 0.0102635682838057 w_k1_MAT_s010
                                 + 0.00597815838214826 w_k1_MAT_s011
                                 + 0.011891121264645 w_k1_MAT_s012
                                 + 0.00778054277150941 w_k1_MAT_s013
                                 + 0.011447570683673 w_k1_MAT_s014
                                 + 0.00731234217857 w_k1_MAT_s015
                                 + 0.00440442106862566 w_k1_MAT_s016
                                 + 0.00439979899623891 w_k1_MAT_s017
                                 + 0.0594866094948554 w_k1_MAT_s100
                                 + 0.026529154308263 w_k1_MAT_s101
                                 + 0.0495591371780684 w_k1_MAT_s102
                                 + 0.0490333097594126 w_k1_MAT_s103
                                 + 0.0780756830594113 w_k1_MAT_s104
                                 + 0.0589896936497967 w_k1_MAT_s105
                                 + 0.0570109596142787 w_k1_MAT_s106
                                 + 0.0320951507591315 w_k1_MAT_s107
                                 + 0.0197871910072265 w_k1_MAT_s110
                                 + 0.0218135910844324 w_k1_MAT_s111
                                 + 0.0341916632287674 w_k1_MAT_s112
                                 + 0.0255741703858932 w_k1_MAT_s113
                                 + 0.0524947399275092 w_k1_MAT_s114
                                 + 0.0225190927497572 w_k1_MAT_s115
                                 + 0.0358683277752138 w_k1_MAT_s116
                                 + 0.0569147184539876 w_k1_MAT_s117
                                 + 0.0251669082301923 w_k1_MSFT_s000
                                 + 0.0522877708164222 w_k1_MSFT_s001
                                 + 0.0411814111231877 w_k1_MSFT_s002
                                 + 0.0283825200711734 w_k1_MSFT_s003
                                 + 0.0279846864622077 w_k1_MSFT_s004
                                 + 0.0331012292121042 w_k1_MSFT_s005
                                 + 0.0356686067803782 w_k1_MSFT_s006
                                 + 0.0236687635162864 w_k1_MSFT_s007
                                 + 0.0106016450407724 w_k1_MSFT_s010
                                 + 0.0057251868426323 w_k1_MSFT_s011
                                 + 0.00964739544070297 w_k1_MSFT_s012
                                 + 0.00880074614765506 w_k1_MSFT_s013
                                 + 0.00808311603616551 w_k1_MSFT_s014
                                 + 0.00776387108127142 w_k1_MSFT_s015
                                 + 0.00549944228141131 w_k1_MSFT_s016
                                 + 0.00350361296328694 w_k1_MSFT_s017
                                 + 0.0541336935232354 w_k1_MSFT_s100
                                 + 0.0264275341394378 w_k1_MSFT_s101
                                 + 0.055816589432429 w_k1_MSFT_s102
                                 + 0.0466617029634376 w_k1_MSFT_s103
                                 + 0.0738338245918106 w_k1_MSFT_s104
                                 + 0.0610478432468452 w_k1_MSFT_s105
                                 + 0.0726188599080331 w_k1_MSFT_s106
                                 + 0.0365172950397331 w_k1_MSFT_s107
                                 + 0.0190535595121932 w_k1_MSFT_s110
                                 + 0.0214321680316451 w_k1_MSFT_s111
                                 + 0.0331152239179091 w_k1_MSFT_s112
                                 + 0.0258942354977133 w_k1_MSFT_s113
                                 + 0.053680631573259 w_k1_MSFT_s114
                                 + 0.0212387194084407 w_k1_MSFT_s115
                                 + 0.0313391556601406 w_k1_MSFT_s116
                                 + 0.05660202947405 w_k1_MSFT_s117
                                 + 0.0303686956544168 w_k1_PG_s000
                                 + 0.0547376447512357 w_k1_PG_s001
                                 + 0.0431354895893094 w_k1_PG_s002
                                 + 0.0253459092841158 w_k1_PG_s003
                                 + 0.0262531319173519 w_k1_PG_s004
                                 + 0.0367846369454086 w_k1_PG_s005
                                 + 0.0372141516521106 w_k1_PG_s006
                                 + 0.0204533502769298 w_k1_PG_s007
                                 + 0.00933421418567996 w_k1_PG_s010
                                 + 0.00597348254558712 w_k1_PG_s011
                                 + 0.0109012073690515 w_k1_PG_s012
                                 + 0.0081477469737389 w_k1_PG_s013
                                 + 0.00885970025758363 w_k1_PG_s014
                                 + 0.00747860131827814 w_k1_PG_s015
                                 + 0.00502859655252657 w_k1_PG_s016
                                 + 0.00366391827927093 w_k1_PG_s017
                                 + 0.0539538132778081 w_k1_PG_s100
                                 + 0.0249822062015182 w_k1_PG_s101
                                 + 0.055609087085419 w_k1_PG_s102
                                 + 0.0455485056323588 w_k1_PG_s103
                                 + 0.0739344205701632 w_k1_PG_s104
                                 + 0.0512291559133888 w_k1_PG_s105
                                 + 0.0747110847408088 w_k1_PG_s106
                                 + 0.0385397521033752 w_k1_PG_s107
                                 + 0.019641391504916 w_k1_PG_s110
                                 + 0.0217915067365551 w_k1_PG_s111
                                 + 0.0336593023250661 w_k1_PG_s112
                                 + 0.0235611027466017 w_k1_PG_s113
                                 + 0.0479869453255919 w_k1_PG_s114
                                 + 0.0218701697290548 w_k1_PG_s115
                                 + 0.033621337985057 w_k1_PG_s116
                                 + 0.0601924316090901 w_k1_PG_s117 >= 1.219
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
