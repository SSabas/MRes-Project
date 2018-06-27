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
 w_k1_ADM_s0_return_constraint:  1.20684 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.986658 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.902755 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.08121 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.17116 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.940106 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.20946 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.01236 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.871187 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.20004 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.771116 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.14971 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.44866 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.11957 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.14835 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.10309 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.79849 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.943448 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.742088 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.847576 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.05052 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.5313 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.946912 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.0572 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.880285 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.01003 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.043 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.10021 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.52048 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.40229 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.26519 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.08056 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.745868 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.950939 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.1284 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.875021 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.907335 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.817911 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.31363 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.920747 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.868778 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.21292 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.44057 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.892508 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.22696 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.0399 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.863264 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.24397 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.725375 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.82679 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.44112 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.0238 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.12133 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.97316 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.779579 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.612809 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.739244 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.708626 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.09405 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.80219 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.880191 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.01654 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.670003 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.0843 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.953785 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.13896 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.42753 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.44269 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.50163 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.25528 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.695422 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.985612 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.963907 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.911217 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.851517 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.08474 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.33804 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.944176 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.946703 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.29676 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.47507 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.888278 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.03012 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.937885 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.868846 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.28811 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.569071 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.8137 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.57869 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.02474 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.01346 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.05057 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.925541 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.89632 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.62443 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.632804 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.36206 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.77544 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.792405 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.27415 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.672314 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.12424 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.904568 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.38038 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.66439 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.32742 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.57895 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.09899 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.599512 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.19945 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.990192 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.761301 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.898771 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.824146 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.26261 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.930722 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.939113 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.10561 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.27897 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.94065 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.0775 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.880654 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.922115 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.22424 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.716903 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.49858 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.49265 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.16799 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.974645 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.967194 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.941927 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.914736 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.746527 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.677651 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.24326 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.50084 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.809104 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.16703 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.824705 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.00239 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.02992 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.3188 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.3589 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.39687 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.26635 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.17081 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.68781 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.01105 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.15805 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.923759 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.911015 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.785828 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15195 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.00564 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.786261 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.33821 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.34266 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.922347 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.06388 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.51991 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.789329 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.24918 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.879152 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.66698 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.931653 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.653838 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.01365 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.45629 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.797666 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.676237 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.773002 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.06471 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.853443 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.44554 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.16102 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.845873 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.639719 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.54515 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.766624 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.907055 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.45694 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.964507 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.47498 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.963771 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.719192 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.2099 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.513808 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.772631 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.02236 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.54873 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.21054 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.937432 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.887245 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.05105 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.22208 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.977518 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.10552 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.05357 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.86739 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.2697 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.784977 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.3866 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.38468 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.09573 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.05199 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.06568 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.00283 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.88807 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.771717 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.762483 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.14835 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.35021 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.882222 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.08259 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.884507 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.0118 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.924332 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.2174 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.30881 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.25152 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.07393 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.02984 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.730185 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.03328 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.912089 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.850216 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.0365 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.862523 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.17741 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.994251 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.877922 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.16368 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.14522 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.929084 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.10275 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.12206 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.865309 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.3254 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.741951 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.58656 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.20194 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.803534 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.03711 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.32417 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.987487 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.826819 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.737711 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.882216 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.11494 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.61775 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.07393 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.1308 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.889481 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.41764 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.00228 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.21852 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.64208 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.1668 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.3721 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.890028 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.647513 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.25773 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.728858 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.774389 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.938154 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.03253 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.34635 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.978772 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.711131 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.1867 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.26886 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.924777 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.07928 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.27537 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.846108 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.25458 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.804736 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 2.06133 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.9368 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.617578 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.0447 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.37711 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.787616 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.603103 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.727008 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.27243 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.01515 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 2.00336 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.34885 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.935759 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.684853 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.97495 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.831588 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.21873 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.73646 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.0219 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.65847 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.08825 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.580376 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.0815 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.602869 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.821065 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.832865 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.39827 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16993 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.969169 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.96898 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.08405 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.18681 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.891799 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.04655 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.847562 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.95759 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.14338 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.703016 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.37297 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.38186 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.00874 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.969087 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.06651 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.0082 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.969333 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.773164 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.759012 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.14219 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.56377 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.859204 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.2235 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.926643 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.05484 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.987855 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.20778 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.35014 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.30491 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.27153 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.10929 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.766423 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.19522 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15807 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.879104 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.944587 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.85447 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16951 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.00865 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.938218 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.07848 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.14374 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.877369 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.10153 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.996824 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.908051 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.10971 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.822238 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.21842 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.15882 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.981865 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.99004 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.976112 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.992739 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.913325 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.852402 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.880487 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.13384 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.28464 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.886691 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.958959 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.810442 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.983883 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.957449 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.15665 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.25043 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.09716 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.23223 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.11767 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.807958 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.09479 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.00789 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.952965 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.965635 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.986267 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.340000265964639 z_k1_s0000
                                 - 0.382999700999797 z_k1_s0001
                                 - 0.258000792221751 z_k1_s0002
                                 - 0.423002028295684 z_k1_s0003
                                 - 0.369001116851997 z_k1_s0004
                                 - 0.281000457612566 z_k1_s0005
                                 - 0.238000743776687 z_k1_s0006
                                 - 0.496000420795558 z_k1_s0007
                                 - 0.0639999794964198 z_k1_s0010
                                 - 0.289999650999483 z_k1_s0011
                                 - 0.193000383771875 z_k1_s0012
                                 - 0.221999201572187 z_k1_s0013
                                 - 0.19799933218599 z_k1_s0014
                                 - 0.262000622881994 z_k1_s0015
                                 - 0.194999963137521 z_k1_s0016
                                 - 0.215000673792425 z_k1_s0017
                                 - 0.344999443449099 z_k1_s0020
                                 - 0.39400036640513 z_k1_s0021
                                 - 0.490000211538793 z_k1_s0022
                                 - 0.302999511203121 z_k1_s0023
                                 - 0.308999501523975 z_k1_s0024
                                 - 0.365001415190061 z_k1_s0025
                                 - 0.464001255484151 z_k1_s0026
                                 - 0.335999457967818 z_k1_s0027
                                 - 0.148000208184483 z_k1_s0030
                                 - 0.455000867374888 z_k1_s0031
                                 - 0.173000020232247 z_k1_s0032
                                 - 0.107000128305644 z_k1_s0033
                                 - 0.146000135580547 z_k1_s0034
                                 - 0.261000084801322 z_k1_s0035
                                 - 0.139000194467182 z_k1_s0036
                                 - 0.136000242061483 z_k1_s0037
                                 - 0.320000946022935 z_k1_s0040
                                 - 0.520999411445387 z_k1_s0041
                                 - 0.405999071468395 z_k1_s0042
                                 - 0.262999713113498 z_k1_s0043
                                 - 0.436999695313917 z_k1_s0044
                                 - 0.394000632591187 z_k1_s0045
                                 - 0.112999801864798 z_k1_s0046
                                 - 0.206999813659259 z_k1_s0047
                                 - 0.0569999532277916 z_k1_s0050
                                 - 0.158000022887497 z_k1_s0051
                                 - 0.169999576714588 z_k1_s0052
                                 - 0.0929999290084997 z_k1_s0053
                                 - 0.0989999081219584 z_k1_s0054
                                 - 0.0859999702261241 z_k1_s0055
                                 - 0.258000011778329 z_k1_s0056
                                 - 0.0899999900017486 z_k1_s0057
                                 - 0.176999501779576 z_k1_s0060
                                 - 0.242999556392334 z_k1_s0061
                                 - 0.15999953730357 z_k1_s0062
                                 - 0.0850000405800632 z_k1_s0063
                                 - 0.0560000562562812 z_k1_s0064
                                 - 0.108999971175599 z_k1_s0065
                                 - 0.0750000422235861 z_k1_s0066
                                 - 0.186000253090515 z_k1_s0067
                                 - 0.0610000537330173 z_k1_s0070
                                 - 0.188999652928355 z_k1_s0071
                                 - 0.157999819442789 z_k1_s0072
                                 - 0.16599983214876 z_k1_s0073
                                 - 0.124000196942557 z_k1_s0074
                                 - 0.0620000121712491 z_k1_s0075
                                 - 0.0519999962887849 z_k1_s0076
                                 - 0.051000037850553 z_k1_s0077
                                 - 0.35000115231807 z_k1_s0100
                                 - 0.299000336368002 z_k1_s0101
                                 - 0.324001371677705 z_k1_s0102
                                 - 0.241000679093147 z_k1_s0103
                                 - 0.236999979762884 z_k1_s0104
                                 - 0.270000507730575 z_k1_s0105
                                 - 0.0910001852425458 z_k1_s0106
                                 - 0.0940000426393547 z_k1_s0107
                                 - 0.0790000649582352 z_k1_s0110
                                 - 0.154000196500923 z_k1_s0111
                                 - 0.364999961673836 z_k1_s0112
                                 - 0.274000958969695 z_k1_s0113
                                 - 0.315999569832352 z_k1_s0114
                                 - 0.190000287241437 z_k1_s0115
                                 - 0.186000008825595 z_k1_s0116
                                 - 0.161000079978131 z_k1_s0117
                                 - 0.0430000253217867 z_k1_s0120
                                 - 0.0359999906786971 z_k1_s0121
                                 - 0.0329999978888074 z_k1_s0122
                                 - 0.0819999315903532 z_k1_s0123
                                 - 0.0419998990584562 z_k1_s0124
                                 - 0.0500001371648966 z_k1_s0125
                                 - 0.0190000058026485 z_k1_s0126
                                 - 0.0810001913271244 z_k1_s0127
                                 - 0.168999963415914 z_k1_s0130
                                 - 0.118999712919959 z_k1_s0131
                                 - 0.0410000371666044 z_k1_s0132
                                 - 0.0440000381763551 z_k1_s0133
                                 - 0.116000132510406 z_k1_s0134
                                 - 0.0999997999915814 z_k1_s0135
                                 - 0.0500000401958566 z_k1_s0136
                                 - 0.0620000442348595 z_k1_s0137
                                 - 0.171999462727968 z_k1_s0140
                                 - 0.070999717720499 z_k1_s0141
                                 - 0.115999869350528 z_k1_s0142
                                 - 0.0839997221648812 z_k1_s0143
                                 - 0.177999379625703 z_k1_s0144
                                 - 0.070999717720499 z_k1_s0145
                                 - 0.114999145203748 z_k1_s0146
                                 - 0.299999165873967 z_k1_s0147
                                 - 0.0589999336613283 z_k1_s0150
                                 - 0.072999957143653 z_k1_s0151
                                 - 0.0989998183537981 z_k1_s0152
                                 - 0.17900006653844 z_k1_s0153
                                 - 0.0749999832982566 z_k1_s0154
                                 - 0.0989998183537981 z_k1_s0155
                                 - 0.0709999309890494 z_k1_s0156
                                 - 0.142999914955376 z_k1_s0157
                                 - 0.174000187428523 z_k1_s0160
                                 - 0.174000187428523 z_k1_s0161
                                 - 0.425000855662135 z_k1_s0162
                                 - 0.267001202689777 z_k1_s0163
                                 - 0.203999795357031 z_k1_s0164
                                 - 0.167000266091854 z_k1_s0165
                                 - 0.245000131287117 z_k1_s0166
                                 - 0.267001202689777 z_k1_s0167
                                 - 0.0610000135525247 z_k1_s0170
                                 - 0.0710000264463832 z_k1_s0171
                                 - 0.100999879127924 z_k1_s0172
                                 - 0.152000158786643 z_k1_s0173
                                 - 0.0859999992871622 z_k1_s0174
                                 - 0.163999951059231 z_k1_s0175
                                 - 0.167000131627422 z_k1_s0176
                                 - 0.12799964424129 z_k1_s0177
                                 - 0.74799924899477 z_k1_s1000
                                 - 0.819997634328809 z_k1_s1001
                                 - 0.212999711812826 z_k1_s1002
                                 - 0.624998387597808 z_k1_s1003
                                 - 0.189999773929221 z_k1_s1004
                                 - 0.274999499042631 z_k1_s1005
                                 - 0.319999271749889 z_k1_s1006
                                 - 0.283999384084217 z_k1_s1007
                                 - 0.214999118801443 z_k1_s1010
                                 - 0.134999833344748 z_k1_s1011
                                 - 0.232000097682485 z_k1_s1012
                                 - 0.211999360469126 z_k1_s1013
                                 - 0.178000220704536 z_k1_s1014
                                 - 0.047999937613691 z_k1_s1015
                                 - 0.214000138689835 z_k1_s1016
                                 - 0.175000462372219 z_k1_s1017
                                 - 0.143999553577485 z_k1_s1020
                                 - 0.189999321840038 z_k1_s1021
                                 - 0.703998076777943 z_k1_s1022
                                 - 0.445999046776042 z_k1_s1023
                                 - 0.219999398993107 z_k1_s1024
                                 - 0.253999253074039 z_k1_s1025
                                 - 0.654998475819494 z_k1_s1026
                                 - 0.303999965060523 z_k1_s1027
                                 - 0.513000900426537 z_k1_s1030
                                 - 0.190000007595156 z_k1_s1031
                                 - 0.209999804078946 z_k1_s1032
                                 - 0.248999911882115 z_k1_s1033
                                 - 0.458999715961061 z_k1_s1034
                                 - 0.333999370438081 z_k1_s1035
                                 - 0.190000007595156 z_k1_s1036
                                 - 0.443000706933668 z_k1_s1037
                                 - 0.633998853798939 z_k1_s1040
                                 - 0.312999149273859 z_k1_s1041
                                 - 0.309998941054651 z_k1_s1042
                                 - 0.824996307970727 z_k1_s1043
                                 - 0.491000216148445 z_k1_s1044
                                 - 1.23599548659299 z_k1_s1045
                                 - 0.398998797244039 z_k1_s1046
                                 - 0.306999184334047 z_k1_s1047
                                 - 0.145999686964573 z_k1_s1050
                                 - 0.0719998805742681 z_k1_s1051
                                 - 0.12200036457239 z_k1_s1052
                                 - 0.303999566624603 z_k1_s1053
                                 - 0.110999390685837 z_k1_s1054
                                 - 0.0869998556939073 z_k1_s1055
                                 - 0.149999751196392 z_k1_s1056
                                 - 0.0709999176662499 z_k1_s1057
                                 - 0.232000951596757 z_k1_s1060
                                 - 0.200000685031491 z_k1_s1061
                                 - 0.128000281459926 z_k1_s1062
                                 - 0.299001387092607 z_k1_s1063
                                 - 0.462000660281401 z_k1_s1064
                                 - 0.116000083397808 z_k1_s1065
                                 - 0.209000392127437 z_k1_s1066
                                 - 0.316000179828442 z_k1_s1067
                                 - 0.113999454325284 z_k1_s1070
                                 - 0.130999963477843 z_k1_s1071
                                 - 0.146999650681876 z_k1_s1072
                                 - 0.206999599694699 z_k1_s1073
                                 - 0.195999534242502 z_k1_s1074
                                 - 0.116999676175465 z_k1_s1075
                                 - 0.135999585229679 z_k1_s1076
                                 - 0.0739998997158916 z_k1_s1077
                                 - 1.45499598347882 z_k1_s1100
                                 - 2.80000304652403 z_k1_s1101
                                 - 0.881000298885101 z_k1_s1102
                                 - 1.03200041715547 z_k1_s1103
                                 - 0.93400052530729 z_k1_s1104
                                 - 0.614000637376317 z_k1_s1105
                                 - 2.1860002613461 z_k1_s1106
                                 - 0.837000435108433 z_k1_s1107
                                 - 1.03400105741443 z_k1_s1110
                                 - 3.33400262736329 z_k1_s1111
                                 - 0.800001007378238 z_k1_s1112
                                 - 0.8620005873644 z_k1_s1113
                                 - 2.10900664775423 z_k1_s1114
                                 - 1.77000188989033 z_k1_s1115
                                 - 0.667000938634561 z_k1_s1116
                                 - 1.21300655692344 z_k1_s1117
                                 - 0.619002248387284 z_k1_s1120
                                 - 1.82800410423251 z_k1_s1121
                                 - 1.33500735358803 z_k1_s1122
                                 - 0.814002640901329 z_k1_s1123
                                 - 1.71800205576745 z_k1_s1124
                                 - 1.22600672433169 z_k1_s1125
                                 - 1.37700918616896 z_k1_s1126
                                 - 0.712002317075141 z_k1_s1127
                                 - 0.695000048851676 z_k1_s1130
                                 - 0.200000300802939 z_k1_s1131
                                 - 0.466001710346899 z_k1_s1132
                                 - 0.311000201410793 z_k1_s1133
                                 - 0.371000328901713 z_k1_s1134
                                 - 0.506999103046222 z_k1_s1135
                                 - 0.874999686323661 z_k1_s1136
                                 - 0.300000264954215 z_k1_s1137
                                 - 0.455000119856595 z_k1_s1140
                                 - 0.645003055676014 z_k1_s1141
                                 - 0.353000125678001 z_k1_s1142
                                 - 1.23000189089055 z_k1_s1143
                                 - 0.689999809766838 z_k1_s1144
                                 - 0.403000223393081 z_k1_s1145
                                 - 0.903002226344728 z_k1_s1146
                                 - 0.450000366535299 z_k1_s1147
                                 - 1.11900009102502 z_k1_s1150
                                 - 1.55999971628893 z_k1_s1151
                                 - 1.00899989760047 z_k1_s1152
                                 - 0.496001053778273 z_k1_s1153
                                 - 1.04999942486694 z_k1_s1154
                                 - 0.452000976408451 z_k1_s1155
                                 - 0.342000782983897 z_k1_s1156
                                 - 0.631001170090035 z_k1_s1157
                                 - 0.437999839177341 z_k1_s1160
                                 - 0.342000292268806 z_k1_s1161
                                 - 0.982002016494615 z_k1_s1162
                                 - 0.404000109795204 z_k1_s1163
                                 - 0.551001487628002 z_k1_s1164
                                 - 0.331000272165711 z_k1_s1165
                                 - 0.804001543299002 z_k1_s1166
                                 - 0.215000023196688 z_k1_s1167
                                 - 0.496999952970941 z_k1_s1170
                                 - 0.248000022818504 z_k1_s1171
                                 - 0.674001954279575 z_k1_s1172
                                 - 0.705997966565299 z_k1_s1173
                                 - 0.920001651230144 z_k1_s1174
                                 - 0.700001589762111 z_k1_s1175
                                 - 0.329000184868117 z_k1_s1176
                                 - 1.03800196303576 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.860452 w_k1_ADM_s000
                                 + 0.896228 w_k1_AXP_s000
                                 + 1.23835 w_k1_DIS_s000 + 0.9903 w_k1_FDX_s000
                                 + 1.22597 w_k1_IBM_s000
                                 + 0.926198 w_k1_INTC_s000
                                 + 1.09104 w_k1_KO_s000 + 1.06521 w_k1_MAT_s000
                                 + 0.999982 w_k1_MSFT_s000
                                 + 0.93813 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.851671 w_k1_ADM_s000
                                 + 0.981613 w_k1_AXP_s000
                                 + 0.712618 w_k1_DIS_s000
                                 + 0.888358 w_k1_FDX_s000
                                 + 0.954722 w_k1_IBM_s000
                                 + 0.854597 w_k1_INTC_s000
                                 + 0.845726 w_k1_KO_s000
                                 + 1.18481 w_k1_MAT_s000
                                 + 0.842322 w_k1_MSFT_s000
                                 + 0.878021 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.03526 w_k1_ADM_s000
                                 + 0.812882 w_k1_AXP_s000
                                 + 1.22213 w_k1_DIS_s000 + 1.1366 w_k1_FDX_s000
                                 + 0.7965 w_k1_IBM_s000
                                 + 0.922331 w_k1_INTC_s000
                                 + 0.965842 w_k1_KO_s000
                                 + 0.633766 w_k1_MAT_s000
                                 + 1.03508 w_k1_MSFT_s000
                                 + 0.892921 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.08436 w_k1_ADM_s000
                                 + 1.32304 w_k1_AXP_s000
                                 + 1.06176 w_k1_DIS_s000
                                 + 1.20952 w_k1_FDX_s000
                                 + 0.894853 w_k1_IBM_s000
                                 + 1.08201 w_k1_INTC_s000
                                 + 0.834022 w_k1_KO_s000
                                 + 1.27225 w_k1_MAT_s000
                                 + 1.0149 w_k1_MSFT_s000
                                 + 0.912478 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.734893 w_k1_ADM_s000
                                 + 0.867711 w_k1_AXP_s000
                                 + 0.876477 w_k1_DIS_s000
                                 + 0.937226 w_k1_FDX_s000
                                 + 1.22792 w_k1_IBM_s000
                                 + 1.03818 w_k1_INTC_s000
                                 + 0.963991 w_k1_KO_s000
                                 + 1.03274 w_k1_MAT_s000
                                 + 0.950609 w_k1_MSFT_s000
                                 + 0.950171 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.28999 w_k1_ADM_s000
                                 + 1.31808 w_k1_AXP_s000
                                 + 1.80425 w_k1_DIS_s000
                                 + 1.33614 w_k1_FDX_s000
                                 + 1.33112 w_k1_IBM_s000
                                 + 1.25194 w_k1_INTC_s000
                                 + 1.32287 w_k1_KO_s000 + 1.42888 w_k1_MAT_s000
                                 + 1.42253 w_k1_MSFT_s000
                                 + 1.23269 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.932398 w_k1_ADM_s000
                                 + 1.41047 w_k1_AXP_s000
                                 + 1.08416 w_k1_DIS_s000
                                 + 1.02201 w_k1_FDX_s000 + 1.5214 w_k1_IBM_s000
                                 + 1.12927 w_k1_INTC_s000
                                 + 1.12474 w_k1_KO_s000 + 1.90771 w_k1_MAT_s000
                                 + 0.95224 w_k1_MSFT_s000
                                 + 0.937326 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.690024 w_k1_ADM_s000
                                 + 0.749753 w_k1_AXP_s000
                                 + 0.678616 w_k1_DIS_s000
                                 + 0.820893 w_k1_FDX_s000
                                 + 0.764053 w_k1_IBM_s000
                                 + 0.86946 w_k1_INTC_s000
                                 + 0.721789 w_k1_KO_s000
                                 + 0.872901 w_k1_MAT_s000
                                 + 0.925728 w_k1_MSFT_s000
                                 + 0.769128 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.937593 w_k1_ADM_s001
                                 + 1.05416 w_k1_AXP_s001
                                 + 0.558503 w_k1_DIS_s001
                                 + 0.751408 w_k1_FDX_s001
                                 + 0.9645 w_k1_IBM_s001
                                 + 0.65382 w_k1_INTC_s001
                                 + 0.783917 w_k1_KO_s001
                                 + 1.06809 w_k1_MAT_s001
                                 + 0.748409 w_k1_MSFT_s001
                                 + 0.91399 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.848867 w_k1_ADM_s001
                                 + 0.781898 w_k1_AXP_s001
                                 + 0.69489 w_k1_DIS_s001
                                 + 0.815488 w_k1_FDX_s001
                                 + 0.892421 w_k1_IBM_s001
                                 + 0.681543 w_k1_INTC_s001
                                 + 0.771312 w_k1_KO_s001
                                 + 0.98807 w_k1_MAT_s001
                                 + 0.782509 w_k1_MSFT_s001
                                 + 0.84208 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.743921 w_k1_ADM_s001
                                 + 0.73654 w_k1_AXP_s001
                                 + 0.781084 w_k1_DIS_s001
                                 + 0.834894 w_k1_FDX_s001
                                 + 0.604751 w_k1_IBM_s001
                                 + 0.714501 w_k1_INTC_s001
                                 + 0.776689 w_k1_KO_s001
                                 + 0.621219 w_k1_MAT_s001
                                 + 0.983733 w_k1_MSFT_s001
                                 + 0.851096 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.32664 w_k1_ADM_s001
                                 + 1.52783 w_k1_AXP_s001
                                 + 1.41843 w_k1_DIS_s001
                                 + 1.37754 w_k1_FDX_s001 + 1.1571 w_k1_IBM_s001
                                 + 1.06372 w_k1_INTC_s001
                                 + 1.19673 w_k1_KO_s001 + 1.21143 w_k1_MAT_s001
                                 + 1.34821 w_k1_MSFT_s001
                                 + 1.14337 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.0767 w_k1_ADM_s001
                                 + 1.17997 w_k1_AXP_s001
                                 + 1.19979 w_k1_DIS_s001
                                 + 1.00235 w_k1_FDX_s001
                                 + 1.55041 w_k1_IBM_s001
                                 + 1.0685 w_k1_INTC_s001 + 1.51031 w_k1_KO_s001
                                 + 2.04089 w_k1_MAT_s001
                                 + 1.19593 w_k1_MSFT_s001
                                 + 1.14243 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.923151 w_k1_ADM_s001
                                 + 0.776473 w_k1_AXP_s001
                                 + 1.06408 w_k1_DIS_s001
                                 + 0.999212 w_k1_FDX_s001
                                 + 0.865996 w_k1_IBM_s001
                                 + 0.995082 w_k1_INTC_s001
                                 + 0.938843 w_k1_KO_s001
                                 + 0.999866 w_k1_MAT_s001
                                 + 1.17245 w_k1_MSFT_s001
                                 + 0.995803 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.761329 w_k1_ADM_s001
                                 + 1.02057 w_k1_AXP_s001
                                 + 0.855519 w_k1_DIS_s001
                                 + 0.841064 w_k1_FDX_s001
                                 + 1.25136 w_k1_IBM_s001
                                 + 0.900525 w_k1_INTC_s001
                                 + 1.06793 w_k1_KO_s001 + 1.37697 w_k1_MAT_s001
                                 + 0.964086 w_k1_MSFT_s001
                                 + 0.851756 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.19751 w_k1_ADM_s001
                                 + 1.28652 w_k1_AXP_s001
                                 + 0.990992 w_k1_DIS_s001
                                 + 1.14134 w_k1_FDX_s001
                                 + 0.926173 w_k1_IBM_s001
                                 + 1.09746 w_k1_INTC_s001
                                 + 0.964934 w_k1_KO_s001
                                 + 1.41829 w_k1_MAT_s001
                                 + 1.15573 w_k1_MSFT_s001
                                 + 0.997779 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.10382 w_k1_ADM_s002
                                 + 0.948012 w_k1_AXP_s002
                                 + 0.98369 w_k1_DIS_s002
                                 + 1.06233 w_k1_FDX_s002
                                 + 0.874973 w_k1_IBM_s002
                                 + 1.1022 w_k1_INTC_s002
                                 + 0.947018 w_k1_KO_s002
                                 + 1.03657 w_k1_MAT_s002
                                 + 0.977122 w_k1_MSFT_s002
                                 + 0.995308 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.43898 w_k1_ADM_s002
                                 + 1.09001 w_k1_AXP_s002
                                 + 1.12734 w_k1_DIS_s002
                                 + 1.19829 w_k1_FDX_s002
                                 + 0.899907 w_k1_IBM_s002
                                 + 1.14543 w_k1_INTC_s002
                                 + 0.930019 w_k1_KO_s002
                                 + 0.903705 w_k1_MAT_s002
                                 + 1.08257 w_k1_MSFT_s002
                                 + 0.999523 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.990584 w_k1_ADM_s002
                                 + 0.804489 w_k1_AXP_s002
                                 + 0.655594 w_k1_DIS_s002
                                 + 0.899597 w_k1_FDX_s002
                                 + 0.691314 w_k1_IBM_s002
                                 + 0.990246 w_k1_INTC_s002
                                 + 0.701994 w_k1_KO_s002
                                 + 0.474782 w_k1_MAT_s002
                                 + 0.849134 w_k1_MSFT_s002
                                 + 0.939014 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.05829 w_k1_ADM_s002
                                 + 0.813666 w_k1_AXP_s002
                                 + 0.821697 w_k1_DIS_s002
                                 + 0.940599 w_k1_FDX_s002
                                 + 0.68496 w_k1_IBM_s002
                                 + 0.896092 w_k1_INTC_s002
                                 + 0.745834 w_k1_KO_s002
                                 + 0.559494 w_k1_MAT_s002
                                 + 0.850479 w_k1_MSFT_s002
                                 + 0.963364 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.56886 w_k1_ADM_s002
                                 + 1.4019 w_k1_AXP_s002 + 1.43938 w_k1_DIS_s002
                                 + 1.42001 w_k1_FDX_s002
                                 + 1.31727 w_k1_IBM_s002
                                 + 1.35474 w_k1_INTC_s002
                                 + 1.34137 w_k1_KO_s002
                                 + 0.999446 w_k1_MAT_s002
                                 + 1.33456 w_k1_MSFT_s002
                                 + 1.26357 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.18601 w_k1_ADM_s002
                                 + 1.30491 w_k1_AXP_s002
                                 + 1.19094 w_k1_DIS_s002
                                 + 1.02053 w_k1_FDX_s002
                                 + 1.51635 w_k1_IBM_s002
                                 + 1.27626 w_k1_INTC_s002
                                 + 1.13261 w_k1_KO_s002 + 1.14378 w_k1_MAT_s002
                                 + 1.01333 w_k1_MSFT_s002
                                 + 1.10471 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.01842 w_k1_ADM_s002
                                 + 0.700797 w_k1_AXP_s002
                                 + 0.76916 w_k1_DIS_s002
                                 + 0.78325 w_k1_FDX_s002 + 1.004 w_k1_IBM_s002
                                 + 0.832819 w_k1_INTC_s002
                                 + 0.831491 w_k1_KO_s002
                                 + 0.70843 w_k1_MAT_s002
                                 + 0.879595 w_k1_MSFT_s002
                                 + 0.854602 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.00751 w_k1_ADM_s002
                                 + 1.221 w_k1_AXP_s002 + 0.888704 w_k1_DIS_s002
                                 + 1.05598 w_k1_FDX_s002
                                 + 1.19117 w_k1_IBM_s002
                                 + 1.36256 w_k1_INTC_s002
                                 + 1.01663 w_k1_KO_s002 + 1.051 w_k1_MAT_s002
                                 + 0.918129 w_k1_MSFT_s002
                                 + 0.972991 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.25119 w_k1_ADM_s003
                                 + 1.10741 w_k1_AXP_s003
                                 + 1.17671 w_k1_DIS_s003
                                 + 1.00571 w_k1_FDX_s003
                                 + 1.07185 w_k1_IBM_s003
                                 + 0.864744 w_k1_INTC_s003
                                 + 1.05569 w_k1_KO_s003 + 1.05475 w_k1_MAT_s003
                                 + 1.06585 w_k1_MSFT_s003
                                 + 1.12172 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.38706 w_k1_ADM_s003
                                 + 1.30596 w_k1_AXP_s003 + 1.4539 w_k1_DIS_s003
                                 + 1.28331 w_k1_FDX_s003
                                 + 1.16825 w_k1_IBM_s003
                                 + 1.15391 w_k1_INTC_s003
                                 + 1.25387 w_k1_KO_s003 + 1.25913 w_k1_MAT_s003
                                 + 1.24902 w_k1_MSFT_s003
                                 + 1.13494 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.19709 w_k1_ADM_s003
                                 + 0.881192 w_k1_AXP_s003
                                 + 0.993355 w_k1_DIS_s003
                                 + 0.863604 w_k1_FDX_s003
                                 + 1.0732 w_k1_IBM_s003
                                 + 0.856304 w_k1_INTC_s003
                                 + 0.965173 w_k1_KO_s003
                                 + 0.714132 w_k1_MAT_s003
                                 + 0.92398 w_k1_MSFT_s003
                                 + 0.92393 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.800616 w_k1_ADM_s003
                                 + 0.65638 w_k1_AXP_s003
                                 + 0.665827 w_k1_DIS_s003
                                 + 0.749334 w_k1_FDX_s003
                                 + 0.647122 w_k1_IBM_s003
                                 + 0.858557 w_k1_INTC_s003
                                 + 0.665984 w_k1_KO_s003
                                 + 0.478758 w_k1_MAT_s003
                                 + 0.96462 w_k1_MSFT_s003
                                 + 0.754629 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.07864 w_k1_ADM_s003
                                 + 1.20208 w_k1_AXP_s003
                                 + 1.03621 w_k1_DIS_s003
                                 + 0.974606 w_k1_FDX_s003
                                 + 1.09116 w_k1_IBM_s003
                                 + 0.902865 w_k1_INTC_s003
                                 + 0.965473 w_k1_KO_s003
                                 + 0.836153 w_k1_MAT_s003
                                 + 0.962664 w_k1_MSFT_s003
                                 + 0.992247 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.08725 w_k1_ADM_s003
                                 + 1.0046 w_k1_AXP_s003
                                 + 0.866036 w_k1_DIS_s003
                                 + 0.966728 w_k1_FDX_s003
                                 + 0.77794 w_k1_IBM_s003
                                 + 0.838235 w_k1_INTC_s003
                                 + 0.789208 w_k1_KO_s003
                                 + 0.758585 w_k1_MAT_s003
                                 + 0.914722 w_k1_MSFT_s003
                                 + 0.955334 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.914387 w_k1_ADM_s003
                                 + 0.975696 w_k1_AXP_s003
                                 + 0.783871 w_k1_DIS_s003
                                 + 0.836381 w_k1_FDX_s003
                                 + 1.08561 w_k1_IBM_s003
                                 + 0.706907 w_k1_INTC_s003
                                 + 0.834587 w_k1_KO_s003
                                 + 1.29182 w_k1_MAT_s003
                                 + 0.82824 w_k1_MSFT_s003
                                 + 0.958848 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.892914 w_k1_ADM_s003
                                 + 1.37453 w_k1_AXP_s003
                                 + 1.05036 w_k1_DIS_s003
                                 + 0.973637 w_k1_FDX_s003
                                 + 1.62754 w_k1_IBM_s003
                                 + 1.0948 w_k1_INTC_s003 + 1.01245 w_k1_KO_s003
                                 + 1.65152 w_k1_MAT_s003
                                 + 1.01269 w_k1_MSFT_s003
                                 + 1.00262 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.988315 w_k1_ADM_s004
                                 + 1.12846 w_k1_AXP_s004
                                 + 0.765044 w_k1_DIS_s004
                                 + 0.893272 w_k1_FDX_s004
                                 + 1.10529 w_k1_IBM_s004
                                 + 0.760185 w_k1_INTC_s004
                                 + 0.834414 w_k1_KO_s004 + 1.2994 w_k1_MAT_s004
                                 + 0.749533 w_k1_MSFT_s004
                                 + 0.891112 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.05917 w_k1_ADM_s004
                                 + 0.813728 w_k1_AXP_s004
                                 + 0.804391 w_k1_DIS_s004
                                 + 0.877344 w_k1_FDX_s004
                                 + 0.690912 w_k1_IBM_s004
                                 + 1.04901 w_k1_INTC_s004
                                 + 0.816358 w_k1_KO_s004
                                 + 0.67455 w_k1_MAT_s004
                                 + 0.988658 w_k1_MSFT_s004
                                 + 0.895299 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.961726 w_k1_ADM_s004
                                 + 1.00087 w_k1_AXP_s004
                                 + 0.994425 w_k1_DIS_s004
                                 + 0.885995 w_k1_FDX_s004
                                 + 1.16696 w_k1_IBM_s004
                                 + 0.880058 w_k1_INTC_s004
                                 + 1.07694 w_k1_KO_s004 + 1.01291 w_k1_MAT_s004
                                 + 0.945642 w_k1_MSFT_s004
                                 + 0.993147 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.30251 w_k1_ADM_s004
                                 + 1.4503 w_k1_AXP_s004 + 1.75904 w_k1_DIS_s004
                                 + 1.31394 w_k1_FDX_s004
                                 + 1.21109 w_k1_IBM_s004
                                 + 1.62831 w_k1_INTC_s004
                                 + 1.53934 w_k1_KO_s004 + 1.25135 w_k1_MAT_s004
                                 + 1.67551 w_k1_MSFT_s004
                                 + 1.25776 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.0516 w_k1_ADM_s004
                                 + 1.44695 w_k1_AXP_s004
                                 + 1.27493 w_k1_DIS_s004
                                 + 1.07732 w_k1_FDX_s004
                                 + 1.62237 w_k1_IBM_s004
                                 + 1.01306 w_k1_INTC_s004
                                 + 1.47547 w_k1_KO_s004 + 2.11339 w_k1_MAT_s004
                                 + 1.03639 w_k1_MSFT_s004
                                 + 1.09455 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.08146 w_k1_ADM_s004
                                 + 1.09758 w_k1_AXP_s004
                                 + 1.06804 w_k1_DIS_s004
                                 + 1.15513 w_k1_FDX_s004
                                 + 0.887585 w_k1_IBM_s004
                                 + 1.07191 w_k1_INTC_s004
                                 + 1.03983 w_k1_KO_s004 + 1.34444 w_k1_MAT_s004
                                 + 1.19319 w_k1_MSFT_s004
                                 + 1.01805 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.56775 w_k1_ADM_s004
                                 + 1.64901 w_k1_AXP_s004
                                 + 1.04445 w_k1_DIS_s004
                                 + 1.33398 w_k1_FDX_s004
                                 + 0.993122 w_k1_IBM_s004
                                 + 1.35863 w_k1_INTC_s004
                                 + 1.15234 w_k1_KO_s004 + 1.13752 w_k1_MAT_s004
                                 + 1.17449 w_k1_MSFT_s004 + 1.0835 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.602998 w_k1_ADM_s004
                                 + 0.619194 w_k1_AXP_s004
                                 + 0.514456 w_k1_DIS_s004
                                 + 0.700234 w_k1_FDX_s004
                                 + 0.708392 w_k1_IBM_s004
                                 + 0.817416 w_k1_INTC_s004
                                 + 0.711946 w_k1_KO_s004
                                 + 0.901507 w_k1_MAT_s004
                                 + 0.753763 w_k1_MSFT_s004
                                 + 0.761012 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.75869 w_k1_ADM_s005
                                 + 1.63949 w_k1_AXP_s005 + 1.2058 w_k1_DIS_s005
                                 + 1.35628 w_k1_FDX_s005
                                 + 0.880323 w_k1_IBM_s005
                                 + 1.34396 w_k1_INTC_s005
                                 + 1.15778 w_k1_KO_s005 + 1.11551 w_k1_MAT_s005
                                 + 1.3452 w_k1_MSFT_s005 + 1.41813 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.09205 w_k1_ADM_s005
                                 + 1.10086 w_k1_AXP_s005
                                 + 1.33667 w_k1_DIS_s005
                                 + 1.11073 w_k1_FDX_s005 + 1.349 w_k1_IBM_s005
                                 + 1.20628 w_k1_INTC_s005
                                 + 1.27781 w_k1_KO_s005 + 1.31092 w_k1_MAT_s005
                                 + 1.26937 w_k1_MSFT_s005
                                 + 1.30744 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.37099 w_k1_ADM_s005
                                 + 0.851648 w_k1_AXP_s005
                                 + 0.904701 w_k1_DIS_s005
                                 + 0.961356 w_k1_FDX_s005
                                 + 0.645906 w_k1_IBM_s005
                                 + 1.04837 w_k1_INTC_s005
                                 + 0.825515 w_k1_KO_s005
                                 + 0.854082 w_k1_MAT_s005
                                 + 1.05388 w_k1_MSFT_s005
                                 + 1.02207 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.03583 w_k1_ADM_s005
                                 + 1.2827 w_k1_AXP_s005 + 1.65861 w_k1_DIS_s005
                                 + 1.35306 w_k1_FDX_s005
                                 + 0.791915 w_k1_IBM_s005
                                 + 1.19222 w_k1_INTC_s005
                                 + 1.08818 w_k1_KO_s005 + 0.99157 w_k1_MAT_s005
                                 + 1.33459 w_k1_MSFT_s005
                                 + 1.09493 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.3911 w_k1_ADM_s005
                                 + 0.911777 w_k1_AXP_s005
                                 + 0.561311 w_k1_DIS_s005
                                 + 0.764293 w_k1_FDX_s005
                                 + 0.85848 w_k1_IBM_s005
                                 + 0.650416 w_k1_INTC_s005
                                 + 0.746382 w_k1_KO_s005
                                 + 0.866929 w_k1_MAT_s005
                                 + 0.658887 w_k1_MSFT_s005
                                 + 0.875687 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.10541 w_k1_ADM_s005
                                 + 1.30898 w_k1_AXP_s005
                                 + 0.889063 w_k1_DIS_s005
                                 + 0.926744 w_k1_FDX_s005
                                 + 1.17542 w_k1_IBM_s005
                                 + 1.0189 w_k1_INTC_s005
                                 + 0.985228 w_k1_KO_s005
                                 + 1.42308 w_k1_MAT_s005
                                 + 0.96398 w_k1_MSFT_s005
                                 + 1.06214 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.00985 w_k1_ADM_s005
                                 + 0.723611 w_k1_AXP_s005
                                 + 0.835635 w_k1_DIS_s005
                                 + 0.732285 w_k1_FDX_s005
                                 + 1.10644 w_k1_IBM_s005
                                 + 0.875091 w_k1_INTC_s005
                                 + 1.00154 w_k1_KO_s005
                                 + 0.766956 w_k1_MAT_s005
                                 + 0.92302 w_k1_MSFT_s005
                                 + 0.966396 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.20505 w_k1_ADM_s005
                                 + 1.23918 w_k1_AXP_s005
                                 + 1.00705 w_k1_DIS_s005
                                 + 1.09022 w_k1_FDX_s005
                                 + 0.98622 w_k1_IBM_s005
                                 + 1.24196 w_k1_INTC_s005
                                 + 1.01559 w_k1_KO_s005
                                 + 0.900939 w_k1_MAT_s005
                                 + 0.983184 w_k1_MSFT_s005
                                 + 1.02206 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.959583 w_k1_ADM_s006
                                 + 0.946825 w_k1_AXP_s006
                                 + 0.770944 w_k1_DIS_s006
                                 + 0.868746 w_k1_FDX_s006
                                 + 1.20814 w_k1_IBM_s006
                                 + 0.967342 w_k1_INTC_s006
                                 + 0.922692 w_k1_KO_s006
                                 + 0.79882 w_k1_MAT_s006
                                 + 0.795757 w_k1_MSFT_s006
                                 + 0.917987 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.860272 w_k1_ADM_s006
                                 + 0.690867 w_k1_AXP_s006
                                 + 0.699837 w_k1_DIS_s006
                                 + 0.827686 w_k1_FDX_s006
                                 + 0.776255 w_k1_IBM_s006
                                 + 0.831278 w_k1_INTC_s006
                                 + 0.721859 w_k1_KO_s006
                                 + 0.666126 w_k1_MAT_s006
                                 + 0.796932 w_k1_MSFT_s006
                                 + 0.955835 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.19552 w_k1_ADM_s006
                                 + 1.27748 w_k1_AXP_s006
                                 + 1.11137 w_k1_DIS_s006
                                 + 1.10375 w_k1_FDX_s006
                                 + 1.09707 w_k1_IBM_s006
                                 + 0.97538 w_k1_INTC_s006
                                 + 1.07328 w_k1_KO_s006 + 1.15487 w_k1_MAT_s006
                                 + 1.12745 w_k1_MSFT_s006
                                 + 0.962958 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.9381 w_k1_ADM_s006
                                 + 0.946769 w_k1_AXP_s006
                                 + 1.05816 w_k1_DIS_s006
                                 + 0.904752 w_k1_FDX_s006
                                 + 1.4385 w_k1_IBM_s006
                                 + 1.15504 w_k1_INTC_s006
                                 + 1.22222 w_k1_KO_s006 + 1.5729 w_k1_MAT_s006
                                 + 0.978977 w_k1_MSFT_s006
                                 + 1.01598 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.32356 w_k1_ADM_s006
                                 + 1.26953 w_k1_AXP_s006
                                 + 1.02725 w_k1_DIS_s006
                                 + 1.21316 w_k1_FDX_s006
                                 + 0.788445 w_k1_IBM_s006
                                 + 0.876428 w_k1_INTC_s006
                                 + 0.864915 w_k1_KO_s006
                                 + 1.13511 w_k1_MAT_s006
                                 + 0.97725 w_k1_MSFT_s006
                                 + 1.11898 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.988735 w_k1_ADM_s006
                                 + 1.10975 w_k1_AXP_s006
                                 + 0.912342 w_k1_DIS_s006
                                 + 1.08976 w_k1_FDX_s006 + 0.9884 w_k1_IBM_s006
                                 + 0.99603 w_k1_INTC_s006
                                 + 0.89391 w_k1_KO_s006 + 1.11981 w_k1_MAT_s006
                                 + 0.972877 w_k1_MSFT_s006
                                 + 1.01227 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.915126 w_k1_ADM_s006
                                 + 0.680081 w_k1_AXP_s006
                                 + 0.901708 w_k1_DIS_s006
                                 + 1.0414 w_k1_FDX_s006
                                 + 0.732285 w_k1_IBM_s006
                                 + 1.06956 w_k1_INTC_s006
                                 + 0.90395 w_k1_KO_s006
                                 + 0.524312 w_k1_MAT_s006
                                 + 1.0181 w_k1_MSFT_s006 + 1.01522 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.21183 w_k1_ADM_s006
                                 + 1.27552 w_k1_AXP_s006
                                 + 1.47767 w_k1_DIS_s006
                                 + 1.41253 w_k1_FDX_s006
                                 + 1.05914 w_k1_IBM_s006
                                 + 1.29634 w_k1_INTC_s006
                                 + 1.29974 w_k1_KO_s006 + 1.08714 w_k1_MAT_s006
                                 + 1.31776 w_k1_MSFT_s006
                                 + 1.18125 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.09464 w_k1_ADM_s007
                                 + 0.796045 w_k1_AXP_s007
                                 + 0.598635 w_k1_DIS_s007
                                 + 0.741439 w_k1_FDX_s007
                                 + 0.738583 w_k1_IBM_s007
                                 + 0.766205 w_k1_INTC_s007
                                 + 0.863095 w_k1_KO_s007
                                 + 1.08161 w_k1_MAT_s007
                                 + 0.911522 w_k1_MSFT_s007
                                 + 0.841682 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.09981 w_k1_ADM_s007
                                 + 1.08254 w_k1_AXP_s007
                                 + 1.40456 w_k1_DIS_s007
                                 + 1.11563 w_k1_FDX_s007
                                 + 1.25686 w_k1_IBM_s007
                                 + 1.23588 w_k1_INTC_s007
                                 + 1.54973 w_k1_KO_s007 + 1.43978 w_k1_MAT_s007
                                 + 1.35692 w_k1_MSFT_s007
                                 + 1.08391 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.03784 w_k1_ADM_s007
                                 + 1.04772 w_k1_AXP_s007
                                 + 0.921706 w_k1_DIS_s007
                                 + 0.964419 w_k1_FDX_s007
                                 + 1.21376 w_k1_IBM_s007
                                 + 1.01241 w_k1_INTC_s007
                                 + 1.13046 w_k1_KO_s007 + 1.48111 w_k1_MAT_s007
                                 + 1.04963 w_k1_MSFT_s007
                                 + 0.998832 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.37586 w_k1_ADM_s007
                                 + 1.47139 w_k1_AXP_s007
                                 + 1.12328 w_k1_DIS_s007
                                 + 1.30681 w_k1_FDX_s007
                                 + 0.943043 w_k1_IBM_s007
                                 + 1.33595 w_k1_INTC_s007
                                 + 1.07564 w_k1_KO_s007 + 1.17061 w_k1_MAT_s007
                                 + 1.35896 w_k1_MSFT_s007
                                 + 1.10952 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.946895 w_k1_ADM_s007
                                 + 0.962222 w_k1_AXP_s007
                                 + 0.932158 w_k1_DIS_s007
                                 + 0.957162 w_k1_FDX_s007
                                 + 0.737557 w_k1_IBM_s007
                                 + 0.879992 w_k1_INTC_s007
                                 + 1.06077 w_k1_KO_s007
                                 + 0.945074 w_k1_MAT_s007
                                 + 1.07027 w_k1_MSFT_s007
                                 + 0.907971 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.04543 w_k1_ADM_s007
                                 + 0.903291 w_k1_AXP_s007
                                 + 0.656533 w_k1_DIS_s007
                                 + 0.85019 w_k1_FDX_s007
                                 + 0.729329 w_k1_IBM_s007
                                 + 0.868813 w_k1_INTC_s007
                                 + 0.852992 w_k1_KO_s007
                                 + 1.33262 w_k1_MAT_s007
                                 + 1.05939 w_k1_MSFT_s007
                                 + 0.858778 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.18192 w_k1_ADM_s007
                                 + 1.35117 w_k1_AXP_s007
                                 + 1.10666 w_k1_DIS_s007
                                 + 1.00116 w_k1_FDX_s007
                                 + 1.64177 w_k1_IBM_s007
                                 + 0.919318 w_k1_INTC_s007
                                 + 1.34395 w_k1_KO_s007 + 2.05437 w_k1_MAT_s007
                                 + 1.13949 w_k1_MSFT_s007
                                 + 1.15849 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.786897 w_k1_ADM_s007
                                 + 0.888381 w_k1_AXP_s007
                                 + 0.673874 w_k1_DIS_s007
                                 + 0.738112 w_k1_FDX_s007
                                 + 1.17375 w_k1_IBM_s007
                                 + 1.09824 w_k1_INTC_s007
                                 + 0.964189 w_k1_KO_s007
                                 + 1.36947 w_k1_MAT_s007
                                 + 1.01783 w_k1_MSFT_s007
                                 + 0.881145 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.951776 w_k1_ADM_s010
                                 + 0.951087 w_k1_AXP_s010
                                 + 1.0319 w_k1_DIS_s010
                                 + 0.994685 w_k1_FDX_s010
                                 + 1.26982 w_k1_IBM_s010
                                 + 0.988963 w_k1_INTC_s010
                                 + 1.17908 w_k1_KO_s010 + 1.16746 w_k1_MAT_s010
                                 + 1.04307 w_k1_MSFT_s010
                                 + 0.92205 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.06446 w_k1_ADM_s010
                                 + 1.53445 w_k1_AXP_s010
                                 + 1.63357 w_k1_DIS_s010
                                 + 1.32171 w_k1_FDX_s010
                                 + 1.36884 w_k1_IBM_s010
                                 + 1.07843 w_k1_INTC_s010
                                 + 1.36927 w_k1_KO_s010 + 1.53678 w_k1_MAT_s010
                                 + 1.17349 w_k1_MSFT_s010
                                 + 1.31116 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.808055 w_k1_ADM_s010
                                 + 0.883371 w_k1_AXP_s010
                                 + 0.811029 w_k1_DIS_s010
                                 + 0.893943 w_k1_FDX_s010
                                 + 0.927029 w_k1_IBM_s010
                                 + 0.902512 w_k1_INTC_s010
                                 + 0.877543 w_k1_KO_s010
                                 + 0.840298 w_k1_MAT_s010
                                 + 0.950823 w_k1_MSFT_s010
                                 + 0.939613 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.669314 w_k1_ADM_s010
                                 + 0.629171 w_k1_AXP_s010
                                 + 0.646877 w_k1_DIS_s010
                                 + 0.768564 w_k1_FDX_s010
                                 + 0.737159 w_k1_IBM_s010
                                 + 0.986662 w_k1_INTC_s010
                                 + 0.693494 w_k1_KO_s010
                                 + 0.606461 w_k1_MAT_s010
                                 + 0.936145 w_k1_MSFT_s010
                                 + 0.772847 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.01071 w_k1_ADM_s010
                                 + 1.00211 w_k1_AXP_s010
                                 + 1.18471 w_k1_DIS_s010
                                 + 1.10101 w_k1_FDX_s010
                                 + 1.14196 w_k1_IBM_s010
                                 + 0.862378 w_k1_INTC_s010
                                 + 1.14278 w_k1_KO_s010 + 1.58968 w_k1_MAT_s010
                                 + 1.28437 w_k1_MSFT_s010
                                 + 1.21104 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.923813 w_k1_ADM_s010
                                 + 0.951588 w_k1_AXP_s010
                                 + 0.849813 w_k1_DIS_s010
                                 + 1.09216 w_k1_FDX_s010
                                 + 0.743119 w_k1_IBM_s010
                                 + 0.902693 w_k1_INTC_s010
                                 + 0.866854 w_k1_KO_s010
                                 + 0.643826 w_k1_MAT_s010
                                 + 1.12248 w_k1_MSFT_s010
                                 + 0.987025 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.80857 w_k1_ADM_s010
                                 + 0.834526 w_k1_AXP_s010
                                 + 1.23575 w_k1_DIS_s010 + 1.0025 w_k1_FDX_s010
                                 + 0.953621 w_k1_IBM_s010
                                 + 0.933669 w_k1_INTC_s010
                                 + 1.20652 w_k1_KO_s010 + 1.04113 w_k1_MAT_s010
                                 + 1.24393 w_k1_MSFT_s010
                                 + 1.16008 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.06291 w_k1_ADM_s010
                                 + 1.21114 w_k1_AXP_s010 + 1.284 w_k1_DIS_s010
                                 + 1.30666 w_k1_FDX_s010
                                 + 0.914767 w_k1_IBM_s010
                                 + 1.03326 w_k1_INTC_s010
                                 + 1.19883 w_k1_KO_s010 + 1.04108 w_k1_MAT_s010
                                 + 1.30736 w_k1_MSFT_s010
                                 + 1.19705 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.864624 w_k1_ADM_s011
                                 + 0.991557 w_k1_AXP_s011
                                 + 0.681653 w_k1_DIS_s011
                                 + 0.842551 w_k1_FDX_s011
                                 + 0.786972 w_k1_IBM_s011
                                 + 0.79556 w_k1_INTC_s011
                                 + 0.758257 w_k1_KO_s011
                                 + 0.917391 w_k1_MAT_s011
                                 + 0.739614 w_k1_MSFT_s011
                                 + 0.904016 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.73097 w_k1_ADM_s011
                                 + 0.680036 w_k1_AXP_s011
                                 + 0.672768 w_k1_DIS_s011
                                 + 0.748242 w_k1_FDX_s011
                                 + 0.652513 w_k1_IBM_s011
                                 + 0.817111 w_k1_INTC_s011
                                 + 0.653747 w_k1_KO_s011
                                 + 0.659276 w_k1_MAT_s011
                                 + 0.806438 w_k1_MSFT_s011
                                 + 0.919556 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.982462 w_k1_ADM_s011
                                 + 1.50609 w_k1_AXP_s011
                                 + 1.65436 w_k1_DIS_s011
                                 + 1.32084 w_k1_FDX_s011
                                 + 1.23916 w_k1_IBM_s011
                                 + 1.2139 w_k1_INTC_s011 + 1.22951 w_k1_KO_s011
                                 + 1.43045 w_k1_MAT_s011
                                 + 1.13862 w_k1_MSFT_s011 + 1.3214 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.04349 w_k1_ADM_s011
                                 + 1.03317 w_k1_AXP_s011
                                 + 1.30227 w_k1_DIS_s011
                                 + 1.04164 w_k1_FDX_s011
                                 + 0.915607 w_k1_IBM_s011
                                 + 0.941821 w_k1_INTC_s011
                                 + 0.947327 w_k1_KO_s011
                                 + 0.889621 w_k1_MAT_s011
                                 + 1.03253 w_k1_MSFT_s011
                                 + 1.12713 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.63051 w_k1_ADM_s011
                                 + 1.04818 w_k1_AXP_s011
                                 + 0.964203 w_k1_DIS_s011
                                 + 0.877509 w_k1_FDX_s011
                                 + 1.38295 w_k1_IBM_s011
                                 + 0.971573 w_k1_INTC_s011
                                 + 0.899868 w_k1_KO_s011
                                 + 1.52998 w_k1_MAT_s011
                                 + 0.859751 w_k1_MSFT_s011
                                 + 1.09758 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.986651 w_k1_ADM_s011
                                 + 1.1506 w_k1_AXP_s011
                                 + 0.849149 w_k1_DIS_s011
                                 + 1.01246 w_k1_FDX_s011
                                 + 0.919332 w_k1_IBM_s011
                                 + 1.04976 w_k1_INTC_s011
                                 + 0.953778 w_k1_KO_s011
                                 + 0.946309 w_k1_MAT_s011
                                 + 1.09246 w_k1_MSFT_s011
                                 + 1.20579 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.488619 w_k1_ADM_s011
                                 + 0.718074 w_k1_AXP_s011
                                 + 0.544445 w_k1_DIS_s011
                                 + 0.654094 w_k1_FDX_s011
                                 + 0.977982 w_k1_IBM_s011
                                 + 0.627369 w_k1_INTC_s011
                                 + 0.737465 w_k1_KO_s011
                                 + 1.00715 w_k1_MAT_s011
                                 + 0.754329 w_k1_MSFT_s011
                                 + 0.86879 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.763783 w_k1_ADM_s011
                                 + 0.837863 w_k1_AXP_s011
                                 + 0.977147 w_k1_DIS_s011
                                 + 0.946623 w_k1_FDX_s011
                                 + 0.709324 w_k1_IBM_s011
                                 + 0.902361 w_k1_INTC_s011
                                 + 0.823317 w_k1_KO_s011
                                 + 0.638578 w_k1_MAT_s011
                                 + 0.877616 w_k1_MSFT_s011
                                 + 0.87329 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.15284 w_k1_ADM_s012
                                 + 1.09077 w_k1_AXP_s012
                                 + 0.935899 w_k1_DIS_s012
                                 + 0.892943 w_k1_FDX_s012
                                 + 0.999177 w_k1_IBM_s012
                                 + 1.08447 w_k1_INTC_s012
                                 + 0.993749 w_k1_KO_s012
                                 + 0.781365 w_k1_MAT_s012
                                 + 0.963972 w_k1_MSFT_s012
                                 + 0.89478 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.31765 w_k1_ADM_s012
                                 + 1.77578 w_k1_AXP_s012
                                 + 1.78624 w_k1_DIS_s012
                                 + 1.39854 w_k1_FDX_s012
                                 + 1.31341 w_k1_IBM_s012
                                 + 1.16381 w_k1_INTC_s012
                                 + 1.41558 w_k1_KO_s012 + 2.0489 w_k1_MAT_s012
                                 + 1.32935 w_k1_MSFT_s012
                                 + 1.04709 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.98494 w_k1_ADM_s012
                                 + 0.785035 w_k1_AXP_s012
                                 + 0.727765 w_k1_DIS_s012
                                 + 0.787579 w_k1_FDX_s012
                                 + 0.922569 w_k1_IBM_s012
                                 + 0.692172 w_k1_INTC_s012
                                 + 0.816488 w_k1_KO_s012
                                 + 0.90468 w_k1_MAT_s012
                                 + 0.898346 w_k1_MSFT_s012
                                 + 0.874188 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.33688 w_k1_ADM_s012
                                 + 1.28337 w_k1_AXP_s012
                                 + 1.04244 w_k1_DIS_s012
                                 + 1.12715 w_k1_FDX_s012
                                 + 0.757083 w_k1_IBM_s012
                                 + 1.16277 w_k1_INTC_s012
                                 + 0.879787 w_k1_KO_s012
                                 + 0.694518 w_k1_MAT_s012
                                 + 1.05818 w_k1_MSFT_s012
                                 + 0.872275 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.07301 w_k1_ADM_s012
                                 + 0.993196 w_k1_AXP_s012
                                 + 1.0085 w_k1_DIS_s012
                                 + 0.885705 w_k1_FDX_s012
                                 + 1.37124 w_k1_IBM_s012
                                 + 0.953858 w_k1_INTC_s012
                                 + 1.15846 w_k1_KO_s012 + 1.4312 w_k1_MAT_s012
                                 + 1.01296 w_k1_MSFT_s012
                                 + 0.959178 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.845773 w_k1_ADM_s012
                                 + 1.07238 w_k1_AXP_s012
                                 + 0.711453 w_k1_DIS_s012
                                 + 0.890872 w_k1_FDX_s012
                                 + 0.731234 w_k1_IBM_s012
                                 + 1.17064 w_k1_INTC_s012
                                 + 0.729791 w_k1_KO_s012
                                 + 0.928999 w_k1_MAT_s012
                                 + 0.921813 w_k1_MSFT_s012
                                 + 0.815565 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.08944 w_k1_ADM_s012
                                 + 1.50518 w_k1_AXP_s012
                                 + 0.807342 w_k1_DIS_s012
                                 + 1.00424 w_k1_FDX_s012
                                 + 1.27076 w_k1_IBM_s012
                                 + 0.989299 w_k1_INTC_s012
                                 + 0.982284 w_k1_KO_s012
                                 + 2.33115 w_k1_MAT_s012
                                 + 0.87346 w_k1_MSFT_s012
                                 + 1.05937 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.934575 w_k1_ADM_s012
                                 + 1.11696 w_k1_AXP_s012 + 1.3144 w_k1_DIS_s012
                                 + 1.07677 w_k1_FDX_s012
                                 + 1.16321 w_k1_IBM_s012
                                 + 1.28696 w_k1_INTC_s012
                                 + 1.26953 w_k1_KO_s012 + 1.52023 w_k1_MAT_s012
                                 + 1.18174 w_k1_MSFT_s012
                                 + 1.10097 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.843377 w_k1_ADM_s013
                                 + 1.06206 w_k1_AXP_s013
                                 + 0.926976 w_k1_DIS_s013
                                 + 0.969904 w_k1_FDX_s013
                                 + 0.970294 w_k1_IBM_s013
                                 + 1.07322 w_k1_INTC_s013
                                 + 0.958317 w_k1_KO_s013
                                 + 0.848644 w_k1_MAT_s013
                                 + 0.915065 w_k1_MSFT_s013
                                 + 0.965279 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.29851 w_k1_ADM_s013
                                 + 1.69226 w_k1_AXP_s013
                                 + 1.24978 w_k1_DIS_s013
                                 + 1.27584 w_k1_FDX_s013
                                 + 1.11738 w_k1_IBM_s013
                                 + 1.09513 w_k1_INTC_s013 + 1.2649 w_k1_KO_s013
                                 + 1.2217 w_k1_MAT_s013
                                 + 1.13828 w_k1_MSFT_s013
                                 + 1.21645 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.709706 w_k1_ADM_s013
                                 + 1.03287 w_k1_AXP_s013
                                 + 1.01411 w_k1_DIS_s013
                                 + 0.83176 w_k1_FDX_s013
                                 + 1.50564 w_k1_IBM_s013
                                 + 1.01153 w_k1_INTC_s013 + 1.2225 w_k1_KO_s013
                                 + 1.19511 w_k1_MAT_s013
                                 + 0.909295 w_k1_MSFT_s013
                                 + 1.09101 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.965237 w_k1_ADM_s013
                                 + 1.17177 w_k1_AXP_s013
                                 + 0.839431 w_k1_DIS_s013
                                 + 1.00057 w_k1_FDX_s013
                                 + 0.882407 w_k1_IBM_s013
                                 + 0.995732 w_k1_INTC_s013
                                 + 0.947795 w_k1_KO_s013
                                 + 0.882686 w_k1_MAT_s013
                                 + 1.02817 w_k1_MSFT_s013
                                 + 0.90696 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.822599 w_k1_ADM_s013
                                 + 1.29797 w_k1_AXP_s013
                                 + 1.28987 w_k1_DIS_s013
                                 + 1.01594 w_k1_FDX_s013
                                 + 1.34277 w_k1_IBM_s013
                                 + 1.18373 w_k1_INTC_s013 + 1.4503 w_k1_KO_s013
                                 + 1.17133 w_k1_MAT_s013
                                 + 1.07221 w_k1_MSFT_s013
                                 + 1.04647 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.02987 w_k1_ADM_s013
                                 + 1.06577 w_k1_AXP_s013
                                 + 0.752015 w_k1_DIS_s013
                                 + 0.859561 w_k1_FDX_s013
                                 + 0.73462 w_k1_IBM_s013
                                 + 0.996661 w_k1_INTC_s013
                                 + 0.806243 w_k1_KO_s013
                                 + 0.817292 w_k1_MAT_s013
                                 + 0.928193 w_k1_MSFT_s013
                                 + 0.970891 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.970736 w_k1_ADM_s013
                                 + 0.74693 w_k1_AXP_s013
                                 + 0.572317 w_k1_DIS_s013
                                 + 0.803522 w_k1_FDX_s013
                                 + 0.57488 w_k1_IBM_s013
                                 + 0.899592 w_k1_INTC_s013
                                 + 0.68719 w_k1_KO_s013
                                 + 0.437402 w_k1_MAT_s013
                                 + 0.808287 w_k1_MSFT_s013
                                 + 0.740061 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.30982 w_k1_ADM_s013
                                 + 1.58622 w_k1_AXP_s013
                                 + 1.03629 w_k1_DIS_s013
                                 + 1.23867 w_k1_FDX_s013
                                 + 0.745024 w_k1_IBM_s013
                                 + 1.29292 w_k1_INTC_s013
                                 + 1.10485 w_k1_KO_s013 + 1.07431 w_k1_MAT_s013
                                 + 1.32234 w_k1_MSFT_s013
                                 + 0.970118 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.16275 w_k1_ADM_s014
                                 + 1.31371 w_k1_AXP_s014
                                 + 0.907614 w_k1_DIS_s014
                                 + 0.997717 w_k1_FDX_s014
                                 + 1.1116 w_k1_IBM_s014
                                 + 0.828294 w_k1_INTC_s014
                                 + 1.06832 w_k1_KO_s014 + 1.88883 w_k1_MAT_s014
                                 + 0.931149 w_k1_MSFT_s014
                                 + 1.07214 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.726222 w_k1_ADM_s014
                                 + 0.579287 w_k1_AXP_s014
                                 + 0.571124 w_k1_DIS_s014
                                 + 0.695725 w_k1_FDX_s014
                                 + 0.591135 w_k1_IBM_s014
                                 + 1.03343 w_k1_INTC_s014
                                 + 0.677864 w_k1_KO_s014
                                 + 0.53517 w_k1_MAT_s014
                                 + 0.777569 w_k1_MSFT_s014
                                 + 0.717054 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.836923 w_k1_ADM_s014
                                 + 0.672792 w_k1_AXP_s014
                                 + 0.708045 w_k1_DIS_s014
                                 + 0.7931 w_k1_FDX_s014
                                 + 0.912626 w_k1_IBM_s014
                                 + 1.04877 w_k1_INTC_s014
                                 + 0.869698 w_k1_KO_s014
                                 + 0.645795 w_k1_MAT_s014
                                 + 0.971493 w_k1_MSFT_s014
                                 + 0.816549 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.897098 w_k1_ADM_s014
                                 + 0.831013 w_k1_AXP_s014
                                 + 0.823042 w_k1_DIS_s014
                                 + 0.82794 w_k1_FDX_s014
                                 + 0.773739 w_k1_IBM_s014
                                 + 0.939032 w_k1_INTC_s014
                                 + 0.826431 w_k1_KO_s014
                                 + 0.671909 w_k1_MAT_s014
                                 + 0.885651 w_k1_MSFT_s014
                                 + 0.881003 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.913474 w_k1_ADM_s014
                                 + 0.805989 w_k1_AXP_s014
                                 + 0.71434 w_k1_DIS_s014
                                 + 0.85236 w_k1_FDX_s014
                                 + 1.23045 w_k1_IBM_s014
                                 + 1.04101 w_k1_INTC_s014
                                 + 1.05382 w_k1_KO_s014 + 1.16729 w_k1_MAT_s014
                                 + 0.913649 w_k1_MSFT_s014
                                 + 0.911236 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.23037 w_k1_ADM_s014
                                 + 0.918098 w_k1_AXP_s014
                                 + 0.687047 w_k1_DIS_s014
                                 + 0.959337 w_k1_FDX_s014
                                 + 0.690441 w_k1_IBM_s014
                                 + 0.918889 w_k1_INTC_s014
                                 + 0.884606 w_k1_KO_s014
                                 + 0.785548 w_k1_MAT_s014
                                 + 0.86993 w_k1_MSFT_s014
                                 + 0.971275 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.1242 w_k1_ADM_s014
                                 + 0.803085 w_k1_AXP_s014
                                 + 1.04347 w_k1_DIS_s014
                                 + 1.07249 w_k1_FDX_s014
                                 + 0.754699 w_k1_IBM_s014
                                 + 1.25229 w_k1_INTC_s014
                                 + 0.984252 w_k1_KO_s014
                                 + 0.498596 w_k1_MAT_s014
                                 + 1.20525 w_k1_MSFT_s014
                                 + 0.955329 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.29637 w_k1_ADM_s014
                                 + 1.25756 w_k1_AXP_s014
                                 + 1.27519 w_k1_DIS_s014
                                 + 1.23126 w_k1_FDX_s014 + 1.1009 w_k1_IBM_s014
                                 + 1.50679 w_k1_INTC_s014 + 1.1143 w_k1_KO_s014
                                 + 0.926476 w_k1_MAT_s014
                                 + 1.0779 w_k1_MSFT_s014 + 1.0446 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.699293 w_k1_ADM_s015
                                 + 0.644032 w_k1_AXP_s015
                                 + 0.64711 w_k1_DIS_s015
                                 + 0.785493 w_k1_FDX_s015
                                 + 0.641511 w_k1_IBM_s015
                                 + 0.803436 w_k1_INTC_s015
                                 + 0.742558 w_k1_KO_s015
                                 + 0.512594 w_k1_MAT_s015
                                 + 0.914965 w_k1_MSFT_s015
                                 + 0.841738 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.11456 w_k1_ADM_s015
                                 + 0.879244 w_k1_AXP_s015
                                 + 1.35415 w_k1_DIS_s015
                                 + 1.19162 w_k1_FDX_s015
                                 + 0.890961 w_k1_IBM_s015
                                 + 1.26339 w_k1_INTC_s015
                                 + 1.18602 w_k1_KO_s015
                                 + 0.667336 w_k1_MAT_s015
                                 + 1.1554 w_k1_MSFT_s015 + 1.15284 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.866091 w_k1_ADM_s015
                                 + 1.028 w_k1_AXP_s015 + 0.865384 w_k1_DIS_s015
                                 + 0.976452 w_k1_FDX_s015
                                 + 0.884247 w_k1_IBM_s015
                                 + 1.23344 w_k1_INTC_s015
                                 + 0.972395 w_k1_KO_s015
                                 + 0.720194 w_k1_MAT_s015
                                 + 0.949655 w_k1_MSFT_s015
                                 + 0.945278 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.810285 w_k1_ADM_s015
                                 + 0.909492 w_k1_AXP_s015
                                 + 0.766945 w_k1_DIS_s015
                                 + 0.812001 w_k1_FDX_s015
                                 + 1.10147 w_k1_IBM_s015
                                 + 0.803356 w_k1_INTC_s015
                                 + 0.856385 w_k1_KO_s015
                                 + 0.843378 w_k1_MAT_s015
                                 + 0.9189 w_k1_MSFT_s015
                                 + 0.959634 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.961204 w_k1_ADM_s015
                                 + 0.828688 w_k1_AXP_s015
                                 + 0.792576 w_k1_DIS_s015
                                 + 1.00921 w_k1_FDX_s015
                                 + 0.654791 w_k1_IBM_s015
                                 + 0.995717 w_k1_INTC_s015
                                 + 0.978889 w_k1_KO_s015
                                 + 0.49566 w_k1_MAT_s015
                                 + 1.01686 w_k1_MSFT_s015
                                 + 0.937248 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.989272 w_k1_ADM_s015
                                 + 1.49804 w_k1_AXP_s015
                                 + 1.07859 w_k1_DIS_s015
                                 + 1.32353 w_k1_FDX_s015
                                 + 0.909167 w_k1_IBM_s015
                                 + 1.21427 w_k1_INTC_s015
                                 + 0.961471 w_k1_KO_s015
                                 + 0.958151 w_k1_MAT_s015
                                 + 1.16617 w_k1_MSFT_s015
                                 + 0.988237 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.879648 w_k1_ADM_s015
                                 + 1.37457 w_k1_AXP_s015
                                 + 1.51839 w_k1_DIS_s015
                                 + 1.42303 w_k1_FDX_s015
                                 + 1.05215 w_k1_IBM_s015
                                 + 1.35075 w_k1_INTC_s015
                                 + 1.42035 w_k1_KO_s015 + 1.16696 w_k1_MAT_s015
                                 + 1.45732 w_k1_MSFT_s015
                                 + 1.28213 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.08153 w_k1_ADM_s015
                                 + 1.22766 w_k1_AXP_s015 + 1.2979 w_k1_DIS_s015
                                 + 1.05266 w_k1_FDX_s015
                                 + 1.41592 w_k1_IBM_s015
                                 + 1.12744 w_k1_INTC_s015 + 1.479 w_k1_KO_s015
                                 + 1.13078 w_k1_MAT_s015 + 1.11 w_k1_MSFT_s015
                                 + 1.08003 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.00158 w_k1_ADM_s016
                                 + 1.08138 w_k1_AXP_s016 + 1.1476 w_k1_DIS_s016
                                 + 1.00478 w_k1_FDX_s016
                                 + 0.913429 w_k1_IBM_s016
                                 + 1.0937 w_k1_INTC_s016 + 1.04368 w_k1_KO_s016
                                 + 0.813504 w_k1_MAT_s016
                                 + 1.1972 w_k1_MSFT_s016
                                 + 0.934441 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.21498 w_k1_ADM_s016
                                 + 1.55394 w_k1_AXP_s016
                                 + 1.38948 w_k1_DIS_s016
                                 + 1.16128 w_k1_FDX_s016
                                 + 1.71098 w_k1_IBM_s016
                                 + 1.28951 w_k1_INTC_s016
                                 + 1.58367 w_k1_KO_s016 + 2.34077 w_k1_MAT_s016
                                 + 1.43249 w_k1_MSFT_s016
                                 + 1.20117 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.25162 w_k1_ADM_s016
                                 + 1.37368 w_k1_AXP_s016
                                 + 0.997556 w_k1_DIS_s016
                                 + 1.14233 w_k1_FDX_s016
                                 + 0.982644 w_k1_IBM_s016
                                 + 1.2169 w_k1_INTC_s016 + 1.043 w_k1_KO_s016
                                 + 0.863212 w_k1_MAT_s016
                                 + 1.20342 w_k1_MSFT_s016
                                 + 0.860585 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.997112 w_k1_ADM_s016
                                 + 0.846484 w_k1_AXP_s016
                                 + 0.659768 w_k1_DIS_s016
                                 + 0.844812 w_k1_FDX_s016
                                 + 0.686695 w_k1_IBM_s016
                                 + 0.87009 w_k1_INTC_s016
                                 + 0.824609 w_k1_KO_s016
                                 + 0.689035 w_k1_MAT_s016
                                 + 0.901412 w_k1_MSFT_s016
                                 + 0.799308 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.01554 w_k1_ADM_s016
                                 + 0.689621 w_k1_AXP_s016
                                 + 0.544124 w_k1_DIS_s016
                                 + 0.698923 w_k1_FDX_s016
                                 + 0.872787 w_k1_IBM_s016
                                 + 0.690232 w_k1_INTC_s016
                                 + 0.765095 w_k1_KO_s016
                                 + 0.761535 w_k1_MAT_s016
                                 + 0.841885 w_k1_MSFT_s016
                                 + 0.804705 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.4001 w_k1_ADM_s016
                                 + 1.34647 w_k1_AXP_s016
                                 + 1.62267 w_k1_DIS_s016
                                 + 1.35779 w_k1_FDX_s016
                                 + 1.01386 w_k1_IBM_s016
                                 + 1.20436 w_k1_INTC_s016
                                 + 1.28678 w_k1_KO_s016 + 1.09864 w_k1_MAT_s016
                                 + 1.55023 w_k1_MSFT_s016
                                 + 1.15321 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.06888 w_k1_ADM_s016
                                 + 1.06856 w_k1_AXP_s016
                                 + 0.818711 w_k1_DIS_s016
                                 + 0.871443 w_k1_FDX_s016
                                 + 0.993917 w_k1_IBM_s016
                                 + 0.995503 w_k1_INTC_s016
                                 + 1.06368 w_k1_KO_s016
                                 + 0.923383 w_k1_MAT_s016
                                 + 1.11533 w_k1_MSFT_s016
                                 + 0.937824 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.02263 w_k1_ADM_s016
                                 + 1.06672 w_k1_AXP_s016
                                 + 0.859159 w_k1_DIS_s016
                                 + 0.929449 w_k1_FDX_s016
                                 + 1.29496 w_k1_IBM_s016
                                 + 0.988734 w_k1_INTC_s016
                                 + 1.0233 w_k1_KO_s016 + 1.00818 w_k1_MAT_s016
                                 + 1.21232 w_k1_MSFT_s016
                                 + 0.994066 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.04482 w_k1_ADM_s017
                                 + 1.41837 w_k1_AXP_s017
                                 + 1.71637 w_k1_DIS_s017
                                 + 1.17571 w_k1_FDX_s017
                                 + 1.36117 w_k1_IBM_s017
                                 + 1.55703 w_k1_INTC_s017
                                 + 1.43438 w_k1_KO_s017
                                 + 0.967778 w_k1_MAT_s017
                                 + 1.19489 w_k1_MSFT_s017 + 1.272 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.16414 w_k1_ADM_s017
                                 + 1.56968 w_k1_AXP_s017
                                 + 0.995189 w_k1_DIS_s017
                                 + 1.06099 w_k1_FDX_s017
                                 + 1.33775 w_k1_IBM_s017
                                 + 1.03603 w_k1_INTC_s017
                                 + 1.13682 w_k1_KO_s017 + 1.46183 w_k1_MAT_s017
                                 + 0.874997 w_k1_MSFT_s017
                                 + 1.31997 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.02853 w_k1_ADM_s017
                                 + 0.869509 w_k1_AXP_s017
                                 + 1.22621 w_k1_DIS_s017
                                 + 1.12097 w_k1_FDX_s017
                                 + 0.67215 w_k1_IBM_s017
                                 + 1.05579 w_k1_INTC_s017
                                 + 1.04517 w_k1_KO_s017
                                 + 0.767295 w_k1_MAT_s017
                                 + 1.00357 w_k1_MSFT_s017
                                 + 1.11204 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.691597 w_k1_ADM_s017
                                 + 0.697488 w_k1_AXP_s017
                                 + 0.786969 w_k1_DIS_s017
                                 + 0.772133 w_k1_FDX_s017
                                 + 0.779282 w_k1_IBM_s017
                                 + 0.773556 w_k1_INTC_s017
                                 + 0.729849 w_k1_KO_s017
                                 + 0.678811 w_k1_MAT_s017
                                 + 0.803703 w_k1_MSFT_s017
                                 + 0.94503 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.952788 w_k1_ADM_s017
                                 + 1.51437 w_k1_AXP_s017
                                 + 1.81018 w_k1_DIS_s017
                                 + 1.39778 w_k1_FDX_s017
                                 + 0.848304 w_k1_IBM_s017
                                 + 1.35773 w_k1_INTC_s017 + 1.1989 w_k1_KO_s017
                                 + 0.865436 w_k1_MAT_s017
                                 + 1.39035 w_k1_MSFT_s017
                                 + 1.27249 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.09183 w_k1_ADM_s017
                                 + 0.859422 w_k1_AXP_s017
                                 + 0.869001 w_k1_DIS_s017
                                 + 0.974449 w_k1_FDX_s017
                                 + 0.759641 w_k1_IBM_s017
                                 + 0.8983 w_k1_INTC_s017
                                 + 0.976888 w_k1_KO_s017
                                 + 0.71955 w_k1_MAT_s017
                                 + 0.823241 w_k1_MSFT_s017
                                 + 1.07101 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.842002 w_k1_ADM_s017
                                 + 0.946577 w_k1_AXP_s017
                                 + 1.09084 w_k1_DIS_s017
                                 + 0.869018 w_k1_FDX_s017
                                 + 1.50887 w_k1_IBM_s017
                                 + 0.801375 w_k1_INTC_s017
                                 + 1.23581 w_k1_KO_s017 + 1.78324 w_k1_MAT_s017
                                 + 0.823004 w_k1_MSFT_s017
                                 + 1.23493 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.935628 w_k1_ADM_s017
                                 + 1.04632 w_k1_AXP_s017
                                 + 1.23098 w_k1_DIS_s017
                                 + 1.06135 w_k1_FDX_s017
                                 + 1.01761 w_k1_IBM_s017
                                 + 1.2893 w_k1_INTC_s017 + 1.19425 w_k1_KO_s017
                                 + 1.00806 w_k1_MAT_s017
                                 + 1.04315 w_k1_MSFT_s017
                                 + 1.00949 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.78435 w_k1_ADM_s100
                                 + 0.729492 w_k1_AXP_s100
                                 + 0.665469 w_k1_DIS_s100
                                 + 0.845321 w_k1_FDX_s100
                                 + 0.690549 w_k1_IBM_s100
                                 + 0.878253 w_k1_INTC_s100
                                 + 0.714278 w_k1_KO_s100
                                 + 0.628901 w_k1_MAT_s100
                                 + 0.960181 w_k1_MSFT_s100
                                 + 0.898534 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.942312 w_k1_ADM_s100
                                 + 1.10648 w_k1_AXP_s100
                                 + 1.02847 w_k1_DIS_s100
                                 + 1.13131 w_k1_FDX_s100
                                 + 0.78897 w_k1_IBM_s100
                                 + 1.15021 w_k1_INTC_s100
                                 + 0.937121 w_k1_KO_s100
                                 + 0.848456 w_k1_MAT_s100
                                 + 1.08576 w_k1_MSFT_s100
                                 + 0.960239 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.960156 w_k1_ADM_s100
                                 + 1.26978 w_k1_AXP_s100
                                 + 1.47216 w_k1_DIS_s100
                                 + 1.20719 w_k1_FDX_s100
                                 + 1.43471 w_k1_IBM_s100
                                 + 1.18838 w_k1_INTC_s100
                                 + 1.43059 w_k1_KO_s100 + 1.24718 w_k1_MAT_s100
                                 + 1.31027 w_k1_MSFT_s100
                                 + 1.13681 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.910404 w_k1_ADM_s100
                                 + 0.956604 w_k1_AXP_s100
                                 + 0.958316 w_k1_DIS_s100
                                 + 0.892043 w_k1_FDX_s100
                                 + 1.15856 w_k1_IBM_s100
                                 + 0.968252 w_k1_INTC_s100
                                 + 1.08891 w_k1_KO_s100 + 1.18898 w_k1_MAT_s100
                                 + 0.985551 w_k1_MSFT_s100
                                 + 1.0415 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.750136 w_k1_ADM_s100
                                 + 0.708825 w_k1_AXP_s100
                                 + 0.641337 w_k1_DIS_s100
                                 + 0.713542 w_k1_FDX_s100
                                 + 1.03124 w_k1_IBM_s100
                                 + 0.655261 w_k1_INTC_s100
                                 + 0.925021 w_k1_KO_s100
                                 + 0.937443 w_k1_MAT_s100
                                 + 0.852904 w_k1_MSFT_s100
                                 + 0.96076 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.873031 w_k1_ADM_s100
                                 + 1.01106 w_k1_AXP_s100
                                 + 1.67015 w_k1_DIS_s100
                                 + 1.26147 w_k1_FDX_s100
                                 + 1.02668 w_k1_IBM_s100
                                 + 1.30369 w_k1_INTC_s100
                                 + 1.32755 w_k1_KO_s100
                                 + 0.831632 w_k1_MAT_s100
                                 + 1.42434 w_k1_MSFT_s100 + 1.1538 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.955002 w_k1_ADM_s100
                                 + 1.50999 w_k1_AXP_s100
                                 + 1.33667 w_k1_DIS_s100
                                 + 1.34926 w_k1_FDX_s100
                                 + 1.11226 w_k1_IBM_s100
                                 + 1.11387 w_k1_INTC_s100
                                 + 1.14788 w_k1_KO_s100 + 1.25412 w_k1_MAT_s100
                                 + 1.22288 w_k1_MSFT_s100
                                 + 1.17421 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.02915 w_k1_ADM_s100
                                 + 1.32354 w_k1_AXP_s100
                                 + 1.03034 w_k1_DIS_s100 + 1.0284 w_k1_FDX_s100
                                 + 1.6398 w_k1_IBM_s100
                                 + 1.28308 w_k1_INTC_s100
                                 + 1.29983 w_k1_KO_s100 + 1.32127 w_k1_MAT_s100
                                 + 1.13432 w_k1_MSFT_s100 + 1.2789 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.22688 w_k1_ADM_s101
                                 + 1.04576 w_k1_AXP_s101
                                 + 1.53389 w_k1_DIS_s101
                                 + 1.40154 w_k1_FDX_s101
                                 + 0.940449 w_k1_IBM_s101
                                 + 1.27971 w_k1_INTC_s101
                                 + 1.18222 w_k1_KO_s101
                                 + 0.827444 w_k1_MAT_s101
                                 + 1.41281 w_k1_MSFT_s101
                                 + 1.25749 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.971246 w_k1_ADM_s101
                                 + 0.698376 w_k1_AXP_s101
                                 + 0.701283 w_k1_DIS_s101
                                 + 0.924282 w_k1_FDX_s101
                                 + 0.565262 w_k1_IBM_s101
                                 + 0.985754 w_k1_INTC_s101
                                 + 0.798811 w_k1_KO_s101
                                 + 0.520131 w_k1_MAT_s101
                                 + 0.905168 w_k1_MSFT_s101
                                 + 0.923131 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.96633 w_k1_ADM_s101
                                 + 0.869933 w_k1_AXP_s101
                                 + 0.911773 w_k1_DIS_s101
                                 + 0.819405 w_k1_FDX_s101
                                 + 1.21446 w_k1_IBM_s101
                                 + 0.928245 w_k1_INTC_s101
                                 + 1.06918 w_k1_KO_s101
                                 + 0.934398 w_k1_MAT_s101
                                 + 0.86494 w_k1_MSFT_s101
                                 + 1.03874 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.883124 w_k1_ADM_s101
                                 + 0.730812 w_k1_AXP_s101
                                 + 0.819213 w_k1_DIS_s101
                                 + 0.787801 w_k1_FDX_s101
                                 + 0.884923 w_k1_IBM_s101
                                 + 0.937654 w_k1_INTC_s101
                                 + 0.831209 w_k1_KO_s101
                                 + 0.606715 w_k1_MAT_s101
                                 + 0.89061 w_k1_MSFT_s101
                                 + 0.964927 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.991224 w_k1_ADM_s101
                                 + 1.29352 w_k1_AXP_s101 + 1.1479 w_k1_DIS_s101
                                 + 1.17276 w_k1_FDX_s101
                                 + 1.11811 w_k1_IBM_s101
                                 + 1.16279 w_k1_INTC_s101
                                 + 1.08217 w_k1_KO_s101 + 1.22648 w_k1_MAT_s101
                                 + 1.08282 w_k1_MSFT_s101
                                 + 1.23808 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.70335 w_k1_ADM_s101
                                 + 0.536774 w_k1_AXP_s101
                                 + 0.70679 w_k1_DIS_s101
                                 + 0.702758 w_k1_FDX_s101
                                 + 0.870407 w_k1_IBM_s101
                                 + 0.737222 w_k1_INTC_s101
                                 + 0.84649 w_k1_KO_s101
                                 + 0.902247 w_k1_MAT_s101
                                 + 0.75124 w_k1_MSFT_s101
                                 + 0.755211 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.00496 w_k1_ADM_s101
                                 + 0.943182 w_k1_AXP_s101
                                 + 1.15583 w_k1_DIS_s101
                                 + 1.10915 w_k1_FDX_s101
                                 + 0.827116 w_k1_IBM_s101
                                 + 0.984187 w_k1_INTC_s101
                                 + 1.05772 w_k1_KO_s101
                                 + 0.643572 w_k1_MAT_s101
                                 + 1.01414 w_k1_MSFT_s101
                                 + 1.07405 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.17131 w_k1_ADM_s101
                                 + 1.12969 w_k1_AXP_s101
                                 + 1.33182 w_k1_DIS_s101
                                 + 1.01936 w_k1_FDX_s101
                                 + 1.52794 w_k1_IBM_s101
                                 + 1.10522 w_k1_INTC_s101 + 1.5127 w_k1_KO_s101
                                 + 1.3912 w_k1_MAT_s101
                                 + 1.13528 w_k1_MSFT_s101
                                 + 1.38471 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.5239 w_k1_ADM_s102
                                 + 1.24979 w_k1_AXP_s102
                                 + 1.77422 w_k1_DIS_s102
                                 + 1.26672 w_k1_FDX_s102
                                 + 1.55762 w_k1_IBM_s102
                                 + 1.11822 w_k1_INTC_s102
                                 + 1.37411 w_k1_KO_s102 + 1.57428 w_k1_MAT_s102
                                 + 1.17764 w_k1_MSFT_s102
                                 + 1.40212 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.839348 w_k1_ADM_s102
                                 + 1.22322 w_k1_AXP_s102 + 1.0949 w_k1_DIS_s102
                                 + 0.972922 w_k1_FDX_s102
                                 + 1.6513 w_k1_IBM_s102
                                 + 1.26745 w_k1_INTC_s102
                                 + 1.14963 w_k1_KO_s102 + 1.81927 w_k1_MAT_s102
                                 + 0.99063 w_k1_MSFT_s102
                                 + 1.25523 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.13089 w_k1_ADM_s102
                                 + 1.05497 w_k1_AXP_s102
                                 + 0.875075 w_k1_DIS_s102
                                 + 0.977352 w_k1_FDX_s102
                                 + 0.909799 w_k1_IBM_s102
                                 + 0.885628 w_k1_INTC_s102
                                 + 0.806914 w_k1_KO_s102
                                 + 0.877543 w_k1_MAT_s102
                                 + 0.786524 w_k1_MSFT_s102
                                 + 0.945934 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.958847 w_k1_ADM_s102
                                 + 0.703128 w_k1_AXP_s102
                                 + 0.862216 w_k1_DIS_s102
                                 + 0.960949 w_k1_FDX_s102
                                 + 0.563722 w_k1_IBM_s102
                                 + 1.14565 w_k1_INTC_s102
                                 + 0.718473 w_k1_KO_s102
                                 + 0.383277 w_k1_MAT_s102
                                 + 0.986016 w_k1_MSFT_s102
                                 + 0.939577 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.0768 w_k1_ADM_s102
                                 + 1.22653 w_k1_AXP_s102 + 1.0223 w_k1_DIS_s102
                                 + 1.19897 w_k1_FDX_s102
                                 + 0.647453 w_k1_IBM_s102
                                 + 0.796396 w_k1_INTC_s102
                                 + 0.736405 w_k1_KO_s102
                                 + 0.674742 w_k1_MAT_s102
                                 + 0.908903 w_k1_MSFT_s102
                                 + 1.08442 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.26385 w_k1_ADM_s102
                                 + 1.6989 w_k1_AXP_s102 + 1.4494 w_k1_DIS_s102
                                 + 1.48685 w_k1_FDX_s102
                                 + 1.17948 w_k1_IBM_s102
                                 + 1.10311 w_k1_INTC_s102
                                 + 1.01882 w_k1_KO_s102 + 1.1904 w_k1_MAT_s102
                                 + 1.20075 w_k1_MSFT_s102
                                 + 1.20875 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.961417 w_k1_ADM_s102
                                 + 1.09447 w_k1_AXP_s102
                                 + 0.844041 w_k1_DIS_s102
                                 + 0.865057 w_k1_FDX_s102
                                 + 1.25791 w_k1_IBM_s102
                                 + 0.870341 w_k1_INTC_s102
                                 + 0.910839 w_k1_KO_s102
                                 + 1.30385 w_k1_MAT_s102
                                 + 0.791648 w_k1_MSFT_s102
                                 + 0.956781 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.939531 w_k1_ADM_s102
                                 + 1.03587 w_k1_AXP_s102
                                 + 1.55145 w_k1_DIS_s102
                                 + 1.20067 w_k1_FDX_s102
                                 + 0.957483 w_k1_IBM_s102
                                 + 1.16811 w_k1_INTC_s102
                                 + 1.22914 w_k1_KO_s102
                                 + 0.903878 w_k1_MAT_s102
                                 + 1.17542 w_k1_MSFT_s102
                                 + 1.11563 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.26196 w_k1_ADM_s103
                                 + 1.25813 w_k1_AXP_s103
                                 + 1.53597 w_k1_DIS_s103
                                 + 1.16917 w_k1_FDX_s103
                                 + 1.10622 w_k1_IBM_s103
                                 + 1.20555 w_k1_INTC_s103
                                 + 1.52045 w_k1_KO_s103 + 1.07348 w_k1_MAT_s103
                                 + 1.33124 w_k1_MSFT_s103
                                 + 1.38955 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.829548 w_k1_ADM_s103
                                 + 1.05576 w_k1_AXP_s103
                                 + 1.12376 w_k1_DIS_s103
                                 + 0.939038 w_k1_FDX_s103
                                 + 1.83065 w_k1_IBM_s103
                                 + 1.00096 w_k1_INTC_s103
                                 + 1.31685 w_k1_KO_s103 + 1.54511 w_k1_MAT_s103
                                 + 0.985256 w_k1_MSFT_s103
                                 + 1.12965 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.810262 w_k1_ADM_s103
                                 + 0.7779 w_k1_AXP_s103
                                 + 0.737168 w_k1_DIS_s103
                                 + 0.684664 w_k1_FDX_s103
                                 + 0.963309 w_k1_IBM_s103
                                 + 0.785792 w_k1_INTC_s103
                                 + 0.890143 w_k1_KO_s103
                                 + 0.801853 w_k1_MAT_s103
                                 + 0.77436 w_k1_MSFT_s103
                                 + 1.04599 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.678526 w_k1_ADM_s103
                                 + 0.544213 w_k1_AXP_s103
                                 + 0.703961 w_k1_DIS_s103
                                 + 0.733026 w_k1_FDX_s103
                                 + 0.733655 w_k1_IBM_s103
                                 + 0.838536 w_k1_INTC_s103
                                 + 0.834425 w_k1_KO_s103
                                 + 0.585348 w_k1_MAT_s103
                                 + 0.906308 w_k1_MSFT_s103
                                 + 0.921666 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.705204 w_k1_ADM_s103
                                 + 0.94597 w_k1_AXP_s103
                                 + 0.944158 w_k1_DIS_s103
                                 + 0.886191 w_k1_FDX_s103
                                 + 1.32009 w_k1_IBM_s103
                                 + 1.00285 w_k1_INTC_s103 + 1.1072 w_k1_KO_s103
                                 + 1.15762 w_k1_MAT_s103
                                 + 1.04748 w_k1_MSFT_s103
                                 + 1.12951 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.00359 w_k1_ADM_s103
                                 + 1.02406 w_k1_AXP_s103
                                 + 0.885151 w_k1_DIS_s103
                                 + 0.930647 w_k1_FDX_s103
                                 + 0.945206 w_k1_IBM_s103
                                 + 1.07121 w_k1_INTC_s103
                                 + 0.882581 w_k1_KO_s103
                                 + 0.759328 w_k1_MAT_s103
                                 + 0.893125 w_k1_MSFT_s103
                                 + 1.0577 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.777004 w_k1_ADM_s103
                                 + 0.899681 w_k1_AXP_s103
                                 + 0.808987 w_k1_DIS_s103
                                 + 0.96079 w_k1_FDX_s103
                                 + 0.596816 w_k1_IBM_s103
                                 + 0.953564 w_k1_INTC_s103
                                 + 0.74996 w_k1_KO_s103 + 0.40823 w_k1_MAT_s103
                                 + 0.954382 w_k1_MSFT_s103
                                 + 0.991748 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.908929 w_k1_ADM_s103
                                 + 0.977027 w_k1_AXP_s103
                                 + 1.11799 w_k1_DIS_s103
                                 + 1.03501 w_k1_FDX_s103
                                 + 0.82975 w_k1_IBM_s103
                                 + 1.1708 w_k1_INTC_s103 + 1.08949 w_k1_KO_s103
                                 + 0.866723 w_k1_MAT_s103
                                 + 1.06423 w_k1_MSFT_s103
                                 + 1.19078 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.05834 w_k1_ADM_s104
                                 + 0.84411 w_k1_AXP_s104
                                 + 1.08842 w_k1_DIS_s104
                                 + 1.05932 w_k1_FDX_s104
                                 + 0.676147 w_k1_IBM_s104
                                 + 1.17522 w_k1_INTC_s104
                                 + 0.82065 w_k1_KO_s104
                                 + 0.503457 w_k1_MAT_s104
                                 + 1.12634 w_k1_MSFT_s104
                                 + 0.962971 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.788953 w_k1_ADM_s104
                                 + 0.670726 w_k1_AXP_s104
                                 + 0.776704 w_k1_DIS_s104
                                 + 0.794873 w_k1_FDX_s104
                                 + 0.655987 w_k1_IBM_s104
                                 + 1.00552 w_k1_INTC_s104
                                 + 0.764986 w_k1_KO_s104
                                 + 0.538638 w_k1_MAT_s104
                                 + 0.941084 w_k1_MSFT_s104
                                 + 0.910557 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.798256 w_k1_ADM_s104
                                 + 0.590525 w_k1_AXP_s104
                                 + 0.765365 w_k1_DIS_s104
                                 + 0.725226 w_k1_FDX_s104
                                 + 0.835177 w_k1_IBM_s104
                                 + 0.858016 w_k1_INTC_s104
                                 + 0.806691 w_k1_KO_s104
                                 + 0.712569 w_k1_MAT_s104
                                 + 0.809076 w_k1_MSFT_s104
                                 + 0.883218 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.994515 w_k1_ADM_s104
                                 + 0.816509 w_k1_AXP_s104
                                 + 1.2539 w_k1_DIS_s104
                                 + 0.992489 w_k1_FDX_s104
                                 + 0.97287 w_k1_IBM_s104
                                 + 1.26197 w_k1_INTC_s104
                                 + 1.25395 w_k1_KO_s104 + 1.02846 w_k1_MAT_s104
                                 + 1.13491 w_k1_MSFT_s104
                                 + 1.11541 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.27335 w_k1_ADM_s104
                                 + 1.43345 w_k1_AXP_s104 + 1.3073 w_k1_DIS_s104
                                 + 1.31595 w_k1_FDX_s104
                                 + 0.874662 w_k1_IBM_s104
                                 + 1.13856 w_k1_INTC_s104
                                 + 0.955102 w_k1_KO_s104
                                 + 0.962372 w_k1_MAT_s104
                                 + 1.13629 w_k1_MSFT_s104
                                 + 1.06477 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.989009 w_k1_ADM_s104
                                 + 1.0942 w_k1_AXP_s104 + 1.3308 w_k1_DIS_s104
                                 + 1.09592 w_k1_FDX_s104
                                 + 1.44782 w_k1_IBM_s104
                                 + 1.03315 w_k1_INTC_s104
                                 + 1.19077 w_k1_KO_s104 + 1.90284 w_k1_MAT_s104
                                 + 1.01157 w_k1_MSFT_s104
                                 + 1.10231 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.1411 w_k1_ADM_s104
                                 + 0.992831 w_k1_AXP_s104
                                 + 0.77292 w_k1_DIS_s104
                                 + 0.948675 w_k1_FDX_s104
                                 + 0.950857 w_k1_IBM_s104
                                 + 0.898754 w_k1_INTC_s104
                                 + 0.830963 w_k1_KO_s104
                                 + 0.969077 w_k1_MAT_s104
                                 + 0.879445 w_k1_MSFT_s104
                                 + 0.921752 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.718599 w_k1_ADM_s104
                                 + 0.72219 w_k1_AXP_s104
                                 + 0.732961 w_k1_DIS_s104
                                 + 0.750964 w_k1_FDX_s104
                                 + 1.09499 w_k1_IBM_s104
                                 + 0.908256 w_k1_INTC_s104
                                 + 0.74859 w_k1_KO_s104
                                 + 0.873737 w_k1_MAT_s104
                                 + 0.86201 w_k1_MSFT_s104
                                 + 0.83376 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.951196 w_k1_ADM_s105
                                 + 1.10318 w_k1_AXP_s105
                                 + 1.08892 w_k1_DIS_s105
                                 + 1.09371 w_k1_FDX_s105
                                 + 0.73372 w_k1_IBM_s105
                                 + 1.17929 w_k1_INTC_s105
                                 + 1.06668 w_k1_KO_s105
                                 + 0.741903 w_k1_MAT_s105
                                 + 1.16005 w_k1_MSFT_s105 + 1.0938 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.778844 w_k1_ADM_s105
                                 + 1.21463 w_k1_AXP_s105
                                 + 1.11321 w_k1_DIS_s105
                                 + 0.857385 w_k1_FDX_s105
                                 + 1.67033 w_k1_IBM_s105
                                 + 0.820905 w_k1_INTC_s105
                                 + 1.40822 w_k1_KO_s105 + 2.08856 w_k1_MAT_s105
                                 + 1.00632 w_k1_MSFT_s105
                                 + 0.966722 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.21666 w_k1_ADM_s105
                                 + 0.900806 w_k1_AXP_s105
                                 + 0.822192 w_k1_DIS_s105
                                 + 0.892115 w_k1_FDX_s105
                                 + 0.619069 w_k1_IBM_s105
                                 + 0.956744 w_k1_INTC_s105
                                 + 0.860229 w_k1_KO_s105
                                 + 0.509934 w_k1_MAT_s105
                                 + 1.02676 w_k1_MSFT_s105
                                 + 0.931291 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.823026 w_k1_ADM_s105
                                 + 0.868378 w_k1_AXP_s105
                                 + 0.74813 w_k1_DIS_s105
                                 + 0.792433 w_k1_FDX_s105
                                 + 1.01767 w_k1_IBM_s105
                                 + 0.814366 w_k1_INTC_s105
                                 + 0.976848 w_k1_KO_s105
                                 + 1.14054 w_k1_MAT_s105
                                 + 0.870242 w_k1_MSFT_s105
                                 + 0.885925 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.980043 w_k1_ADM_s105
                                 + 1.21346 w_k1_AXP_s105
                                 + 1.61813 w_k1_DIS_s105
                                 + 1.20464 w_k1_FDX_s105
                                 + 1.15099 w_k1_IBM_s105
                                 + 1.38544 w_k1_INTC_s105
                                 + 1.54647 w_k1_KO_s105 + 1.1224 w_k1_MAT_s105
                                 + 1.50225 w_k1_MSFT_s105
                                 + 1.34623 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.55455 w_k1_ADM_s105
                                 + 1.23048 w_k1_AXP_s105 + 1.4033 w_k1_DIS_s105
                                 + 1.36132 w_k1_FDX_s105
                                 + 0.797477 w_k1_IBM_s105
                                 + 1.07655 w_k1_INTC_s105
                                 + 1.09276 w_k1_KO_s105
                                 + 0.816519 w_k1_MAT_s105
                                 + 1.27345 w_k1_MSFT_s105
                                 + 1.15878 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.884458 w_k1_ADM_s105
                                 + 1.23914 w_k1_AXP_s105
                                 + 1.02094 w_k1_DIS_s105
                                 + 1.03885 w_k1_FDX_s105
                                 + 1.14334 w_k1_IBM_s105
                                 + 1.04193 w_k1_INTC_s105
                                 + 1.16377 w_k1_KO_s105 + 1.18195 w_k1_MAT_s105
                                 + 1.03927 w_k1_MSFT_s105
                                 + 1.10119 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.15395 w_k1_ADM_s105
                                 + 1.04668 w_k1_AXP_s105
                                 + 1.22056 w_k1_DIS_s105
                                 + 1.02617 w_k1_FDX_s105
                                 + 1.43837 w_k1_IBM_s105
                                 + 1.02745 w_k1_INTC_s105
                                 + 1.31316 w_k1_KO_s105 + 1.75474 w_k1_MAT_s105
                                 + 1.09622 w_k1_MSFT_s105
                                 + 1.12864 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.712701 w_k1_ADM_s106
                                 + 1.13301 w_k1_AXP_s106
                                 + 1.19581 w_k1_DIS_s106
                                 + 0.992804 w_k1_FDX_s106
                                 + 1.83956 w_k1_IBM_s106
                                 + 1.1051 w_k1_INTC_s106 + 1.43966 w_k1_KO_s106
                                 + 2.32556 w_k1_MAT_s106
                                 + 1.09587 w_k1_MSFT_s106
                                 + 1.10975 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.925629 w_k1_ADM_s106
                                 + 1.33601 w_k1_AXP_s106
                                 + 1.19799 w_k1_DIS_s106
                                 + 1.15207 w_k1_FDX_s106
                                 + 1.25948 w_k1_IBM_s106
                                 + 1.11076 w_k1_INTC_s106
                                 + 1.33079 w_k1_KO_s106 + 1.66684 w_k1_MAT_s106
                                 + 1.26783 w_k1_MSFT_s106 + 1.3174 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.12557 w_k1_ADM_s106
                                 + 1.1389 w_k1_AXP_s106 + 1.70849 w_k1_DIS_s106
                                 + 1.34594 w_k1_FDX_s106
                                 + 0.829492 w_k1_IBM_s106
                                 + 1.00229 w_k1_INTC_s106 + 1.179 w_k1_KO_s106
                                 + 0.885318 w_k1_MAT_s106
                                 + 1.35821 w_k1_MSFT_s106
                                 + 1.06927 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.984596 w_k1_ADM_s106
                                 + 0.977073 w_k1_AXP_s106
                                 + 1.07424 w_k1_DIS_s106
                                 + 0.924682 w_k1_FDX_s106
                                 + 1.1446 w_k1_IBM_s106
                                 + 0.97886 w_k1_INTC_s106
                                 + 0.945142 w_k1_KO_s106
                                 + 1.02173 w_k1_MAT_s106
                                 + 1.02711 w_k1_MSFT_s106
                                 + 0.923525 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.784211 w_k1_ADM_s106
                                 + 0.704533 w_k1_AXP_s106
                                 + 0.791372 w_k1_DIS_s106
                                 + 0.800337 w_k1_FDX_s106
                                 + 0.753771 w_k1_IBM_s106
                                 + 0.924655 w_k1_INTC_s106
                                 + 0.82509 w_k1_KO_s106
                                 + 0.606445 w_k1_MAT_s106
                                 + 0.870522 w_k1_MSFT_s106
                                 + 0.800521 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.17474 w_k1_ADM_s106
                                 + 1.66368 w_k1_AXP_s106
                                 + 1.70654 w_k1_DIS_s106
                                 + 1.30622 w_k1_FDX_s106
                                 + 1.31928 w_k1_IBM_s106
                                 + 1.28807 w_k1_INTC_s106
                                 + 1.34883 w_k1_KO_s106 + 1.33664 w_k1_MAT_s106
                                 + 1.15521 w_k1_MSFT_s106 + 1.1611 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.694011 w_k1_ADM_s106
                                 + 0.839313 w_k1_AXP_s106
                                 + 0.763658 w_k1_DIS_s106
                                 + 0.771533 w_k1_FDX_s106
                                 + 1.21098 w_k1_IBM_s106
                                 + 0.691844 w_k1_INTC_s106
                                 + 0.959471 w_k1_KO_s106 + 1.066 w_k1_MAT_s106
                                 + 0.765536 w_k1_MSFT_s106
                                 + 0.862207 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.772562 w_k1_ADM_s106
                                 + 1.16696 w_k1_AXP_s106
                                 + 1.03648 w_k1_DIS_s106
                                 + 1.09144 w_k1_FDX_s106
                                 + 0.857311 w_k1_IBM_s106
                                 + 1.21073 w_k1_INTC_s106
                                 + 0.939912 w_k1_KO_s106
                                 + 0.714499 w_k1_MAT_s106
                                 + 1.09708 w_k1_MSFT_s106
                                 + 1.05027 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.10957 w_k1_ADM_s107
                                 + 1.17789 w_k1_AXP_s107
                                 + 1.08151 w_k1_DIS_s107
                                 + 1.16452 w_k1_FDX_s107
                                 + 0.762439 w_k1_IBM_s107
                                 + 0.986867 w_k1_INTC_s107
                                 + 0.932864 w_k1_KO_s107
                                 + 1.56816 w_k1_MAT_s107
                                 + 1.14407 w_k1_MSFT_s107 + 1.0172 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.24186 w_k1_ADM_s107
                                 + 1.32141 w_k1_AXP_s107 + 1.8596 w_k1_DIS_s107
                                 + 1.36581 w_k1_FDX_s107
                                 + 1.05141 w_k1_IBM_s107
                                 + 1.25155 w_k1_INTC_s107
                                 + 1.35011 w_k1_KO_s107 + 1.33652 w_k1_MAT_s107
                                 + 1.22811 w_k1_MSFT_s107
                                 + 1.23954 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.970722 w_k1_ADM_s107
                                 + 1.22645 w_k1_AXP_s107
                                 + 0.995027 w_k1_DIS_s107
                                 + 0.86684 w_k1_FDX_s107
                                 + 1.36049 w_k1_IBM_s107
                                 + 1.04847 w_k1_INTC_s107
                                 + 1.17615 w_k1_KO_s107 + 1.48727 w_k1_MAT_s107
                                 + 0.877894 w_k1_MSFT_s107
                                 + 0.97927 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.763702 w_k1_ADM_s107
                                 + 0.723983 w_k1_AXP_s107
                                 + 0.737892 w_k1_DIS_s107
                                 + 0.756664 w_k1_FDX_s107
                                 + 0.829183 w_k1_IBM_s107
                                 + 0.78719 w_k1_INTC_s107
                                 + 0.840302 w_k1_KO_s107
                                 + 1.14522 w_k1_MAT_s107
                                 + 0.717417 w_k1_MSFT_s107
                                 + 0.864535 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.953691 w_k1_ADM_s107
                                 + 0.940431 w_k1_AXP_s107
                                 + 1.18136 w_k1_DIS_s107
                                 + 1.00299 w_k1_FDX_s107
                                 + 1.10487 w_k1_IBM_s107
                                 + 0.95193 w_k1_INTC_s107
                                 + 1.05529 w_k1_KO_s107 + 1.18661 w_k1_MAT_s107
                                 + 0.981326 w_k1_MSFT_s107 + 1.221 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.01288 w_k1_ADM_s107
                                 + 1.22282 w_k1_AXP_s107
                                 + 0.922603 w_k1_DIS_s107
                                 + 0.940004 w_k1_FDX_s107
                                 + 0.738409 w_k1_IBM_s107
                                 + 0.98013 w_k1_INTC_s107
                                 + 0.854679 w_k1_KO_s107
                                 + 1.05183 w_k1_MAT_s107
                                 + 0.956677 w_k1_MSFT_s107
                                 + 0.939852 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.866966 w_k1_ADM_s107
                                 + 0.884176 w_k1_AXP_s107
                                 + 0.822328 w_k1_DIS_s107
                                 + 0.872083 w_k1_FDX_s107
                                 + 1.03471 w_k1_IBM_s107
                                 + 1.14711 w_k1_INTC_s107
                                 + 1.13767 w_k1_KO_s107 + 1.36227 w_k1_MAT_s107
                                 + 0.992239 w_k1_MSFT_s107
                                 + 0.992502 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.1027 w_k1_ADM_s107
                                 + 1.65822 w_k1_AXP_s107
                                 + 1.37052 w_k1_DIS_s107
                                 + 1.13939 w_k1_FDX_s107
                                 + 1.30679 w_k1_IBM_s107
                                 + 1.31852 w_k1_INTC_s107
                                 + 1.45247 w_k1_KO_s107 + 2.04957 w_k1_MAT_s107
                                 + 0.937681 w_k1_MSFT_s107
                                 + 1.28176 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.780422 w_k1_ADM_s110
                                 + 0.648822 w_k1_AXP_s110
                                 + 0.606956 w_k1_DIS_s110
                                 + 0.75742 w_k1_FDX_s110
                                 + 0.785913 w_k1_IBM_s110
                                 + 0.824273 w_k1_INTC_s110
                                 + 0.745328 w_k1_KO_s110
                                 + 0.712916 w_k1_MAT_s110
                                 + 0.77581 w_k1_MSFT_s110
                                 + 0.753602 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.899587 w_k1_ADM_s110
                                 + 0.973173 w_k1_AXP_s110
                                 + 0.99916 w_k1_DIS_s110
                                 + 0.943853 w_k1_FDX_s110
                                 + 1.26081 w_k1_IBM_s110
                                 + 0.922509 w_k1_INTC_s110
                                 + 1.11845 w_k1_KO_s110 + 1.16973 w_k1_MAT_s110
                                 + 0.882832 w_k1_MSFT_s110
                                 + 1.03147 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.01615 w_k1_ADM_s110
                                 + 1.18311 w_k1_AXP_s110
                                 + 1.17261 w_k1_DIS_s110
                                 + 1.00015 w_k1_FDX_s110
                                 + 1.95016 w_k1_IBM_s110
                                 + 0.947264 w_k1_INTC_s110
                                 + 1.37294 w_k1_KO_s110 + 1.76417 w_k1_MAT_s110
                                 + 1.05708 w_k1_MSFT_s110
                                 + 1.17934 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.21514 w_k1_ADM_s110
                                 + 1.4293 w_k1_AXP_s110 + 1.63367 w_k1_DIS_s110
                                 + 1.36092 w_k1_FDX_s110
                                 + 1.12852 w_k1_IBM_s110 + 1.501 w_k1_INTC_s110
                                 + 1.38009 w_k1_KO_s110
                                 + 0.975585 w_k1_MAT_s110
                                 + 1.44956 w_k1_MSFT_s110
                                 + 1.29918 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.981595 w_k1_ADM_s110
                                 + 0.906615 w_k1_AXP_s110
                                 + 0.818229 w_k1_DIS_s110
                                 + 0.994882 w_k1_FDX_s110
                                 + 0.628804 w_k1_IBM_s110
                                 + 0.896811 w_k1_INTC_s110
                                 + 0.784598 w_k1_KO_s110
                                 + 0.709153 w_k1_MAT_s110
                                 + 0.965136 w_k1_MSFT_s110
                                 + 0.96181 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.34571 w_k1_ADM_s110
                                 + 1.06982 w_k1_AXP_s110
                                 + 1.46187 w_k1_DIS_s110
                                 + 1.39927 w_k1_FDX_s110
                                 + 0.724429 w_k1_IBM_s110
                                 + 1.37438 w_k1_INTC_s110
                                 + 1.10262 w_k1_KO_s110 + 0.58864 w_k1_MAT_s110
                                 + 1.31613 w_k1_MSFT_s110
                                 + 1.13898 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.765659 w_k1_ADM_s110
                                 + 0.846132 w_k1_AXP_s110
                                 + 0.921393 w_k1_DIS_s110
                                 + 0.93133 w_k1_FDX_s110
                                 + 0.929047 w_k1_IBM_s110
                                 + 0.931128 w_k1_INTC_s110
                                 + 0.975746 w_k1_KO_s110
                                 + 0.88586 w_k1_MAT_s110
                                 + 0.893073 w_k1_MSFT_s110
                                 + 0.980657 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.09589 w_k1_ADM_s110
                                 + 1.3116 w_k1_AXP_s110 + 1.09767 w_k1_DIS_s110
                                 + 1.24287 w_k1_FDX_s110
                                 + 0.76733 w_k1_IBM_s110
                                 + 1.09865 w_k1_INTC_s110
                                 + 0.870371 w_k1_KO_s110
                                 + 0.735937 w_k1_MAT_s110
                                 + 1.15625 w_k1_MSFT_s110
                                 + 1.10107 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.01981 w_k1_ADM_s111
                                 + 1.08235 w_k1_AXP_s111
                                 + 1.03637 w_k1_DIS_s111
                                 + 0.962364 w_k1_FDX_s111
                                 + 1.15604 w_k1_IBM_s111
                                 + 0.937664 w_k1_INTC_s111
                                 + 1.09785 w_k1_KO_s111 + 1.124 w_k1_MAT_s111
                                 + 0.85968 w_k1_MSFT_s111
                                 + 0.985475 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.03846 w_k1_ADM_s111
                                 + 0.925176 w_k1_AXP_s111
                                 + 1.00147 w_k1_DIS_s111
                                 + 0.981328 w_k1_FDX_s111
                                 + 0.775155 w_k1_IBM_s111
                                 + 0.978602 w_k1_INTC_s111
                                 + 0.96614 w_k1_KO_s111
                                 + 0.669503 w_k1_MAT_s111
                                 + 1.02986 w_k1_MSFT_s111
                                 + 0.978277 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.896649 w_k1_ADM_s111
                                 + 1.11217 w_k1_AXP_s111
                                 + 0.948497 w_k1_DIS_s111
                                 + 0.866818 w_k1_FDX_s111
                                 + 1.51892 w_k1_IBM_s111
                                 + 0.982539 w_k1_INTC_s111
                                 + 1.30791 w_k1_KO_s111 + 1.88572 w_k1_MAT_s111
                                 + 0.881861 w_k1_MSFT_s111
                                 + 1.04697 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.92693 w_k1_ADM_s111
                                 + 0.772387 w_k1_AXP_s111
                                 + 0.855369 w_k1_DIS_s111
                                 + 0.829408 w_k1_FDX_s111
                                 + 1.1199 w_k1_IBM_s111
                                 + 0.999985 w_k1_INTC_s111
                                 + 1.16016 w_k1_KO_s111
                                 + 0.814444 w_k1_MAT_s111
                                 + 0.938603 w_k1_MSFT_s111
                                 + 0.934177 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.14141 w_k1_ADM_s111
                                 + 1.20679 w_k1_AXP_s111
                                 + 1.17371 w_k1_DIS_s111
                                 + 1.14156 w_k1_FDX_s111
                                 + 1.04076 w_k1_IBM_s111
                                 + 1.28626 w_k1_INTC_s111
                                 + 1.26828 w_k1_KO_s111
                                 + 0.745722 w_k1_MAT_s111
                                 + 1.27603 w_k1_MSFT_s111
                                 + 1.08439 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.777894 w_k1_ADM_s111
                                 + 0.681943 w_k1_AXP_s111
                                 + 0.651489 w_k1_DIS_s111
                                 + 0.752283 w_k1_FDX_s111
                                 + 0.68724 w_k1_IBM_s111
                                 + 0.870862 w_k1_INTC_s111
                                 + 0.744282 w_k1_KO_s111
                                 + 0.51789 w_k1_MAT_s111
                                 + 0.758178 w_k1_MSFT_s111
                                 + 0.763424 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.22922 w_k1_ADM_s111
                                 + 1.09598 w_k1_AXP_s111
                                 + 1.63154 w_k1_DIS_s111
                                 + 1.38084 w_k1_FDX_s111
                                 + 0.735328 w_k1_IBM_s111
                                 + 1.42923 w_k1_INTC_s111
                                 + 1.15425 w_k1_KO_s111
                                 + 0.637446 w_k1_MAT_s111
                                 + 1.22082 w_k1_MSFT_s111
                                 + 1.09618 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.3284 w_k1_ADM_s111
                                 + 1.56108 w_k1_AXP_s111
                                 + 1.75467 w_k1_DIS_s111
                                 + 1.36412 w_k1_FDX_s111
                                 + 1.44639 w_k1_IBM_s111
                                 + 1.30941 w_k1_INTC_s111
                                 + 1.64853 w_k1_KO_s111 + 1.32483 w_k1_MAT_s111
                                 + 1.38612 w_k1_MSFT_s111
                                 + 1.29638 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.36436 w_k1_ADM_s112
                                 + 1.63896 w_k1_AXP_s112
                                 + 1.51835 w_k1_DIS_s112
                                 + 1.29644 w_k1_FDX_s112
                                 + 1.82595 w_k1_IBM_s112
                                 + 1.40183 w_k1_INTC_s112
                                 + 1.65686 w_k1_KO_s112 + 2.06178 w_k1_MAT_s112
                                 + 1.20861 w_k1_MSFT_s112
                                 + 1.34186 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17805 w_k1_ADM_s112
                                 + 1.28952 w_k1_AXP_s112
                                 + 1.18625 w_k1_DIS_s112
                                 + 1.25225 w_k1_FDX_s112
                                 + 0.949872 w_k1_IBM_s112
                                 + 1.15892 w_k1_INTC_s112
                                 + 1.21567 w_k1_KO_s112 + 1.13186 w_k1_MAT_s112
                                 + 1.12622 w_k1_MSFT_s112
                                 + 1.13422 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.80797 w_k1_ADM_s112
                                 + 0.722352 w_k1_AXP_s112
                                 + 0.60711 w_k1_DIS_s112
                                 + 0.758031 w_k1_FDX_s112
                                 + 0.899292 w_k1_IBM_s112
                                 + 0.882324 w_k1_INTC_s112
                                 + 0.860087 w_k1_KO_s112 + 1.1323 w_k1_MAT_s112
                                 + 0.766039 w_k1_MSFT_s112
                                 + 0.82007 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.891044 w_k1_ADM_s112
                                 + 0.906307 w_k1_AXP_s112
                                 + 0.793375 w_k1_DIS_s112
                                 + 0.843085 w_k1_FDX_s112
                                 + 1.46224 w_k1_IBM_s112
                                 + 0.895983 w_k1_INTC_s112
                                 + 1.07384 w_k1_KO_s112 + 1.3322 w_k1_MAT_s112
                                 + 0.771625 w_k1_MSFT_s112
                                 + 0.943521 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.917115 w_k1_ADM_s112
                                 + 0.951333 w_k1_AXP_s112
                                 + 1.14523 w_k1_DIS_s112 + 1.0196 w_k1_FDX_s112
                                 + 1.27393 w_k1_IBM_s112
                                 + 1.05686 w_k1_INTC_s112
                                 + 1.24786 w_k1_KO_s112 + 1.3018 w_k1_MAT_s112
                                 + 1.02407 w_k1_MSFT_s112
                                 + 1.05011 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.12663 w_k1_ADM_s112
                                 + 0.989945 w_k1_AXP_s112
                                 + 0.813535 w_k1_DIS_s112
                                 + 1.04411 w_k1_FDX_s112
                                 + 0.578601 w_k1_IBM_s112
                                 + 0.988369 w_k1_INTC_s112
                                 + 0.77409 w_k1_KO_s112
                                 + 0.651391 w_k1_MAT_s112
                                 + 0.959458 w_k1_MSFT_s112
                                 + 0.94712 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.88972 w_k1_ADM_s112
                                 + 0.670882 w_k1_AXP_s112
                                 + 0.814732 w_k1_DIS_s112
                                 + 0.892604 w_k1_FDX_s112
                                 + 0.661381 w_k1_IBM_s112
                                 + 0.85879 w_k1_INTC_s112
                                 + 0.814293 w_k1_KO_s112
                                 + 0.613224 w_k1_MAT_s112
                                 + 0.910597 w_k1_MSFT_s112
                                 + 0.880863 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.09758 w_k1_ADM_s112
                                 + 0.961082 w_k1_AXP_s112
                                 + 1.56283 w_k1_DIS_s112
                                 + 1.33393 w_k1_FDX_s112
                                 + 1.03351 w_k1_IBM_s112
                                 + 1.10943 w_k1_INTC_s112
                                 + 1.41606 w_k1_KO_s112 + 1.12889 w_k1_MAT_s112
                                 + 1.39453 w_k1_MSFT_s112
                                 + 1.21911 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.856664 w_k1_ADM_s113
                                 + 0.897151 w_k1_AXP_s113
                                 + 0.974332 w_k1_DIS_s113
                                 + 0.955333 w_k1_FDX_s113
                                 + 1.07886 w_k1_IBM_s113
                                 + 1.0093 w_k1_INTC_s113
                                 + 0.889159 w_k1_KO_s113
                                 + 1.00157 w_k1_MAT_s113
                                 + 0.935279 w_k1_MSFT_s113
                                 + 1.01639 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.27439 w_k1_ADM_s113
                                 + 1.50765 w_k1_AXP_s113
                                 + 1.59982 w_k1_DIS_s113
                                 + 1.39248 w_k1_FDX_s113
                                 + 1.54337 w_k1_IBM_s113
                                 + 1.39217 w_k1_INTC_s113
                                 + 1.28428 w_k1_KO_s113 + 2.05672 w_k1_MAT_s113
                                 + 1.32687 w_k1_MSFT_s113
                                 + 1.28294 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.887987 w_k1_ADM_s113
                                 + 0.707029 w_k1_AXP_s113
                                 + 0.630597 w_k1_DIS_s113
                                 + 0.719602 w_k1_FDX_s113
                                 + 1.01553 w_k1_IBM_s113
                                 + 0.815289 w_k1_INTC_s113
                                 + 0.784058 w_k1_KO_s113
                                 + 0.968299 w_k1_MAT_s113
                                 + 0.792264 w_k1_MSFT_s113
                                 + 0.911936 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.995106 w_k1_ADM_s113
                                 + 1.22744 w_k1_AXP_s113
                                 + 1.32846 w_k1_DIS_s113
                                 + 1.01554 w_k1_FDX_s113
                                 + 1.48344 w_k1_IBM_s113
                                 + 0.953173 w_k1_INTC_s113
                                 + 1.11899 w_k1_KO_s113 + 1.6617 w_k1_MAT_s113
                                 + 1.0735 w_k1_MSFT_s113 + 1.1331 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.895916 w_k1_ADM_s113
                                 + 0.981242 w_k1_AXP_s113
                                 + 0.866002 w_k1_DIS_s113
                                 + 0.831949 w_k1_FDX_s113
                                 + 1.25105 w_k1_IBM_s113
                                 + 0.990593 w_k1_INTC_s113
                                 + 0.939315 w_k1_KO_s113 + 1.4523 w_k1_MAT_s113
                                 + 0.84352 w_k1_MSFT_s113
                                 + 0.960183 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.23755 w_k1_ADM_s113
                                 + 1.07734 w_k1_AXP_s113
                                 + 1.42791 w_k1_DIS_s113
                                 + 1.24537 w_k1_FDX_s113
                                 + 0.925559 w_k1_IBM_s113
                                 + 1.19675 w_k1_INTC_s113
                                 + 1.05601 w_k1_KO_s113 + 0.99565 w_k1_MAT_s113
                                 + 1.18795 w_k1_MSFT_s113
                                 + 1.10174 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.08716 w_k1_ADM_s113
                                 + 0.966142 w_k1_AXP_s113
                                 + 0.820381 w_k1_DIS_s113
                                 + 0.933202 w_k1_FDX_s113
                                 + 0.763609 w_k1_IBM_s113
                                 + 0.967827 w_k1_INTC_s113
                                 + 0.720955 w_k1_KO_s113
                                 + 0.658001 w_k1_MAT_s113
                                 + 0.853667 w_k1_MSFT_s113
                                 + 0.858283 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.41445 w_k1_ADM_s113
                                 + 1.29908 w_k1_AXP_s113 + 1.0742 w_k1_DIS_s113
                                 + 1.11074 w_k1_FDX_s113
                                 + 1.12367 w_k1_IBM_s113
                                 + 1.16421 w_k1_INTC_s113
                                 + 1.11459 w_k1_KO_s113 + 1.36217 w_k1_MAT_s113
                                 + 1.11397 w_k1_MSFT_s113
                                 + 1.16421 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.03201 w_k1_ADM_s114
                                 + 1.26169 w_k1_AXP_s114
                                 + 1.07424 w_k1_DIS_s114 + 1.0834 w_k1_FDX_s114
                                 + 1.33833 w_k1_IBM_s114
                                 + 1.23623 w_k1_INTC_s114
                                 + 1.14972 w_k1_KO_s114
                                 + 0.982545 w_k1_MAT_s114
                                 + 0.919858 w_k1_MSFT_s114
                                 + 1.09012 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.04571 w_k1_ADM_s114
                                 + 0.969655 w_k1_AXP_s114
                                 + 0.982214 w_k1_DIS_s114
                                 + 0.973178 w_k1_FDX_s114
                                 + 0.963473 w_k1_IBM_s114
                                 + 1.04523 w_k1_INTC_s114
                                 + 1.08348 w_k1_KO_s114
                                 + 0.728025 w_k1_MAT_s114
                                 + 0.975566 w_k1_MSFT_s114
                                 + 1.01406 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.840908 w_k1_ADM_s114
                                 + 0.961934 w_k1_AXP_s114
                                 + 1.21801 w_k1_DIS_s114
                                 + 1.02795 w_k1_FDX_s114
                                 + 1.20246 w_k1_IBM_s114
                                 + 1.25599 w_k1_INTC_s114
                                 + 1.13992 w_k1_KO_s114
                                 + 0.821238 w_k1_MAT_s114
                                 + 1.10576 w_k1_MSFT_s114
                                 + 1.05955 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.7388 w_k1_ADM_s114
                                 + 0.736358 w_k1_AXP_s114
                                 + 0.665707 w_k1_DIS_s114
                                 + 0.776353 w_k1_FDX_s114
                                 + 0.775874 w_k1_IBM_s114
                                 + 0.907118 w_k1_INTC_s114
                                 + 0.791943 w_k1_KO_s114
                                 + 0.504466 w_k1_MAT_s114
                                 + 0.714807 w_k1_MSFT_s114
                                 + 0.763004 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.964391 w_k1_ADM_s114
                                 + 1.34505 w_k1_AXP_s114
                                 + 1.33469 w_k1_DIS_s114
                                 + 1.22734 w_k1_FDX_s114 + 1.3777 w_k1_IBM_s114
                                 + 1.14025 w_k1_INTC_s114 + 1.2258 w_k1_KO_s114
                                 + 1.17089 w_k1_MAT_s114
                                 + 1.21958 w_k1_MSFT_s114
                                 + 1.07692 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.48279 w_k1_ADM_s114
                                 + 1.37326 w_k1_AXP_s114
                                 + 1.89735 w_k1_DIS_s114
                                 + 1.51649 w_k1_FDX_s114
                                 + 1.28011 w_k1_IBM_s114
                                 + 1.49701 w_k1_INTC_s114
                                 + 1.46632 w_k1_KO_s114
                                 + 0.950885 w_k1_MAT_s114
                                 + 1.42359 w_k1_MSFT_s114 + 1.2064 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.703591 w_k1_ADM_s114
                                 + 0.915337 w_k1_AXP_s114
                                 + 0.707042 w_k1_DIS_s114
                                 + 0.825098 w_k1_FDX_s114
                                 + 1.2262 w_k1_IBM_s114
                                 + 1.05544 w_k1_INTC_s114
                                 + 0.910321 w_k1_KO_s114
                                 + 1.07002 w_k1_MAT_s114
                                 + 0.792562 w_k1_MSFT_s114
                                 + 0.950675 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.980009 w_k1_ADM_s114
                                 + 1.10437 w_k1_AXP_s114 + 0.9787 w_k1_DIS_s114
                                 + 1.17034 w_k1_FDX_s114
                                 + 0.873377 w_k1_IBM_s114
                                 + 1.23059 w_k1_INTC_s114
                                 + 1.09625 w_k1_KO_s114
                                 + 0.727849 w_k1_MAT_s114
                                 + 1.08991 w_k1_MSFT_s114
                                 + 1.00061 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.875561 w_k1_ADM_s115
                                 + 0.855061 w_k1_AXP_s115
                                 + 0.76258 w_k1_DIS_s115
                                 + 0.856619 w_k1_FDX_s115
                                 + 0.975642 w_k1_IBM_s115
                                 + 0.929983 w_k1_INTC_s115
                                 + 0.887918 w_k1_KO_s115
                                 + 1.10332 w_k1_MAT_s115
                                 + 0.856161 w_k1_MSFT_s115
                                 + 0.929092 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.14659 w_k1_ADM_s115
                                 + 1.11513 w_k1_AXP_s115
                                 + 1.13872 w_k1_DIS_s115
                                 + 1.10584 w_k1_FDX_s115
                                 + 0.905128 w_k1_IBM_s115
                                 + 0.893331 w_k1_INTC_s115
                                 + 1.06748 w_k1_KO_s115 + 1.12924 w_k1_MAT_s115
                                 + 1.07906 w_k1_MSFT_s115 + 1.0675 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.93406 w_k1_ADM_s115
                                 + 0.751441 w_k1_AXP_s115
                                 + 1.02921 w_k1_DIS_s115
                                 + 0.933311 w_k1_FDX_s115
                                 + 0.806633 w_k1_IBM_s115
                                 + 1.25722 w_k1_INTC_s115
                                 + 0.973676 w_k1_KO_s115
                                 + 0.505294 w_k1_MAT_s115
                                 + 1.09863 w_k1_MSFT_s115
                                 + 0.928524 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.960779 w_k1_ADM_s115
                                 + 1.32676 w_k1_AXP_s115
                                 + 0.967608 w_k1_DIS_s115
                                 + 1.01596 w_k1_FDX_s115
                                 + 1.17207 w_k1_IBM_s115
                                 + 1.03323 w_k1_INTC_s115
                                 + 0.986191 w_k1_KO_s115
                                 + 1.00183 w_k1_MAT_s115
                                 + 0.971985 w_k1_MSFT_s115
                                 + 0.951977 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.20685 w_k1_ADM_s115
                                 + 1.11306 w_k1_AXP_s115
                                 + 0.992586 w_k1_DIS_s115
                                 + 0.935206 w_k1_FDX_s115
                                 + 1.44265 w_k1_IBM_s115
                                 + 0.896976 w_k1_INTC_s115
                                 + 1.20022 w_k1_KO_s115 + 1.34535 w_k1_MAT_s115
                                 + 0.985062 w_k1_MSFT_s115
                                 + 1.08142 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.743452 w_k1_ADM_s115
                                 + 0.585891 w_k1_AXP_s115
                                 + 0.609668 w_k1_DIS_s115
                                 + 0.712931 w_k1_FDX_s115
                                 + 0.696145 w_k1_IBM_s115
                                 + 0.914989 w_k1_INTC_s115
                                 + 0.773666 w_k1_KO_s115
                                 + 0.590619 w_k1_MAT_s115
                                 + 0.832397 w_k1_MSFT_s115
                                 + 0.762031 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.913654 w_k1_ADM_s115
                                 + 1.50595 w_k1_AXP_s115
                                 + 1.52544 w_k1_DIS_s115
                                 + 1.26332 w_k1_FDX_s115
                                 + 1.57833 w_k1_IBM_s115
                                 + 1.19844 w_k1_INTC_s115
                                 + 1.29873 w_k1_KO_s115 + 1.74745 w_k1_MAT_s115
                                 + 1.21738 w_k1_MSFT_s115
                                 + 1.21835 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.45223 w_k1_ADM_s115
                                 + 1.2207 w_k1_AXP_s115 + 1.72996 w_k1_DIS_s115
                                 + 1.32653 w_k1_FDX_s115
                                 + 1.15542 w_k1_IBM_s115
                                 + 1.22388 w_k1_INTC_s115 + 1.4762 w_k1_KO_s115
                                 + 0.761401 w_k1_MAT_s115
                                 + 1.39018 w_k1_MSFT_s115
                                 + 1.16275 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.798687 w_k1_ADM_s116
                                 + 0.800826 w_k1_AXP_s116
                                 + 0.706944 w_k1_DIS_s116
                                 + 0.742825 w_k1_FDX_s116
                                 + 0.88335 w_k1_IBM_s116
                                 + 0.680207 w_k1_INTC_s116
                                 + 0.793757 w_k1_KO_s116 + 1.0688 w_k1_MAT_s116
                                 + 0.742192 w_k1_MSFT_s116
                                 + 0.856204 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.849289 w_k1_ADM_s116
                                 + 0.721448 w_k1_AXP_s116
                                 + 1.15496 w_k1_DIS_s116
                                 + 0.980099 w_k1_FDX_s116
                                 + 0.731635 w_k1_IBM_s116
                                 + 0.858097 w_k1_INTC_s116
                                 + 0.792588 w_k1_KO_s116
                                 + 0.727481 w_k1_MAT_s116
                                 + 1.15459 w_k1_MSFT_s116
                                 + 0.955077 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.986904 w_k1_ADM_s116
                                 + 0.807299 w_k1_AXP_s116
                                 + 0.907933 w_k1_DIS_s116
                                 + 0.921778 w_k1_FDX_s116
                                 + 0.901408 w_k1_IBM_s116
                                 + 0.945104 w_k1_INTC_s116
                                 + 0.961432 w_k1_KO_s116
                                 + 1.00275 w_k1_MAT_s116
                                 + 1.01895 w_k1_MSFT_s116
                                 + 0.949618 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.02469 w_k1_ADM_s116
                                 + 1.1198 w_k1_AXP_s116 + 1.1637 w_k1_DIS_s116
                                 + 1.04441 w_k1_FDX_s116
                                 + 1.02805 w_k1_IBM_s116
                                 + 0.85388 w_k1_INTC_s116
                                 + 1.10336 w_k1_KO_s116 + 1.16798 w_k1_MAT_s116
                                 + 1.14293 w_k1_MSFT_s116
                                 + 1.11924 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.13392 w_k1_ADM_s116
                                 + 1.21325 w_k1_AXP_s116
                                 + 1.41598 w_k1_DIS_s116
                                 + 1.22485 w_k1_FDX_s116
                                 + 1.09885 w_k1_IBM_s116
                                 + 1.17912 w_k1_INTC_s116
                                 + 1.23647 w_k1_KO_s116 + 1.16248 w_k1_MAT_s116
                                 + 1.30415 w_k1_MSFT_s116
                                 + 1.09288 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.09031 w_k1_ADM_s116
                                 + 1.16836 w_k1_AXP_s116
                                 + 0.980861 w_k1_DIS_s116
                                 + 1.09351 w_k1_FDX_s116
                                 + 0.996451 w_k1_IBM_s116
                                 + 1.1194 w_k1_INTC_s116 + 1.04716 w_k1_KO_s116
                                 + 1.12539 w_k1_MAT_s116
                                 + 1.13084 w_k1_MSFT_s116 + 1.087 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.25354 w_k1_ADM_s116
                                 + 1.41171 w_k1_AXP_s116
                                 + 1.23888 w_k1_DIS_s116
                                 + 1.07828 w_k1_FDX_s116
                                 + 1.47175 w_k1_IBM_s116
                                 + 0.987686 w_k1_INTC_s116
                                 + 1.36822 w_k1_KO_s116 + 1.95841 w_k1_MAT_s116
                                 + 1.12328 w_k1_MSFT_s116
                                 + 1.11959 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.683792 w_k1_ADM_s116
                                 + 0.712848 w_k1_AXP_s116
                                 + 0.784276 w_k1_DIS_s116
                                 + 0.702326 w_k1_FDX_s116
                                 + 1.1312 w_k1_IBM_s116
                                 + 0.788984 w_k1_INTC_s116
                                 + 0.964461 w_k1_KO_s116
                                 + 1.22609 w_k1_MAT_s116
                                 + 0.799896 w_k1_MSFT_s116
                                 + 0.871623 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.11217 w_k1_ADM_s117
                                 + 1.42423 w_k1_AXP_s117
                                 + 1.37924 w_k1_DIS_s117
                                 + 1.31416 w_k1_FDX_s117
                                 + 0.959033 w_k1_IBM_s117
                                 + 1.331 w_k1_INTC_s117 + 1.15428 w_k1_KO_s117
                                 + 1.15358 w_k1_MAT_s117
                                 + 1.26992 w_k1_MSFT_s117
                                 + 1.17272 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.760781 w_k1_ADM_s117
                                 + 0.471627 w_k1_AXP_s117
                                 + 0.473997 w_k1_DIS_s117
                                 + 0.64417 w_k1_FDX_s117
                                 + 0.761147 w_k1_IBM_s117
                                 + 0.83932 w_k1_INTC_s117
                                 + 0.730595 w_k1_KO_s117
                                 + 0.619032 w_k1_MAT_s117
                                 + 0.751676 w_k1_MSFT_s117
                                 + 0.787554 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.832634 w_k1_ADM_s117
                                 + 0.765006 w_k1_AXP_s117
                                 + 0.798367 w_k1_DIS_s117
                                 + 0.883387 w_k1_FDX_s117
                                 + 0.641516 w_k1_IBM_s117
                                 + 0.796589 w_k1_INTC_s117
                                 + 0.782541 w_k1_KO_s117
                                 + 0.661716 w_k1_MAT_s117
                                 + 0.92666 w_k1_MSFT_s117
                                 + 0.93459 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.745203 w_k1_ADM_s117
                                 + 0.740772 w_k1_AXP_s117
                                 + 0.917681 w_k1_DIS_s117
                                 + 0.932034 w_k1_FDX_s117
                                 + 0.786054 w_k1_IBM_s117
                                 + 0.97275 w_k1_INTC_s117
                                 + 0.862441 w_k1_KO_s117
                                 + 0.927914 w_k1_MAT_s117
                                 + 1.00022 w_k1_MSFT_s117
                                 + 0.931019 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.06158 w_k1_ADM_s117
                                 + 1.0287 w_k1_AXP_s117 + 1.23015 w_k1_DIS_s117
                                 + 1.02913 w_k1_FDX_s117 + 1.5196 w_k1_IBM_s117
                                 + 0.964836 w_k1_INTC_s117
                                 + 1.07687 w_k1_KO_s117 + 1.51795 w_k1_MAT_s117
                                 + 0.902783 w_k1_MSFT_s117
                                 + 1.05856 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.964828 w_k1_ADM_s117
                                 + 0.692574 w_k1_AXP_s117
                                 + 0.965949 w_k1_DIS_s117
                                 + 0.988232 w_k1_FDX_s117
                                 + 0.98578 w_k1_IBM_s117
                                 + 1.08392 w_k1_INTC_s117
                                 + 1.06484 w_k1_KO_s117
                                 + 0.883003 w_k1_MAT_s117
                                 + 0.970854 w_k1_MSFT_s117
                                 + 1.06482 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.756917 w_k1_ADM_s117
                                 + 0.916496 w_k1_AXP_s117
                                 + 0.729343 w_k1_DIS_s117
                                 + 0.871935 w_k1_FDX_s117
                                 + 0.928392 w_k1_IBM_s117
                                 + 0.955497 w_k1_INTC_s117
                                 + 0.865018 w_k1_KO_s117
                                 + 1.01205 w_k1_MAT_s117
                                 + 0.924627 w_k1_MSFT_s117
                                 + 0.876741 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.31764 w_k1_ADM_s117
                                 + 1.17465 w_k1_AXP_s117
                                 + 1.37044 w_k1_DIS_s117
                                 + 1.19645 w_k1_FDX_s117
                                 + 1.18867 w_k1_IBM_s117
                                 + 1.00097 w_k1_INTC_s117
                                 + 1.18598 w_k1_KO_s117 + 1.38419 w_k1_MAT_s117
                                 + 1.18831 w_k1_MSFT_s117
                                 + 1.05745 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0254235575669548 w_k1_ADM_s000
                                 + 0.0160524372757813 w_k1_ADM_s001
                                 + 0.0348256691407353 w_k1_ADM_s002
                                 + 0.0179884291192392 w_k1_ADM_s003
                                 + 0.0278872901764837 w_k1_ADM_s004
                                 + 0.0120396880800335 w_k1_ADM_s005
                                 + 0.0112583930849967 w_k1_ADM_s006
                                 + 0.00950831320382381 w_k1_ADM_s007
                                 + 0.0173697448541416 w_k1_ADM_s010
                                 + 0.0142594785340235 w_k1_ADM_s011
                                 + 0.00422889821205132 w_k1_ADM_s012
                                 + 0.00696775560575638 w_k1_ADM_s013
                                 + 0.0119213855939855 w_k1_ADM_s014
                                 + 0.0074054667100188 w_k1_ADM_s015
                                 + 0.0215975773459572 w_k1_ADM_s016
                                 + 0.0087676659430161 w_k1_ADM_s017
                                 + 0.0311338747382147 w_k1_ADM_s100
                                 + 0.0143654676246748 w_k1_ADM_s101
                                 + 0.0307596207316963 w_k1_ADM_s102
                                 + 0.023532823373917 w_k1_ADM_s103
                                 + 0.0450939030248957 w_k1_ADM_s104
                                 + 0.0105221214055887 w_k1_ADM_s105
                                 + 0.0167669683065771 w_k1_ADM_s106
                                 + 0.0109489212201637 w_k1_ADM_s107
                                 + 0.10137678880942 w_k1_ADM_s110
                                 + 0.122484112264369 w_k1_ADM_s111
                                 + 0.0976547373074116 w_k1_ADM_s112
                                 + 0.0390896343677191 w_k1_ADM_s113
                                 + 0.0468895980214763 w_k1_ADM_s114
                                 + 0.0701950880262812 w_k1_ADM_s115
                                 + 0.0416394635841125 w_k1_ADM_s116
                                 + 0.0509750611519772 w_k1_ADM_s117
                                 + 0.0284818617120376 w_k1_AXP_s000
                                 + 0.0168822936918176 w_k1_AXP_s001
                                 + 0.0304217455613191 w_k1_AXP_s002
                                 + 0.0174105169597993 w_k1_AXP_s003
                                 + 0.0295211562056066 w_k1_AXP_s004
                                 + 0.0103251348134999 w_k1_AXP_s005
                                 + 0.0110064915610169 w_k1_AXP_s006
                                 + 0.00953836606014796 w_k1_AXP_s007
                                 + 0.0191374519116309 w_k1_AXP_s010
                                 + 0.0183416764804858 w_k1_AXP_s011
                                 + 0.00456379287892343 w_k1_AXP_s012
                                 + 0.00867605907474884 w_k1_AXP_s013
                                 + 0.0109320655326231 w_k1_AXP_s014
                                 + 0.00850360486158394 w_k1_AXP_s015
                                 + 0.0218052984977776 w_k1_AXP_s016
                                 + 0.00954994557605036 w_k1_AXP_s017
                                 + 0.03593107010633 w_k1_AXP_s100
                                 + 0.0133142365483556 w_k1_AXP_s101
                                 + 0.0320180726087866 w_k1_AXP_s102
                                 + 0.0252488247263542 w_k1_AXP_s103
                                 + 0.042758811120361 w_k1_AXP_s104
                                 + 0.0112433213425912 w_k1_AXP_s105
                                 + 0.0203064606213056 w_k1_AXP_s106
                                 + 0.0120788465674403 w_k1_AXP_s107
                                 + 0.106373828205928 w_k1_AXP_s110
                                 + 0.12136013923162 w_k1_AXP_s111
                                 + 0.0953002386502569 w_k1_AXP_s112
                                 + 0.0378161106236623 w_k1_AXP_s113
                                 + 0.0524981067898968 w_k1_AXP_s114
                                 + 0.068315289419957 w_k1_AXP_s115
                                 + 0.0418617504300709 w_k1_AXP_s116
                                 + 0.0481542977346494 w_k1_AXP_s117
                                 + 0.0288344972751967 w_k1_DIS_s000
                                 + 0.0159913924002188 w_k1_DIS_s001
                                 + 0.0288871761950229 w_k1_DIS_s002
                                 + 0.0170790280119608 w_k1_DIS_s003
                                 + 0.027327320796941 w_k1_DIS_s004
                                 + 0.0102623129223179 w_k1_DIS_s005
                                 + 0.0107372555470932 w_k1_DIS_s006
                                 + 0.00882278537690768 w_k1_DIS_s007
                                 + 0.0201165219448117 w_k1_DIS_s010
                                 + 0.0184273540800799 w_k1_DIS_s011
                                 + 0.00413780356151071 w_k1_DIS_s012
                                 + 0.00701588254394759 w_k1_DIS_s013
                                 + 0.0102641350425123 w_k1_DIS_s014
                                 + 0.00819616979551134 w_k1_DIS_s015
                                 + 0.0185354169380435 w_k1_DIS_s016
                                 + 0.0105675053915857 w_k1_DIS_s017
                                 + 0.0355512254006476 w_k1_DIS_s100
                                 + 0.0152833136896926 w_k1_DIS_s101
                                 + 0.0308165399390089 w_k1_DIS_s102
                                 + 0.0270954579115915 w_k1_DIS_s103
                                 + 0.0502504988941759 w_k1_DIS_s104
                                 + 0.0110837063274981 w_k1_DIS_s105
                                 + 0.0210762018622137 w_k1_DIS_s106
                                 + 0.012186559234603 w_k1_DIS_s107
                                 + 0.109945185020743 w_k1_DIS_s110
                                 + 0.127518023034417 w_k1_DIS_s111
                                 + 0.0976417649641268 w_k1_DIS_s112
                                 + 0.037894653952741 w_k1_DIS_s113
                                 + 0.0513553309073854 w_k1_DIS_s114
                                 + 0.0707923105761994 w_k1_DIS_s115
                                 + 0.0433591811503996 w_k1_DIS_s116
                                 + 0.0545939457908436 w_k1_DIS_s117
                                 + 0.0285351203122152 w_k1_FDX_s000
                                 + 0.0162118408956063 w_k1_FDX_s001
                                 + 0.0309394868274249 w_k1_FDX_s002
                                 + 0.0160561548354661 w_k1_FDX_s003
                                 + 0.0266982082875506 w_k1_FDX_s004
                                 + 0.00984482079912185 w_k1_FDX_s005
                                 + 0.0113595520432463 w_k1_FDX_s006
                                 + 0.00886493893229966 w_k1_FDX_s007
                                 + 0.0198807271856523 w_k1_FDX_s010
                                 + 0.0169303593286572 w_k1_FDX_s011
                                 + 0.00395202711438528 w_k1_FDX_s012
                                 + 0.00714644473154394 w_k1_FDX_s013
                                 + 0.0109509390654689 w_k1_FDX_s014
                                 + 0.00833634332485949 w_k1_FDX_s015
                                 + 0.0191895098399194 w_k1_FDX_s016
                                 + 0.00938628027219041 w_k1_FDX_s017
                                 + 0.0358093117636305 w_k1_FDX_s100
                                 + 0.0144145373328953 w_k1_FDX_s101
                                 + 0.030569440244365 w_k1_FDX_s102
                                 + 0.0246316240351915 w_k1_FDX_s103
                                 + 0.0457377071886118 w_k1_FDX_s104
                                 + 0.0105198475564669 w_k1_FDX_s105
                                 + 0.0193693194859577 w_k1_FDX_s106
                                 + 0.0110521343544569 w_k1_FDX_s107
                                 + 0.108949801265855 w_k1_FDX_s110
                                 + 0.119900490213783 w_k1_FDX_s111
                                 + 0.10000513725606 w_k1_FDX_s112
                                 + 0.0368345038500345 w_k1_FDX_s113
                                 + 0.0516816098597467 w_k1_FDX_s114
                                 + 0.067026044408234 w_k1_FDX_s115
                                 + 0.0404246615455803 w_k1_FDX_s116
                                 + 0.0523365980130157 w_k1_FDX_s117
                                 + 0.029347247532678 w_k1_IBM_s000
                                 + 0.0167113681894243 w_k1_IBM_s001
                                 + 0.030293108413465 w_k1_IBM_s002
                                 + 0.0167969229418658 w_k1_IBM_s003
                                 + 0.028235034383758 w_k1_IBM_s004
                                 + 0.0100706903537803 w_k1_IBM_s005
                                 + 0.0110408329933429 w_k1_IBM_s006
                                 + 0.00912827495489101 w_k1_IBM_s007
                                 + 0.0197579139970566 w_k1_IBM_s010
                                 + 0.0177361934841841 w_k1_IBM_s011
                                 + 0.00395291335130705 w_k1_IBM_s012
                                 + 0.00701663682436434 w_k1_IBM_s013
                                 + 0.0108912265305703 w_k1_IBM_s014
                                 + 0.00803888415820526 w_k1_IBM_s015
                                 + 0.0199424952167524 w_k1_IBM_s016
                                 + 0.00944120167124275 w_k1_IBM_s017
                                 + 0.0349307104798248 w_k1_IBM_s100
                                 + 0.0143306038442977 w_k1_IBM_s101
                                 + 0.029869889268552 w_k1_IBM_s102
                                 + 0.0270288412668173 w_k1_IBM_s103
                                 + 0.046300279923239 w_k1_IBM_s104
                                 + 0.0108304877321278 w_k1_IBM_s105
                                 + 0.0215237374519116 w_k1_IBM_s106
                                 + 0.0113665448445268 w_k1_IBM_s107
                                 + 0.112617524158176 w_k1_IBM_s110
                                 + 0.116165382545412 w_k1_IBM_s111
                                 + 0.0980203802747958 w_k1_IBM_s112
                                 + 0.0393172755012764 w_k1_IBM_s113
                                 + 0.055759570116355 w_k1_IBM_s114
                                 + 0.0699728483727462 w_k1_IBM_s115
                                 + 0.0429943416280096 w_k1_IBM_s116
                                 + 0.0528010440197478 w_k1_IBM_s117
                                 + 0.0277277743332462 w_k1_INTC_s000
                                 + 0.0149736702816129 w_k1_INTC_s001
                                 + 0.0331699358761424 w_k1_INTC_s002
                                 + 0.0149076975535499 w_k1_INTC_s003
                                 + 0.0276311131535615 w_k1_INTC_s004
                                 + 0.0104586312614567 w_k1_INTC_s005
                                 + 0.0110644190470255 w_k1_INTC_s006
                                 + 0.00928847922232432 w_k1_INTC_s007
                                 + 0.0182899213322583 w_k1_INTC_s010
                                 + 0.0165826004119626 w_k1_INTC_s011
                                 + 0.00428352914864225 w_k1_INTC_s012
                                 + 0.00759098780823774 w_k1_INTC_s013
                                 + 0.0126296298784244 w_k1_INTC_s014
                                 + 0.00857559448664211 w_k1_INTC_s015
                                 + 0.0201400476444138 w_k1_INTC_s016
                                 + 0.00955697912587651 w_k1_INTC_s017
                                 + 0.0366222765704757 w_k1_INTC_s100
                                 + 0.0146874177921513 w_k1_INTC_s101
                                 + 0.0291685022299632 w_k1_INTC_s102
                                 + 0.0270037592494738 w_k1_INTC_s103
                                 + 0.0484035723897351 w_k1_INTC_s104
                                 + 0.0107225185732884 w_k1_INTC_s105
                                 + 0.0200330492535851 w_k1_INTC_s106
                                 + 0.0114835760364396 w_k1_INTC_s107
                                 + 0.108024231284865 w_k1_INTC_s110
                                 + 0.126760059975862 w_k1_INTC_s111
                                 + 0.0989340113718589 w_k1_INTC_s112
                                 + 0.0382663416673739 w_k1_INTC_s113
                                 + 0.05692681093502 w_k1_INTC_s114
                                 + 0.0675280252419187 w_k1_INTC_s115
                                 + 0.0384841285430221 w_k1_INTC_s116
                                 + 0.0509308433221243 w_k1_INTC_s117
                                 + 0.0264998512938828 w_k1_KO_s000
                                 + 0.0165015085528838 w_k1_KO_s001
                                 + 0.0281839996673287 w_k1_KO_s002
                                 + 0.0156563334353092 w_k1_KO_s003
                                 + 0.0286647842456328 w_k1_KO_s004
                                 + 0.0101784702814613 w_k1_KO_s005
                                 + 0.0106976020407911 w_k1_KO_s006
                                 + 0.0100619608264928 w_k1_KO_s007
                                 + 0.0200092268152488 w_k1_KO_s010
                                 + 0.0160421772227879 w_k1_KO_s011
                                 + 0.00399419102819108 w_k1_KO_s012
                                 + 0.00756023243852367 w_k1_KO_s013
                                 + 0.0110004673916675 w_k1_KO_s014
                                 + 0.00860893921464457 w_k1_KO_s015
                                 + 0.0202540277583504 w_k1_KO_s016
                                 + 0.0100726964820003 w_k1_KO_s017
                                 + 0.0356529971191905 w_k1_KO_s100
                                 + 0.0151061464311564 w_k1_KO_s101
                                 + 0.026958491743967 w_k1_KO_s102
                                 + 0.0285301856589562 w_k1_KO_s103
                                 + 0.045464199936838 w_k1_KO_s104
                                 + 0.0119356289054196 w_k1_KO_s105
                                 + 0.0206887119598844 w_k1_KO_s106
                                 + 0.0119908496161625 w_k1_KO_s107
                                 + 0.111212329034746 w_k1_KO_s110
                                 + 0.13164434308809 w_k1_KO_s111
                                 + 0.104925734131847 w_k1_KO_s112
                                 + 0.0343729541005889 w_k1_KO_s113
                                 + 0.0535051778518639 w_k1_KO_s114
                                 + 0.0711601828078895 w_k1_KO_s115
                                 + 0.0434393367321403 w_k1_KO_s116
                                 + 0.0514293295901072 w_k1_KO_s117
                                 + 0.0318722382914642 w_k1_MAT_s000
                                 + 0.0198323246816849 w_k1_MAT_s001
                                 + 0.0252400333738711 w_k1_MAT_s002
                                 + 0.0162802003127724 w_k1_MAT_s003
                                 + 0.0327600690492846 w_k1_MAT_s004
                                 + 0.00995288812918618 w_k1_MAT_s005
                                 + 0.0104889151284949 w_k1_MAT_s006
                                 + 0.0114291443705385 w_k1_MAT_s007
                                 + 0.0202971736164257 w_k1_MAT_s010
                                 + 0.0189328680606859 w_k1_MAT_s011
                                 + 0.00468155079612049 w_k1_MAT_s012
                                 + 0.00682721383195197 w_k1_MAT_s013
                                 + 0.010930576921677 w_k1_MAT_s014
                                 + 0.00677809094669508 w_k1_MAT_s015
                                 + 0.0193392366476653 w_k1_MAT_s016
                                 + 0.00962766818082444 w_k1_MAT_s017
                                 + 0.0335827864634418 w_k1_MAT_s100
                                 + 0.0123632759934729 w_k1_MAT_s101
                                 + 0.0294068763284199 w_k1_MAT_s102
                                 + 0.0240489262160881 w_k1_MAT_s103
                                 + 0.0503647935051074 w_k1_MAT_s104
                                 + 0.0116513082868552 w_k1_MAT_s105
                                 + 0.021755254876412 w_k1_MAT_s106
                                 + 0.0150211661366481 w_k1_MAT_s107
                                 + 0.1044982029486 w_k1_MAT_s110
                                 + 0.101265745468397 w_k1_MAT_s111
                                 + 0.106246265250425 w_k1_MAT_s112
                                 + 0.0410345330653257 w_k1_MAT_s113
                                 + 0.0431191119716933 w_k1_MAT_s114
                                 + 0.0676063087107972 w_k1_MAT_s115
                                 + 0.0502470615949512 w_k1_MAT_s116
                                 + 0.056123290147589 w_k1_MAT_s117
                                 + 0.0279526210808889 w_k1_MSFT_s000
                                 + 0.0174444332246229 w_k1_MSFT_s001
                                 + 0.0293627937322487 w_k1_MSFT_s002
                                 + 0.0162125742800928 w_k1_MSFT_s003
                                 + 0.027912955953198 w_k1_MSFT_s004
                                 + 0.01055270480101 w_k1_MSFT_s005
                                 + 0.0108033813450657 w_k1_MSFT_s006
                                 + 0.0101304751038133 w_k1_MSFT_s007
                                 + 0.0209318473372448 w_k1_MSFT_s010
                                 + 0.0164198173784608 w_k1_MSFT_s011
                                 + 0.00406675417089854 w_k1_MSFT_s012
                                 + 0.00712216730208786 w_k1_MSFT_s013
                                 + 0.010888173665225 w_k1_MSFT_s014
                                 + 0.00850740484642906 w_k1_MSFT_s015
                                 + 0.022372792554099 w_k1_MSFT_s016
                                 + 0.00884082736587049 w_k1_MSFT_s017
                                 + 0.0377080146108192 w_k1_MSFT_s100
                                 + 0.014599276467888 w_k1_MSFT_s101
                                 + 0.0273207583110369 w_k1_MSFT_s102
                                 + 0.0269029546637773 w_k1_MSFT_s103
                                 + 0.046195144776052 w_k1_MSFT_s104
                                 + 0.011429007139308 w_k1_MSFT_s105
                                 + 0.0203162932491351 w_k1_MSFT_s106
                                 + 0.0107746359456537 w_k1_MSFT_s107
                                 + 0.106575487985073 w_k1_MSFT_s110
                                 + 0.123658249807176 w_k1_MSFT_s111
                                 + 0.0964010492570512 w_k1_MSFT_s112
                                 + 0.0361483387935195 w_k1_MSFT_s113
                                 + 0.049386881378785 w_k1_MSFT_s114
                                 + 0.0693610958259073 w_k1_MSFT_s115
                                 + 0.0435029834426255 w_k1_MSFT_s116
                                 + 0.0519612154325812 w_k1_MSFT_s117
                                 + 0.0257317331124926 w_k1_PG_s000
                                 + 0.0158850425833697 w_k1_PG_s001
                                 + 0.0300633307710146 w_k1_PG_s002
                                 + 0.0158684585916262 w_k1_PG_s003
                                 + 0.0264500920210556 w_k1_PG_s004
                                 + 0.010823412833202 w_k1_PG_s005
                                 + 0.0110403658389668 w_k1_PG_s006
                                 + 0.00869209403774012 w_k1_PG_s007
                                 + 0.0197705234878088 w_k1_PG_s010
                                 + 0.0188230473937734 w_k1_PG_s011
                                 + 0.00366916043218138 w_k1_PG_s012
                                 + 0.00708157044783409 w_k1_PG_s013
                                 + 0.0105844256068613 w_k1_PG_s014
                                 + 0.0081278086440517 w_k1_PG_s015
                                 + 0.0180269212582377 w_k1_PG_s016
                                 + 0.0104779809788346 w_k1_PG_s017
                                 + 0.0359136425063795 w_k1_PG_s100
                                 + 0.0156933379476939 w_k1_PG_s101
                                 + 0.0303681858906977 w_k1_PG_s102
                                 + 0.0296429077024333 w_k1_PG_s103
                                 + 0.0459852591746336 w_k1_PG_s104
                                 + 0.0110779295933403 w_k1_PG_s105
                                 + 0.019505621128321 w_k1_PG_s106
                                 + 0.0118035846927673 w_k1_PG_s107
                                 + 0.110273382385674 w_k1_PG_s110
                                 + 0.118653166284644 w_k1_PG_s111
                                 + 0.0981303345338319 w_k1_PG_s112
                                 + 0.0375540761434522 w_k1_PG_s113
                                 + 0.0500058474814075 w_k1_PG_s114
                                 + 0.0674431834436504 w_k1_PG_s115
                                 + 0.041358805559529 w_k1_PG_s116
                                 + 0.0517070396973299 w_k1_PG_s117 >= 1.08
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
