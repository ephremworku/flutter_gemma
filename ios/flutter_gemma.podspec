Pod::Spec.new do |s|
  s.name             = 'flutter_gemma'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter project.'
  s.description      = <<-DESC
    A new Flutter project.
  DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.resources        = ['Assets/**.*']
  s.resource_bundles = {
    'flutter_gemma_resources' => ['Assets/**/*']
  }
  s.dependency       'Flutter'
#   s.dependency       'MediaPipeTasksGenAI', '= 0.10.21'
#   s.dependency       'MediaPipeTasksGenAIC', '= 0.10.21'
  s.dependency       'MediaPipeTasksText', '= 0.10.21'
  s.platform         = :ios, '13.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }
  s.swift_version = '5.0'
end